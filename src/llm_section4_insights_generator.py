"""
Generate high-impact Gemini insights for Section 4 reports (Q15, Q16, Q17, Q18) using compact evidence payloads.

Usage:
  python src/llm_section4_insights_generator.py --all
  python src/llm_section4_insights_generator.py reports/section4/q15_project_complexity_size_impact.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from google import genai
from google.genai import errors, types


BASE = Path(__file__).resolve().parents[1]
SECTION4_DIR = BASE / "reports" / "section4"

if load_dotenv:
    load_dotenv(BASE / ".env")

DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]
PRIMARY_MODEL = "gemini-2.5-flash"
SECONDARY_MODEL = "gemini-2.5-flash-lite"

RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_SEC = 10
RETRY_MAX_DELAY_SEC = 80
MIN_REQUEST_GAP_SEC = float(os.environ.get("GEMINI_MIN_REQUEST_GAP_SEC", "13"))
INTER_FILE_DELAY_SEC = int(os.environ.get("GEMINI_INTER_FILE_DELAY_SEC", "12"))
OVERLOAD_RETRY_LIMIT = int(os.environ.get("GEMINI_OVERLOAD_RETRY_LIMIT", "3"))
PARSE_RETRY_LIMIT = int(os.environ.get("GEMINI_PARSE_RETRY_LIMIT", "4"))
MAX_EVIDENCE_CHARS = int(os.environ.get("GEMINI_MAX_EVIDENCE_CHARS", "5000"))
ALLOW_PARTIAL_INSIGHTS = os.environ.get("GEMINI_ALLOW_PARTIAL_INSIGHTS", "0").strip().lower() in {"1", "true", "yes"}
FALLBACK_TEXT = "Derived with partial model output; verify before publishing."

INSIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "key_insight_1": {"type": "string"},
        "key_insight_2": {"type": "string"},
        "hidden_insight": {"type": "string"},
        "risk_alert": {"type": "string"},
        "business_implication": {"type": "string"},
        "action_plan": {"type": "array", "items": {"type": "string"}},
        "standout_statement": {"type": "string"},
        "confidence_note": {"type": "string"},
    },
    "required": [
        "headline",
        "key_insight_1",
        "key_insight_2",
        "hidden_insight",
        "risk_alert",
        "business_implication",
        "action_plan",
        "standout_statement",
        "confidence_note",
    ],
}

SYSTEM_INSTRUCTION = (
    "You are a principal HR analytics strategist preparing competition-grade insights. "
    "You must use only provided evidence, cite concrete numbers inside each insight, "
    "and avoid speculation without a data anchor. Always include one non-obvious insight "
    "and one high-stakes risk signal. Never claim causality from correlational evidence. "
    "Use the phrase 'statistically significant' only when p-value < 0.05 is explicitly provided. "
    "Respond with valid JSON only."
)

PROMPT_TEMPLATE = """Question:
{question}

Evidence (compact JSON):
{evidence_json}

Rules:
1) Use only the evidence above.
2) Quote concrete figures (rates, deltas, lift, effect sizes, p-values) in each insight where available.
3) Explain likely organizational mechanisms behind the patterns.
4) Prioritize strategic implications for HR and delivery leadership.
5) If evidence is correlational, explicitly mark uncertainty in confidence_note.
6) Use 'statistically significant' only when p-value < 0.05; otherwise use wording like 'directional' or 'indicative'.

Return ONLY JSON matching the requested schema.
"""

_last_request_ts = 0.0


def _model_candidates(preferred_model: str | None = None, strict_model: bool = False) -> list[str]:
    if preferred_model:
        if strict_model:
            return [preferred_model]
        return [preferred_model] + [m for m in DEFAULT_MODEL_CANDIDATES if m != preferred_model]

    configured = [m.strip() for m in os.environ.get("GEMINI_MODEL_NAME", "").split(",") if m.strip()]
    if configured:
        if strict_model:
            return configured
        merged = configured + [m for m in DEFAULT_MODEL_CANDIDATES if m not in configured]
    else:
        merged = DEFAULT_MODEL_CANDIDATES.copy()

    # Keep preferred defaults first unless a preferred model is explicitly passed.
    if PRIMARY_MODEL in merged:
        merged = [PRIMARY_MODEL] + [m for m in merged if m != PRIMARY_MODEL]
    else:
        merged = [PRIMARY_MODEL] + merged
    if SECONDARY_MODEL in merged:
        merged = [merged[0], SECONDARY_MODEL] + [m for m in merged[1:] if m != SECONDARY_MODEL]
    return merged


def _init_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in environment or .env.")
    return genai.Client(api_key=api_key)


def _throttle(verbose: bool = False) -> None:
    global _last_request_ts
    if MIN_REQUEST_GAP_SEC <= 0:
        return
    now = time.time()
    if _last_request_ts:
        elapsed = now - _last_request_ts
        if elapsed < MIN_REQUEST_GAP_SEC:
            wait_s = MIN_REQUEST_GAP_SEC - elapsed
            if verbose:
                print(f"    [wait] throttling for {wait_s:.1f}s to respect RPM limits")
            time.sleep(wait_s)
    _last_request_ts = time.time()


def _retry_wait(attempt: int, message: str) -> float:
    m = re.search(r"retry(?:[- ]after)?\D+(\d+(?:\.\d+)?)\s*s", message, flags=re.IGNORECASE)
    if m:
        base = float(m.group(1))
        return round(base + random.uniform(0.4, 1.6), 1)
    base = min(RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1)), RETRY_MAX_DELAY_SEC)
    return round(base + random.uniform(0.6, 2.4), 1)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in {429, 500, 502, 503, 504}
    text = str(exc).lower()
    return any(
        token in text
        for token in [
            "rate",
            "quota",
            "429",
            "503",
            "unavailable",
            "resource exhausted",
            "connecterror",
            "network",
            "socket",
            "timed out",
            "getaddrinfo",
            "winerror 10051",
            "winerror 10013",
        ]
    )


def _is_overload_503(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError) and exc.code == 503:
        return True
    text = str(exc).lower()
    return "503" in text and "unavailable" in text


def _is_hard_quota_exhausted(exc: Exception) -> bool:
    text = str(exc).lower()
    if "resource_exhausted" not in text and "quota exceeded" not in text and "429" not in text:
        return False
    hard_signals = [
        "limit: 0",
        "generaterequestsperdayperprojectpermodel-freetier",
        "generaterequestsperminuteperprojectpermodel-freetier",
        "generatecontentinputtokenspermodelperminute-freetier",
    ]
    return any(sig in text for sig in hard_signals)


def _is_parse_error(exc: Exception) -> bool:
    return isinstance(exc, (ValueError, SyntaxError, json.JSONDecodeError))


def _extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join([line for line in text.splitlines() if not line.strip().startswith("```")]).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _validate_insights(payload: Any) -> dict[str, Any]:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object.")
    return _coerce_insights(payload)


def _coerce_action_plan(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(x).strip(" -\t\r\n") for x in value if str(x).strip()]
    elif isinstance(value, str):
        raw_lines = re.split(r"[\n;]+", value)
        items = [line.strip(" -\t\r\n") for line in raw_lines if line.strip()]
    else:
        items = []

    if not items:
        items = [
            "Investigate root causes in the highest-risk segment first.",
            "Prioritize interventions on variables with strongest measured effect.",
            "Track post-intervention outcome lift with monthly monitoring.",
        ]
    return items[:6]


def _coerce_insights(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in INSIGHT_SCHEMA["required"]:
        if key == "action_plan":
            out[key] = _coerce_action_plan(payload.get(key))
            continue
        value = payload.get(key)
        if value is None or str(value).strip() == "":
            out[key] = FALLBACK_TEXT
        else:
            out[key] = str(value).strip()
    return out


def _extract_keyed_fields(raw_text: str) -> dict[str, Any]:
    # Best-effort extraction when JSON is malformed.
    keys = [
        "headline",
        "key_insight_1",
        "key_insight_2",
        "hidden_insight",
        "risk_alert",
        "business_implication",
        "standout_statement",
        "confidence_note",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        pattern = rf'"?{re.escape(key)}"?\s*:\s*"([^"]+)"'
        match = re.search(pattern, raw_text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            out[key] = match.group(1).strip()

    ap_match = re.search(r'"?action_plan"?\s*:\s*\[(.*?)\]', raw_text, flags=re.IGNORECASE | re.DOTALL)
    if ap_match:
        quoted = re.findall(r'"([^"]+)"', ap_match.group(1))
        if quoted:
            out["action_plan"] = [q.strip() for q in quoted if q.strip()]

    return out


def _placeholder_field_count(insights: dict[str, Any]) -> int:
    count = 0
    for k, v in insights.items():
        if k == "action_plan":
            continue
        if isinstance(v, str) and v.strip() == FALLBACK_TEXT:
            count += 1
    return count


def _parse_response(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        return _validate_insights(parsed)

    text = (getattr(response, "text", "") or "").strip()
    if not text:
        raise ValueError("Gemini returned empty response text.")
    cleaned = _extract_json_text(text)
    try:
        return _validate_insights(json.loads(cleaned))
    except json.JSONDecodeError:
        pythonish = re.sub(r"\btrue\b", "True", cleaned, flags=re.IGNORECASE)
        pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
        pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)
        try:
            return _validate_insights(ast.literal_eval(pythonish))
        except Exception:
            recovered = _extract_keyed_fields(cleaned)
            if recovered:
                recovered.setdefault("confidence_note", "Recovered from malformed model JSON; validate manually.")
                return _coerce_insights(recovered)
            raise


def _build_evidence_payload(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("llm_evidence")
    if isinstance(evidence, dict) and evidence:
        return evidence

    # Generic fallback if llm_evidence is missing (works across sections, not only Section 4).
    payload: dict[str, Any] = {"question": report.get("question")}
    skip_keys = {
        "llm_evidence",
        "llm_insights",
        "llm_reasoning",
        "llm_profile_description",
        "llm_generation_meta",
        "sample_employees",
        "top20_wasted_potential",
        "top20_at_risk",
        "top20_candidates",
    }

    preferred_keys = [
        "dataset_scope",
        "dataset_summary",
        "target_definition",
        "segment_sizes",
        "overall_baseline_rates_pct",
        "overall_outcome_baseline_pct",
        "overall_resignation_rate_pct",
        "modeling",
        "feature_influence",
        "factor_evidence",
        "numeric_comparison",
        "categorical_patterns",
        "interaction_patterns",
        "multivariable_models",
        "multivariable_regression",
        "pairwise_correlations",
        "correlations",
        "key_correlations",
        "quartile_analysis",
        "role_performance_profile",
        "performance_tests",
        "summary_signals",
    ]

    def _trim_value(key: str, value: Any) -> Any:
        if isinstance(value, list):
            if not value:
                return value
            # Large person-level lists are usually not helpful and can blow token budget.
            if key.lower().endswith("_employees"):
                return value[:10]
            if isinstance(value[0], dict):
                return value[:12]
            return value[:25]
        return value

    for key in preferred_keys:
        if key in report and key not in skip_keys:
            payload[key] = _trim_value(key, report[key])

    # Include additional top-level keys as secondary evidence if compact/structured.
    for key, value in report.items():
        if key in payload or key in skip_keys:
            continue
        if isinstance(value, (int, float, str, bool, dict, list)):
            payload[key] = _trim_value(key, value)

    return payload


def _compact_json_for_prompt(obj: dict[str, Any], max_chars: int = MAX_EVIDENCE_CHARS) -> str:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if len(text) <= max_chars:
        return text

    # Hard trim as a safety rail for token efficiency.
    trimmed = text[: max_chars - 200] + ',"_truncated_notice":"evidence_trimmed_for_token_budget"}'
    return trimmed


def _generate_insights(
    client: genai.Client,
    question: str,
    evidence_payload: dict[str, Any],
    model_candidates: list[str],
    verbose: bool = True,
) -> tuple[dict[str, Any], str]:
    prompt = PROMPT_TEMPLATE.format(
        question=question or "Section 4 analytics question",
        evidence_json=_compact_json_for_prompt(evidence_payload),
    )

    last_error = ""
    for model_name in model_candidates:
        if verbose:
            print(f"  [model] trying {model_name}")
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                if verbose:
                    print(f"    [call] attempt {attempt}/{RETRY_ATTEMPTS}")
                _throttle(verbose=verbose)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                        max_output_tokens=1100,
                        response_mime_type="application/json",
                        response_schema=INSIGHT_SCHEMA,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )
                insights = _parse_response(response)
                placeholders = _placeholder_field_count(insights)
                if placeholders > 0 and not ALLOW_PARTIAL_INSIGHTS:
                    if verbose:
                        print(
                            f"    [retry-partial] response recovered but {placeholders} fields are placeholders; "
                            "treating as invalid and retrying/falling back"
                        )
                    if attempt < RETRY_ATTEMPTS:
                        time.sleep(1.2)
                        continue
                    break
                if verbose:
                    print(f"    [ok] response parsed successfully from {model_name}")
                return insights, model_name
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if _is_parse_error(exc):
                    if attempt < min(RETRY_ATTEMPTS, max(1, PARSE_RETRY_LIMIT)):
                        if verbose:
                            print(
                                f"    [retry-parse] malformed model JSON on attempt {attempt}; "
                                "retrying same model quickly"
                            )
                        time.sleep(1.5)
                        continue
                    if verbose:
                        print(f"    [switch] parse failed for {model_name}; moving to next model")
                    break

                if _is_hard_quota_exhausted(exc):
                    if verbose:
                        print(
                            "    [skip-quota] hard quota exhaustion detected for this model; "
                            "skipping further retries"
                        )
                    break

                if _is_overload_503(exc) and attempt >= max(1, OVERLOAD_RETRY_LIMIT):
                    if verbose:
                        print(
                            f"    [switch] model appears overloaded after {attempt} attempt(s); "
                            "moving to next fallback model"
                        )
                    break
                if _is_retryable(exc) and attempt < RETRY_ATTEMPTS:
                    wait_s = _retry_wait(attempt, str(exc))
                    if verbose:
                        print(f"    [retry] {last_error} | waiting {wait_s}s")
                    time.sleep(wait_s)
                    continue
                if verbose:
                    print(f"    [fail] {last_error}")
                break
    raise RuntimeError(f"Unable to generate insights after retries. Last error: {last_error}")


def _discover_default_files() -> list[Path]:
    if not SECTION4_DIR.exists():
        return []
    q15 = sorted(SECTION4_DIR.glob("q15_*.json"))
    q16 = sorted(SECTION4_DIR.glob("q16_*.json"))
    q17 = sorted(SECTION4_DIR.glob("q17_*.json"))
    q18 = sorted(SECTION4_DIR.glob("q18_*.json"))
    return q15 + q16 + q17 + q18


def process_file(
    path: Path,
    client: genai.Client,
    model_candidates: list[str],
    verbose: bool = True,
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[FAIL] Could not read {path}: {exc}")
        return False

    question = str(report.get("question", path.stem))
    evidence = _build_evidence_payload(report)
    evidence_chars = len(_compact_json_for_prompt(evidence))
    if verbose:
        print(f"[RUN] {path.name}")
        print(f"  [info] evidence chars: {evidence_chars}")
        print(f"  [info] model order: {', '.join(model_candidates)}")

    try:
        insights, model_name = _generate_insights(
            client=client,
            question=question,
            evidence_payload=evidence,
            model_candidates=model_candidates,
            verbose=verbose,
        )
    except Exception as exc:
        print(f"[FAIL] Gemini generation failed for {path.name}: {exc}")
        return False

    report["llm_insights"] = insights
    report["llm_generation_meta"] = {
        "model_used": model_name,
        "generated_at_epoch": int(time.time()),
        "evidence_char_length": evidence_chars,
    }

    try:
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as exc:
        print(f"[FAIL] Could not save {path}: {exc}")
        return False

    print(f"[OK] Updated LLM insights -> {path}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Gemini insights for Section 4 Q15/Q16/Q17/Q18 reports.")
    parser.add_argument("files", nargs="*", help="Specific report files to process.")
    parser.add_argument("--all", action="store_true", help="Process all q15/q16/q17/q18 JSONs in reports/section4.")
    parser.add_argument("--delay", type=int, default=INTER_FILE_DELAY_SEC, help="Delay between files in seconds.")
    parser.add_argument("--model", type=str, default=None, help="Preferred Gemini model (e.g., gemini-2.5-flash).")
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="If set, only use --model (or GEMINI_MODEL_NAME list) and do not auto-fallback to defaults.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce progress output.")
    args = parser.parse_args()

    if args.all:
        targets = _discover_default_files()
    else:
        targets = [Path(p) for p in args.files]

    if not targets:
        print("No target files found. Run Q15/Q16/Q17/Q18 analysis scripts first, then rerun with --all.")
        sys.exit(1)

    model_candidates = _model_candidates(preferred_model=args.model, strict_model=args.strict_model)
    verbose = not args.quiet

    client = _init_client()
    all_ok = True
    for idx, target in enumerate(targets):
        ok = process_file(
            path=target,
            client=client,
            model_candidates=model_candidates,
            verbose=verbose,
        )
        all_ok = all_ok and ok
        if idx < len(targets) - 1 and args.delay > 0:
            if verbose:
                print(f"[wait] sleeping {args.delay}s before next file")
            time.sleep(args.delay)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
