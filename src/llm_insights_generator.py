"""
llm_insights_generator.py
─────────────────────────
Reads a Section-3 JSON report (Q11-Q14), extracts a concise analytics
summary, calls the Gemini API to generate HR-expert insights, and
overwrites the "llm_insights" field in the same file.

Usage
─────
    # Single file
    python src/llm_insights_generator.py reports/section3/q12_conflict_teamwork_contradiction.json

    # All Q11-Q14 files at once
    python src/llm_insights_generator.py --all

    # Custom inter-file delay (seconds)
    python src/llm_insights_generator.py --all --delay 20

Environment
───────────
    Set GEMINI_API_KEY in a .env file at project root, or as a real env var.

Rate-limit safety
─────────────────
    * Exponential back-off with jitter on 429 / 503 errors
    * INTER_FILE_DELAY_SEC between files when --all is used
    * RETRY_ATTEMPTS retries per request before giving up
"""

from __future__ import annotations

import json
import ast
import os
import sys
import time
import random
import re
import argparse
import logging
from pathlib import Path
from typing import Any

# ── Load .env before anything else ───────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    # dotenv not installed — rely on real env vars
    pass

from google import genai
from google.genai import errors, types

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
# Accept a comma-separated list so we can fall back across models if one is unavailable.
DEFAULT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
]


def _build_model_candidates() -> list[str]:
    configured = [m.strip() for m in os.environ.get("GEMINI_MODEL_NAME", "").split(",") if m.strip()]
    if not configured:
        return DEFAULT_MODEL_CANDIDATES.copy()

    strict_only = os.environ.get("GEMINI_STRICT_MODEL_ONLY", "").strip().lower() in {"1", "true", "yes"}
    if strict_only:
        return configured

    # If a single model is configured, keep it first but still include resilient fallbacks.
    merged = configured + [m for m in DEFAULT_MODEL_CANDIDATES if m not in configured]
    return merged


MODEL_CANDIDATES    = _build_model_candidates()
RETRY_ATTEMPTS      = 5
RETRY_BASE_DELAY    = 12      # seconds (doubles on each attempt)
RETRY_MAX_DELAY     = 120     # seconds cap
INTER_FILE_DELAY    = 15      # seconds between files when --all is used
MIN_REQUEST_GAP_SEC = float(os.environ.get("GEMINI_MIN_REQUEST_GAP_SEC", "12"))
OVERLOAD_RETRY_LIMIT = int(os.environ.get("GEMINI_OVERLOAD_RETRY_LIMIT", "2"))

BASE_DIR      = Path(__file__).resolve().parents[1]
SECTION3_DIR  = BASE_DIR / "reports" / "section3"


# ── Gemini client ─────────────────────────────────────────────────────────────
def _init_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.error("GEMINI_API_KEY is not set. Add it to .env or set it as an environment variable.")
        sys.exit(1)
    return genai.Client(api_key=api_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY EXTRACTORS
#  Each extractor returns a concise, structured text block.
#  ONLY statistics / correlations / counts / deltas are included.
#  Raw employee arrays are intentionally excluded.
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_q11(data: dict) -> str:
    """Q11 - Soft Skill Cluster Analysis."""
    clusters = data.get("clusters", [])
    lines = ["=== Q11: Soft-Skill Cluster Analysis (KMeans, k=4) ==="]
    meth = data.get("methodology", {})
    lines.append(f"Algorithm  : {meth.get('algorithm', 'N/A')}")
    lines.append(f"Features   : {', '.join(meth.get('features', []))}")
    lines.append(f"Rationale  : {meth.get('cluster_count_rationale', '')}\n")

    lines.append("Cluster Profiles:")
    for c in clusters:
        lines.append(
            f"  [{c.get('cluster_label')} | {c.get('name')}]"
            f"  n={c.get('size')} ({c.get('size_pct')}%)"
            f"  | Attrition={c.get('attrition_rate_pct')}%"
            f"  | AvgSoftScore={c.get('avg_soft_score')}"
            f"  | Leadership={c.get('leadership')}"
            f"  | Teamwork={c.get('teamwork')}"
            f"  | Adaptability={c.get('adaptability')}"
            f"  | Creativity={c.get('creativity')}"
            f"  | Performance={c.get('performance_rating')}"
            f"  | Engagement={c.get('engagement_score')}"
            f"  | Promotions={c.get('promotions')}"
        )
    return "\n".join(lines)


def _extract_q12(data: dict) -> str:
    """Q12 - High Conflict / Low Teamwork Contradiction."""
    lines = ["=== Q12: Conflict-Teamwork Contradiction Analysis ==="]

    th = data.get("thresholds", {})
    lines.append(f"High-conflict threshold (Q75) : {th.get('high_conflict_q75')}")
    lines.append(f"Low-teamwork  threshold (Q25) : {th.get('low_teamwork_q25')}\n")

    cnts = data.get("counts", {})
    lines.append(
        f"Contradictory employees : {cnts.get('contradictory_employees')}"
        f" ({cnts.get('contradictory_pct')}% of {cnts.get('total_employees')} total)"
    )

    attr = data.get("attrition_rates", {})
    lines.append(
        f"\nAttrition — Contradictory    : {attr.get('contradictory_group_pct')}%"
        f"\nAttrition — Non-Contradictory: {attr.get('non_contradictory_group_pct')}%"
        f"\nAttrition Delta              : {attr.get('delta')} pp"
    )

    pc = data.get("profile_comparison", {})
    delta = pc.get("delta", {})
    if delta:
        lines.append("\nKey deltas (Contradictory minus Non-Contradictory):")
        for k, v in delta.items():
            lines.append(f"  {k:45s}: {v:+.3f}")

    ratio = data.get("conflict_to_teamwork_ratio", {})
    rs = ratio.get("ratio_stats", {})
    if rs:
        lines.append(
            f"\nConflict-to-Teamwork Ratio — "
            f"mean={rs.get('mean')}, std={rs.get('std')}, "
            f"Q25={rs.get('25%')}, median={rs.get('50%')}, Q75={rs.get('75%')}"
        )

    dept = data.get("department_breakdown", [])
    if dept:
        lines.append("\nTop departments by contradictory count:")
        for row in dept[:5]:
            lines.append(
                f"  {row.get('Department'):12s}: n={row.get('count')}"
                f"  avg_conflict={row.get('avg_conflict')}"
                f"  avg_teamwork={row.get('avg_teamwork')}"
                f"  avg_performance={row.get('avg_performance')}"
            )

    archetypes = [a.get("name") for a in data.get("contradiction_archetypes", [])]
    if archetypes:
        lines.append("\nIdentified contradiction archetypes: " + " | ".join(archetypes))

    return "\n".join(lines)


def _extract_q13(data: dict) -> str:
    """Q13 - Engagement Score Impact on Satisfaction & Retention."""
    lines = ["=== Q13: Engagement -> Satisfaction & Retention Impact ==="]

    kc = data.get("key_correlations", {})
    lines.append(f"Correlation — Engagement vs Satisfaction : r = {kc.get('engagement_vs_satisfaction')}")
    lines.append(f"Correlation — Engagement vs Retention    : r = {kc.get('engagement_vs_retention')}\n")

    ret = data.get("retention_analysis", {})
    lines.append(f"Retention — Very High band : {ret.get('very_high_band_retention_pct')}%")
    lines.append(f"Retention — Very Low  band : {ret.get('very_low_band_retention_pct')}%")
    lines.append(f"Lift (Very High vs Very Low): {ret.get('retention_lift_pct_points')} pp")
    lines.append(
        f"Attrition — Disengaged bottom-25%  : {ret.get('disengaged_bottom25_attrition_pct')}%"
        f"  (n={ret.get('disengaged_count')})"
    )
    lines.append(
        f"Attrition — Highly-engaged top-25% : {ret.get('highly_engaged_top25_attrition_pct')}%"
        f"  (n={ret.get('highly_engaged_count')})"
    )

    bands = data.get("engagement_band_profile", [])
    if bands:
        lines.append("\nEngagement-band profile (Very Low -> Very High):")
        for b in bands:
            lines.append(
                f"  {b.get('Engagement_Band'):9s}: n={b.get('count')}"
                f"  avg_eng={b.get('avg_engagement')}"
                f"  avg_sat={b.get('avg_satisfaction')}"
                f"  retention={b.get('retention_rate')}"
                f"  avg_perf={b.get('avg_performance')}"
                f"  avg_wlb={b.get('avg_wlb')}"
            )

    drivers = data.get("engagement_drivers", {})
    if drivers:
        lines.append("\nEngagement feature correlations (sorted by |r|):")
        for feat, corr in sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True):
            lines.append(f"  {feat:45s}: {corr:+.4f}")

    rri = data.get("retention_risk_index", {})
    stats = rri.get("stats", {})
    if stats:
        lines.append(
            f"\nRetention Risk Index — "
            f"mean={stats.get('mean')}, std={stats.get('std')}, "
            f"Q25={stats.get('25%')}, median={stats.get('50%')}, "
            f"Q75={stats.get('75%')}, max={stats.get('max')}"
        )
        lines.append(f"Formula: {rri.get('formula')}")

    return "\n".join(lines)


def _extract_q14(data: dict) -> str:
    """Q14 - High Initiative + Low Innovation Gap Analysis."""
    lines = ["=== Q14: Initiative-Innovation Gap Analysis ==="]

    th = data.get("thresholds", {})
    lines.append(f"High-initiative threshold (Q75): {th.get('high_initiative_q75')}")
    lines.append(f"Low-innovation definition      : {th.get('low_innovation_definition')}\n")

    cnts = data.get("counts", {})
    lines.append(
        f"Gap employees    : {cnts.get('gap_employees')} ({cnts.get('gap_pct')}%)"
        f"  |  Aligned: {cnts.get('aligned_employees')}"
        f"  |  Total: {cnts.get('total_employees')}"
    )

    attr = data.get("attrition", {})
    lines.append(
        f"\nAttrition — Gap group     : {attr.get('gap_group_pct')}%"
        f"\nAttrition — Aligned group : {attr.get('aligned_group_pct')}%"
        f"\nAttrition — Overall       : {attr.get('overall_pct')}%"
        f"\nDelta (Gap vs Aligned)    : {attr.get('delta_vs_aligned')} pp"
    )

    pc = data.get("profile_comparison", {})
    delta = pc.get("delta", {})
    if delta:
        lines.append("\nKey deltas (Gap group minus Aligned):")
        for k, v in delta.items():
            lines.append(f"  {k:45s}: {v:+.3f}")

    gm = data.get("innovation_gap_metrics", {})
    gs = gm.get("gap_stats", {})
    if gs:
        lines.append(
            f"\nInnovation Gap Score — "
            f"mean={gs.get('mean')}, std={gs.get('std')}, "
            f"Q25={gs.get('25%')}, median={gs.get('50%')}, "
            f"Q75={gs.get('75%')}, max={gs.get('max')}"
        )

    dept = data.get("department_breakdown", [])
    if dept:
        lines.append("\nDepartment breakdown (gap employees):")
        for row in dept:
            lines.append(
                f"  {row.get('Department'):12s}: n={row.get('count')}"
                f"  avg_initiative={row.get('avg_initiative')}"
                f"  avg_engagement={row.get('avg_engagement')}"
                f"  avg_wlb={row.get('avg_wlb')}"
            )

    blockers = [b.get("blocker") for b in data.get("innovation_blockers", [])]
    if blockers:
        lines.append("\nIdentified innovation blockers: " + " | ".join(blockers))

    return "\n".join(lines)


# ── Dispatcher ────────────────────────────────────────────────────────────────
_EXTRACTORS = {
    "q11": _extract_q11,
    "q12": _extract_q12,
    "q13": _extract_q13,
    "q14": _extract_q14,
}


def build_summary_text(json_path: Path, data: dict) -> str:
    """Auto-detect question from filename and return concise summary text."""
    stem = json_path.stem.lower()
    for key, extractor in _EXTRACTORS.items():
        if stem.startswith(key):
            return extractor(data)
    return _generic_extractor(json_path, data)


def _generic_extractor(json_path: Path, data: dict) -> str:
    """Fallback: pull top-level scalars/dicts, skip large raw arrays."""
    skip_keys = {"llm_insights", "sample_employees", "top20_wasted_potential",
                 "top20_at_risk", "department_breakdown"}
    lines = [f"=== {data.get('question', json_path.stem)} ==="]
    for k, v in data.items():
        if k in skip_keys:
            continue
        if isinstance(v, (str, int, float, bool)):
            lines.append(f"{k}: {v}")
        elif isinstance(v, dict):
            lines.append(f"\n{k}:")
            for kk, vv in v.items():
                if isinstance(vv, (str, int, float, bool)):
                    lines.append(f"  {kk}: {vv}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

SYSTEM_INSTRUCTION = (
    "You are an expert HR analytics consultant with 20+ years of experience "
    "in workforce analytics, organizational behaviour, and people strategy. "
    "You generate deep, non-obvious insights from structured HR data. "
    "Your analysis goes beyond the obvious and always identifies at least one "
    "counter-intuitive or hidden signal that a typical analyst would miss. "
    "You ALWAYS respond with ONLY valid JSON — no markdown fences, no preamble."
)

USER_PROMPT_TEMPLATE = """\
Context:
{summary}

Task:
Based ONLY on the structured statistics above:
1. Identify key patterns or contradictions
2. Explain WHY they exist — include behavioral and organizational reasoning
3. Highlight at least one non-obvious or hidden insight
4. Provide a strong, actionable business implication

Respond with ONLY this JSON (no markdown, no extra text):
{{
  "headline": "One powerful sentence summarising the single most important finding",
  "key_insight_1": "First key pattern with root-cause explanation",
  "key_insight_2": "Second key pattern with root-cause explanation",
  "hidden_insight": "A non-obvious insight a typical analyst would miss",
  "business_implication": "Specific, actionable recommendation for HR leadership"
}}
"""


def build_prompt(summary_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(summary=summary_text)


LLM_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "key_insight_1": {"type": "string"},
        "key_insight_2": {"type": "string"},
        "hidden_insight": {"type": "string"},
        "business_implication": {"type": "string"},
    },
    "required": [
        "headline",
        "key_insight_1",
        "key_insight_2",
        "hidden_insight",
        "business_implication",
    ],
}

REQUIRED_INSIGHT_KEYS = tuple(LLM_RESPONSE_SCHEMA["required"])
_last_request_ts = 0.0


def _extract_json_object(raw_text: str) -> str:
    """Extract the first JSON object from a response that may include wrappers."""
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start : end + 1]
    return raw_text


def _strip_markdown_fences(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    return text


def _repair_json_text(raw_text: str) -> str:
    """Best-effort cleanup for common LLM JSON issues."""
    cleaned = _extract_json_object(_strip_markdown_fences(raw_text))
    cleaned = (cleaned
               .replace("\u201c", "\"")
               .replace("\u201d", "\"")
               .replace("\u2018", "'")
               .replace("\u2019", "'"))

    # Remove trailing commas before object/array endings.
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Quote unquoted object keys: {key: "..."} or , key: "..."
    cleaned = re.sub(
        r'(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_\- ]*)(?P<colon>\s*:)',
        lambda m: f'{m.group("prefix")}"{m.group("key").strip()}"{m.group("colon")}',
        cleaned,
    )
    return cleaned


def _validate_insights_schema(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ValueError("Gemini response is not a JSON object.")

    normalized: dict[str, str] = {}
    for key in REQUIRED_INSIGHT_KEYS:
        value = payload.get(key)
        if value is None:
            raise ValueError(f"Missing required key '{key}' in LLM response.")
        normalized[key] = str(value).strip()
    return normalized


def _parse_insights_response(response: Any) -> dict[str, str]:
    # First preference: parsed object provided by SDK when response_schema is used.
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            parsed = parsed.model_dump()
        return _validate_insights_schema(parsed)

    raw_text = (getattr(response, "text", "") or "").strip()
    if not raw_text:
        raise ValueError("Gemini response contained no text payload.")

    raw_text = _repair_json_text(raw_text)
    try:
        return _validate_insights_schema(json.loads(raw_text))
    except json.JSONDecodeError:
        # Fallback for Python-style dict strings (single quotes, True/False/None).
        pythonish_text = re.sub(r"\btrue\b", "True", raw_text, flags=re.IGNORECASE)
        pythonish_text = re.sub(r"\bfalse\b", "False", pythonish_text, flags=re.IGNORECASE)
        pythonish_text = re.sub(r"\bnull\b", "None", pythonish_text, flags=re.IGNORECASE)
        return _validate_insights_schema(ast.literal_eval(pythonish_text))


def _throttle_request_rate() -> None:
    """Avoid exceeding low free-tier RPM limits (default ~=5 RPM)."""
    global _last_request_ts
    if MIN_REQUEST_GAP_SEC <= 0:
        return

    now = time.time()
    elapsed = now - _last_request_ts
    if _last_request_ts and elapsed < MIN_REQUEST_GAP_SEC:
        wait_s = round(MIN_REQUEST_GAP_SEC - elapsed, 1)
        log.info(f"  Throttling {wait_s}s to respect model RPM limits.")
        time.sleep(wait_s)
    _last_request_ts = time.time()


def _recommended_wait_seconds(attempt: int, error_message: str) -> float:
    retry_after = re.search(r"retry(?:[- ]after)?\D+(\d+(?:\.\d+)?)\s*s", error_message, re.IGNORECASE)
    if retry_after:
        base = float(retry_after.group(1))
        jitter = random.uniform(0, max(base * 0.1, 0.5))
        return round(base + jitter, 1)

    base_delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
    jitter = random.uniform(0, max(base_delay * 0.2, 0.5))
    return round(base_delay + jitter, 1)


def _is_retryable_api_error(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in {429, 500, 502, 503, 504}
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "quota", "rate", "503", "unavailable", "resource exhausted"))


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM CALL  -  rate-limit safe with exponential back-off + jitter
# ═══════════════════════════════════════════════════════════════════════════════

def generate_llm_insights(summary_text: str, client: genai.Client) -> dict:
    """
    Call the Gemini API with the summary text.

    Returns a dict with structured insights, or an error dict on failure.
    Implements exponential back-off with jitter for 429/503 rate-limit errors.
    """
    prompt = build_prompt(summary_text)

    last_error = ""

    for model_name in MODEL_CANDIDATES:
        log.info(f"  Using Gemini model: {model_name}")
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                _throttle_request_rate()
                log.info(f"  -> Calling Gemini [{model_name}], attempt {attempt}/{RETRY_ATTEMPTS} ...")

                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.2,
                        max_output_tokens=1024,
                        response_mime_type="application/json",
                        response_schema=LLM_RESPONSE_SCHEMA,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    ),
                )

                insights = _parse_insights_response(response)
                log.info("  OK  Insights generated successfully.")
                return insights

            except (json.JSONDecodeError, ValueError) as parse_err:
                last_error = f"{type(parse_err).__name__}: {parse_err}"
                raw_text = (getattr(locals().get("response"), "text", "") or "").strip()
                preview = raw_text.replace("\n", " ")[:260] if raw_text else "<empty>"
                log.error(
                    f"  Invalid JSON payload on attempt {attempt} for model {model_name}: {parse_err}. "
                    f"Payload preview: {preview!r}"
                )
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(3)
                    continue
                break

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                err_text = str(e)

                if _is_retryable_api_error(e):
                    is_api_error = isinstance(e, errors.APIError)
                    err_lower = err_text.lower()

                    # Quota-exhausted models usually won't recover immediately; switch models quickly.
                    if is_api_error and e.code == 429 and "quota" in err_lower:
                        log.warning(
                            f"  Quota exhausted for model {model_name}. "
                            "Switching to next candidate model."
                        )
                        break

                    # Persistent 503 overload on one model: fail over sooner instead of spending all attempts.
                    if is_api_error and e.code == 503 and attempt >= max(1, OVERLOAD_RETRY_LIMIT):
                        log.warning(
                            f"  Model {model_name} appears overloaded after {attempt} attempt(s). "
                            "Switching to next candidate model."
                        )
                        break

                    if attempt >= RETRY_ATTEMPTS:
                        break
                    wait_s = _recommended_wait_seconds(attempt, err_text)
                    log.warning(
                        f"  Model {model_name} unavailable or rate-limited ({type(e).__name__}). "
                        f"Waiting {wait_s}s before retry {attempt + 1} ..."
                    )
                    time.sleep(wait_s)
                    continue

                log.error(f"  Unexpected error on attempt {attempt} for model {model_name}: {type(e).__name__}: {e}")
                if attempt < RETRY_ATTEMPTS:
                    time.sleep(5)
                    continue
                break

        log.warning(f"  Falling back from model {model_name} to the next candidate (if any).")

    return {
        "error": (
            "All model candidates and retry attempts exhausted without a successful response."
            + (f" Last error: {last_error}" if last_error else "")
        )
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(json_path: Path, client: genai.Client) -> bool:
    """
    Full pipeline for a single JSON file:
      read -> extract summary -> call LLM -> update json -> save

    Returns True on success, False on failure.
    """
    log.info(f"Processing: {json_path.name}")

    # 1. Read
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except Exception as e:
        log.error(f"  Failed to read {json_path}: {e}")
        return False

    # 2. Extract summary (no raw data sent to LLM)
    summary_text = build_summary_text(json_path, data)
    log.info(f"  Summary length: {len(summary_text)} chars")

    # 3. Call Gemini
    insights = generate_llm_insights(summary_text, client)

    if "error" in insights:
        log.error(f"  LLM call failed: {insights}")
        return False

    # 4. Update the llm_insights field
    data["llm_insights"] = insights

    # 5. Save (overwrite original file)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"  Saved -> {json_path}")
        return True
    except Exception as e:
        log.error(f"  Failed to save {json_path}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_section3_files() -> list[Path]:
    """Return all Q11-Q14 JSON files from reports/section3/."""
    if not SECTION3_DIR.exists():
        log.error(f"Section-3 reports dir not found: {SECTION3_DIR}")
        return []
    return sorted(SECTION3_DIR.glob("q1[1-4]_*.json"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemini LLM insights for Section-3 HR analytics JSON files (Q11-Q14)."
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Path(s) to specific JSON file(s) to process.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all Q11-Q14 JSON files in reports/section3/.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=INTER_FILE_DELAY,
        metavar="SECONDS",
        help=f"Wait between files when using --all (default: {INTER_FILE_DELAY}s).",
    )
    args = parser.parse_args()

    client = _init_client()

    if args.all:
        paths = _discover_section3_files()
        if not paths:
            log.error("No Q11-Q14 JSON files found.")
            sys.exit(1)
        log.info(f"Found {len(paths)} file(s): {[p.name for p in paths]}")
    elif args.files:
        paths = [Path(p) for p in args.files]
    else:
        parser.print_help()
        sys.exit(0)

    results: dict[str, bool] = {}
    for idx, path in enumerate(paths):
        success = process_file(path, client)
        results[path.name] = success

        # Rate-limit buffer between files (skip after last file)
        if idx < len(paths) - 1:
            log.info(f"  Waiting {args.delay}s before next file ...\n")
            time.sleep(args.delay)

    # Final summary
    log.info("\n============= RESULTS =============")
    for fname, ok in results.items():
        status = "OK     " if ok else "FAILED "
        log.info(f"  {status}  {fname}")
    log.info("====================================\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
