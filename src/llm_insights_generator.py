"""
llm_insights_generator.py
─────────────────────────
Reads a Section-3 JSON report (Q11–Q14), extracts a concise analytics
summary, calls the Gemini API to generate HR-expert insights, and
overwrites the "llm_insights" field in the same file.

Usage
─────
    # Single file
    python src/llm_insights_generator.py reports/section3/q12_conflict_teamwork_contradiction.json

    # All Q11-Q14 files at once
    python src/llm_insights_generator.py --all

Environment variable required
─────────────────────────────
    GEMINI_API_KEY=<your_free_tier_key>

Rate-limit safety
─────────────────
    * Exponential back-off with jitter on 429 / 503 errors
    * Configurable INTER_FILE_DELAY_SEC between files when --all is used
    * Max RETRY_ATTEMPTS retries per file
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import argparse
import logging
from pathlib import Path
from typing import Any

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME          = "gemini-1.5-flash"   # Free-tier friendly; change to gemini-1.5-pro if needed
RETRY_ATTEMPTS      = 5
RETRY_BASE_DELAY    = 10      # seconds (doubles on each attempt)
RETRY_MAX_DELAY     = 120     # seconds cap
INTER_FILE_DELAY    = 15      # seconds between files when --all is used

BASE_DIR = Path(__file__).resolve().parents[1]           # project root
SECTION3_DIR = BASE_DIR / "reports" / "section3"

# ── Gemini client initialisation ──────────────────────────────────────────────
def _init_gemini() -> genai.GenerativeModel:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        log.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)


# ═══════════════════════════════════════════════════════════════════════════════
#  SUMMARY EXTRACTORS  –  one per question, returns a compact text block
#  ONLY summary statistics / correlations / counts / deltas are extracted.
#  Raw employee records and full arrays are intentionally excluded.
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_q11(data: dict) -> str:
    """Q11 – Soft Skill Cluster Analysis."""
    clusters = data.get("clusters", [])
    lines = ["=== Q11: Soft-Skill Cluster Analysis ==="]
    lines.append(f"Algorithm : {data.get('methodology', {}).get('algorithm', 'N/A')}")
    lines.append(f"Features  : {', '.join(data.get('methodology', {}).get('features', []))}\n")

    lines.append("Cluster Profiles:")
    for c in clusters:
        lines.append(
            f"  [{c.get('cluster_label')} / {c.get('name')}]"
            f"  n={c.get('size')} ({c.get('size_pct')}%)"
            f"  | Attrition: {c.get('attrition_rate_pct')}%"
            f"  | Avg soft score: {c.get('avg_soft_score')}"
            f"  | Leadership: {c.get('leadership')}"
            f"  | Teamwork: {c.get('teamwork')}"
            f"  | Adaptability: {c.get('adaptability')}"
            f"  | Creativity: {c.get('creativity')}"
            f"  | Performance: {c.get('performance_rating')}"
            f"  | Engagement: {c.get('engagement_score')}"
            f"  | Promotions: {c.get('promotions')}"
        )
    return "\n".join(lines)


def _extract_q12(data: dict) -> str:
    """Q12 – High Conflict / Low Teamwork Contradiction."""
    lines = ["=== Q12: Conflict-Teamwork Contradiction Analysis ==="]

    th = data.get("thresholds", {})
    lines.append(f"High-conflict threshold (Q75): {th.get('high_conflict_q75')}")
    lines.append(f"Low-teamwork  threshold (Q25): {th.get('low_teamwork_q25')}\n")

    cnts = data.get("counts", {})
    lines.append(
        f"Contradictory employees: {cnts.get('contradictory_employees')}"
        f" ({cnts.get('contradictory_pct')}% of {cnts.get('total_employees')} total)"
    )

    attr = data.get("attrition_rates", {})
    lines.append(
        f"\nAttrition — Contradictory: {attr.get('contradictory_group_pct')}%"
        f"  |  Non-Contradictory: {attr.get('non_contradictory_group_pct')}%"
        f"  |  Delta: {attr.get('delta')} pp"
    )

    pc = data.get("profile_comparison", {})
    delta = pc.get("delta", {})
    if delta:
        lines.append("\nKey deltas (Contradictory minus Non-Contradictory):")
        for k, v in delta.items():
            lines.append(f"  {k}: {v:+.3f}")

    ratio = data.get("conflict_to_teamwork_ratio", {})
    rs = ratio.get("ratio_stats", {})
    if rs:
        lines.append(
            f"\nConflict-to-Teamwork Ratio stats — "
            f"mean={rs.get('mean')}, std={rs.get('std')}, "
            f"Q25={rs.get('25%')}, median={rs.get('50%')}, Q75={rs.get('75%')}"
        )

    dept = data.get("department_breakdown", [])
    if dept:
        lines.append("\nTop departments (by contradictory count):")
        for row in dept[:5]:
            lines.append(
                f"  {row.get('Department')}: n={row.get('count')}"
                f"  avg_conflict={row.get('avg_conflict')}"
                f"  avg_teamwork={row.get('avg_teamwork')}"
                f"  avg_perf={row.get('avg_performance')}"
            )

    return "\n".join(lines)


def _extract_q13(data: dict) -> str:
    """Q13 – Engagement Score Impact on Satisfaction & Retention."""
    lines = ["=== Q13: Engagement → Satisfaction & Retention Impact ==="]

    kc = data.get("key_correlations", {})
    lines.append(f"Correlation — Engagement vs Satisfaction : {kc.get('engagement_vs_satisfaction')}")
    lines.append(f"Correlation — Engagement vs Retention    : {kc.get('engagement_vs_retention')}\n")

    ret = data.get("retention_analysis", {})
    lines.append(f"Retention — Very High engagement band : {ret.get('very_high_band_retention_pct')}%")
    lines.append(f"Retention — Very Low  engagement band : {ret.get('very_low_band_retention_pct')}%")
    lines.append(f"Lift (Very High vs Very Low)           : {ret.get('retention_lift_pct_points')} pp")
    lines.append(
        f"Attrition — Disengaged bottom-25%     : {ret.get('disengaged_bottom25_attrition_pct')}%"
        f"  (n={ret.get('disengaged_count')})"
    )
    lines.append(
        f"Attrition — Highly-engaged top-25%    : {ret.get('highly_engaged_top25_attrition_pct')}%"
        f"  (n={ret.get('highly_engaged_count')})"
    )

    bands = data.get("engagement_band_profile", [])
    if bands:
        lines.append("\nEngagement-band profile (ordered Very Low → Very High):")
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
        lines.append("\nEngagement correlations (top drivers):")
        sorted_d = sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True)
        for feature, corr in sorted_d:
            lines.append(f"  {feature}: {corr:+.4f}")

    rri = data.get("retention_risk_index", {})
    stats = rri.get("stats", {})
    if stats:
        lines.append(
            f"\nRetention Risk Index — mean={stats.get('mean')}, std={stats.get('std')}, "
            f"Q25={stats.get('25%')}, median={stats.get('50%')}, Q75={stats.get('75%')}, "
            f"max={stats.get('max')}"
        )

    return "\n".join(lines)


def _extract_q14(data: dict) -> str:
    """Q14 – High Initiative + Low Innovation Gap Analysis."""
    lines = ["=== Q14: Initiative-Innovation Gap Analysis ==="]

    th = data.get("thresholds", {})
    lines.append(f"High-initiative threshold (Q75): {th.get('high_initiative_q75')}")
    lines.append(f"Low-innovation definition      : {th.get('low_innovation_definition')}\n")

    cnts = data.get("counts", {})
    lines.append(
        f"Gap employees   : {cnts.get('gap_employees')} ({cnts.get('gap_pct')}%)"
        f"  |  Aligned: {cnts.get('aligned_employees')}"
        f"  |  Total: {cnts.get('total_employees')}"
    )

    attr = data.get("attrition", {})
    lines.append(
        f"\nAttrition — Gap group     : {attr.get('gap_group_pct')}%"
        f"  |  Aligned group: {attr.get('aligned_group_pct')}%"
        f"  |  Overall: {attr.get('overall_pct')}%"
        f"  |  Delta vs aligned: {attr.get('delta_vs_aligned')} pp"
    )

    pc = data.get("profile_comparison", {})
    delta = pc.get("delta", {})
    if delta:
        lines.append("\nKey deltas (Gap minus Aligned):")
        for k, v in delta.items():
            lines.append(f"  {k}: {v:+.3f}")

    gap_m = data.get("innovation_gap_metrics", {})
    gs = gap_m.get("gap_stats", {})
    if gs:
        lines.append(
            f"\nInnovation Gap Score stats — "
            f"mean={gs.get('mean')}, std={gs.get('std')}, "
            f"Q25={gs.get('25%')}, median={gs.get('50%')}, "
            f"Q75={gs.get('75%')}, max={gs.get('max')}"
        )

    dept = data.get("department_breakdown", [])
    if dept:
        lines.append("\nDepartment breakdown (gap employees):")
        for row in dept:
            lines.append(
                f"  {row.get('Department')}: n={row.get('count')}"
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
    """Auto-detect question and return a concise summary text."""
    stem = json_path.stem.lower()
    for key, extractor in _EXTRACTORS.items():
        if stem.startswith(key):
            return extractor(data)
    # Fallback: generic extractor (skip raw arrays, keep scalars/dicts)
    return _generic_extractor(json_path, data)


def _generic_extractor(json_path: Path, data: dict) -> str:
    """Fallback: pull top-level scalar/dict fields, skip large arrays."""
    lines = [f"=== {data.get('question', json_path.stem)} ==="]
    for k, v in data.items():
        if k in ("llm_insights", "sample_employees", "top20_wasted_potential", "top20_at_risk"):
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
    "counter-intuitive or hidden signal that a typical analyst would miss."
)

USER_PROMPT_TEMPLATE = """\
Context:
{summary}

Task:
Based ONLY on the structured data above:
1. Identify key patterns or contradictions in the data
2. Explain WHY they exist — include behavioral and organizational reasoning
3. Highlight at least one non-obvious or hidden insight
4. Provide a strong, actionable business implication

IMPORTANT: Respond with ONLY valid JSON — no markdown, no preamble, no explanation outside the JSON block.

Output format (STRICT JSON):
{{
  "headline": "One powerful sentence summarising the core finding",
  "key_insight_1": "First key pattern or finding with explanation of root cause",
  "key_insight_2": "Second key pattern or finding with explanation of root cause",
  "hidden_insight": "A non-obvious or counter-intuitive insight a typical analyst would miss",
  "business_implication": "A specific, actionable recommendation for HR leadership"
}}
"""

def build_prompt(summary_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(summary=summary_text)


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM CALL  –  rate-limit safe with exponential back-off + jitter
# ═══════════════════════════════════════════════════════════════════════════════

def generate_llm_insights(summary_text: str, model: genai.GenerativeModel) -> dict:
    """
    Call Gemini API with the given summary text.

    Returns a dict with the structured insights or an error dict on failure.
    """
    prompt = build_prompt(summary_text)

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            log.info(f"  → Calling Gemini ({MODEL_NAME}), attempt {attempt}/{RETRY_ATTEMPTS} …")
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1024,
                ),
            )
            raw_text = response.text.strip()

            # Strip markdown code fences if model wraps JSON in them
            if raw_text.startswith("```"):
                raw_text = "\n".join(
                    line for line in raw_text.splitlines()
                    if not line.strip().startswith("```")
                ).strip()

            insights = json.loads(raw_text)
            log.info("  ✓ Insights generated successfully.")
            return insights

        except (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable) as e:
            delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
            jitter = random.uniform(0, delay * 0.25)
            wait = round(delay + jitter, 1)
            log.warning(f"  Rate-limit / service error: {e}")
            log.warning(f"  Waiting {wait}s before retry {attempt + 1} …")
            time.sleep(wait)

        except json.JSONDecodeError as e:
            log.error(f"  ✗ JSON parse error on attempt {attempt}: {e}")
            log.debug(f"  Raw response was:\n{raw_text}")
            if attempt == RETRY_ATTEMPTS:
                return {
                    "error": "JSON parse failed after all retries",
                    "raw_response": raw_text,
                }
            time.sleep(3)

        except Exception as e:
            log.error(f"  ✗ Unexpected error on attempt {attempt}: {type(e).__name__}: {e}")
            if attempt == RETRY_ATTEMPTS:
                return {"error": str(e)}
            time.sleep(5)

    return {"error": "All retry attempts exhausted without a successful response."}


# ═══════════════════════════════════════════════════════════════════════════════
#  FILE PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(json_path: Path, model: genai.GenerativeModel) -> bool:
    """
    Full pipeline for a single JSON file:
      read → extract summary → call LLM → update json → save

    Returns True on success, False on failure.
    """
    log.info(f"Processing: {json_path.name}")

    # 1. Read
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
    except Exception as e:
        log.error(f"  ✗ Failed to read {json_path}: {e}")
        return False

    # 2. Extract summary (no raw data sent to LLM)
    summary_text = build_summary_text(json_path, data)
    log.info(f"  Summary length: {len(summary_text)} chars")

    # 3. Call LLM
    insights = generate_llm_insights(summary_text, model)

    if "error" in insights:
        log.error(f"  ✗ LLM call failed: {insights}")
        return False

    # 4. Update the JSON
    data["llm_insights"] = insights

    # 5. Save (overwrite)
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        log.info(f"  ✓ Saved → {json_path}")
        return True
    except Exception as e:
        log.error(f"  ✗ Failed to save {json_path}: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_section3_files() -> list[Path]:
    """Return Q11–Q14 JSON files in the section3 reports directory."""
    if not SECTION3_DIR.exists():
        log.error(f"Section3 reports directory not found: {SECTION3_DIR}")
        return []
    files = sorted(
        p for p in SECTION3_DIR.glob("q1[1-4]_*.json")
    )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Gemini LLM insights for Section-3 HR analytics JSON files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="Path(s) to JSON file(s) to process. Ignored when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all Q11–Q14 JSON files in reports/section3/.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=INTER_FILE_DELAY,
        metavar="SECONDS",
        help=f"Seconds to wait between files when using --all (default: {INTER_FILE_DELAY}).",
    )
    args = parser.parse_args()

    model = _init_gemini()

    if args.all:
        paths = _discover_section3_files()
        if not paths:
            log.error("No Q11–Q14 JSON files found.")
            sys.exit(1)
        log.info(f"Found {len(paths)} files: {[p.name for p in paths]}")
    elif args.files:
        paths = [Path(p) for p in args.files]
    else:
        parser.print_help()
        sys.exit(0)

    results: dict[str, bool] = {}
    for idx, path in enumerate(paths):
        success = process_file(path, model)
        results[path.name] = success

        # Inter-file delay to respect rate limits (skip after last file)
        if idx < len(paths) - 1:
            log.info(f"  ⏳ Waiting {args.delay}s before next file …\n")
            time.sleep(args.delay)

    # Summary
    log.info("\n══════════════ RESULTS ══════════════")
    for fname, ok in results.items():
        status = "✓ OK" if ok else "✗ FAILED"
        log.info(f"  {status}  {fname}")
    log.info("════════════════════════════════════\n")

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
