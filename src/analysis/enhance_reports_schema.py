"""Normalize report schema for stronger grounding and consistent LLM insight format.

Usage:
  python src/analysis/enhance_reports_schema.py reports/section2/q6_*.json --ensure-evidence --normalize-insights
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


CANONICAL_INSIGHT_KEYS = [
    "headline",
    "key_insight_1",
    "key_insight_2",
    "hidden_insight",
    "risk_alert",
    "business_implication",
    "action_plan",
    "standout_statement",
    "confidence_note",
]


def _first_non_empty(data: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            text = _to_text(value)
        else:
            text = str(value).strip()
        if text:
            return text
    return ""


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
        return " | ".join(parts)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            sv = str(v).strip()
            if sv:
                parts.append(f"{k}: {sv}")
        return " | ".join(parts)
    return str(value).strip()


def _to_action_plan(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()][:6]
    if isinstance(value, dict):
        out = []
        for k, v in value.items():
            sv = str(v).strip()
            if sv:
                out.append(f"{k}: {sv}")
        return out[:6]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [text]
    return []


def _looks_canonical_insights(payload: dict[str, Any]) -> bool:
    return all(key in payload for key in CANONICAL_INSIGHT_KEYS)


def _canonicalize_insights(existing: dict[str, Any], question: str) -> dict[str, Any]:
    if _looks_canonical_insights(existing):
        # Keep canonical payload but coerce action plan to list[str].
        normalized = dict(existing)
        normalized["action_plan"] = _to_action_plan(existing.get("action_plan"))
        if not normalized["action_plan"]:
            normalized["action_plan"] = ["Review this question with the analytics owner and define intervention steps."]
        return normalized

    headline = _first_non_empty(existing, ["headline", "headline_finding", "standout_statement", "power_statement"])
    key_1 = _first_non_empty(
        existing,
        [
            "key_insight_1",
            "key_finding_1",
            "advanced_insight",
            "training_impact_statement",
            "mentorship_dependency_statement",
            "training_roi_statement",
            "readiness_score_statement",
            "archetype_insight",
            "dimensional_insight",
            "engagement_necessity_insight",
            "enabling_environment_insight",
        ],
    )
    key_2 = _first_non_empty(
        existing,
        [
            "key_insight_2",
            "key_finding_2",
            "promotion_insight",
            "internship_conversion_insight",
            "career_growth_insight",
            "department_insight",
            "satisfaction_pathway",
            "retention_pathway",
        ],
    )
    hidden = _first_non_empty(
        existing,
        [
            "hidden_insight",
            "standout_insight",
            "connecting_to_q3",
            "connecting_q3_q6_q7",
            "connecting_q6_q7_q8_q9_q10",
            "disengagement_as_leading_indicator",
        ],
    )
    risk = _first_non_empty(
        existing,
        [
            "risk_alert",
            "key_insight_3",
            "key_insight_wlb",
            "system_improvement_statement",
        ],
    )
    business = _first_non_empty(
        existing,
        [
            "business_implication",
            "strategic_recommendation",
            "system_improvement",
            "effectiveness_redefinition",
            "prediction_without_ml",
        ],
    )
    standout = _first_non_empty(existing, ["standout_statement", "power_statement", "standout_insight", "headline"])
    confidence = _first_non_empty(
        existing,
        ["confidence_note"],
    )

    if not confidence:
        confidence = (
            "Directional interpretation derived from report statistics; validate with business context "
            "before high-stakes decisions."
        )

    action = _to_action_plan(existing.get("action_plan"))
    if not action:
        for candidate in ["recommendation", "strategic_recommendation", "system_proposal", "tier_action_map", "roi_level_action_map"]:
            action = _to_action_plan(existing.get(candidate))
            if action:
                break
    if not action:
        action = ["Prioritize this segment for targeted intervention and monitor monthly outcome lift."]

    if not headline:
        headline = f"Insights generated for: {question or 'report'}"
    if not key_1:
        key_1 = "Primary pattern detected from section-level analytics."
    if not key_2:
        key_2 = "Secondary pattern indicates contextual drivers beyond single metrics."
    if not hidden:
        hidden = "Cross-metric interaction suggests non-obvious segment behavior worth deeper review."
    if not risk:
        risk = "If left unaddressed, this pattern may reduce retention, productivity, or training ROI."
    if not business:
        business = "Translate this segment signal into targeted policy and manager-level interventions."
    if not standout:
        standout = headline

    return {
        "headline": headline,
        "key_insight_1": key_1,
        "key_insight_2": key_2,
        "hidden_insight": hidden,
        "risk_alert": risk,
        "business_implication": business,
        "action_plan": action[:6],
        "standout_statement": standout,
        "confidence_note": confidence,
    }


def _trim_value(key: str, value: Any) -> Any:
    if isinstance(value, list):
        if not value:
            return value
        if key.lower().endswith("_employees"):
            return value[:10]
        if isinstance(value[0], dict):
            return value[:12]
        return value[:25]
    return value


def _build_evidence_payload(report: dict[str, Any]) -> dict[str, Any]:
    existing = report.get("llm_evidence")
    if isinstance(existing, dict) and existing:
        return existing

    payload: dict[str, Any] = {"question": report.get("question")}
    skip_keys = {
        "llm_evidence",
        "llm_insights",
        "llm_generation_meta",
        "llm_reasoning",
        "llm_profile_description",
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
        "training_roi_metric",
        "training_effectiveness_metric",
        "readiness_score_formula",
        "retention_analysis",
        "innovation_gap_metrics",
        "mentorship_dependency_metric",
        "cross_segment_program_x_roi_level",
        "program_kpi_summary",
        "tier_numeric_profile",
        "roi_level_numeric_profile",
        "profile_comparison",
    ]

    for key in preferred_keys:
        if key in report and key not in skip_keys:
            payload[key] = _trim_value(key, report[key])

    for key, value in report.items():
        if key in payload or key in skip_keys:
            continue
        if isinstance(value, (int, float, str, bool, dict, list)):
            payload[key] = _trim_value(key, value)

    return payload


def process_file(
    path: Path,
    ensure_evidence: bool,
    normalize_insights: bool,
    add_local_meta: bool,
    verbose: bool,
) -> bool:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[FAIL] Could not parse {path}: {exc}")
        return False

    changed = False

    if ensure_evidence:
        evidence = _build_evidence_payload(report)
        if report.get("llm_evidence") != evidence:
            report["llm_evidence"] = evidence
            changed = True

    if normalize_insights and isinstance(report.get("llm_insights"), dict):
        canonical = _canonicalize_insights(report["llm_insights"], str(report.get("question", path.stem)))
        if report["llm_insights"] != canonical:
            report["llm_insights"] = canonical
            changed = True
            if add_local_meta and not isinstance(report.get("llm_generation_meta"), dict):
                report["llm_generation_meta"] = {
                    "model_used": "local-schema-normalizer",
                    "generated_at_epoch": int(time.time()),
                    "note": "Canonicalized legacy insight schema into standard 9-field structure.",
                }
                changed = True

    if not changed:
        if verbose:
            print(f"[SKIP] {path.name} (no changes)")
        return True

    try:
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    except Exception as exc:
        print(f"[FAIL] Could not write {path}: {exc}")
        return False

    if verbose:
        print(f"[OK] Enhanced schema -> {path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Enhance JSON report schema for evidence and insight consistency.")
    parser.add_argument("files", nargs="+", help="Report JSON files to enhance.")
    parser.add_argument("--ensure-evidence", action="store_true", help="Ensure llm_evidence exists and is populated.")
    parser.add_argument("--normalize-insights", action="store_true", help="Canonicalize llm_insights to 9-field schema.")
    parser.add_argument("--add-local-meta", action="store_true", help="Add local llm_generation_meta when canonicalizing.")
    parser.add_argument("--quiet", action="store_true", help="Reduce output.")
    args = parser.parse_args()

    if not args.ensure_evidence and not args.normalize_insights:
        parser.error("Specify at least one of --ensure-evidence or --normalize-insights.")

    ok_all = True
    for raw in args.files:
        ok = process_file(
            path=Path(raw),
            ensure_evidence=args.ensure_evidence,
            normalize_insights=args.normalize_insights,
            add_local_meta=args.add_local_meta,
            verbose=not args.quiet,
        )
        ok_all = ok_all and ok

    if not ok_all:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

