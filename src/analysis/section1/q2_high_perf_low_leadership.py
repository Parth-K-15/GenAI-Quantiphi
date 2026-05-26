"""Section 1 - Q2: High performance but low leadership potential."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys

import pandas as pd


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section1", "q2_high_perf_low_leadership.json")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q2 high-performance/low-leadership analysis.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q2 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q2_llm_generation(args: argparse.Namespace) -> None:
    if not args.with_llm:
        return

    llm_cmd = [
        sys.executable,
        "src/llm_section4_insights_generator.py",
        OUT,
        "--delay",
        str(max(0, args.llm_delay)),
    ]
    if args.llm_model:
        llm_cmd.extend(["--model", args.llm_model])
    if args.llm_strict_model:
        llm_cmd.append("--strict-model")
    if args.llm_quiet:
        llm_cmd.append("--quiet")

    print("[RUN] Q2 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q2 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q2 Gemini insights generation")


def main() -> None:
    args = _parse_cli_args()

    leadership = DF["Leadership_Potential"].astype(str).str.strip().str.lower()
    high_perf = DF["Performance_Rating"] >= 12
    low_lead = leadership == "low"

    segment = DF[high_perf & low_lead].copy()
    high_perf_only = DF[high_perf & ~low_lead].copy()
    total_high = int(high_perf.sum())

    ctx_cols = [
        "Department",
        "Job_Title",
        "Project_Role",
        "Project_Outcome",
        "Highest_Education_Level",
        "Certifications",
        "Training_Program",
        "Innovation_Projects_Involvement",
        "Work_Quality_Improvement_Plan",
        "Career_Goals_Achievement_Status",
        "Employee_Resignation_Status",
        "Mentor_Experience_Level",
        "Internship_Conversion_Status",
    ]
    num_cols = [
        "Performance_Rating",
        "Leadership_Qualities_Rating",
        "Technical_Skills_Rating",
        "Communication_Skills_Rating",
        "Problem_Solving_Skills_Rating",
        "Employee_Engagement_Score",
        "Employee_Job_Satisfaction_Score",
        "Overtime_Hours_Per_Week",
        "Professional_Development_Hours",
        "Number_Of_Promotions",
        "Conflict_Resolution_Cases",
        "Feedback_From_Supervisors",
        "Feedback_From_Colleagues",
        "Avg_Skills_Score",
        "Engagement_Index",
    ]
    num_cols = [c for c in num_cols if c in DF.columns]

    seg_means = segment[num_cols].mean().round(2).to_dict()
    rest_means = high_perf_only[num_cols].mean().round(2).to_dict()
    mean_deltas = {c: round(float(seg_means.get(c, 0) - rest_means.get(c, 0)), 3) for c in num_cols}
    top_abs_deltas = dict(
        sorted(mean_deltas.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
    )

    cat_profiles: dict[str, dict[str, float]] = {}
    for col in ctx_cols:
        if col in segment.columns:
            cat_profiles[col] = (
                segment[col].value_counts(normalize=True).round(3).mul(100).to_dict()
            )

    reasons = []
    if seg_means.get("Conflict_Resolution_Cases", 0) < rest_means.get("Conflict_Resolution_Cases", 999):
        reasons.append("Lower conflict resolution cases suggest limited team leadership exposure.")
    if seg_means.get("Professional_Development_Hours", 0) < rest_means.get("Professional_Development_Hours", 999):
        reasons.append("Slightly lower development hours may reflect less leadership-focused upskilling.")
    if seg_means.get("Number_Of_Promotions", 0) < rest_means.get("Number_Of_Promotions", 999):
        reasons.append("Lower promotion velocity can signal weaker visibility of leadership behaviors.")
    reasons.extend(
        [
            "High performers can be strong individual contributors without formal leadership orientation.",
            "Role design may reward output delivery more than people-management capability.",
            "Leadership potential labels may capture future readiness, not current task performance.",
        ]
    )

    pct = round((len(segment) / total_high) * 100, 1) if total_high else 0.0

    result = {
        "question": "Q2 - High Performance, Low Leadership Potential",
        "total_high_performers": total_high,
        "high_perf_low_leadership_count": int(len(segment)),
        "percentage_of_high_performers": pct,
        "segment_numeric_means": seg_means,
        "comparison_means": rest_means,
        "categorical_profiles": cat_profiles,
        "possible_reasons": reasons,
        "dept_breakdown": segment["Department"].value_counts().to_dict() if "Department" in segment.columns else {},
        "role_breakdown": segment["Job_Title"].value_counts().to_dict() if "Job_Title" in segment.columns else {},
        "project_role_breakdown": (
            segment["Project_Role"].value_counts().to_dict() if "Project_Role" in segment.columns else {}
        ),
        "outcome_breakdown": (
            segment["Project_Outcome"].value_counts().to_dict() if "Project_Outcome" in segment.columns else {}
        ),
        "resignation_breakdown": (
            segment["Employee_Resignation_Status"].value_counts().to_dict()
            if "Employee_Resignation_Status" in segment.columns
            else {}
        ),
        "llm_evidence": {
            "total_high_performers": total_high,
            "high_perf_low_leadership_count": int(len(segment)),
            "percentage_of_high_performers": pct,
            "segment_numeric_means": seg_means,
            "comparison_means": rest_means,
            "numeric_deltas_segment_minus_comparison_top_abs": top_abs_deltas,
            "dept_breakdown": segment["Department"].value_counts().to_dict() if "Department" in segment.columns else {},
            "role_breakdown": segment["Job_Title"].value_counts().to_dict() if "Job_Title" in segment.columns else {},
            "outcome_breakdown": (
                segment["Project_Outcome"].value_counts().to_dict() if "Project_Outcome" in segment.columns else {}
            ),
            "resignation_breakdown": (
                segment["Employee_Resignation_Status"].value_counts().to_dict()
                if "Employee_Resignation_Status" in segment.columns
                else {}
            ),
        },
        "llm_insights": {
            "headline": "Pending Gemini generation.",
            "key_insight_1": "Pending Gemini generation.",
            "key_insight_2": "Pending Gemini generation.",
            "hidden_insight": "Pending Gemini generation.",
            "risk_alert": "Pending Gemini generation.",
            "business_implication": "Pending Gemini generation.",
            "action_plan": ["Pending Gemini generation."],
            "standout_statement": "Pending Gemini generation.",
            "confidence_note": "Pending Gemini generation.",
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    print(f"[OK] Saved -> {OUT}")

    _run_q2_llm_generation(args)


if __name__ == "__main__":
    main()
