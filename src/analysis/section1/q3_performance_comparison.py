"""Section 1 - Q3: High (>=10) vs Low (<=5) performance behavioral comparison."""

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
OUT = os.path.join(BASE, "reports", "section1", "q3_performance_comparison.json")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q3 performance comparison.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q3 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q3_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q3 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q3 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q3 Gemini insights generation")


def main() -> None:
    args = _parse_cli_args()

    high_seg = DF[DF["Performance_Rating"] >= 10].copy()
    low_seg = DF[DF["Performance_Rating"] <= 5].copy()

    num_cols = [
        c
        for c in [
            "Technical_Skills_Rating",
            "Communication_Skills_Rating",
            "Problem_Solving_Skills_Rating",
            "Leadership_Qualities_Rating",
            "Initiative_Rating",
            "Adaptability_Rating",
            "Creativity_Rating",
            "Strategic_Thinking_Rating",
            "Teamwork_Skills_Rating",
            "Employee_Engagement_Score",
            "Employee_Job_Satisfaction_Score",
            "Professional_Development_Hours",
            "Overtime_Hours_Per_Week",
            "Number_Of_Promotions",
            "Conflict_Resolution_Cases",
            "Feedback_From_Colleagues",
            "Feedback_From_Supervisors",
            "Mentor_Rating",
            "Employee_Work_Life_Balance_Rating",
            "Avg_Skills_Score",
            "Avg_Soft_Skills_Score",
            "Engagement_Index",
            "Training_Efficiency",
            "Onboarding_Delay_Days",
            "Tenure_Years",
        ]
        if c in DF.columns
    ]

    high_m = high_seg[num_cols].mean().round(3)
    low_m = low_seg[num_cols].mean().round(3)
    diff = (high_m - low_m).round(3)
    top10 = diff.abs().sort_values(ascending=False).head(10).index.tolist()

    cat_cols = [
        c
        for c in [
            "Department",
            "Job_Title",
            "Project_Role",
            "Project_Outcome",
            "Highest_Education_Level",
            "Training_Program",
            "Leadership_Potential",
            "Career_Goals_Achievement_Status",
            "Employee_Resignation_Status",
            "Mentor_Experience_Level",
            "Project_Complexity",
        ]
        if c in DF.columns
    ]

    cat_high = {
        c: high_seg[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict() for c in cat_cols
    }
    cat_low = {
        c: low_seg[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict() for c in cat_cols
    }

    result = {
        "question": "Q3 - High (>=10) vs Low (<=5) Performance Behavioral Comparison",
        "segment_sizes": {"high_ge10": int(len(high_seg)), "low_le5": int(len(low_seg))},
        "numeric_means_high": high_m.to_dict(),
        "numeric_means_low": low_m.to_dict(),
        "numeric_diff": diff.to_dict(),
        "top_differentiators": top10,
        "categorical_high": cat_high,
        "categorical_low": cat_low,
        "llm_evidence": {
            "segment_sizes": {"high_ge10": int(len(high_seg)), "low_le5": int(len(low_seg))},
            "top_differentiators": top10,
            "top_differentiator_deltas": {k: float(diff[k]) for k in top10},
            "skills_summary": {
                "high_avg_skills_score": round(float(high_m.get("Avg_Skills_Score", 0)), 3),
                "low_avg_skills_score": round(float(low_m.get("Avg_Skills_Score", 0)), 3),
                "high_training_efficiency": round(float(high_m.get("Training_Efficiency", 0)), 3),
                "low_training_efficiency": round(float(low_m.get("Training_Efficiency", 0)), 3),
                "high_wlb": round(float(high_m.get("Employee_Work_Life_Balance_Rating", 0)), 3),
                "low_wlb": round(float(low_m.get("Employee_Work_Life_Balance_Rating", 0)), 3),
            },
            "resignation_high_pct": cat_high.get("Employee_Resignation_Status", {}),
            "resignation_low_pct": cat_low.get("Employee_Resignation_Status", {}),
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

    _run_q3_llm_generation(args)


if __name__ == "__main__":
    main()
