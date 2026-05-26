"""Section 1 - Q5: Ideal employee profile (top performers + composite score)."""

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
OUT = os.path.join(BASE, "reports", "section1", "q5_ideal_employee_profile.json")
TOP_OUT = os.path.join(BASE, "reports", "section1", "q5_top_employees.csv")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q5 ideal employee profile analysis.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q5 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q5_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q5 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q5 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q5 Gemini insights generation")


def main() -> None:
    args = _parse_cli_args()

    work_df = DF.copy()

    eng_max = float(work_df["Employee_Engagement_Score"].max()) if "Employee_Engagement_Score" in work_df.columns else 100.0
    wlb_max = float(work_df["Employee_Work_Life_Balance_Rating"].max()) if "Employee_Work_Life_Balance_Rating" in work_df.columns else 15.0

    work_df["Ideal_Employee_Score"] = (
        0.25 * work_df.get("Avg_Skills_Score", 0) / 20
        + 0.25 * work_df.get("Avg_Soft_Skills_Score", 0) / 20
        + 0.20 * work_df.get("Training_Efficiency", 0) * 10
        + 0.15 * work_df.get("Employee_Engagement_Score", 0) / eng_max
        + 0.15 * work_df.get("Employee_Work_Life_Balance_Rating", 0) / wlb_max
    ).round(6)

    p90 = float(work_df["Performance_Rating"].quantile(0.90))
    top10_perf = work_df[work_df["Performance_Rating"] >= p90].copy()

    s90 = float(work_df["Ideal_Employee_Score"].quantile(0.90))
    top10_score = work_df[work_df["Ideal_Employee_Score"] >= s90].copy()

    overlap = work_df[
        (work_df["Performance_Rating"] >= p90) & (work_df["Ideal_Employee_Score"] >= s90)
    ].copy()

    all_rating_cols = [
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
            "Mentor_Rating",
            "Employee_Work_Life_Balance_Rating",
            "Avg_Skills_Score",
            "Avg_Soft_Skills_Score",
            "Training_Efficiency",
            "Engagement_Index",
            "Number_Of_Promotions",
            "Feedback_From_Colleagues",
            "Feedback_From_Supervisors",
            "Conflict_Resolution_Cases",
            "Overtime_Hours_Per_Week",
            "Tenure_Years",
        ]
        if c in work_df.columns
    ]

    top_means = top10_perf[all_rating_cols].mean().round(3)
    all_means = work_df[all_rating_cols].mean().round(3)
    gap = (top_means - all_means).round(3)

    cat_cols = [
        c
        for c in [
            "Department",
            "Job_Title",
            "Project_Role",
            "Highest_Education_Level",
            "Certifications",
            "Training_Program",
            "Hiring_Source",
            "Leadership_Potential",
            "Project_Outcome",
            "Mentor_Experience_Level",
            "Career_Goals_Achievement_Status",
            "Employee_Resignation_Status",
            "Internship_Conversion_Status",
        ]
        if c in work_df.columns
    ]
    ideal_cats = {
        c: top10_perf[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict()
        for c in cat_cols
    }

    standout = gap.sort_values(ascending=False).head(5)

    top_rank_df = work_df[["Employee_ID", "Performance_Rating", "Ideal_Employee_Score"]].sort_values(
        "Ideal_Employee_Score", ascending=False
    )
    top_rank_df.head(20).to_csv(TOP_OUT, index=False)

    result = {
        "question": "Q5 - Ideal Employee Profile (Top 10% Performers)",
        "performance_90th_pct_threshold": p90,
        "score_90th_pct_threshold": round(s90, 6),
        "top10_by_performance_count": int(len(top10_perf)),
        "top10_by_score_count": int(len(top10_score)),
        "overlap_both_criteria": int(len(overlap)),
        "ideal_numeric_profile": top_means.to_dict(),
        "population_means": all_means.to_dict(),
        "gap_from_population": gap.to_dict(),
        "top5_standout_traits": standout.to_dict(),
        "ideal_categorical_profile": ideal_cats,
        "scoring_formula": {
            "formula": (
                "Ideal_Employee_Score = "
                "0.25×(Avg_Skills/20) + 0.25×(Avg_Soft_Skills/20) + "
                "0.20×(Training_Efficiency×10) + 0.15×(Engagement/max) + "
                "0.15×(WLB/max)"
            ),
            "weights_rationale": {
                "Avg_Skills_Score": "25% — core competency foundation",
                "Avg_Soft_Skills_Score": "25% — collaboration and leadership enablers",
                "Training_Efficiency": "20% — output-per-effort and execution quality",
                "Employee_Engagement_Score": "15% — motivation and consistency",
                "Employee_Work_Life_Balance_Rating": "15% — sustainability and long-term effectiveness",
            },
            "score_stats": {
                "mean": round(float(work_df["Ideal_Employee_Score"].mean()), 6),
                "top10_mean": round(float(top10_score["Ideal_Employee_Score"].mean()), 6),
                "max": round(float(work_df["Ideal_Employee_Score"].max()), 6),
            },
        },
        "llm_profile_description": {
            "headline": (
                "Ideal employees are differentiated more by skill utilization quality, contextual balance, and "
                "execution consistency than by raw skill levels alone."
            ),
            "trait_summary": {
                "Skills": (
                    f"Avg_Skills_Score: {float(top_means.get('Avg_Skills_Score', 0)):.2f} vs "
                    f"population {float(all_means.get('Avg_Skills_Score', 0)):.2f}"
                ),
                "Soft_Skills": (
                    f"Avg_Soft_Skills_Score: {float(top_means.get('Avg_Soft_Skills_Score', 0)):.2f} vs "
                    f"population {float(all_means.get('Avg_Soft_Skills_Score', 0)):.2f}"
                ),
                "Efficiency": (
                    f"Training_Efficiency: {float(top_means.get('Training_Efficiency', 0)):.3f} vs "
                    f"population {float(all_means.get('Training_Efficiency', 0)):.3f}"
                ),
                "Engagement": f"Employee_Engagement_Score: {float(top_means.get('Employee_Engagement_Score', 0)):.2f}",
                "WLB": f"Employee_Work_Life_Balance_Rating: {float(top_means.get('Employee_Work_Life_Balance_Rating', 0)):.2f}",
            },
        },
        "score_distribution_deciles": {
            f"D{i}": round(float(work_df["Ideal_Employee_Score"].quantile(i / 10)), 6)
            for i in range(1, 11)
        },
        "llm_evidence": {
            "performance_90th_pct_threshold": p90,
            "score_90th_pct_threshold": round(s90, 6),
            "top10_by_performance_count": int(len(top10_perf)),
            "top10_by_score_count": int(len(top10_score)),
            "overlap_both_criteria": int(len(overlap)),
            "top5_standout_traits": standout.to_dict(),
            "score_stats": {
                "mean": round(float(work_df["Ideal_Employee_Score"].mean()), 6),
                "top10_mean": round(float(top10_score["Ideal_Employee_Score"].mean()), 6),
                "max": round(float(work_df["Ideal_Employee_Score"].max()), 6),
            },
            "trait_gaps": {
                "Avg_Skills_Score": round(float(gap.get("Avg_Skills_Score", 0)), 3),
                "Avg_Soft_Skills_Score": round(float(gap.get("Avg_Soft_Skills_Score", 0)), 3),
                "Training_Efficiency": round(float(gap.get("Training_Efficiency", 0)), 3),
                "Employee_Engagement_Score": round(float(gap.get("Employee_Engagement_Score", 0)), 3),
                "Employee_Work_Life_Balance_Rating": round(
                    float(gap.get("Employee_Work_Life_Balance_Rating", 0)), 3
                ),
            },
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
    print(f"[OK] Saved -> {TOP_OUT}")

    _run_q5_llm_generation(args)


if __name__ == "__main__":
    main()
