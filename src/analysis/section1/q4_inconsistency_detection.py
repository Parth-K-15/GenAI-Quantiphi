"""Section 1 - Q4: Skill-performance inconsistency detection."""

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
OUT = os.path.join(BASE, "reports", "section1", "q4_inconsistency_detection.json")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q4 inconsistency detection.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q4 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q4_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q4 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q4 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q4 Gemini insights generation")


def main() -> None:
    args = _parse_cli_args()

    skill_cols = [
        "Technical_Skills_Rating",
        "Communication_Skills_Rating",
        "Problem_Solving_Skills_Rating",
        "Leadership_Qualities_Rating",
        "Initiative_Rating",
        "Adaptability_Rating",
        "Creativity_Rating",
        "Teamwork_Skills_Rating",
        "Strategic_Thinking_Rating",
    ]
    skill_cols = [c for c in skill_cols if c in DF.columns]

    work_df = DF.copy()
    work_df["Composite_Skill_Score"] = work_df[skill_cols].mean(axis=1).round(3)
    skill_thresh = float(work_df["Composite_Skill_Score"].quantile(0.75))

    outcome_norm = work_df["Project_Outcome"].astype(str).str.strip().str.lower()
    high_skill = work_df["Composite_Skill_Score"] >= skill_thresh
    failed_proj = outcome_norm == "failed"
    success_proj = outcome_norm == "successful"

    anomaly_a = work_df[high_skill & failed_proj].copy()
    anomaly_b = work_df[~high_skill & success_proj].copy()
    expected = work_df[high_skill & success_proj].copy()

    ctx_num = [
        c
        for c in [
            "Performance_Rating",
            "Employee_Engagement_Score",
            "Employee_Job_Satisfaction_Score",
            "Overtime_Hours_Per_Week",
            "Conflict_Resolution_Cases",
            "Mentor_Rating",
            "Work_Hours_Per_Week",
            "Employee_Work_Life_Balance_Rating",
            "Training_Efficiency",
            "Engagement_Index",
        ]
        if c in work_df.columns
    ]

    a_means = anomaly_a[ctx_num].mean().round(3).to_dict()
    e_means = expected[ctx_num].mean().round(3).to_dict()
    deltas = {c: round(float(a_means.get(c, 0) - e_means.get(c, 0)), 3) for c in ctx_num}

    cat_ctx = [
        c
        for c in [
            "Department",
            "Job_Title",
            "Project_Role",
            "Project_Complexity",
            "Project_Size",
            "Training_Program",
            "Leadership_Potential",
            "Employee_Resignation_Status",
            "Career_Goals_Achievement_Status",
        ]
        if c in work_df.columns
    ]
    a_cats = {
        c: anomaly_a[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict()
        for c in cat_ctx
    }
    e_cats = {
        c: expected[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict()
        for c in cat_ctx
    }

    reasons = []
    if a_means.get("Employee_Engagement_Score", 999) < e_means.get("Employee_Engagement_Score", 0):
        reasons.append("Lower engagement despite high skills suggests execution friction.")
    if a_means.get("Overtime_Hours_Per_Week", 0) > e_means.get("Overtime_Hours_Per_Week", 0):
        reasons.append("Higher overtime indicates possible burnout and delivery instability.")
    if a_means.get("Employee_Work_Life_Balance_Rating", 999) < e_means.get("Employee_Work_Life_Balance_Rating", 0):
        reasons.append("Poorer work-life balance can reduce effective output under project pressure.")
    if a_means.get("Training_Efficiency", 999) < e_means.get("Training_Efficiency", 0):
        reasons.append("Lower training efficiency suggests skills are not translating into outcomes.")
    reasons.extend(
        [
            "Project-role alignment and team dynamics can outweigh individual skill ratings.",
            "Skill depth may be domain-specific and mismatched to assigned project demands.",
        ]
    )

    high_skill_total = len(anomaly_a) + len(expected)
    anomaly_pct_of_high_skill = round((len(anomaly_a) / high_skill_total) * 100, 1) if high_skill_total else 0.0

    result = {
        "question": "Q4 - Skill-Performance Inconsistency Detection",
        "skill_threshold_75pct": round(skill_thresh, 3),
        "segment_counts": {
            "high_skill_failed_anomaly_A": int(len(anomaly_a)),
            "low_skill_successful_anomaly_B": int(len(anomaly_b)),
            "high_skill_successful_expected": int(len(expected)),
            "total": int(len(work_df)),
        },
        "anomaly_A_pct_of_high_skill": anomaly_pct_of_high_skill,
        "anomaly_A_numeric_means": a_means,
        "expected_numeric_means": e_means,
        "anomaly_A_categorical": a_cats,
        "expected_categorical": e_cats,
        "anomaly_B_counts": {
            "department": anomaly_b["Department"].value_counts().head(4).to_dict()
            if "Department" in anomaly_b.columns
            else {},
            "job_title": anomaly_b["Job_Title"].value_counts().head(4).to_dict()
            if "Job_Title" in anomaly_b.columns
            else {},
        },
        "llm_reasoning": {
            "anomaly_A_headline": (
                f"{len(anomaly_a)} employees ({round(len(anomaly_a) / len(work_df) * 100, 1)}%) show high skills "
                "yet failed project outcomes."
            ),
            "anomaly_B_headline": (
                f"{len(anomaly_b)} employees with below-threshold skill composites still achieved success."
            ),
            "reasons_for_anomaly_A": reasons,
            "key_insight": (
                "Skill ratings are necessary but not sufficient for project success; operating context mediates outcomes."
            ),
        },
        "outcome_distribution": work_df["Project_Outcome"].value_counts().to_dict()
        if "Project_Outcome" in work_df.columns
        else {},
        "llm_evidence": {
            "skill_threshold_75pct": round(skill_thresh, 3),
            "segment_counts": {
                "high_skill_failed_anomaly_A": int(len(anomaly_a)),
                "low_skill_successful_anomaly_B": int(len(anomaly_b)),
                "high_skill_successful_expected": int(len(expected)),
                "total": int(len(work_df)),
            },
            "anomaly_A_pct_of_high_skill": anomaly_pct_of_high_skill,
            "numeric_deltas_anomaly_A_minus_expected": deltas,
            "anomaly_A_numeric_means": a_means,
            "expected_numeric_means": e_means,
            "outcome_distribution": work_df["Project_Outcome"].value_counts().to_dict()
            if "Project_Outcome" in work_df.columns
            else {},
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

    _run_q4_llm_generation(args)


if __name__ == "__main__":
    main()
