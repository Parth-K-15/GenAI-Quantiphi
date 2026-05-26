"""Section 1 - Q1: Skill influence on performance rating."""

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
OUT = os.path.join(BASE, "reports", "section1", "q1_skill_influence.json")

SKILL_COLS = [
    "Technical_Skills_Rating",
    "Communication_Skills_Rating",
    "Problem_Solving_Skills_Rating",
]
TARGET = "Performance_Rating"


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q1 skill influence analysis.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q1 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q1_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q1 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q1 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q1 Gemini insights generation")


def main() -> None:
    args = _parse_cli_args()

    corr_matrix = DF[SKILL_COLS + [TARGET]].corr().round(4)

    ind_corr: dict[str, float] = {}
    for col in SKILL_COLS:
        ind_corr[col] = round(float(DF[col].corr(DF[TARGET])), 6)

    abs_corrs = {c: abs(v) for c, v in ind_corr.items()}
    total_abs = sum(abs_corrs.values())
    weights = {c: round(v / total_abs, 4) if total_abs else 0.0 for c, v in abs_corrs.items()}

    df_out = DF.copy()
    df_out["Weighted_Skill_Score"] = sum(df_out[c] * w for c, w in weights.items())
    df_out["Weighted_Skill_Score"] = df_out["Weighted_Skill_Score"].round(4)
    wss_perf_corr = round(float(df_out["Weighted_Skill_Score"].corr(df_out[TARGET])), 6)

    bins = [0, 5, 10, 15, 25]
    labels = ["Low(1-5)", "Medium(6-10)", "High(11-15)", "Top(16+)"]
    df_out["Perf_Band"] = pd.cut(df_out[TARGET], bins=bins, labels=labels, include_lowest=True)
    band_skill = df_out.groupby("Perf_Band", observed=True)[SKILL_COLS].mean().round(2)

    scatter_data = []
    for rating in sorted(df_out[TARGET].unique()):
        sub = df_out[df_out[TARGET] == rating]
        scatter_data.append(
            {
                "rating": int(rating),
                "count": int(len(sub)),
                "avg_tech": round(float(sub["Technical_Skills_Rating"].mean()), 2),
                "avg_comm": round(float(sub["Communication_Skills_Rating"].mean()), 2),
                "avg_prob": round(float(sub["Problem_Solving_Skills_Rating"].mean()), 2),
            }
        )

    strongest_skill = max(ind_corr.items(), key=lambda kv: abs(kv[1]))[0]

    result = {
        "question": "Q1 - Skill Influence on Performance Rating",
        "correlation_matrix": corr_matrix.to_dict(),
        "individual_correlations": ind_corr,
        "normalized_weights": weights,
        "weighted_score_vs_performance_corr": wss_perf_corr,
        "skill_means": {c: round(float(df_out[c].mean()), 2) for c in SKILL_COLS},
        "performance_stats": {
            "mean": round(float(df_out[TARGET].mean()), 2),
            "median": float(df_out[TARGET].median()),
            "std": round(float(df_out[TARGET].std()), 2),
            "min": int(df_out[TARGET].min()),
            "max": int(df_out[TARGET].max()),
        },
        "band_skill_avg": band_skill.to_dict(),
        "scatter_by_rating": scatter_data,
        "perf_distribution": df_out[TARGET].value_counts().sort_index().to_dict(),
        "llm_reasoning": {
            "finding": "Low linear correlations indicate these 3 skills are not sole predictors of performance.",
            "communication_dominance": (
                f"Communication_Skills_Rating carries {weights.get('Communication_Skills_Rating', 0.0) * 100:.1f}% "
                "weight — the strongest individual predictor."
            ),
            "technical_insight": (
                "Technical_Skills_Rating shows near-zero correlation, suggesting technical skill alone does not "
                "drive performance scores in this dataset."
            ),
            "problem_solving_insight": (
                "Problem_Solving_Skills_Rating is near-zero in linear correlation, hinting that role context may "
                "matter more than isolated skill rating."
            ),
            "conclusion": (
                "Performance_Rating appears influenced by broader contextual factors beyond these three skill "
                "dimensions in isolation."
            ),
            "weighted_formula": (
                f"Weighted_Score = {weights.get('Technical_Skills_Rating', 0.0):.4f}×Tech + "
                f"{weights.get('Communication_Skills_Rating', 0.0):.4f}×Comm + "
                f"{weights.get('Problem_Solving_Skills_Rating', 0.0):.4f}×ProbSolve"
            ),
        },
        "llm_evidence": {
            "rows_used": int(len(df_out)),
            "target": TARGET,
            "skills_evaluated": SKILL_COLS,
            "individual_correlations": ind_corr,
            "normalized_weights": weights,
            "weighted_score_vs_performance_corr": wss_perf_corr,
            "strongest_skill_by_abs_correlation": strongest_skill,
            "performance_stats": {
                "mean": round(float(df_out[TARGET].mean()), 2),
                "median": float(df_out[TARGET].median()),
                "std": round(float(df_out[TARGET].std()), 2),
            },
            "skill_means": {c: round(float(df_out[c].mean()), 2) for c in SKILL_COLS},
            "band_skill_avg": band_skill.to_dict(),
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

    _run_q1_llm_generation(args)


if __name__ == "__main__":
    main()
