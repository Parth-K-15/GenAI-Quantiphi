"""Section 6 - Q22: Analyze relationship between salary increase %, bonus %, and performance rating."""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section6", "q22_compensation_performance_relationship.json")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q22 compensation-performance relationship analysis.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q22 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q22_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q22 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q22 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q22 Gemini insights generation")


def _safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float((num / den) * 100)


ARGS = _parse_cli_args()


required_cols = [
    "Annual_Salary_Increase_Percentage",
    "Performance_Bonus_Percentage",
    "Performance_Rating",
]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q22: {missing_cols}")

work_df = DF[required_cols].copy()
for c in required_cols:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
work_df = work_df.dropna().copy()

salary_col = "Annual_Salary_Increase_Percentage"
bonus_col = "Performance_Bonus_Percentage"
perf_col = "Performance_Rating"

pairwise_corr = {}
for a, b in [(salary_col, bonus_col), (salary_col, perf_col), (bonus_col, perf_col)]:
    pear_r, pear_p = pearsonr(work_df[a], work_df[b])
    spear_rho, spear_p = spearmanr(work_df[a], work_df[b], nan_policy="omit")
    pairwise_corr[f"{a}_vs_{b}"] = {
        "pearson_r": round(float(pear_r), 4),
        "pearson_p_value": round(float(pear_p), 6),
        "spearman_rho": round(float(spear_rho), 4),
        "spearman_p_value": round(float(spear_p), 6),
    }

# Multiple linear model: performance as function of salary increase and bonus %
X = work_df[[salary_col, bonus_col]].copy()
y = work_df[perf_col].copy()

lin = LinearRegression()
lin.fit(X, y)
pred = lin.predict(X)
in_sample_r2 = r2_score(y, pred)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_pred = cross_val_predict(LinearRegression(), X, y, cv=cv, n_jobs=1)
cv_r2 = r2_score(y, cv_pred)

coef_rows = [
    {"feature": salary_col, "coefficient": round(float(lin.coef_[0]), 6)},
    {"feature": bonus_col, "coefficient": round(float(lin.coef_[1]), 6)},
]

# Quartile interaction grid.
grid_df = work_df.copy()
grid_df["salary_band"] = pd.qcut(grid_df[salary_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
grid_df["bonus_band"] = pd.qcut(grid_df[bonus_col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
interaction_grid = (
    grid_df.groupby(["salary_band", "bonus_band"], observed=True)
    .agg(
        count=(perf_col, "count"),
        avg_performance=(perf_col, "mean"),
        median_performance=(perf_col, "median"),
    )
    .reset_index()
)
interaction_grid["avg_performance"] = interaction_grid["avg_performance"].round(4)
interaction_grid["median_performance"] = interaction_grid["median_performance"].round(4)

# High-high vs low-low contrast.
high_high = grid_df[(grid_df["salary_band"] == "Q4") & (grid_df["bonus_band"] == "Q4")]
low_low = grid_df[(grid_df["salary_band"] == "Q1") & (grid_df["bonus_band"] == "Q1")]
hh_perf = float(high_high[perf_col].mean()) if len(high_high) else None
ll_perf = float(low_low[perf_col].mean()) if len(low_low) else None

# Compensation alignment ratio.
work_df["compensation_reward_ratio"] = (work_df[salary_col] + work_df[bonus_col]) / (work_df[perf_col] + 1e-9)
ratio_stats = {
    "mean": round(float(work_df["compensation_reward_ratio"].mean()), 6),
    "median": round(float(work_df["compensation_reward_ratio"].median()), 6),
    "std": round(float(work_df["compensation_reward_ratio"].std()), 6),
    "p25": round(float(work_df["compensation_reward_ratio"].quantile(0.25)), 6),
    "p75": round(float(work_df["compensation_reward_ratio"].quantile(0.75)), 6),
}

result = {
    "question": "Q22 - Relationship Between Salary Increase %, Bonus %, and Performance Rating",
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "features_used": [salary_col, bonus_col, perf_col],
    },
    "pairwise_correlations": pairwise_corr,
    "multivariable_regression": {
        "target": perf_col,
        "predictors": [salary_col, bonus_col],
        "coefficients": coef_rows,
        "intercept": round(float(lin.intercept_), 6),
        "in_sample_r2": round(float(in_sample_r2), 6),
        "cross_validated_r2": round(float(cv_r2), 6),
        "model_note": "Linear model quantifies directional relationship of compensation levers with performance.",
    },
    "interaction_patterns": {
        "salary_bonus_quartile_grid": interaction_grid.to_dict(orient="records"),
        "high_salary_high_bonus_segment": {
            "count": int(len(high_high)),
            "avg_performance": round(hh_perf, 4) if hh_perf is not None else None,
            "segment_share_pct": round(_safe_pct(len(high_high), len(work_df)), 3),
        },
        "low_salary_low_bonus_segment": {
            "count": int(len(low_low)),
            "avg_performance": round(ll_perf, 4) if ll_perf is not None else None,
            "segment_share_pct": round(_safe_pct(len(low_low), len(work_df)), 3),
        },
        "avg_performance_gap_high_high_vs_low_low": round(hh_perf - ll_perf, 4)
        if hh_perf is not None and ll_perf is not None
        else None,
    },
    "compensation_reward_ratio": {
        "formula": "(Salary Increase % + Bonus %) / Performance Rating",
        "distribution_stats": ratio_stats,
    },
    "llm_evidence": {
        "rows_used": int(len(work_df)),
        "pairwise_correlations": pairwise_corr,
        "multivariable_regression": {
            "coefficients": coef_rows,
            "intercept": round(float(lin.intercept_), 6),
            "in_sample_r2": round(float(in_sample_r2), 6),
            "cross_validated_r2": round(float(cv_r2), 6),
        },
        "high_high_segment": {
            "count": int(len(high_high)),
            "avg_performance": round(hh_perf, 4) if hh_perf is not None else None,
        },
        "low_low_segment": {
            "count": int(len(low_low)),
            "avg_performance": round(ll_perf, 4) if ll_perf is not None else None,
        },
        "avg_performance_gap_high_high_vs_low_low": round(hh_perf - ll_perf, 4)
        if hh_perf is not None and ll_perf is not None
        else None,
        "compensation_reward_ratio_stats": ratio_stats,
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
_run_q22_llm_generation(ARGS)

