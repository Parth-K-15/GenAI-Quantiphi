"""Section 5 - Q21: Compare work-life balance, overtime, and engagement between resigned vs retained employees."""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import argparse
import json
import math
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section5", "q21_resigned_vs_retained_wlb_overtime_engagement.json")


def _normalize_label_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float((num / den) * 100)


def _cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    sx = np.var(x, ddof=1)
    sy = np.var(y, ddof=1)
    pooled = ((len(x) - 1) * sx + (len(y) - 1) * sy) / (len(x) + len(y) - 2)
    if pooled <= 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / math.sqrt(pooled))


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q21 resigned vs retained comparison.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q21 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q21_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q21 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q21 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q21 Gemini insights generation")


ARGS = _parse_cli_args()


required_cols = [
    "Employee_Resignation_Status",
    "Employee_Work_Life_Balance_Rating",
    "Overtime_Hours_Per_Week",
    "Employee_Engagement_Score",
]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q21: {missing_cols}")

work_df = DF.copy()
work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])
work_df = work_df[work_df["Employee_Resignation_Status"].isin(["Yes", "No"])].copy()
work_df["Resignation_Target"] = (work_df["Employee_Resignation_Status"] == "Yes").astype(int)

resigned_df = work_df[work_df["Resignation_Target"] == 1].copy()
retained_df = work_df[work_df["Resignation_Target"] == 0].copy()

if len(resigned_df) == 0 or len(retained_df) == 0:
    raise ValueError("Q21 requires both resigned and retained employees.")

features = [
    ("Employee_Work_Life_Balance_Rating", "work_life_balance"),
    ("Overtime_Hours_Per_Week", "overtime_hours"),
    ("Employee_Engagement_Score", "engagement_score"),
]

comparison_rows = []
for col, alias in features:
    resigned_vals = pd.to_numeric(resigned_df[col], errors="coerce").dropna().to_numpy()
    retained_vals = pd.to_numeric(retained_df[col], errors="coerce").dropna().to_numpy()
    p_value = None
    try:
        _, p = mannwhitneyu(resigned_vals, retained_vals, alternative="two-sided")
        p_value = float(p)
    except Exception:
        p_value = None

    comparison_rows.append(
        {
            "feature": col,
            "feature_alias": alias,
            "resigned_mean": round(float(np.mean(resigned_vals)), 4),
            "retained_mean": round(float(np.mean(retained_vals)), 4),
            "delta_resigned_minus_retained": round(float(np.mean(resigned_vals) - np.mean(retained_vals)), 4),
            "resigned_median": round(float(np.median(resigned_vals)), 4),
            "retained_median": round(float(np.median(retained_vals)), 4),
            "cohen_d": round(float(_cohen_d(resigned_vals, retained_vals)), 4),
            "mannwhitney_p_value": round(p_value, 6) if p_value is not None else None,
            "is_statistically_significant_0_05": bool(p_value is not None and p_value < 0.05),
        }
    )

comparison_rows = sorted(comparison_rows, key=lambda r: abs(r["cohen_d"]), reverse=True)

overall_resignation_rate = _safe_pct((work_df["Resignation_Target"] == 1).sum(), len(work_df))

# Univariate risk gradients via quartiles.
quartile_summaries = {}
for col, _ in features:
    q_df = work_df[[col, "Resignation_Target"]].copy()
    q_df[col] = pd.to_numeric(q_df[col], errors="coerce")
    q_df = q_df.dropna()
    q_df["band"] = pd.qcut(q_df[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    band_rows = []
    for band in ["Q1", "Q2", "Q3", "Q4"]:
        sub = q_df[q_df["band"] == band]
        if len(sub) == 0:
            continue
        rate = _safe_pct((sub["Resignation_Target"] == 1).sum(), len(sub))
        band_rows.append(
            {
                "band": band,
                "count": int(len(sub)),
                "resignation_rate_pct": round(rate, 3),
                "resignation_lift_vs_overall_pp": round(rate - overall_resignation_rate, 3),
                "band_mean_value": round(float(sub[col].mean()), 4),
            }
        )
    quartile_summaries[col] = band_rows

# Combination pattern: low WLB + high OT + low engagement vs opposite.
wlb_q25 = float(work_df["Employee_Work_Life_Balance_Rating"].quantile(0.25))
wlb_q75 = float(work_df["Employee_Work_Life_Balance_Rating"].quantile(0.75))
ot_q25 = float(work_df["Overtime_Hours_Per_Week"].quantile(0.25))
ot_q75 = float(work_df["Overtime_Hours_Per_Week"].quantile(0.75))
eng_q25 = float(work_df["Employee_Engagement_Score"].quantile(0.25))
eng_q75 = float(work_df["Employee_Engagement_Score"].quantile(0.75))

high_risk_mask = (
    (pd.to_numeric(work_df["Employee_Work_Life_Balance_Rating"], errors="coerce") <= wlb_q25)
    & (pd.to_numeric(work_df["Overtime_Hours_Per_Week"], errors="coerce") >= ot_q75)
    & (pd.to_numeric(work_df["Employee_Engagement_Score"], errors="coerce") <= eng_q25)
)
protective_mask = (
    (pd.to_numeric(work_df["Employee_Work_Life_Balance_Rating"], errors="coerce") >= wlb_q75)
    & (pd.to_numeric(work_df["Overtime_Hours_Per_Week"], errors="coerce") <= ot_q25)
    & (pd.to_numeric(work_df["Employee_Engagement_Score"], errors="coerce") >= eng_q75)
)

high_risk_group = work_df[high_risk_mask]
protective_group = work_df[protective_mask]

high_risk_rate = _safe_pct((high_risk_group["Resignation_Target"] == 1).sum(), len(high_risk_group))
protective_rate = _safe_pct((protective_group["Resignation_Target"] == 1).sum(), len(protective_group))

segment_sizes = {
    "resigned_employees": int(len(resigned_df)),
    "retained_employees": int(len(retained_df)),
    "total_employees": int(len(work_df)),
}

result = {
    "question": "Q21 - Comparison of Work-Life Balance, Overtime, and Engagement: Resigned vs Retained",
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "resignation_status_classes": ["Yes", "No"],
        "features_compared": [f[0] for f in features],
    },
    "segment_sizes": segment_sizes,
    "overall_resignation_rate_pct": round(overall_resignation_rate, 3),
    "feature_comparison": comparison_rows,
    "quartile_risk_gradients": quartile_summaries,
    "combination_risk_pattern": {
        "high_risk_definition": {
            "work_life_balance_le_q25": round(wlb_q25, 4),
            "overtime_ge_q75": round(ot_q75, 4),
            "engagement_le_q25": round(eng_q25, 4),
        },
        "protective_definition": {
            "work_life_balance_ge_q75": round(wlb_q75, 4),
            "overtime_le_q25": round(ot_q25, 4),
            "engagement_ge_q75": round(eng_q75, 4),
        },
        "high_risk_group_count": int(len(high_risk_group)),
        "high_risk_group_resignation_rate_pct": round(high_risk_rate, 3),
        "high_risk_lift_vs_overall_pp": round(high_risk_rate - overall_resignation_rate, 3),
        "protective_group_count": int(len(protective_group)),
        "protective_group_resignation_rate_pct": round(protective_rate, 3),
        "protective_lift_vs_overall_pp": round(protective_rate - overall_resignation_rate, 3),
    },
    "llm_evidence": {
        "segment_sizes": segment_sizes,
        "overall_resignation_rate_pct": round(overall_resignation_rate, 3),
        "feature_comparison": comparison_rows,
        "quartile_risk_gradients": {
            k: v for k, v in quartile_summaries.items()
        },
        "combination_risk_pattern": {
            "high_risk_group_count": int(len(high_risk_group)),
            "high_risk_group_resignation_rate_pct": round(high_risk_rate, 3),
            "high_risk_lift_vs_overall_pp": round(high_risk_rate - overall_resignation_rate, 3),
            "protective_group_count": int(len(protective_group)),
            "protective_group_resignation_rate_pct": round(protective_rate, 3),
            "protective_lift_vs_overall_pp": round(protective_rate - overall_resignation_rate, 3),
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
_run_q21_llm_generation(ARGS)

