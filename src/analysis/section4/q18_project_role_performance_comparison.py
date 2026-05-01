"""Section 4 - Q18: Compare performance across Project_Roles (Manager/Developer/Analyst)."""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import math
import os
import argparse
import subprocess
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, kruskal, mannwhitneyu


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section4", "q18_project_role_performance_comparison.json")


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


def _cramers_v(table: pd.DataFrame) -> float:
    n = table.to_numpy().sum()
    if n == 0:
        return 0.0
    r, c = table.shape
    k = min(r - 1, c - 1)
    if k <= 0:
        return 0.0
    chi2, _, _, _ = chi2_contingency(table)
    return float(np.sqrt((chi2 / n) / k))


def _chi_square_summary(table: pd.DataFrame) -> dict:
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {
            "chi2_statistic": None,
            "p_value": None,
            "degrees_of_freedom": None,
            "cramers_v": 0.0,
            "interpretation": "Not enough category variation to run chi-square.",
        }

    chi2, p_value, dof, expected = chi2_contingency(table)
    min_expected = float(np.min(expected))
    cramers_v = _cramers_v(table)
    if cramers_v >= 0.35:
        strength = "strong"
    elif cramers_v >= 0.2:
        strength = "moderate"
    elif cramers_v >= 0.1:
        strength = "weak"
    else:
        strength = "very weak"
    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(float(cramers_v), 4),
        "min_expected_cell_count": round(min_expected, 4),
        "interpretation": (
            f"Association is {strength} (Cramer's V={cramers_v:.3f}) and "
            f"{'statistically significant' if p_value < 0.05 else 'not statistically significant'} at alpha=0.05."
        ),
    }


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q18 project-role performance comparison.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q18 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to Section 4 LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q18_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q18 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q18 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q18 Gemini insights generation")


ARGS = _parse_cli_args()


required_cols = ["Project_Role", "Performance_Rating", "Project_Outcome", "Employee_Resignation_Status", "Number_Of_Promotions"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q18: {missing_cols}")

work_df = DF.copy()
work_df["Project_Role"] = _normalize_label_col(work_df["Project_Role"])
work_df["Project_Outcome"] = _normalize_label_col(work_df["Project_Outcome"])
work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])

role_order = [r for r in ["Manager", "Developer", "Analyst"] if r in set(work_df["Project_Role"])]
role_df = work_df[work_df["Project_Role"].isin(role_order)].copy()
if len(role_order) < 2:
    raise ValueError("Q18 requires at least two project roles among Manager/Developer/Analyst.")

overall_perf_mean = float(pd.to_numeric(role_df["Performance_Rating"], errors="coerce").mean())

role_profiles = []
for role in role_order:
    sub = role_df[role_df["Project_Role"] == role]
    perf_vals = pd.to_numeric(sub["Performance_Rating"], errors="coerce").dropna()
    success_rate = _safe_pct((sub["Project_Outcome"] == "Successful").sum(), len(sub))
    fail_rate = _safe_pct((sub["Project_Outcome"] == "Failed").sum(), len(sub))
    attrition_rate = _safe_pct((sub["Employee_Resignation_Status"] == "Yes").sum(), len(sub))
    promoted_pct = _safe_pct((pd.to_numeric(sub["Number_Of_Promotions"], errors="coerce").fillna(0) > 0).sum(), len(sub))

    role_profiles.append(
        {
            "project_role": role,
            "count": int(len(sub)),
            "avg_performance_rating": round(float(perf_vals.mean()), 4),
            "median_performance_rating": round(float(perf_vals.median()), 4),
            "std_performance_rating": round(float(perf_vals.std(ddof=1)), 4),
            "iqr_performance_rating": round(float(perf_vals.quantile(0.75) - perf_vals.quantile(0.25)), 4),
            "successful_rate_pct": round(success_rate, 3),
            "failed_rate_pct": round(fail_rate, 3),
            "attrition_rate_pct": round(attrition_rate, 3),
            "promoted_pct": round(promoted_pct, 3),
            "performance_lift_vs_overall": round(float(perf_vals.mean()) - overall_perf_mean, 4),
        }
    )

role_profiles_sorted = sorted(role_profiles, key=lambda r: r["avg_performance_rating"], reverse=True)

perf_groups = []
for role in role_order:
    vals = pd.to_numeric(role_df[role_df["Project_Role"] == role]["Performance_Rating"], errors="coerce").dropna().to_numpy()
    if len(vals) > 0:
        perf_groups.append(vals)

anova_summary = {
    "f_statistic": None,
    "p_value": None,
    "is_statistically_significant_0_05": None,
}
if len(perf_groups) >= 2:
    f_stat, p_val = f_oneway(*perf_groups)
    anova_summary = {
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p_val), 6),
        "is_statistically_significant_0_05": bool(p_val < 0.05),
    }

kruskal_summary = {
    "h_statistic": None,
    "p_value": None,
    "is_statistically_significant_0_05": None,
}
if len(perf_groups) >= 2:
    h_stat, p_val = kruskal(*perf_groups)
    kruskal_summary = {
        "h_statistic": round(float(h_stat), 4),
        "p_value": round(float(p_val), 6),
        "is_statistically_significant_0_05": bool(p_val < 0.05),
    }

pairwise_rows = []
for role_a, role_b in combinations(role_order, 2):
    a_vals = pd.to_numeric(
        role_df[role_df["Project_Role"] == role_a]["Performance_Rating"], errors="coerce"
    ).dropna().to_numpy()
    b_vals = pd.to_numeric(
        role_df[role_df["Project_Role"] == role_b]["Performance_Rating"], errors="coerce"
    ).dropna().to_numpy()
    if len(a_vals) == 0 or len(b_vals) == 0:
        continue
    p_value = None
    try:
        _, p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        p_value = float(p)
    except Exception:
        p_value = None

    pairwise_rows.append(
        {
            "role_a": role_a,
            "role_b": role_b,
            "mean_role_a": round(float(np.mean(a_vals)), 4),
            "mean_role_b": round(float(np.mean(b_vals)), 4),
            "delta_a_minus_b": round(float(np.mean(a_vals) - np.mean(b_vals)), 4),
            "cohen_d": round(float(_cohen_d(a_vals, b_vals)), 4),
            "mannwhitney_p_value": round(p_value, 6) if p_value is not None else None,
            "is_statistically_significant_0_05": bool(p_value is not None and p_value < 0.05),
        }
    )

pairwise_rows = sorted(pairwise_rows, key=lambda r: abs(r["cohen_d"]), reverse=True)

role_vs_outcome = pd.crosstab(role_df["Project_Role"], role_df["Project_Outcome"]).reindex(index=role_order, fill_value=0)
role_vs_attrition = pd.crosstab(role_df["Project_Role"], role_df["Employee_Resignation_Status"]).reindex(index=role_order, fill_value=0)

outcome_assoc = _chi_square_summary(role_vs_outcome)
attrition_assoc = _chi_square_summary(role_vs_attrition)

best_role = role_profiles_sorted[0]["project_role"] if role_profiles_sorted else None
lowest_role = role_profiles_sorted[-1]["project_role"] if role_profiles_sorted else None

result = {
    "question": "Q18 - Performance Comparison Across Project Roles (Manager vs Developer vs Analyst)",
    "dataset_scope": {
        "rows_used": int(len(role_df)),
        "project_roles_included": role_order,
    },
    "role_performance_profile": role_profiles_sorted,
    "performance_tests": {
        "anova": anova_summary,
        "kruskal_wallis": kruskal_summary,
        "pairwise_role_differences": pairwise_rows,
    },
    "role_outcome_association": {
        "contingency_counts": role_vs_outcome.to_dict(),
        "association_test": outcome_assoc,
    },
    "role_attrition_association": {
        "contingency_counts": role_vs_attrition.to_dict(),
        "association_test": attrition_assoc,
    },
    "summary_signals": {
        "best_avg_performance_role": best_role,
        "lowest_avg_performance_role": lowest_role,
        "performance_gap_best_vs_lowest": round(
            role_profiles_sorted[0]["avg_performance_rating"] - role_profiles_sorted[-1]["avg_performance_rating"], 4
        )
        if len(role_profiles_sorted) >= 2
        else None,
    },
    "llm_evidence": {
        "role_performance_profile": [
            {
                "project_role": r["project_role"],
                "count": r["count"],
                "avg_performance_rating": r["avg_performance_rating"],
                "performance_lift_vs_overall": r["performance_lift_vs_overall"],
                "successful_rate_pct": r["successful_rate_pct"],
                "failed_rate_pct": r["failed_rate_pct"],
                "attrition_rate_pct": r["attrition_rate_pct"],
            }
            for r in role_profiles_sorted
        ],
        "anova": anova_summary,
        "kruskal_wallis": kruskal_summary,
        "top_pairwise_effects": pairwise_rows[:3],
        "role_outcome_association": outcome_assoc,
        "role_attrition_association": attrition_assoc,
        "summary_signals": {
            "best_avg_performance_role": best_role,
            "lowest_avg_performance_role": lowest_role,
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
_run_q18_llm_generation(ARGS)
