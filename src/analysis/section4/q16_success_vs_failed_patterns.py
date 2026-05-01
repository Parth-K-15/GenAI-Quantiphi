"""Section 4 - Q16: Identify patterns among employees involved in successful vs failed projects."""
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section4", "q16_success_vs_failed_patterns.json")


def _normalize_label_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float((numerator / denominator) * 100)


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


def _effect_strength(d_value: float) -> str:
    ad = abs(d_value)
    if ad >= 0.8:
        return "large"
    if ad >= 0.5:
        return "medium"
    if ad >= 0.2:
        return "small"
    return "very_small"


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


required_cols = ["Project_Outcome"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q16: {missing_cols}")

work_df = DF.copy()
work_df["Project_Outcome"] = _normalize_label_col(work_df["Project_Outcome"])
work_df = work_df[work_df["Project_Outcome"].isin(["Successful", "Failed"])].copy()

success_df = work_df[work_df["Project_Outcome"] == "Successful"].copy()
failed_df = work_df[work_df["Project_Outcome"] == "Failed"].copy()

if len(success_df) == 0 or len(failed_df) == 0:
    raise ValueError("Q16 requires both 'Successful' and 'Failed' outcomes to be present.")

segment_sizes = {
    "successful_projects": int(len(success_df)),
    "failed_projects": int(len(failed_df)),
    "total_binary_outcome_projects": int(len(work_df)),
}

overall_success_rate = _safe_pct(len(success_df), len(work_df))
overall_failure_rate = _safe_pct(len(failed_df), len(work_df))

numeric_candidates = [
    "Performance_Rating",
    "Technical_Skills_Rating",
    "Communication_Skills_Rating",
    "Problem_Solving_Skills_Rating",
    "Leadership_Qualities_Rating",
    "Initiative_Rating",
    "Adaptability_Rating",
    "Creativity_Rating",
    "Strategic_Thinking_Rating",
    "Teamwork_Skills_Rating",
    "Avg_Skills_Score",
    "Avg_Soft_Skills_Score",
    "Training_Efficiency",
    "Employee_Engagement_Score",
    "Employee_Job_Satisfaction_Score",
    "Employee_Work_Life_Balance_Rating",
    "Overtime_Hours_Per_Week",
    "Professional_Development_Hours",
    "Number_Of_Promotions",
    "Conflict_Resolution_Cases",
    "Feedback_From_Colleagues",
    "Feedback_From_Supervisors",
    "Mentor_Rating",
    "Onboarding_Delay_Days",
    "Tenure_Years",
]
numeric_cols = [c for c in numeric_candidates if c in work_df.columns]

numeric_comparison = []
for col in numeric_cols:
    s_vals = pd.to_numeric(success_df[col], errors="coerce").dropna().to_numpy()
    f_vals = pd.to_numeric(failed_df[col], errors="coerce").dropna().to_numpy()
    if len(s_vals) == 0 or len(f_vals) == 0:
        continue

    s_mean = float(np.mean(s_vals))
    f_mean = float(np.mean(f_vals))
    delta = s_mean - f_mean
    d_val = _cohen_d(s_vals, f_vals)
    p_value = None
    try:
        if len(s_vals) >= 20 and len(f_vals) >= 20:
            _, p = mannwhitneyu(s_vals, f_vals, alternative="two-sided")
            p_value = float(p)
    except Exception:
        p_value = None

    numeric_comparison.append(
        {
            "feature": col,
            "successful_mean": round(s_mean, 4),
            "failed_mean": round(f_mean, 4),
            "delta_success_minus_failed": round(delta, 4),
            "relative_diff_vs_failed_pct": round(_safe_pct(delta, f_mean), 3) if f_mean != 0 else None,
            "cohen_d": round(d_val, 4),
            "effect_strength": _effect_strength(d_val),
            "p_value_mannwhitney": round(p_value, 6) if p_value is not None else None,
            "is_statistically_significant_0_05": bool(p_value is not None and p_value < 0.05),
        }
    )

numeric_ranked = sorted(numeric_comparison, key=lambda x: abs(x["cohen_d"]), reverse=True)
top_numeric_differentiators = numeric_ranked[:12]

categorical_candidates = [
    "Project_Complexity",
    "Project_Size",
    "Project_Role",
    "Department",
    "Job_Title",
    "Training_Program",
    "Leadership_Potential",
    "Innovation_Projects_Involvement",
    "Employee_Resignation_Status",
    "Development_Plan_Completion",
    "Career_Goals_Achievement_Status",
    "Mentor_Experience_Level",
]
categorical_cols = [c for c in categorical_candidates if c in work_df.columns]

categorical_pattern_summary = {}
all_category_rows = []
association_strength_ranking = []
min_category_count = 60

for col in categorical_cols:
    col_df = work_df[[col, "Project_Outcome"]].copy()
    col_df[col] = _normalize_label_col(col_df[col])
    table = pd.crosstab(col_df[col], col_df["Project_Outcome"]).reindex(columns=["Successful", "Failed"], fill_value=0)
    assoc = _chi_square_summary(table)
    association_strength_ranking.append(
        {
            "feature": col,
            "cramers_v": assoc["cramers_v"],
            "p_value": assoc["p_value"],
            "interpretation": assoc["interpretation"],
        }
    )

    profiles = []
    for category, row in table.iterrows():
        total = int(row.sum())
        if total == 0:
            continue
        s_count = int(row.get("Successful", 0))
        f_count = int(row.get("Failed", 0))
        s_rate = _safe_pct(s_count, total)
        f_rate = _safe_pct(f_count, total)
        record = {
            "feature": col,
            "category": str(category),
            "count": total,
            "successful_count": s_count,
            "failed_count": f_count,
            "successful_rate_pct": round(s_rate, 3),
            "failed_rate_pct": round(f_rate, 3),
            "successful_lift_vs_overall_pp": round(s_rate - overall_success_rate, 3),
            "failed_lift_vs_overall_pp": round(f_rate - overall_failure_rate, 3),
        }
        profiles.append(record)
        if total >= min_category_count:
            all_category_rows.append(record)

    categorical_pattern_summary[col] = {
        "contingency_counts": table.to_dict(),
        "association_test": assoc,
        "profiles": sorted(profiles, key=lambda x: x["count"], reverse=True),
    }

top_failure_risk_categories = sorted(
    all_category_rows,
    key=lambda x: (x["failed_lift_vs_overall_pp"], x["count"]),
    reverse=True,
)[:12]
top_success_advantage_categories = sorted(
    all_category_rows,
    key=lambda x: (x["successful_lift_vs_overall_pp"], x["count"]),
    reverse=True,
)[:12]

combo_cols = [c for c in ["Project_Complexity", "Project_Size", "Project_Role"] if c in work_df.columns]
combo_profiles = []
combo_min_count = 50
if len(combo_cols) == 3:
    combo_df = work_df[combo_cols + ["Project_Outcome"]].copy()
    for col in combo_cols:
        combo_df[col] = _normalize_label_col(combo_df[col])
    combo_table = pd.crosstab(
        [combo_df["Project_Complexity"], combo_df["Project_Size"], combo_df["Project_Role"]],
        combo_df["Project_Outcome"],
    ).reindex(columns=["Successful", "Failed"], fill_value=0)

    for idx, row in combo_table.iterrows():
        total = int(row.sum())
        if total < combo_min_count:
            continue
        success_rate = _safe_pct(int(row.get("Successful", 0)), total)
        fail_rate = _safe_pct(int(row.get("Failed", 0)), total)
        combo_profiles.append(
            {
                "project_complexity": str(idx[0]),
                "project_size": str(idx[1]),
                "project_role": str(idx[2]),
                "count": total,
                "successful_rate_pct": round(success_rate, 3),
                "failed_rate_pct": round(fail_rate, 3),
                "successful_lift_vs_overall_pp": round(success_rate - overall_success_rate, 3),
                "failed_lift_vs_overall_pp": round(fail_rate - overall_failure_rate, 3),
            }
        )

top_combo_success_patterns = sorted(
    combo_profiles,
    key=lambda x: (x["successful_lift_vs_overall_pp"], x["count"]),
    reverse=True,
)[:8]
top_combo_failure_patterns = sorted(
    combo_profiles,
    key=lambda x: (x["failed_lift_vs_overall_pp"], x["count"]),
    reverse=True,
)[:8]

strongest_associations = sorted(
    association_strength_ranking,
    key=lambda x: x["cramers_v"],
    reverse=True,
)[:8]

compact_numeric_evidence = [
    {
        "feature": row["feature"],
        "delta": row["delta_success_minus_failed"],
        "cohen_d": row["cohen_d"],
        "p_value": row["p_value_mannwhitney"],
    }
    for row in top_numeric_differentiators[:8]
]

compact_failure_categories = [
    {
        "feature": row["feature"],
        "category": row["category"],
        "failed_lift_pp": row["failed_lift_vs_overall_pp"],
        "count": row["count"],
    }
    for row in top_failure_risk_categories[:8]
]

compact_success_categories = [
    {
        "feature": row["feature"],
        "category": row["category"],
        "success_lift_pp": row["successful_lift_vs_overall_pp"],
        "count": row["count"],
    }
    for row in top_success_advantage_categories[:8]
]

result = {
    "question": "Q16 - Patterns Among Employees Involved in Successful vs Failed Projects",
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "outcomes_included": ["Successful", "Failed"],
    },
    "segment_sizes": segment_sizes,
    "overall_baseline_rates_pct": {
        "successful_rate_pct": round(overall_success_rate, 3),
        "failed_rate_pct": round(overall_failure_rate, 3),
    },
    "numeric_comparison": {
        "all_features": numeric_comparison,
        "top_differentiators_by_effect_size": top_numeric_differentiators,
    },
    "categorical_patterns": {
        "min_category_count_for_ranking": min_category_count,
        "feature_wise": categorical_pattern_summary,
        "top_failure_risk_categories": top_failure_risk_categories,
        "top_success_advantage_categories": top_success_advantage_categories,
        "strongest_association_features": strongest_associations,
    },
    "interaction_patterns": {
        "combo_features": combo_cols,
        "min_combo_count_for_ranking": combo_min_count,
        "stable_combo_count": int(len(combo_profiles)),
        "top_combo_success_patterns": top_combo_success_patterns,
        "top_combo_failure_patterns": top_combo_failure_patterns,
    },
    "llm_evidence": {
        "segment_sizes": segment_sizes,
        "overall_baseline_rates_pct": {
            "successful_rate_pct": round(overall_success_rate, 3),
            "failed_rate_pct": round(overall_failure_rate, 3),
        },
        "top_numeric_effects": compact_numeric_evidence,
        "top_failure_categories": compact_failure_categories,
        "top_success_categories": compact_success_categories,
        "strongest_associations": strongest_associations,
        "top_combo_failure_patterns": top_combo_failure_patterns[:5],
        "top_combo_success_patterns": top_combo_success_patterns[:5],
    },
    "llm_insights": {
        "headline": "Pending Gemini generation.",
        "key_insight_1": "Pending Gemini generation.",
        "key_insight_2": "Pending Gemini generation.",
        "hidden_insight": "Pending Gemini generation.",
        "business_implication": "Pending Gemini generation.",
        "standout_statement": "Pending Gemini generation.",
    },
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False, default=str)

print(f"[OK] Saved -> {OUT}")
