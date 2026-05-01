"""Section 4 - Q15: Analyze how Project_Complexity and Project_Size influence Project_Outcome."""
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section4", "q15_project_complexity_size_impact.json")


def _normalize_label_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _to_pct(series: pd.Series) -> dict:
    return (series * 100).round(2).to_dict()


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
    elif cramers_v >= 0.20:
        strength = "moderate"
    elif cramers_v >= 0.10:
        strength = "weak"
    else:
        strength = "very weak"

    interpretation = (
        f"Association is {strength} (Cramer's V={cramers_v:.3f}); "
        f"{'statistically significant' if p_value < 0.05 else 'not statistically significant'} "
        f"at alpha=0.05."
    )
    if min_expected < 5:
        interpretation += " Some expected cell counts are below 5, so interpret with caution."

    return {
        "chi2_statistic": round(float(chi2), 4),
        "p_value": round(float(p_value), 6),
        "degrees_of_freedom": int(dof),
        "cramers_v": round(cramers_v, 4),
        "min_expected_cell_count": round(min_expected, 4),
        "interpretation": interpretation,
    }


def _row_profiles(table: pd.DataFrame, baseline: dict, segment_label: str) -> list[dict]:
    rows = []
    for idx, row in table.iterrows():
        total = int(row.sum())
        if total == 0:
            continue
        rates = (row / total * 100).round(3)
        success_rate = float(rates.get("Successful", 0.0))
        fail_rate = float(rates.get("Failed", 0.0))
        rows.append(
            {
                segment_label: str(idx),
                "count": total,
                "successful_rate_pct": round(success_rate, 3),
                "failed_rate_pct": round(fail_rate, 3),
                "successful_lift_vs_overall_pp": round(success_rate - baseline["successful_rate_pct"], 3),
                "failed_lift_vs_overall_pp": round(fail_rate - baseline["failed_rate_pct"], 3),
                "outcome_distribution_pct": _to_pct((row / total).round(6)),
            }
        )
    return rows


required_cols = ["Project_Complexity", "Project_Size", "Project_Outcome"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q15: {missing_cols}")

work_df = DF[required_cols].copy()
work_df["Project_Complexity"] = _normalize_label_col(work_df["Project_Complexity"])
work_df["Project_Size"] = _normalize_label_col(work_df["Project_Size"])
work_df["Project_Outcome"] = _normalize_label_col(work_df["Project_Outcome"])
work_df = work_df.dropna()

outcome_order = [c for c in ["Successful", "In Progress", "Failed"] if c in set(work_df["Project_Outcome"])]
complexity_order = [c for c in ["Simple", "Moderate", "Complex"] if c in set(work_df["Project_Complexity"])]
size_order = [c for c in ["Small", "Medium", "Large"] if c in set(work_df["Project_Size"])]

complexity_table = pd.crosstab(work_df["Project_Complexity"], work_df["Project_Outcome"]).reindex(
    index=complexity_order, columns=outcome_order, fill_value=0
)
size_table = pd.crosstab(work_df["Project_Size"], work_df["Project_Outcome"]).reindex(
    index=size_order, columns=outcome_order, fill_value=0
)

overall_outcome_dist = work_df["Project_Outcome"].value_counts(normalize=True)
baseline = {
    "successful_rate_pct": round(float(overall_outcome_dist.get("Successful", 0.0) * 100), 3),
    "failed_rate_pct": round(float(overall_outcome_dist.get("Failed", 0.0) * 100), 3),
    "in_progress_rate_pct": round(float(overall_outcome_dist.get("In Progress", 0.0) * 100), 3),
}

complexity_profiles = _row_profiles(complexity_table, baseline, "project_complexity")
size_profiles = _row_profiles(size_table, baseline, "project_size")

combo_table = pd.crosstab(
    [work_df["Project_Complexity"], work_df["Project_Size"]],
    work_df["Project_Outcome"],
).reindex(columns=outcome_order, fill_value=0)

fail_rate_by_complexity = {
    r["project_complexity"]: r["failed_rate_pct"] for r in complexity_profiles
}
fail_rate_by_size = {
    r["project_size"]: r["failed_rate_pct"] for r in size_profiles
}

combo_profiles = []
for (complexity, size), row in combo_table.iterrows():
    total = int(row.sum())
    if total == 0:
        continue
    rates = (row / total * 100).round(3)
    success_rate = float(rates.get("Successful", 0.0))
    fail_rate = float(rates.get("Failed", 0.0))
    expected_fail_additive = (
        fail_rate_by_complexity.get(complexity, baseline["failed_rate_pct"])
        + fail_rate_by_size.get(size, baseline["failed_rate_pct"])
        - baseline["failed_rate_pct"]
    )
    combo_profiles.append(
        {
            "project_complexity": str(complexity),
            "project_size": str(size),
            "count": total,
            "successful_rate_pct": round(success_rate, 3),
            "failed_rate_pct": round(fail_rate, 3),
            "successful_lift_vs_overall_pp": round(success_rate - baseline["successful_rate_pct"], 3),
            "failed_lift_vs_overall_pp": round(fail_rate - baseline["failed_rate_pct"], 3),
            "failure_synergy_vs_additive_expectation_pp": round(fail_rate - expected_fail_additive, 3),
            "outcome_distribution_pct": _to_pct((row / total).round(6)),
        }
    )

min_segment_size = 60
stable_combo_profiles = [r for r in combo_profiles if r["count"] >= min_segment_size]
top_success_segments = sorted(
    stable_combo_profiles,
    key=lambda x: (x["successful_lift_vs_overall_pp"], x["count"]),
    reverse=True,
)[:5]
top_failure_risk_segments = sorted(
    stable_combo_profiles,
    key=lambda x: (x["failed_lift_vs_overall_pp"], x["count"]),
    reverse=True,
)[:5]
top_failure_synergy_segments = sorted(
    stable_combo_profiles,
    key=lambda x: (x["failure_synergy_vs_additive_expectation_pp"], x["count"]),
    reverse=True,
)[:5]

compact_complexity_profiles = [
    {
        "project_complexity": r["project_complexity"],
        "count": r["count"],
        "successful_rate_pct": r["successful_rate_pct"],
        "failed_rate_pct": r["failed_rate_pct"],
        "failed_lift_vs_overall_pp": r["failed_lift_vs_overall_pp"],
    }
    for r in complexity_profiles
]
compact_size_profiles = [
    {
        "project_size": r["project_size"],
        "count": r["count"],
        "successful_rate_pct": r["successful_rate_pct"],
        "failed_rate_pct": r["failed_rate_pct"],
        "failed_lift_vs_overall_pp": r["failed_lift_vs_overall_pp"],
    }
    for r in size_profiles
]

# Ordinal trend checks to capture directional signal.
complexity_rank_map = {"Simple": 1, "Moderate": 2, "Complex": 3}
size_rank_map = {"Small": 1, "Medium": 2, "Large": 3}
trend_df = work_df.copy()
trend_df["complexity_rank"] = trend_df["Project_Complexity"].map(complexity_rank_map)
trend_df["size_rank"] = trend_df["Project_Size"].map(size_rank_map)
trend_df["is_failed"] = (trend_df["Project_Outcome"] == "Failed").astype(int)
trend_df["is_successful"] = (trend_df["Project_Outcome"] == "Successful").astype(int)

complexity_fail_spearman = spearmanr(trend_df["complexity_rank"], trend_df["is_failed"], nan_policy="omit")
size_fail_spearman = spearmanr(trend_df["size_rank"], trend_df["is_failed"], nan_policy="omit")
complexity_success_spearman = spearmanr(trend_df["complexity_rank"], trend_df["is_successful"], nan_policy="omit")
size_success_spearman = spearmanr(trend_df["size_rank"], trend_df["is_successful"], nan_policy="omit")

result = {
    "question": "Q15 - Impact of Project_Complexity and Project_Size on Project_Outcome",
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "project_complexity_levels": complexity_order,
        "project_size_levels": size_order,
        "project_outcome_levels": outcome_order,
    },
    "overall_outcome_baseline_pct": baseline,
    "complexity_vs_outcome": {
        "contingency_counts": complexity_table.to_dict(),
        "row_profiles": complexity_profiles,
        "association_test": _chi_square_summary(complexity_table),
    },
    "size_vs_outcome": {
        "contingency_counts": size_table.to_dict(),
        "row_profiles": size_profiles,
        "association_test": _chi_square_summary(size_table),
    },
    "complexity_size_interaction": {
        "min_segment_size_for_ranking": min_segment_size,
        "stable_segment_count": int(len(stable_combo_profiles)),
        "top_success_segments": top_success_segments,
        "top_failure_risk_segments": top_failure_risk_segments,
        "top_failure_synergy_segments": top_failure_synergy_segments,
    },
    "ordinal_trend_checks": {
        "complexity_rank_vs_failure_spearman_rho": round(float(complexity_fail_spearman.correlation), 4),
        "complexity_rank_vs_failure_p_value": round(float(complexity_fail_spearman.pvalue), 6),
        "size_rank_vs_failure_spearman_rho": round(float(size_fail_spearman.correlation), 4),
        "size_rank_vs_failure_p_value": round(float(size_fail_spearman.pvalue), 6),
        "complexity_rank_vs_success_spearman_rho": round(float(complexity_success_spearman.correlation), 4),
        "complexity_rank_vs_success_p_value": round(float(complexity_success_spearman.pvalue), 6),
        "size_rank_vs_success_spearman_rho": round(float(size_success_spearman.correlation), 4),
        "size_rank_vs_success_p_value": round(float(size_success_spearman.pvalue), 6),
    },
    "llm_evidence": {
        "overall_outcome_baseline_pct": baseline,
        "complexity_association": _chi_square_summary(complexity_table),
        "size_association": _chi_square_summary(size_table),
        "complexity_profiles": compact_complexity_profiles,
        "size_profiles": compact_size_profiles,
        "top_success_segments": top_success_segments,
        "top_failure_risk_segments": top_failure_risk_segments,
        "top_failure_synergy_segments": top_failure_synergy_segments,
        "ordinal_trend_checks": {
            "complexity_rank_vs_failure_spearman_rho": round(float(complexity_fail_spearman.correlation), 4),
            "complexity_rank_vs_failure_p_value": round(float(complexity_fail_spearman.pvalue), 6),
            "size_rank_vs_failure_spearman_rho": round(float(size_fail_spearman.correlation), 4),
            "size_rank_vs_failure_p_value": round(float(size_fail_spearman.pvalue), 6),
        },
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
