"""Section 7 - Q25: Impact of hiring source, time to hire, and recruitment cost on performance, retention, and satisfaction."""

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
from scipy.stats import chi2_contingency, f_oneway, kruskal, pearsonr, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section7", "q25_recruitment_hiring_impact.json")


def _normalize_label_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float((num / den) * 100)


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


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
    }


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q25 recruitment and hiring impact analysis.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q25 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q25_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q25 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q25 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q25 Gemini insights generation")


ARGS = _parse_cli_args()


required_cols = [
    "Hiring_Source",
    "Time_to_Hire",
    "Recruitment_Cost",
    "Performance_Rating",
    "Employee_Resignation_Status",
    "Employee_Job_Satisfaction_Score",
]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q25: {missing_cols}")

work_df = DF.copy()
work_df["Hiring_Source"] = _normalize_label_col(work_df["Hiring_Source"])
work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])
work_df = work_df[work_df["Employee_Resignation_Status"].isin(["Yes", "No"])].copy()
work_df["Resignation_Target"] = (work_df["Employee_Resignation_Status"] == "Yes").astype(int)
for c in ["Time_to_Hire", "Recruitment_Cost", "Performance_Rating", "Employee_Job_Satisfaction_Score"]:
    work_df[c] = pd.to_numeric(work_df[c], errors="coerce")
work_df = work_df.dropna(subset=["Time_to_Hire", "Recruitment_Cost", "Performance_Rating", "Employee_Job_Satisfaction_Score"]).copy()

overall_resignation_rate = _safe_pct((work_df["Resignation_Target"] == 1).sum(), len(work_df))
overall_perf = float(work_df["Performance_Rating"].mean())
overall_sat = float(work_df["Employee_Job_Satisfaction_Score"].mean())

# Hiring-source segment profiles.
source_profiles = (
    work_df.groupby("Hiring_Source", observed=True)
    .agg(
        count=("Hiring_Source", "count"),
        avg_performance=("Performance_Rating", "mean"),
        avg_satisfaction=("Employee_Job_Satisfaction_Score", "mean"),
        resignation_rate=("Resignation_Target", "mean"),
        avg_time_to_hire=("Time_to_Hire", "mean"),
        avg_recruitment_cost=("Recruitment_Cost", "mean"),
    )
    .reset_index()
)
source_profiles["avg_performance"] = source_profiles["avg_performance"].round(4)
source_profiles["avg_satisfaction"] = source_profiles["avg_satisfaction"].round(4)
source_profiles["resignation_rate_pct"] = (source_profiles["resignation_rate"] * 100).round(3)
source_profiles["resignation_lift_vs_overall_pp"] = (
    source_profiles["resignation_rate_pct"] - overall_resignation_rate
).round(3)
source_profiles["performance_delta_vs_overall"] = (
    source_profiles["avg_performance"] - overall_perf
).round(4)
source_profiles["satisfaction_delta_vs_overall"] = (
    source_profiles["avg_satisfaction"] - overall_sat
).round(4)
source_profiles["avg_time_to_hire"] = source_profiles["avg_time_to_hire"].round(4)
source_profiles["avg_recruitment_cost"] = source_profiles["avg_recruitment_cost"].round(4)

# Source significance tests.
source_groups_perf = [
    work_df[work_df["Hiring_Source"] == s]["Performance_Rating"].dropna().to_numpy()
    for s in source_profiles["Hiring_Source"].tolist()
]
source_groups_sat = [
    work_df[work_df["Hiring_Source"] == s]["Employee_Job_Satisfaction_Score"].dropna().to_numpy()
    for s in source_profiles["Hiring_Source"].tolist()
]

perf_source_anova = {"f_statistic": None, "p_value": None, "is_statistically_significant_0_05": None}
perf_source_kruskal = {"h_statistic": None, "p_value": None, "is_statistically_significant_0_05": None}
sat_source_anova = {"f_statistic": None, "p_value": None, "is_statistically_significant_0_05": None}
sat_source_kruskal = {"h_statistic": None, "p_value": None, "is_statistically_significant_0_05": None}
if len(source_groups_perf) >= 2:
    f_stat, f_p = f_oneway(*source_groups_perf)
    h_stat, h_p = kruskal(*source_groups_perf)
    perf_source_anova = {
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(f_p), 6),
        "is_statistically_significant_0_05": bool(f_p < 0.05),
    }
    perf_source_kruskal = {
        "h_statistic": round(float(h_stat), 4),
        "p_value": round(float(h_p), 6),
        "is_statistically_significant_0_05": bool(h_p < 0.05),
    }
if len(source_groups_sat) >= 2:
    f_stat, f_p = f_oneway(*source_groups_sat)
    h_stat, h_p = kruskal(*source_groups_sat)
    sat_source_anova = {
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(f_p), 6),
        "is_statistically_significant_0_05": bool(f_p < 0.05),
    }
    sat_source_kruskal = {
        "h_statistic": round(float(h_stat), 4),
        "p_value": round(float(h_p), 6),
        "is_statistically_significant_0_05": bool(h_p < 0.05),
    }

source_ret_table = pd.crosstab(work_df["Hiring_Source"], work_df["Employee_Resignation_Status"]).reindex(columns=["Yes", "No"], fill_value=0)
source_ret_assoc = _chi_square_summary(source_ret_table)

# Numeric hiring variable correlations.
numeric_corr = {}
for var in ["Time_to_Hire", "Recruitment_Cost"]:
    pear_perf = pearsonr(work_df[var], work_df["Performance_Rating"])
    spear_perf = spearmanr(work_df[var], work_df["Performance_Rating"], nan_policy="omit")
    pear_sat = pearsonr(work_df[var], work_df["Employee_Job_Satisfaction_Score"])
    spear_sat = spearmanr(work_df[var], work_df["Employee_Job_Satisfaction_Score"], nan_policy="omit")
    pear_ret = pearsonr(work_df[var], work_df["Resignation_Target"])
    spear_ret = spearmanr(work_df[var], work_df["Resignation_Target"], nan_policy="omit")
    numeric_corr[var] = {
        "vs_performance": {
            "pearson_r": round(float(pear_perf.statistic), 4),
            "pearson_p_value": round(float(pear_perf.pvalue), 6),
            "spearman_rho": round(float(spear_perf.correlation), 4),
            "spearman_p_value": round(float(spear_perf.pvalue), 6),
        },
        "vs_job_satisfaction": {
            "pearson_r": round(float(pear_sat.statistic), 4),
            "pearson_p_value": round(float(pear_sat.pvalue), 6),
            "spearman_rho": round(float(spear_sat.correlation), 4),
            "spearman_p_value": round(float(spear_sat.pvalue), 6),
        },
        "vs_retention_target_yes": {
            "pearson_r": round(float(pear_ret.statistic), 4),
            "pearson_p_value": round(float(pear_ret.pvalue), 6),
            "spearman_rho": round(float(spear_ret.correlation), 4),
            "spearman_p_value": round(float(spear_ret.pvalue), 6),
        },
    }

# Multivariable models with same predictors across outcomes.
feature_cols = ["Hiring_Source", "Time_to_Hire", "Recruitment_Cost"]
numeric_cols = ["Time_to_Hire", "Recruitment_Cost"]
categorical_cols = ["Hiring_Source"]
X = work_df[feature_cols].copy()

prep = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ],
    remainder="drop",
)

# Performance model
y_perf = work_df["Performance_Rating"].copy()
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X, y_perf, test_size=0.2, random_state=42)
perf_model = Pipeline(
    steps=[
        ("prep", prep),
        (
            "reg",
            RandomForestRegressor(
                n_estimators=450,
                max_depth=10,
                min_samples_leaf=6,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)
perf_model.fit(X_train_p, y_train_p)
perf_pred = perf_model.predict(X_test_p)
perf_r2 = r2_score(y_test_p, perf_pred)
perf_cv = KFold(n_splits=5, shuffle=True, random_state=42)
perf_cv_r2 = cross_val_score(perf_model, X, y_perf, cv=perf_cv, scoring="r2", n_jobs=1)

# Satisfaction model
y_sat = work_df["Employee_Job_Satisfaction_Score"].copy()
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X, y_sat, test_size=0.2, random_state=42)
sat_model = Pipeline(
    steps=[
        ("prep", prep),
        (
            "reg",
            RandomForestRegressor(
                n_estimators=450,
                max_depth=10,
                min_samples_leaf=6,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)
sat_model.fit(X_train_s, y_train_s)
sat_pred = sat_model.predict(X_test_s)
sat_r2 = r2_score(y_test_s, sat_pred)
sat_cv = KFold(n_splits=5, shuffle=True, random_state=42)
sat_cv_r2 = cross_val_score(sat_model, X, y_sat, cv=sat_cv, scoring="r2", n_jobs=1)

# Retention model
y_ret = work_df["Resignation_Target"].copy()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_ret, test_size=0.2, random_state=42, stratify=y_ret)
ret_model = Pipeline(
    steps=[
        ("prep", prep),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=450,
                max_depth=10,
                min_samples_leaf=6,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)
ret_model.fit(X_train_r, y_train_r)
ret_prob = ret_model.predict_proba(X_test_r)[:, 1]
ret_pred = (ret_prob >= 0.5).astype(int)
ret_metrics = _classification_metrics(y_test_r.to_numpy(), ret_pred, ret_prob)
ret_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ret_cv_auc = cross_val_score(ret_model, X, y_ret, cv=ret_cv, scoring="roc_auc", n_jobs=1)
ret_cv_pr = cross_val_score(ret_model, X, y_ret, cv=ret_cv, scoring="average_precision", n_jobs=1)

# Unified feature influence (per outcome).
perm_perf = permutation_importance(
    perf_model,
    X_test_p,
    y_test_p,
    n_repeats=12,
    random_state=42,
    scoring="r2",
    n_jobs=1,
)
perf_imp = []
for i, c in enumerate(feature_cols):
    perf_imp.append(
        {
            "feature": c,
            "importance_mean_r2_drop": round(float(perm_perf.importances_mean[i]), 6),
            "importance_std": round(float(perm_perf.importances_std[i]), 6),
        }
    )
perf_imp = sorted(perf_imp, key=lambda x: x["importance_mean_r2_drop"], reverse=True)

perm_sat = permutation_importance(
    sat_model,
    X_test_s,
    y_test_s,
    n_repeats=12,
    random_state=42,
    scoring="r2",
    n_jobs=1,
)
sat_imp = []
for i, c in enumerate(feature_cols):
    sat_imp.append(
        {
            "feature": c,
            "importance_mean_r2_drop": round(float(perm_sat.importances_mean[i]), 6),
            "importance_std": round(float(perm_sat.importances_std[i]), 6),
        }
    )
sat_imp = sorted(sat_imp, key=lambda x: x["importance_mean_r2_drop"], reverse=True)

perm_ret = permutation_importance(
    ret_model,
    X_test_r,
    y_test_r,
    n_repeats=12,
    random_state=42,
    scoring="roc_auc",
    n_jobs=1,
)
ret_imp = []
for i, c in enumerate(feature_cols):
    ret_imp.append(
        {
            "feature": c,
            "importance_mean_auc_drop": round(float(perm_ret.importances_mean[i]), 6),
            "importance_std": round(float(perm_ret.importances_std[i]), 6),
        }
    )
ret_imp = sorted(ret_imp, key=lambda x: x["importance_mean_auc_drop"], reverse=True)

# Interaction summaries by hiring source and time/cost quartiles.
inter_df = work_df.copy()
inter_df["time_band"] = pd.qcut(inter_df["Time_to_Hire"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
inter_df["cost_band"] = pd.qcut(inter_df["Recruitment_Cost"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

source_time_profiles = (
    inter_df.groupby(["Hiring_Source", "time_band"], observed=True)
    .agg(
        count=("Hiring_Source", "count"),
        avg_performance=("Performance_Rating", "mean"),
        resignation_rate=("Resignation_Target", "mean"),
        avg_satisfaction=("Employee_Job_Satisfaction_Score", "mean"),
    )
    .reset_index()
)
source_time_profiles["avg_performance"] = source_time_profiles["avg_performance"].round(4)
source_time_profiles["resignation_rate_pct"] = (source_time_profiles["resignation_rate"] * 100).round(3)
source_time_profiles["avg_satisfaction"] = source_time_profiles["avg_satisfaction"].round(4)

source_cost_profiles = (
    inter_df.groupby(["Hiring_Source", "cost_band"], observed=True)
    .agg(
        count=("Hiring_Source", "count"),
        avg_performance=("Performance_Rating", "mean"),
        resignation_rate=("Resignation_Target", "mean"),
        avg_satisfaction=("Employee_Job_Satisfaction_Score", "mean"),
    )
    .reset_index()
)
source_cost_profiles["avg_performance"] = source_cost_profiles["avg_performance"].round(4)
source_cost_profiles["resignation_rate_pct"] = (source_cost_profiles["resignation_rate"] * 100).round(3)
source_cost_profiles["avg_satisfaction"] = source_cost_profiles["avg_satisfaction"].round(4)

result = {
    "question": "Q25 - Impact of Hiring Source, Time to Hire, and Recruitment Cost on Performance, Retention, and Job Satisfaction",
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "features_used": feature_cols,
        "targets": ["Performance_Rating", "Employee_Resignation_Status", "Employee_Job_Satisfaction_Score"],
    },
    "hiring_source_profiles": source_profiles.drop(columns=["resignation_rate"]).to_dict(orient="records"),
    "hiring_source_significance": {
        "performance_anova": perf_source_anova,
        "performance_kruskal_wallis": perf_source_kruskal,
        "satisfaction_anova": sat_source_anova,
        "satisfaction_kruskal_wallis": sat_source_kruskal,
        "retention_association": {
            "contingency_counts": source_ret_table.to_dict(),
            "association_test": source_ret_assoc,
        },
    },
    "numeric_hiring_correlations": numeric_corr,
    "multivariable_models": {
        "performance_model": {
            "algorithm": "RandomForestRegressor",
            "test_r2": round(float(perf_r2), 4),
            "cv_r2_mean": round(float(perf_cv_r2.mean()), 4),
            "cv_r2_std": round(float(perf_cv_r2.std()), 4),
            "top_feature_importance": perf_imp,
        },
        "retention_model": {
            "algorithm": "RandomForestClassifier",
            "test_metrics": ret_metrics,
            "cv_roc_auc_mean": round(float(ret_cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(ret_cv_auc.std()), 4),
            "cv_pr_auc_mean": round(float(ret_cv_pr.mean()), 4),
            "cv_pr_auc_std": round(float(ret_cv_pr.std()), 4),
            "top_feature_importance": ret_imp,
        },
        "job_satisfaction_model": {
            "algorithm": "RandomForestRegressor",
            "test_r2": round(float(sat_r2), 4),
            "cv_r2_mean": round(float(sat_cv_r2.mean()), 4),
            "cv_r2_std": round(float(sat_cv_r2.std()), 4),
            "top_feature_importance": sat_imp,
        },
    },
    "interaction_patterns": {
        "source_time_quartile_profiles": source_time_profiles.to_dict(orient="records"),
        "source_cost_quartile_profiles": source_cost_profiles.to_dict(orient="records"),
    },
    "llm_evidence": {
        "rows_used": int(len(work_df)),
        "hiring_source_profiles": source_profiles.drop(columns=["resignation_rate"]).to_dict(orient="records"),
        "hiring_source_significance": {
            "performance_anova": perf_source_anova,
            "satisfaction_anova": sat_source_anova,
            "retention_association_test": source_ret_assoc,
        },
        "numeric_hiring_correlations": numeric_corr,
        "model_metrics": {
            "performance_model_cv_r2_mean": round(float(perf_cv_r2.mean()), 4),
            "retention_model_cv_roc_auc_mean": round(float(ret_cv_auc.mean()), 4),
            "retention_model_cv_pr_auc_mean": round(float(ret_cv_pr.mean()), 4),
            "satisfaction_model_cv_r2_mean": round(float(sat_cv_r2.mean()), 4),
        },
        "top_feature_influence": {
            "performance": perf_imp,
            "retention": ret_imp,
            "satisfaction": sat_imp,
        },
        "interaction_patterns_top": {
            "source_time_quartile_profiles": source_time_profiles.to_dict(orient="records")[:12],
            "source_cost_quartile_profiles": source_cost_profiles.to_dict(orient="records")[:12],
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
_run_q25_llm_generation(ARGS)

