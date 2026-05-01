"""Section 6 - Q24: Evaluate whether compensation benefits influence retention and satisfaction."""

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
from scipy.stats import chi2_contingency, f_oneway, kruskal
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
OUT = os.path.join(BASE, "reports", "section6", "q24_comp_benefits_influence_retention_satisfaction.json")


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


def _metric_pack(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
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
    parser = argparse.ArgumentParser(description="Run Q24 compensation-benefits influence analysis.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q24 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q24_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q24 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q24 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q24 Gemini insights generation")


ARGS = _parse_cli_args()


required_cols = ["Employee_Resignation_Status", "Employee_Job_Satisfaction_Score"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q24: {missing_cols}")

work_df = DF.copy()
work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])
work_df = work_df[work_df["Employee_Resignation_Status"].isin(["Yes", "No"])].copy()
work_df["Resignation_Target"] = (work_df["Employee_Resignation_Status"] == "Yes").astype(int)
work_df["Employee_Job_Satisfaction_Score"] = pd.to_numeric(work_df["Employee_Job_Satisfaction_Score"], errors="coerce")

categorical_benefits = [
    "Employee_Stock_Options",
    "Employee_Health_Insurance_Coverage",
    "Employee_Retirement_Benefits",
]
numeric_benefits = [
    "Employee_Compensation_Benefits",
    "Employee_Travel_Allowance",
    "Employee_Savings_Plans",
]
categorical_cols = [c for c in categorical_benefits if c in work_df.columns]
numeric_cols = [c for c in numeric_benefits if c in work_df.columns]
feature_cols = numeric_cols + categorical_cols

if len(feature_cols) < 3:
    raise ValueError("Q24 requires at least three compensation-benefit features in dataset.")

overall_resignation_rate = _safe_pct((work_df["Resignation_Target"] == 1).sum(), len(work_df))
overall_satisfaction = float(work_df["Employee_Job_Satisfaction_Score"].mean())

# Categorical benefit effect summaries for retention + satisfaction.
categorical_effects = {}
for col in categorical_cols:
    col_df = work_df[[col, "Resignation_Target", "Employee_Job_Satisfaction_Score"]].copy()
    col_df[col] = _normalize_label_col(col_df[col])
    retention_table = pd.crosstab(col_df[col], col_df["Resignation_Target"]).reindex(columns=[1, 0], fill_value=0)
    retention_assoc = _chi_square_summary(retention_table)

    sat_profiles = (
        col_df.groupby(col, observed=True)
        .agg(
            count=("Employee_Job_Satisfaction_Score", "count"),
            avg_satisfaction=("Employee_Job_Satisfaction_Score", "mean"),
            resignation_rate=("Resignation_Target", "mean"),
        )
        .reset_index()
    )
    sat_profiles["avg_satisfaction"] = sat_profiles["avg_satisfaction"].round(4)
    sat_profiles["resignation_rate_pct"] = (sat_profiles["resignation_rate"] * 100).round(3)
    sat_profiles["resignation_lift_vs_overall_pp"] = (
        sat_profiles["resignation_rate_pct"] - overall_resignation_rate
    ).round(3)
    sat_profiles["satisfaction_delta_vs_overall"] = (
        sat_profiles["avg_satisfaction"] - overall_satisfaction
    ).round(4)

    groups = []
    for level in sat_profiles[col].tolist():
        vals = col_df[col_df[col] == level]["Employee_Job_Satisfaction_Score"].dropna().to_numpy()
        if len(vals):
            groups.append(vals)
    anova = {"f_statistic": None, "p_value": None, "is_statistically_significant_0_05": None}
    krus = {"h_statistic": None, "p_value": None, "is_statistically_significant_0_05": None}
    if len(groups) >= 2:
        f_stat, f_p = f_oneway(*groups)
        h_stat, h_p = kruskal(*groups)
        anova = {
            "f_statistic": round(float(f_stat), 4),
            "p_value": round(float(f_p), 6),
            "is_statistically_significant_0_05": bool(f_p < 0.05),
        }
        krus = {
            "h_statistic": round(float(h_stat), 4),
            "p_value": round(float(h_p), 6),
            "is_statistically_significant_0_05": bool(h_p < 0.05),
        }

    categorical_effects[col] = {
        "retention_association": {
            "contingency_counts": retention_table.to_dict(),
            "association_test": retention_assoc,
        },
        "satisfaction_profiles": sat_profiles.drop(columns=["resignation_rate"]).to_dict(orient="records"),
        "satisfaction_tests": {
            "anova": anova,
            "kruskal_wallis": krus,
        },
    }

# Numeric benefit gradients.
numeric_gradients = []
for col in numeric_cols:
    sub = work_df[[col, "Resignation_Target", "Employee_Job_Satisfaction_Score"]].copy()
    sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=[col])
    sub["benefit_band"] = pd.qcut(sub[col], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    band = (
        sub.groupby("benefit_band", observed=True)
        .agg(
            count=("Resignation_Target", "count"),
            resignation_rate=("Resignation_Target", "mean"),
            avg_satisfaction=("Employee_Job_Satisfaction_Score", "mean"),
            avg_benefit_value=(col, "mean"),
        )
        .reset_index()
    )
    band["resignation_rate_pct"] = (band["resignation_rate"] * 100).round(3)
    band["resignation_lift_vs_overall_pp"] = (band["resignation_rate_pct"] - overall_resignation_rate).round(3)
    band["avg_satisfaction"] = band["avg_satisfaction"].round(4)
    band["satisfaction_delta_vs_overall"] = (band["avg_satisfaction"] - overall_satisfaction).round(4)
    band["avg_benefit_value"] = band["avg_benefit_value"].round(4)
    numeric_gradients.append(
        {
            "feature": col,
            "quartile_profiles": band.drop(columns=["resignation_rate"]).to_dict(orient="records"),
        }
    )

# Multivariable retention model (classification).
X = work_df[feature_cols].copy()
y_ret = work_df["Resignation_Target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y_ret, test_size=0.2, random_state=42, stratify=y_ret
)

numeric_transform = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)
categorical_transform = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]
)
prep = ColumnTransformer(
    transformers=[
        ("num", numeric_transform, numeric_cols),
        ("cat", categorical_transform, categorical_cols),
    ],
    remainder="drop",
)

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
ret_model.fit(X_train, y_train)
ret_prob = ret_model.predict_proba(X_test)[:, 1]
ret_pred = (ret_prob >= 0.5).astype(int)
ret_metrics = _metric_pack(y_test.to_numpy(), ret_pred, ret_prob)

ret_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ret_cv_auc = cross_val_score(ret_model, X, y_ret, cv=ret_cv, scoring="roc_auc", n_jobs=1)
ret_cv_pr = cross_val_score(ret_model, X, y_ret, cv=ret_cv, scoring="average_precision", n_jobs=1)

# Satisfaction model (regression).
y_sat = work_df["Employee_Job_Satisfaction_Score"].copy()
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
sat_model.fit(X_train, y_sat.loc[X_train.index])
sat_test_pred = sat_model.predict(X_test)
sat_r2 = r2_score(y_sat.loc[X_test.index], sat_test_pred)
sat_cv = KFold(n_splits=5, shuffle=True, random_state=42)
sat_cv_r2 = cross_val_score(sat_model, X, y_sat, cv=sat_cv, scoring="r2", n_jobs=1)

# Permutation importance for retention and satisfaction.
perm_ret = permutation_importance(
    ret_model,
    X_test,
    y_test,
    n_repeats=12,
    random_state=42,
    scoring="roc_auc",
    n_jobs=1,
)
ret_perm_rows = []
for i, col in enumerate(feature_cols):
    ret_perm_rows.append(
        {
            "feature": col,
            "importance_mean_auc_drop": round(float(perm_ret.importances_mean[i]), 6),
            "importance_std": round(float(perm_ret.importances_std[i]), 6),
        }
    )
ret_perm_rows = sorted(ret_perm_rows, key=lambda r: r["importance_mean_auc_drop"], reverse=True)

perm_sat = permutation_importance(
    sat_model,
    X_test,
    y_sat.loc[X_test.index],
    n_repeats=12,
    random_state=42,
    scoring="r2",
    n_jobs=1,
)
sat_perm_rows = []
for i, col in enumerate(feature_cols):
    sat_perm_rows.append(
        {
            "feature": col,
            "importance_mean_r2_drop": round(float(perm_sat.importances_mean[i]), 6),
            "importance_std": round(float(perm_sat.importances_std[i]), 6),
        }
    )
sat_perm_rows = sorted(sat_perm_rows, key=lambda r: r["importance_mean_r2_drop"], reverse=True)

result = {
    "question": "Q24 - Influence of Compensation Benefits on Retention and Satisfaction",
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "categorical_benefits": categorical_cols,
        "numeric_benefits": numeric_cols,
    },
    "baseline_metrics": {
        "overall_resignation_rate_pct": round(overall_resignation_rate, 3),
        "overall_avg_satisfaction": round(overall_satisfaction, 4),
    },
    "categorical_benefit_effects": categorical_effects,
    "numeric_benefit_gradients": numeric_gradients,
    "multivariable_benefit_models": {
        "retention_model": {
            "algorithm": "RandomForestClassifier",
            "test_metrics": ret_metrics,
            "cv_roc_auc_mean": round(float(ret_cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(ret_cv_auc.std()), 4),
            "cv_pr_auc_mean": round(float(ret_cv_pr.mean()), 4),
            "cv_pr_auc_std": round(float(ret_cv_pr.std()), 4),
            "top_feature_importance": ret_perm_rows[:10],
        },
        "satisfaction_model": {
            "algorithm": "RandomForestRegressor",
            "test_r2": round(float(sat_r2), 4),
            "cv_r2_mean": round(float(sat_cv_r2.mean()), 4),
            "cv_r2_std": round(float(sat_cv_r2.std()), 4),
            "top_feature_importance": sat_perm_rows[:10],
        },
    },
    "llm_evidence": {
        "baseline_metrics": {
            "overall_resignation_rate_pct": round(overall_resignation_rate, 3),
            "overall_avg_satisfaction": round(overall_satisfaction, 4),
        },
        "categorical_effect_summary": {
            k: {
                "retention_association_test": v["retention_association"]["association_test"],
                "satisfaction_profiles_top": sorted(
                    v["satisfaction_profiles"], key=lambda x: abs(x["satisfaction_delta_vs_overall"]), reverse=True
                )[:4],
            }
            for k, v in categorical_effects.items()
        },
        "numeric_benefit_gradients": numeric_gradients,
        "retention_model_metrics": {
            "test_metrics": ret_metrics,
            "cv_roc_auc_mean": round(float(ret_cv_auc.mean()), 4),
            "cv_pr_auc_mean": round(float(ret_cv_pr.mean()), 4),
        },
        "satisfaction_model_metrics": {
            "test_r2": round(float(sat_r2), 4),
            "cv_r2_mean": round(float(sat_cv_r2.mean()), 4),
        },
        "top_retention_benefit_features": ret_perm_rows[:8],
        "top_satisfaction_benefit_features": sat_perm_rows[:8],
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
_run_q24_llm_generation(ARGS)
