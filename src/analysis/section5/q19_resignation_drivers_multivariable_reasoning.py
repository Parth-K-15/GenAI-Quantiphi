"""Section 5 - Q19: Identify factors contributing to employee resignation using multi-variable reasoning."""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section5", "q19_resignation_drivers_multivariable_reasoning.json")


def _normalize_label_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float((num / den) * 100)


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


required_cols = ["Employee_Resignation_Status"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q19: {missing_cols}")

work_df = DF.copy()
work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])
work_df = work_df[work_df["Employee_Resignation_Status"].isin(["Yes", "No"])].copy()
work_df["Resignation_Target"] = (work_df["Employee_Resignation_Status"] == "Yes").astype(int)

if work_df["Resignation_Target"].nunique() < 2:
    raise ValueError("Q19 needs both resignation classes: Yes and No.")

numeric_candidates = [
    "Performance_Rating",
    "Employee_Engagement_Score",
    "Employee_Job_Satisfaction_Score",
    "Employee_Work_Life_Balance_Rating",
    "Overtime_Hours_Per_Week",
    "Work_Hours_Per_Week",
    "Number_Of_Promotions",
    "Professional_Development_Hours",
    "Annual_Salary_Increase_Percentage",
    "Performance_Bonus_Percentage",
    "Bonus",
    "Employee_Annual_Salary_Adjustment",
    "Employee_Compensation_Benefits",
    "Employee_Travel_Allowance",
    "Employee_Savings_Plans",
    "Compensation_Score",
    "Tenure_Years",
    "Onboarding_Delay_Days",
    "Feedback_From_Colleagues",
    "Feedback_From_Supervisors",
    "Mentor_Rating",
    "Conflict_Resolution_Cases",
    "Initiative_Rating",
    "Adaptability_Rating",
    "Creativity_Rating",
    "Strategic_Thinking_Rating",
    "Teamwork_Skills_Rating",
    "Avg_Skills_Score",
    "Avg_Soft_Skills_Score",
    "Training_Efficiency",
]
categorical_candidates = [
    "Department",
    "Job_Title",
    "Project_Role",
    "Project_Complexity",
    "Project_Size",
    "Training_Program",
    "Leadership_Potential",
    "Development_Plan_Completion",
    "Career_Goals_Achievement_Status",
    "Mentor_Experience_Level",
    "Employee_Stock_Options",
    "Employee_Health_Insurance_Coverage",
    "Hiring_Source",
]

numeric_cols = [c for c in numeric_candidates if c in work_df.columns]
categorical_cols = [c for c in categorical_candidates if c in work_df.columns]

if len(numeric_cols) + len(categorical_cols) < 6:
    raise ValueError("Not enough candidate features found for Q19.")

feature_cols = numeric_cols + categorical_cols
X = work_df[feature_cols].copy()
y = work_df["Resignation_Target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transform, numeric_cols),
        ("cat", categorical_transform, categorical_cols),
    ],
    remainder="drop",
)

logistic_model = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "clf",
            LogisticRegression(
                max_iter=3500,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

rf_model = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

models = {
    "logistic_regression": logistic_model,
    "random_forest": rf_model,
}

model_eval = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(model, X, y, cv=fold, scoring="roc_auc", n_jobs=1)
    cv_pr = cross_val_score(model, X, y, cv=fold, scoring="average_precision", n_jobs=1)
    model_eval[name] = {
        "test_metrics": _metric_pack(y_test.to_numpy(), pred, prob),
        "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
        "cv_roc_auc_std": round(float(cv_auc.std()), 4),
        "cv_pr_auc_mean": round(float(cv_pr.mean()), 4),
        "cv_pr_auc_std": round(float(cv_pr.std()), 4),
    }

selected_model_name = max(models.keys(), key=lambda m: model_eval[m]["cv_roc_auc_mean"])
selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)

test_prob = selected_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)
selected_metrics = _metric_pack(y_test.to_numpy(), test_pred, test_prob)

# Out-of-fold probabilities for unbiased risk profiling.
oof_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = cross_val_predict(
    selected_model,
    X,
    y,
    cv=oof_fold,
    method="predict_proba",
    n_jobs=1,
)[:, 1]
work_df["Predicted_Resignation_Probability"] = oof_prob

perm = permutation_importance(
    selected_model,
    X_test,
    y_test,
    n_repeats=12,
    random_state=42,
    scoring="roc_auc",
    n_jobs=1,
)
perm_rows = []
for i, col in enumerate(feature_cols):
    perm_rows.append(
        {
            "feature": col,
            "importance_mean_auc_drop": round(float(perm.importances_mean[i]), 6),
            "importance_std": round(float(perm.importances_std[i]), 6),
        }
    )
perm_rows = sorted(perm_rows, key=lambda r: r["importance_mean_auc_drop"], reverse=True)
top_perm_features = perm_rows[:12]

# Logistic coefficients for interpretable factor direction.
logistic_model.fit(X_train, y_train)
prep = logistic_model.named_steps["prep"]
coef = logistic_model.named_steps["clf"].coef_[0]
feature_names = [str(f).replace("num__", "").replace("cat__", "") for f in prep.get_feature_names_out()]
coef_rows = []
for fname, c in zip(feature_names, coef):
    coef_rows.append(
        {
            "feature": fname,
            "coefficient": round(float(c), 6),
            "odds_ratio": round(float(np.exp(c)), 6),
        }
    )
coef_rows = sorted(coef_rows, key=lambda r: abs(r["coefficient"]), reverse=True)

resigned_df = work_df[work_df["Resignation_Target"] == 1].copy()
retained_df = work_df[work_df["Resignation_Target"] == 0].copy()

numeric_differences = []
for col in numeric_cols:
    yes_vals = pd.to_numeric(resigned_df[col], errors="coerce").dropna().to_numpy()
    no_vals = pd.to_numeric(retained_df[col], errors="coerce").dropna().to_numpy()
    if len(yes_vals) == 0 or len(no_vals) == 0:
        continue
    p_value = None
    try:
        _, p = mannwhitneyu(yes_vals, no_vals, alternative="two-sided")
        p_value = float(p)
    except Exception:
        p_value = None

    numeric_differences.append(
        {
            "feature": col,
            "resigned_mean": round(float(np.mean(yes_vals)), 4),
            "retained_mean": round(float(np.mean(no_vals)), 4),
            "delta_resigned_minus_retained": round(float(np.mean(yes_vals) - np.mean(no_vals)), 4),
            "cohen_d": round(float(_cohen_d(yes_vals, no_vals)), 4),
            "mannwhitney_p_value": round(p_value, 6) if p_value is not None else None,
            "is_statistically_significant_0_05": bool(p_value is not None and p_value < 0.05),
        }
    )
numeric_differences = sorted(numeric_differences, key=lambda r: abs(r["cohen_d"]), reverse=True)

overall_resignation_rate = _safe_pct((work_df["Resignation_Target"] == 1).sum(), len(work_df))
categorical_risk_rows = []
min_category_count = 70
for col in categorical_cols:
    col_df = work_df[[col, "Employee_Resignation_Status"]].copy()
    col_df[col] = _normalize_label_col(col_df[col])
    table = pd.crosstab(col_df[col], col_df["Employee_Resignation_Status"]).reindex(columns=["Yes", "No"], fill_value=0)
    for category, row in table.iterrows():
        total = int(row.sum())
        if total < min_category_count:
            continue
        yes_count = int(row.get("Yes", 0))
        yes_rate = _safe_pct(yes_count, total)
        categorical_risk_rows.append(
            {
                "feature": col,
                "category": str(category),
                "count": total,
                "resignation_rate_pct": round(yes_rate, 3),
                "resignation_lift_vs_overall_pp": round(yes_rate - overall_resignation_rate, 3),
            }
        )
categorical_risk_rows = sorted(
    categorical_risk_rows,
    key=lambda r: (r["resignation_lift_vs_overall_pp"], r["count"]),
    reverse=True,
)

# Risk combination from strongest numeric signals.
numeric_diff_map = {r["feature"]: r for r in numeric_differences}
top_numeric_candidates = [r["feature"] for r in top_perm_features if r["feature"] in numeric_diff_map][:3]
if len(top_numeric_candidates) < 3:
    top_numeric_candidates = [r["feature"] for r in numeric_differences[:3]]

risk_thresholds = {}
protective_thresholds = {}
risk_directions = {}
risk_mask = np.ones(len(work_df), dtype=bool)
protective_mask = np.ones(len(work_df), dtype=bool)
for feature in top_numeric_candidates:
    delta = numeric_diff_map[feature]["delta_resigned_minus_retained"]
    if delta >= 0:
        risk_directions[feature] = "higher_is_riskier"
        risk_t = float(work_df[feature].quantile(0.75))
        protective_t = float(work_df[feature].quantile(0.25))
        risk_mask &= (pd.to_numeric(work_df[feature], errors="coerce") >= risk_t).fillna(False).to_numpy()
        protective_mask &= (pd.to_numeric(work_df[feature], errors="coerce") <= protective_t).fillna(False).to_numpy()
    else:
        risk_directions[feature] = "lower_is_riskier"
        risk_t = float(work_df[feature].quantile(0.25))
        protective_t = float(work_df[feature].quantile(0.75))
        risk_mask &= (pd.to_numeric(work_df[feature], errors="coerce") <= risk_t).fillna(False).to_numpy()
        protective_mask &= (pd.to_numeric(work_df[feature], errors="coerce") >= protective_t).fillna(False).to_numpy()
    risk_thresholds[feature] = risk_t
    protective_thresholds[feature] = protective_t

risk_combo_df = work_df[risk_mask]
protective_combo_df = work_df[protective_mask]
risk_combo_rate = _safe_pct((risk_combo_df["Resignation_Target"] == 1).sum(), len(risk_combo_df))
protective_combo_rate = _safe_pct((protective_combo_df["Resignation_Target"] == 1).sum(), len(protective_combo_df))

resignation_prevalence = float((work_df["Resignation_Target"] == 1).mean())
weak_signal_flag = bool(
    selected_metrics["roc_auc"] < 0.55
    and abs(selected_metrics["pr_auc"] - round(resignation_prevalence, 4)) <= 0.03
)
signal_note = (
    "Predictive signal is weak in current data (performance is near random baseline). Treat drivers as directional and triangulate with business context."
    if weak_signal_flag
    else "Model shows measurable predictive separation above random baseline for resignation risk."
)

result = {
    "question": "Q19 - Multi-Variable Drivers of Employee Resignation (Yes vs No)",
    "target_definition": {
        "resignation_class": "Employee_Resignation_Status == 'Yes'",
        "retained_class": "Employee_Resignation_Status == 'No'",
    },
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "resignation_count": int((work_df["Resignation_Target"] == 1).sum()),
        "retained_count": int((work_df["Resignation_Target"] == 0).sum()),
        "numeric_feature_count": int(len(numeric_cols)),
        "categorical_feature_count": int(len(categorical_cols)),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    },
    "modeling": {
        "train_test_split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2},
        "candidate_models": model_eval,
        "selected_model": selected_model_name,
        "selected_model_test_metrics": selected_metrics,
        "target_prevalence_resignation": round(resignation_prevalence, 4),
        "model_signal_assessment": {
            "is_weak_signal": weak_signal_flag,
            "note": signal_note,
        },
    },
    "factor_evidence": {
        "permutation_importance_top": top_perm_features,
        "logistic_coefficients_top_abs": coef_rows[:20],
        "numeric_effect_differences_top": numeric_differences[:15],
        "high_resignation_categories_top": categorical_risk_rows[:15],
    },
    "risk_combination_reasoning": {
        "features_used": top_numeric_candidates,
        "risk_directions": risk_directions,
        "risk_thresholds": {k: round(v, 4) for k, v in risk_thresholds.items()},
        "protective_thresholds": {k: round(v, 4) for k, v in protective_thresholds.items()},
        "risk_combo_count": int(len(risk_combo_df)),
        "risk_combo_resignation_rate_pct": round(risk_combo_rate, 3),
        "risk_combo_lift_vs_overall_pp": round(risk_combo_rate - overall_resignation_rate, 3),
        "protective_combo_count": int(len(protective_combo_df)),
        "protective_combo_resignation_rate_pct": round(protective_combo_rate, 3),
        "protective_combo_lift_vs_overall_pp": round(protective_combo_rate - overall_resignation_rate, 3),
    },
    "llm_evidence": {
        "selected_model": selected_model_name,
        "selected_metrics": selected_metrics,
        "target_prevalence_resignation": round(resignation_prevalence, 4),
        "model_signal_assessment": {
            "is_weak_signal": weak_signal_flag,
            "note": signal_note,
        },
        "top_permutation_features": top_perm_features[:10],
        "top_logistic_coefficients": coef_rows[:12],
        "top_numeric_effects": [
            {
                "feature": r["feature"],
                "delta_resigned_minus_retained": r["delta_resigned_minus_retained"],
                "cohen_d": r["cohen_d"],
                "p_value": r["mannwhitney_p_value"],
            }
            for r in numeric_differences[:10]
        ],
        "top_high_resignation_categories": categorical_risk_rows[:10],
        "risk_combo_summary": {
            "features_used": top_numeric_candidates,
            "risk_combo_resignation_rate_pct": round(risk_combo_rate, 3),
            "risk_combo_lift_vs_overall_pp": round(risk_combo_rate - overall_resignation_rate, 3),
            "protective_combo_resignation_rate_pct": round(protective_combo_rate, 3),
            "protective_combo_lift_vs_overall_pp": round(protective_combo_rate - overall_resignation_rate, 3),
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

