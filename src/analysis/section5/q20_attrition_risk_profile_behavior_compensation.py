"""Section 5 - Q20: Risk profile of employees likely to resign using behavioral and compensation features."""

import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os

import numpy as np
import pandas as pd
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
OUT = os.path.join(BASE, "reports", "section5", "q20_attrition_risk_profile_behavior_compensation.json")
TOP_OUT = os.path.join(BASE, "reports", "section5", "q20_top_attrition_risk_employees.csv")


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


required_cols = ["Employee_Resignation_Status"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q20: {missing_cols}")

work_df = DF.copy()
work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])
work_df = work_df[work_df["Employee_Resignation_Status"].isin(["Yes", "No"])].copy()
work_df["Resignation_Target"] = (work_df["Employee_Resignation_Status"] == "Yes").astype(int)

if work_df["Resignation_Target"].nunique() < 2:
    raise ValueError("Q20 needs both resignation classes: Yes and No.")

behavioral_numeric_candidates = [
    "Employee_Engagement_Score",
    "Employee_Job_Satisfaction_Score",
    "Employee_Work_Life_Balance_Rating",
    "Overtime_Hours_Per_Week",
    "Work_Hours_Per_Week",
    "Conflict_Resolution_Cases",
    "Feedback_From_Colleagues",
    "Feedback_From_Supervisors",
    "Mentor_Rating",
    "Initiative_Rating",
    "Adaptability_Rating",
    "Creativity_Rating",
    "Strategic_Thinking_Rating",
    "Leadership_Qualities_Rating",
    "Teamwork_Skills_Rating",
    "Problem_Solving_Skills_Rating",
    "Technical_Skills_Rating",
    "Communication_Skills_Rating",
    "Performance_Rating",
    "Number_Of_Promotions",
]
comp_numeric_candidates = [
    "Annual_Salary_Increase_Percentage",
    "Performance_Bonus_Percentage",
    "Bonus",
    "Employee_Annual_Salary_Adjustment",
    "Employee_Compensation_Benefits",
    "Employee_Travel_Allowance",
    "Employee_Savings_Plans",
    "Compensation_Score",
]
comp_categorical_candidates = [
    "Employee_Stock_Options",
    "Employee_Health_Insurance_Coverage",
    "Employee_Retirement_Benefits",
    "Employee_Recognition_Programs_Participation",
]

numeric_cols = [c for c in behavioral_numeric_candidates + comp_numeric_candidates if c in work_df.columns]
categorical_cols = [c for c in comp_categorical_candidates if c in work_df.columns]

if len(numeric_cols) + len(categorical_cols) < 6:
    raise ValueError("Not enough behavioral/compensation features found for Q20.")

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

# Risk tiers from out-of-fold probabilities.
q50 = float(work_df["Predicted_Resignation_Probability"].quantile(0.50))
q75 = float(work_df["Predicted_Resignation_Probability"].quantile(0.75))
q90 = float(work_df["Predicted_Resignation_Probability"].quantile(0.90))

def _risk_tier(prob: float) -> str:
    if prob <= q50:
        return "Low"
    if prob <= q75:
        return "Medium"
    if prob <= q90:
        return "High"
    return "Critical"


work_df["Attrition_Risk_Tier"] = work_df["Predicted_Resignation_Probability"].apply(_risk_tier)
tier_order = ["Critical", "High", "Medium", "Low"]
overall_resignation_rate = _safe_pct((work_df["Resignation_Target"] == 1).sum(), len(work_df))
tier_summary = []
for tier in tier_order:
    sub = work_df[work_df["Attrition_Risk_Tier"] == tier]
    if len(sub) == 0:
        continue
    tier_rate = _safe_pct((sub["Resignation_Target"] == 1).sum(), len(sub))
    tier_summary.append(
        {
            "risk_tier": tier,
            "count": int(len(sub)),
            "population_pct": round(_safe_pct(len(sub), len(work_df)), 3),
            "actual_resignation_rate_pct": round(tier_rate, 3),
            "resignation_lift_vs_overall_pp": round(tier_rate - overall_resignation_rate, 3),
            "avg_predicted_probability": round(float(sub["Predicted_Resignation_Probability"].mean()), 6),
        }
    )

critical_df = work_df[work_df["Attrition_Risk_Tier"] == "Critical"].copy()
critical_profile_gaps = []
for col in numeric_cols:
    crit_mean = float(pd.to_numeric(critical_df[col], errors="coerce").mean())
    all_mean = float(pd.to_numeric(work_df[col], errors="coerce").mean())
    critical_profile_gaps.append(
        {
            "feature": col,
            "critical_tier_mean": round(crit_mean, 4),
            "overall_mean": round(all_mean, 4),
            "gap": round(crit_mean - all_mean, 4),
        }
    )
critical_profile_gaps = sorted(critical_profile_gaps, key=lambda r: abs(r["gap"]), reverse=True)

# Concentration by role/department in critical tier.
critical_concentration = {}
for col in ["Department", "Project_Role", "Job_Title"]:
    if col not in work_df.columns:
        continue
    all_share = work_df[col].astype(str).value_counts(normalize=True)
    crit_share = critical_df[col].astype(str).value_counts(normalize=True)
    rows = []
    for category, c_share in crit_share.items():
        base_share = float(all_share.get(category, 0.0))
        rows.append(
            {
                "feature": col,
                "category": category,
                "critical_tier_share_pct": round(float(c_share * 100), 3),
                "overall_share_pct": round(float(base_share * 100), 3),
                "over_index_ratio": round(float((c_share / base_share) if base_share > 0 else 0.0), 3),
            }
        )
    critical_concentration[col] = sorted(rows, key=lambda r: r["over_index_ratio"], reverse=True)[:10]

# Risk-trigger combination from top numeric permutation features.
resigned_df = work_df[work_df["Resignation_Target"] == 1].copy()
retained_df = work_df[work_df["Resignation_Target"] == 0].copy()
top_numeric_for_combo = [r["feature"] for r in top_perm_features if r["feature"] in numeric_cols][:3]
if len(top_numeric_for_combo) < 3:
    top_numeric_for_combo = numeric_cols[:3]

risk_directions = {}
risk_thresholds = {}
risk_mask = np.ones(len(work_df), dtype=bool)
for feature in top_numeric_for_combo:
    resigned_mean = float(pd.to_numeric(resigned_df[feature], errors="coerce").mean())
    retained_mean = float(pd.to_numeric(retained_df[feature], errors="coerce").mean())
    series = pd.to_numeric(work_df[feature], errors="coerce")
    if resigned_mean >= retained_mean:
        risk_directions[feature] = "higher_is_riskier"
        threshold = float(series.quantile(0.75))
        risk_mask &= (series >= threshold).fillna(False).to_numpy()
    else:
        risk_directions[feature] = "lower_is_riskier"
        threshold = float(series.quantile(0.25))
        risk_mask &= (series <= threshold).fillna(False).to_numpy()
    risk_thresholds[feature] = threshold

risk_combo_df = work_df[risk_mask]
risk_combo_rate = _safe_pct((risk_combo_df["Resignation_Target"] == 1).sum(), len(risk_combo_df))

# Explainable risk score weights from permutation importance.
positive_perm = [max(0.0, r["importance_mean_auc_drop"]) for r in perm_rows]
total_pos = sum(positive_perm)
risk_score_weights = []
if total_pos > 0:
    for r in perm_rows[:10]:
        w = max(0.0, r["importance_mean_auc_drop"]) / total_pos
        risk_score_weights.append({"feature": r["feature"], "weight": round(float(w), 4)})

# Save top at-risk employees.
top_cols = [
    "Employee_ID",
    "Predicted_Resignation_Probability",
    "Attrition_Risk_Tier",
    "Employee_Resignation_Status",
    "Department",
    "Job_Title",
    "Project_Role",
    "Performance_Rating",
    "Employee_Engagement_Score",
    "Employee_Job_Satisfaction_Score",
    "Employee_Work_Life_Balance_Rating",
    "Compensation_Score",
]
top_cols = [c for c in top_cols if c in work_df.columns]
top_risk = (
    work_df.sort_values("Predicted_Resignation_Probability", ascending=False)
    .head(40)[top_cols]
    .copy()
)
os.makedirs(os.path.dirname(TOP_OUT), exist_ok=True)
top_risk.to_csv(TOP_OUT, index=False)

resignation_prevalence = float((work_df["Resignation_Target"] == 1).mean())
weak_signal_flag = bool(
    selected_metrics["roc_auc"] < 0.55
    and abs(selected_metrics["pr_auc"] - round(resignation_prevalence, 4)) <= 0.03
)
signal_note = (
    "Predictive signal is weak in current data (performance is near random baseline). Use tiers as directional prioritization, not deterministic labels."
    if weak_signal_flag
    else "Model shows measurable predictive separation above random baseline for attrition risk."
)

result = {
    "question": "Q20 - Attrition Risk Profile Using Behavioral and Compensation Features",
    "target_definition": {
        "resignation_class": "Employee_Resignation_Status == 'Yes'",
        "retained_class": "Employee_Resignation_Status == 'No'",
    },
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "resignation_count": int((work_df["Resignation_Target"] == 1).sum()),
        "retained_count": int((work_df["Resignation_Target"] == 0).sum()),
        "feature_count": int(len(feature_cols)),
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
    "risk_tiering": {
        "tier_thresholds_probability": {
            "q50": round(q50, 6),
            "q75": round(q75, 6),
            "q90": round(q90, 6),
        },
        "tier_summary": tier_summary,
        "overall_resignation_rate_pct": round(overall_resignation_rate, 3),
    },
    "critical_tier_profile": {
        "critical_tier_count": int(len(critical_df)),
        "critical_tier_actual_resignation_rate_pct": round(
            _safe_pct((critical_df["Resignation_Target"] == 1).sum(), len(critical_df)), 3
        ),
        "critical_profile_feature_gaps": critical_profile_gaps[:15],
        "critical_tier_concentration": critical_concentration,
    },
    "risk_combination_reasoning": {
        "features_used": top_numeric_for_combo,
        "risk_directions": risk_directions,
        "risk_thresholds": {k: round(v, 4) for k, v in risk_thresholds.items()},
        "risk_combo_count": int(len(risk_combo_df)),
        "risk_combo_resignation_rate_pct": round(risk_combo_rate, 3),
        "risk_combo_lift_vs_overall_pp": round(risk_combo_rate - overall_resignation_rate, 3),
    },
    "risk_scorecard": {
        "formula": "Attrition_Risk = SUM(weight_i * normalized_feature_i) across top predictors",
        "weights_top_features": risk_score_weights,
        "note": "Weights are normalized from permutation-importance AUC drop; higher weight means stronger risk contribution.",
    },
    "feature_influence": {
        "permutation_importance_top": top_perm_features,
    },
    "outputs": {
        "top_attrition_risk_csv": TOP_OUT,
        "top_attrition_risk_note": "Top employees ranked by out-of-fold predicted resignation probability.",
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
        "tier_thresholds_probability": {
            "q50": round(q50, 6),
            "q75": round(q75, 6),
            "q90": round(q90, 6),
        },
        "tier_summary": tier_summary,
        "critical_profile_feature_gaps": critical_profile_gaps[:12],
        "risk_combo_summary": {
            "features_used": top_numeric_for_combo,
            "risk_combo_resignation_rate_pct": round(risk_combo_rate, 3),
            "risk_combo_lift_vs_overall_pp": round(risk_combo_rate - overall_resignation_rate, 3),
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
print(f"[OK] Saved -> {TOP_OUT}")
