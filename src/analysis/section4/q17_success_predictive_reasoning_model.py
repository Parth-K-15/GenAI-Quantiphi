"""Section 4 - Q17: Predictive reasoning model for successful project outcomes.

Question:
What combination of skills and ratings leads to successful project outcomes?
"""
import sys
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import json
import os
import argparse
import subprocess
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
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section4", "q17_success_predictive_reasoning_model.json")
TOP_OUT = os.path.join(BASE, "reports", "section4", "q17_top_success_propensity_employees.csv")


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


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q17 predictive reasoning model.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q17 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to Section 4 LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q17_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q17 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q17 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q17 Gemini insights generation")


ARGS = _parse_cli_args()


required_cols = ["Project_Outcome"]
missing_cols = [c for c in required_cols if c not in DF.columns]
if missing_cols:
    raise ValueError(f"Missing required columns for Q17: {missing_cols}")

work_df = DF.copy()
work_df["Project_Outcome"] = _normalize_label_col(work_df["Project_Outcome"])
work_df = work_df[work_df["Project_Outcome"].isin(["Successful", "Failed"])].copy()
work_df["Success_Target"] = (work_df["Project_Outcome"] == "Successful").astype(int)

if work_df["Success_Target"].nunique() < 2:
    raise ValueError("Q17 needs both successful and failed outcomes.")

# Skills + ratings only (plus effectiveness/tenure rating-derived context features).
feature_candidates = [
    "Technical_Skills_Rating",
    "Communication_Skills_Rating",
    "Problem_Solving_Skills_Rating",
    "Leadership_Qualities_Rating",
    "Initiative_Rating",
    "Adaptability_Rating",
    "Creativity_Rating",
    "Strategic_Thinking_Rating",
    "Teamwork_Skills_Rating",
    "Feedback_From_Colleagues",
    "Feedback_From_Supervisors",
    "Mentor_Rating",
    "Employee_Engagement_Score",
    "Employee_Job_Satisfaction_Score",
    "Employee_Work_Life_Balance_Rating",
    "Performance_Rating",
    "Avg_Skills_Score",
    "Avg_Soft_Skills_Score",
    "Training_Efficiency",
    "Professional_Development_Hours",
    "Conflict_Resolution_Cases",
    "Number_Of_Promotions",
    "Overtime_Hours_Per_Week",
    "Onboarding_Delay_Days",
    "Tenure_Years",
]
feature_cols = [c for c in feature_candidates if c in work_df.columns]
if len(feature_cols) < 5:
    raise ValueError("Not enough feature columns found for Q17 predictive model.")

X = work_df[feature_cols].copy()
y = work_df["Success_Target"].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

numeric_transform = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transform, feature_cols),
    ],
    remainder="drop",
)

logistic_model = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "clf",
            LogisticRegression(
                max_iter=3000,
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

selected_model_name = max(models.keys(), key=lambda m: model_eval[m]["test_metrics"]["roc_auc"])
selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)

test_prob = selected_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= 0.5).astype(int)
selected_metrics = _metric_pack(y_test.to_numpy(), test_pred, test_prob)

# Out-of-fold probabilities for unbiased profiling/ranking.
oof_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_prob = cross_val_predict(
    selected_model,
    X,
    y,
    cv=oof_fold,
    method="predict_proba",
    n_jobs=1,
)[:, 1]

work_df = work_df.copy()
work_df["Predicted_Success_Probability"] = oof_prob

# Permutation importance on holdout for model-agnostic feature ranking.
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
top_features = perm_rows[:8]

# Interpretable linear coefficients from logistic model for reasoning formula.
logistic_model.fit(X_train, y_train)
coef = logistic_model.named_steps["clf"].coef_[0]
intercept = float(logistic_model.named_steps["clf"].intercept_[0])
coef_rows = []
for c, w in zip(feature_cols, coef):
    coef_rows.append({"feature": c, "coefficient": round(float(w), 6)})
coef_rows = sorted(coef_rows, key=lambda r: abs(r["coefficient"]), reverse=True)

# High-propensity success profile (top decile by predicted probability).
p90 = float(work_df["Predicted_Success_Probability"].quantile(0.90))
top_decile = work_df[work_df["Predicted_Success_Probability"] >= p90].copy()
base_success_rate = _safe_pct((work_df["Success_Target"] == 1).sum(), len(work_df))
top_success_rate = _safe_pct((top_decile["Success_Target"] == 1).sum(), len(top_decile))
success_prevalence = float((work_df["Success_Target"] == 1).mean())

profile_gap_rows = []
for col in feature_cols:
    top_mean = float(top_decile[col].mean())
    base_mean = float(work_df[col].mean())
    profile_gap_rows.append(
        {
            "feature": col,
            "top_decile_mean": round(top_mean, 4),
            "overall_mean": round(base_mean, 4),
            "gap": round(top_mean - base_mean, 4),
        }
    )
profile_gap_rows = sorted(profile_gap_rows, key=lambda r: abs(r["gap"]), reverse=True)

# Combination reasoning: top 3 important features high together.
top3 = [r["feature"] for r in top_features[:3]]
high_thresholds = {f: float(work_df[f].quantile(0.75)) for f in top3}
low_thresholds = {f: float(work_df[f].quantile(0.25)) for f in top3}

high_combo_mask = np.ones(len(work_df), dtype=bool)
low_combo_mask = np.ones(len(work_df), dtype=bool)
for f in top3:
    high_combo_mask &= (work_df[f] >= high_thresholds[f]).to_numpy()
    low_combo_mask &= (work_df[f] <= low_thresholds[f]).to_numpy()

high_combo = work_df[high_combo_mask]
low_combo = work_df[low_combo_mask]

high_combo_success = _safe_pct((high_combo["Success_Target"] == 1).sum(), len(high_combo))
low_combo_success = _safe_pct((low_combo["Success_Target"] == 1).sum(), len(low_combo))

weak_signal_flag = bool(
    selected_metrics["roc_auc"] < 0.55
    and abs(selected_metrics["pr_auc"] - round(success_prevalence, 4)) <= 0.03
)
signal_note = (
    "Predictive signal is weak in current data (performance is near random baseline). Use the output as directional, not deterministic."
    if weak_signal_flag
    else "Model shows measurable predictive separation above random baseline for this target."
)

# Recommended weighted readiness score based on normalized permutation importance.
positive_perm = [max(0.0, r["importance_mean_auc_drop"]) for r in perm_rows]
total_pos = sum(positive_perm)
readiness_weights = []
if total_pos > 0:
    for r in perm_rows[:8]:
        w = max(0.0, r["importance_mean_auc_drop"]) / total_pos
        readiness_weights.append({"feature": r["feature"], "weight": round(float(w), 4)})

# Save top propensity employees.
top_cols = [
    "Employee_ID",
    "Predicted_Success_Probability",
    "Project_Outcome",
    "Department",
    "Job_Title",
    "Project_Role",
    "Performance_Rating",
]
top_cols = [c for c in top_cols if c in work_df.columns]
top_success_candidates = (
    work_df.sort_values("Predicted_Success_Probability", ascending=False)
    .head(30)[top_cols]
    .copy()
)
top_success_candidates.to_csv(TOP_OUT, index=False)

result = {
    "question": "Q17 - Predictive Reasoning Model: Combination of Skills and Ratings for Successful Outcomes",
    "target_definition": {
        "success_class": "Project_Outcome == 'Successful'",
        "failure_class": "Project_Outcome == 'Failed'",
        "in_progress_excluded": True,
    },
    "dataset_scope": {
        "rows_used": int(len(work_df)),
        "success_count": int((work_df["Success_Target"] == 1).sum()),
        "failed_count": int((work_df["Success_Target"] == 0).sum()),
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
    },
    "modeling": {
        "train_test_split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2},
        "candidate_models": model_eval,
        "selected_model": selected_model_name,
        "selected_model_test_metrics": selected_metrics,
        "target_prevalence_success": round(success_prevalence, 4),
        "model_signal_assessment": {
            "is_weak_signal": weak_signal_flag,
            "note": signal_note,
        },
    },
    "feature_influence": {
        "permutation_importance_top": top_features,
        "logistic_coefficients_top_abs": coef_rows[:12],
        "logistic_intercept": round(intercept, 6),
    },
    "success_profile": {
        "top_decile_probability_threshold": round(p90, 6),
        "top_decile_count": int(len(top_decile)),
        "overall_success_rate_pct": round(base_success_rate, 3),
        "top_decile_actual_success_rate_pct": round(top_success_rate, 3),
        "success_lift_top_decile_pp": round(top_success_rate - base_success_rate, 3),
        "probability_basis": "Out-of-fold (5-fold) predicted probability to avoid in-sample inflation",
        "top_profile_feature_gaps": profile_gap_rows[:12],
    },
    "reasoning_combinations": {
        "top3_features_by_importance": top3,
        "high_thresholds_q75": {k: round(v, 4) for k, v in high_thresholds.items()},
        "low_thresholds_q25": {k: round(v, 4) for k, v in low_thresholds.items()},
        "high_combo_count": int(len(high_combo)),
        "high_combo_success_rate_pct": round(high_combo_success, 3),
        "high_combo_lift_vs_overall_pp": round(high_combo_success - base_success_rate, 3),
        "low_combo_count": int(len(low_combo)),
        "low_combo_success_rate_pct": round(low_combo_success, 3),
        "low_combo_lift_vs_overall_pp": round(low_combo_success - base_success_rate, 3),
    },
    "readiness_scorecard": {
        "formula": "Success_Readiness = SUM(weight_i * normalized_feature_i) across top predictors",
        "weights_top_features": readiness_weights,
        "note": "Weights are normalized from permutation-importance AUC drop; higher weight means stronger predictive contribution.",
    },
    "outputs": {
        "top_success_propensity_csv": TOP_OUT,
        "top_success_propensity_note": "Ranked by out-of-fold predicted success probability.",
    },
    "llm_evidence": {
        "selected_model": selected_model_name,
        "selected_metrics": selected_metrics,
        "top_features": top_features,
        "top3_features_by_importance": top3,
        "high_combo_success_rate_pct": round(high_combo_success, 3),
        "high_combo_lift_vs_overall_pp": round(high_combo_success - base_success_rate, 3),
        "low_combo_success_rate_pct": round(low_combo_success, 3),
        "low_combo_lift_vs_overall_pp": round(low_combo_success - base_success_rate, 3),
        "top_decile_probability_threshold": round(p90, 6),
        "top_decile_actual_success_rate_pct": round(top_success_rate, 3),
        "success_lift_top_decile_pp": round(top_success_rate - base_success_rate, 3),
        "target_prevalence_success": round(success_prevalence, 4),
        "model_signal_assessment": {
            "is_weak_signal": weak_signal_flag,
            "note": signal_note,
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
_run_q17_llm_generation(ARGS)
