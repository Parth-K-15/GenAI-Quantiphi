"""Section 6 - Q23: Identify employees underpaid relative to performance and skills."""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT = os.path.join(BASE, "reports", "section6", "q23_underpaid_relative_to_performance_skills.json")
TOP_OUT = os.path.join(BASE, "reports", "section6", "q23_top_underpaid_employees.csv")


def _normalize_label_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.title()


def _safe_pct(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return float((num / den) * 100)


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q23 underpaid employee detection.")
    parser.add_argument("--with-llm", action="store_true", help="Also generate Gemini insights for Q23 report.")
    parser.add_argument("--llm-delay", type=int, default=12, help="Delay option passed to report LLM generator.")
    parser.add_argument("--llm-model", type=str, default=None, help="Preferred Gemini model for LLM generation.")
    parser.add_argument(
        "--llm-strict-model",
        action="store_true",
        help="If set, only use --llm-model (or GEMINI_MODEL_NAME) and skip auto-fallback models.",
    )
    parser.add_argument("--llm-quiet", action="store_true", help="Reduce LLM generation logs.")
    return parser.parse_args()


def _run_q23_llm_generation(args: argparse.Namespace) -> None:
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

    print("[RUN] Q23 Gemini insights generation")
    proc = subprocess.run(llm_cmd, cwd=BASE)
    if proc.returncode != 0:
        raise RuntimeError(f"Q23 Gemini insights generation failed with exit code {proc.returncode}")
    print("[OK]  Q23 Gemini insights generation")


ARGS = _parse_cli_args()


# Use Compensation_Score as a consistent internal compensation proxy in this dataset.
target_col = "Compensation_Score" if "Compensation_Score" in DF.columns else "Employee_Compensation_Benefits"
if target_col not in DF.columns:
    raise ValueError("Q23 requires either Compensation_Score or Employee_Compensation_Benefits in dataset.")

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
    "Number_Of_Promotions",
    "Tenure_Years",
    "Professional_Development_Hours",
    "Employee_Engagement_Score",
    "Employee_Job_Satisfaction_Score",
    "Employee_Work_Life_Balance_Rating",
    "Feedback_From_Supervisors",
    "Mentor_Rating",
]
categorical_candidates = [
    "Department",
    "Job_Title",
    "Project_Role",
    "Highest_Education_Level",
    "Certifications",
]

work_df = DF.copy()
if "Employee_Resignation_Status" in work_df.columns:
    work_df["Employee_Resignation_Status"] = _normalize_label_col(work_df["Employee_Resignation_Status"])

numeric_cols = [c for c in numeric_candidates if c in work_df.columns]
categorical_cols = [c for c in categorical_candidates if c in work_df.columns]

if len(numeric_cols) + len(categorical_cols) < 8:
    raise ValueError("Not enough feature columns found for Q23 underpaid model.")

feature_cols = numeric_cols + categorical_cols
model_df = work_df[feature_cols + [target_col]].copy()
model_df[target_col] = pd.to_numeric(model_df[target_col], errors="coerce")
model_df = model_df.dropna(subset=[target_col]).copy()

X = model_df[feature_cols].copy()
y = model_df[target_col].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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

linear_model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("reg", LinearRegression()),
    ]
)
rf_model = Pipeline(
    steps=[
        ("prep", preprocess),
        (
            "reg",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=1,
            ),
        ),
    ]
)

models = {
    "linear_regression": linear_model,
    "random_forest_regressor": rf_model,
}

model_eval = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=1)
    model_eval[name] = {
        "test_r2": round(float(r2_score(y_test, pred)), 4),
        "test_mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
        "cv_r2_mean": round(float(cv_r2.mean()), 4),
        "cv_r2_std": round(float(cv_r2.std()), 4),
    }

selected_model_name = max(models.keys(), key=lambda m: model_eval[m]["cv_r2_mean"])
selected_model = models[selected_model_name]
selected_model.fit(X_train, y_train)

# Use out-of-fold expected compensation to avoid in-sample inflation.
oof_cv = KFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = cross_val_predict(selected_model, X, y, cv=oof_cv, n_jobs=1)

model_df = model_df.copy()
model_df["Expected_Compensation"] = oof_pred
model_df["Compensation_Gap"] = model_df[target_col] - model_df["Expected_Compensation"]
model_df["Compensation_Gap_Pct"] = (model_df["Compensation_Gap"] / model_df["Expected_Compensation"]).replace(
    [np.inf, -np.inf], np.nan
) * 100

gap_p10 = float(model_df["Compensation_Gap_Pct"].quantile(0.10))
gap_p05 = float(model_df["Compensation_Gap_Pct"].quantile(0.05))
perf_p75 = float(pd.to_numeric(model_df["Performance_Rating"], errors="coerce").quantile(0.75)) if "Performance_Rating" in model_df.columns else None
skills_p75 = float(pd.to_numeric(model_df["Avg_Skills_Score"], errors="coerce").quantile(0.75)) if "Avg_Skills_Score" in model_df.columns else None

underpaid_mask = model_df["Compensation_Gap_Pct"] <= gap_p10
if perf_p75 is not None:
    underpaid_mask &= pd.to_numeric(model_df["Performance_Rating"], errors="coerce") >= perf_p75
if skills_p75 is not None:
    underpaid_mask &= pd.to_numeric(model_df["Avg_Skills_Score"], errors="coerce") >= skills_p75

underpaid_df = model_df[underpaid_mask].copy()
severe_underpaid_df = model_df[model_df["Compensation_Gap_Pct"] <= gap_p05].copy()

underpaid_share = _safe_pct(len(underpaid_df), len(model_df))
underpaid_gap_mean = float(underpaid_df["Compensation_Gap_Pct"].mean()) if len(underpaid_df) else None

group_concentration = {}
for col in ["Department", "Project_Role", "Job_Title"]:
    if col not in model_df.columns:
        continue
    all_share = model_df[col].astype(str).value_counts(normalize=True)
    underpaid_share_dist = underpaid_df[col].astype(str).value_counts(normalize=True) if len(underpaid_df) else pd.Series(dtype=float)
    rows = []
    for category, u_share in underpaid_share_dist.items():
        base = float(all_share.get(category, 0.0))
        rows.append(
            {
                "feature": col,
                "category": category,
                "underpaid_share_pct": round(float(u_share * 100), 3),
                "overall_share_pct": round(float(base * 100), 3),
                "over_index_ratio": round(float((u_share / base) if base > 0 else 0.0), 3),
            }
        )
    group_concentration[col] = sorted(rows, key=lambda r: r["over_index_ratio"], reverse=True)[:12]

perm = permutation_importance(
    selected_model,
    X_test,
    y_test,
    n_repeats=12,
    random_state=42,
    scoring="r2",
    n_jobs=1,
)
perm_rows = []
for i, col in enumerate(feature_cols):
    perm_rows.append(
        {
            "feature": col,
            "importance_mean_r2_drop": round(float(perm.importances_mean[i]), 6),
            "importance_std": round(float(perm.importances_std[i]), 6),
        }
    )
perm_rows = sorted(perm_rows, key=lambda r: r["importance_mean_r2_drop"], reverse=True)

top_cols = [
    "Employee_ID",
    target_col,
    "Expected_Compensation",
    "Compensation_Gap",
    "Compensation_Gap_Pct",
    "Performance_Rating",
    "Avg_Skills_Score",
    "Avg_Soft_Skills_Score",
    "Department",
    "Job_Title",
    "Project_Role",
]
top_cols = [c for c in top_cols if c in model_df.columns]
top_underpaid = underpaid_df.sort_values("Compensation_Gap_Pct", ascending=True).head(40)[top_cols].copy()

os.makedirs(os.path.dirname(TOP_OUT), exist_ok=True)
top_underpaid.to_csv(TOP_OUT, index=False)

result = {
    "question": "Q23 - Identify Employees Underpaid Relative to Performance and Skills",
    "compensation_proxy_used": target_col,
    "method_note": (
        "Expected compensation is estimated from performance, skills, and context. "
        "Underpaid employees are those with actual pay materially below expected pay while maintaining high performance/skills."
    ),
    "dataset_scope": {
        "rows_used": int(len(model_df)),
        "feature_count": int(len(feature_cols)),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
    },
    "modeling": {
        "train_test_split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.2},
        "candidate_models": model_eval,
        "selected_model": selected_model_name,
        "selection_basis": "Highest cross-validated R2",
    },
    "underpaid_detection": {
        "thresholds": {
            "underpaid_gap_pct_p10": round(gap_p10, 4),
            "severe_underpaid_gap_pct_p05": round(gap_p05, 4),
            "performance_min_q75": round(perf_p75, 4) if perf_p75 is not None else None,
            "avg_skills_min_q75": round(skills_p75, 4) if skills_p75 is not None else None,
        },
        "underpaid_count": int(len(underpaid_df)),
        "underpaid_share_pct": round(underpaid_share, 3),
        "avg_underpaid_gap_pct": round(underpaid_gap_mean, 4) if underpaid_gap_mean is not None else None,
        "severe_underpaid_count": int(len(severe_underpaid_df)),
    },
    "feature_influence": {
        "permutation_importance_top": perm_rows[:15],
    },
    "underpaid_concentration": group_concentration,
    "outputs": {
        "top_underpaid_csv": TOP_OUT,
        "top_underpaid_note": "Ranked by most negative compensation gap percentage.",
    },
    "llm_evidence": {
        "compensation_proxy_used": target_col,
        "selected_model": selected_model_name,
        "candidate_models": model_eval,
        "underpaid_detection": {
            "underpaid_count": int(len(underpaid_df)),
            "underpaid_share_pct": round(underpaid_share, 3),
            "avg_underpaid_gap_pct": round(underpaid_gap_mean, 4) if underpaid_gap_mean is not None else None,
            "severe_underpaid_count": int(len(severe_underpaid_df)),
        },
        "thresholds": {
            "underpaid_gap_pct_p10": round(gap_p10, 4),
            "severe_underpaid_gap_pct_p05": round(gap_p05, 4),
            "performance_min_q75": round(perf_p75, 4) if perf_p75 is not None else None,
            "avg_skills_min_q75": round(skills_p75, 4) if skills_p75 is not None else None,
        },
        "top_permutation_features": perm_rows[:12],
        "underpaid_concentration": group_concentration,
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
_run_q23_llm_generation(ARGS)

