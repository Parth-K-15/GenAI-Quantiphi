"""
====================================================================
Employee Data Preprocessing - Phase 2 (Advanced Cleaning)
====================================================================
Steps:
  P2-1. Target Leakage Fix  -> Drop Performance_Category
  P2-2. Unique Employee ID Audit
  P2-3. Low / Zero Variance Column Detection & Drop
  P2-4. Outlier Detection & Clipping (IQR method)
  P2-5. Advanced Feature Engineering
        - Training_Efficiency
        - Engagement_Index
        - Compensation_Score
        - High_Performer flag
  P2-6. Final Export & Report
====================================================================
"""

import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json, os
from datetime import datetime

# ── Paths ────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "employee_data_cleaned.csv")
OUT_PATH   = os.path.join(BASE_DIR, "data", "processed", "employee_data_final.csv")
RPT_PATH   = os.path.join(BASE_DIR, "reports", "preprocessing_phase2_report.json")

report = {"pipeline": "Phase-2 Preprocessing", "timestamp": datetime.now().isoformat(), "steps": []}

def log(name, details):
    report["steps"].append({"step": name, "details": details})
    print(f"\n{'='*70}\n  STEP: {name}\n{'='*70}")
    for k, v in details.items():
        print(f"  {k}: {v}")

# ── Load Phase-1 output ──────────────────────────────────────────
df = pd.read_csv(INPUT_PATH, parse_dates=[
    'Hire_Date','Onboarding_Date','First_Project_Start_Date',
    'Offer_Acceptance_Date','Future_Date'])
print(f"\n  Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

# ============================================================
# P2-1  Target Leakage Fix
# ============================================================
dropped_leakage = []
for col in ['Performance_Category']:          # derived from Performance_Rating
    if col in df.columns:
        df.drop(columns=[col], inplace=True)
        dropped_leakage.append(col)

log("P2-1. Target Leakage Fix", {
    "columns_dropped": dropped_leakage,
    "reason": "Performance_Category is directly derived from Performance_Rating; keeping both causes data leakage."
})

# ============================================================
# P2-2  Unique Employee ID Audit
# ============================================================
total_rows      = len(df)
unique_emp_ids  = df['Employee_ID'].nunique() if 'Employee_ID' in df.columns else 0
is_unique       = unique_emp_ids == total_rows
dup_emp_ids     = df[df.duplicated('Employee_ID', keep=False)]['Employee_ID'].unique().tolist()[:10]

log("P2-2. Unique Employee ID Audit", {
    "total_rows": total_rows,
    "unique_employee_ids": unique_emp_ids,
    "ids_are_unique": is_unique,
    "duplicate_id_samples (up to 10)": dup_emp_ids,
    "note": "Multiple records per Employee_ID found - same employee appears in different project rows. Analysis must be grouped by Employee_ID for per-employee insights."
})

# ============================================================
# P2-3  Low / Zero Variance Column Detection & Drop
# ============================================================
low_var_cols   = [c for c in df.select_dtypes(include=np.number).columns if df[c].nunique() <= 1]
const_obj_cols = [c for c in df.select_dtypes(include='object').columns if df[c].nunique() <= 1]
all_drop       = low_var_cols + const_obj_cols

if all_drop:
    df.drop(columns=all_drop, inplace=True)

log("P2-3. Low/Zero Variance Columns", {
    "numeric_constant_cols": low_var_cols,
    "categorical_constant_cols": const_obj_cols,
    "dropped": all_drop,
    "columns_remaining": df.shape[1]
})

# ============================================================
# P2-4  Outlier Detection & Clipping (IQR)
# ============================================================
numeric_cols   = df.select_dtypes(include=np.number).columns.tolist()
# Exclude engineered/date-derived cols that can legitimately be extreme
exclude_clip   = ['Tenure_Days','Tenure_Years','Onboarding_Delay_Days',
                  'Project_Start_Delay_Days','Total_Skills_Score',
                  'Avg_Skills_Score','Total_Soft_Skills_Score','Avg_Soft_Skills_Score']
clip_targets   = [c for c in numeric_cols if c not in exclude_clip]

outlier_report = {}
total_clipped  = 0
for col in clip_targets:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo  = Q1 - 3 * IQR      # 3x IQR = extreme outliers only
    hi  = Q3 + 3 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out > 0:
        df[col] = df[col].clip(lower=lo, upper=hi)
        total_clipped += n_out
        outlier_report[col] = {
            "outliers_found": int(n_out),
            "clipped_to": [round(float(lo),2), round(float(hi),2)]
        }

log("P2-4. Outlier Detection & Clipping (3xIQR)", {
    "columns_checked": len(clip_targets),
    "columns_with_outliers": len(outlier_report),
    "total_values_clipped": total_clipped,
    "details": outlier_report
})

# ============================================================
# P2-5  Advanced Feature Engineering
# ============================================================
feats = {}

# Already have Tenure_Years from Phase 1 – alias as Experience_Years
if 'Tenure_Years' in df.columns:
    df['Experience_Years'] = df['Tenure_Years']
    feats['Experience_Years'] = "Alias of Tenure_Years"

# Training Efficiency = Performance / (Dev Hours + 1)
if 'Performance_Rating' in df.columns and 'Professional_Development_Hours' in df.columns:
    df['Training_Efficiency'] = round(
        df['Performance_Rating'] / (df['Professional_Development_Hours'] + 1), 4)
    feats['Training_Efficiency'] = {
        "formula": "Performance_Rating / (Professional_Development_Hours + 1)",
        "mean": round(float(df['Training_Efficiency'].mean()), 4),
        "median": round(float(df['Training_Efficiency'].median()), 4)
    }

# Engagement Index = mean of engagement + satisfaction + colleague feedback + supervisor feedback
eng_cols = [c for c in [
    'Employee_Engagement_Score','Employee_Job_Satisfaction_Score',
    'Feedback_From_Colleagues','Feedback_From_Supervisors'] if c in df.columns]
if eng_cols:
    df['Engagement_Index'] = round(df[eng_cols].mean(axis=1), 2)
    feats['Engagement_Index'] = {
        "formula": f"Mean of {eng_cols}",
        "mean": round(float(df['Engagement_Index'].mean()), 2),
        "median": round(float(df['Engagement_Index'].median()), 2)
    }

# Compensation Score = Bonus + Salary Increase + Bonus Percentage (normalised proxy)
comp_cols = [c for c in [
    'Bonus','Annual_Salary_Increase_Percentage','Performance_Bonus_Percentage'] if c in df.columns]
if comp_cols:
    df['Compensation_Score'] = df[comp_cols[0]]   # Bonus dominates
    feats['Compensation_Score'] = {
        "formula": "Proxy using Bonus as primary compensation indicator",
        "mean": round(float(df['Compensation_Score'].mean()), 2)
    }

# High Performer flag (Performance_Rating >= 12)
if 'Performance_Rating' in df.columns:
    df['Is_High_Performer'] = (df['Performance_Rating'] >= 12).astype(int)
    hp_count = int(df['Is_High_Performer'].sum())
    feats['Is_High_Performer'] = {
        "formula": "Performance_Rating >= 12",
        "high_performers": hp_count,
        "pct": round(hp_count / len(df) * 100, 2)
    }

# Attrition Risk flag  (resigned = Yes -> 1)
if 'Employee_Resignation_Status' in df.columns:
    df['Attrition_Risk'] = (df['Employee_Resignation_Status'].str.lower() == 'yes').astype(int)
    at_count = int(df['Attrition_Risk'].sum())
    feats['Attrition_Risk'] = {
        "formula": "Employee_Resignation_Status == 'Yes'",
        "at_risk_count": at_count,
        "pct": round(at_count / len(df) * 100, 2)
    }

log("P2-5. Advanced Feature Engineering", {
    "new_features": len(feats),
    "details": feats
})

# ============================================================
# P2-6  Final Validation & Export
# ============================================================
final_shape   = df.shape
final_missing = int(df.isnull().sum().sum())

log("P2-6. Final Validation", {
    "final_rows": final_shape[0],
    "final_columns": final_shape[1],
    "remaining_missing": final_missing,
    "dtypes": {str(k): int(v) for k, v in df.dtypes.value_counts().items()}
})

df.to_csv(OUT_PATH, index=False)

# ── Summary dict for dashboard ────────────────────────────────
report["summary"] = {
    "input_shape":  [total_rows, df.shape[1] + len(dropped_leakage) + len(all_drop)],
    "output_shape": list(final_shape),
    "leakage_cols_dropped": dropped_leakage,
    "low_var_cols_dropped":  all_drop,
    "outlier_cols_clipped":  len(outlier_report),
    "new_features": len(feats),
    "remaining_missing": final_missing,
    "unique_employee_ids": unique_emp_ids,
    "total_rows": total_rows,
    "is_employee_unique": is_unique,
    "perf_rating_mean": round(float(df['Performance_Rating'].mean()), 2),
    "high_performer_pct": round(int(df['Is_High_Performer'].sum()) / len(df) * 100, 2) if 'Is_High_Performer' in df.columns else 0,
    "attrition_pct": round(int(df['Attrition_Risk'].sum()) / len(df) * 100, 2) if 'Attrition_Risk' in df.columns else 0,
    "dept_dist": df['Department'].value_counts().to_dict() if 'Department' in df.columns else {},
    "perf_cat_dist": {
        "Low (1-5)": int((df['Performance_Rating'] <= 5).sum()),
        "Medium (6-10)": int(((df['Performance_Rating'] >= 6) & (df['Performance_Rating'] <= 10)).sum()),
        "High (11-15)": int(((df['Performance_Rating'] >= 11) & (df['Performance_Rating'] <= 15)).sum()),
        "Exceptional (16+)": int((df['Performance_Rating'] >= 16).sum())
    },
    "resignation_dist": df['Employee_Resignation_Status'].value_counts().to_dict() if 'Employee_Resignation_Status' in df.columns else {},
    "project_outcome_dist": df['Project_Outcome'].value_counts().to_dict() if 'Project_Outcome' in df.columns else {},
    "training_efficiency_mean": round(float(df['Training_Efficiency'].mean()), 4) if 'Training_Efficiency' in df.columns else 0,
    "engagement_index_mean": round(float(df['Engagement_Index'].mean()), 2) if 'Engagement_Index' in df.columns else 0,
    "onboarding_delay_mean": round(float(df['Onboarding_Delay_Days'].mean()), 1) if 'Onboarding_Delay_Days' in df.columns else 0,
    "project_delay_mean": round(float(df['Project_Start_Delay_Days'].mean()), 1) if 'Project_Start_Delay_Days' in df.columns else 0,
    "tenure_mean": round(float(df['Tenure_Years'].mean()), 2) if 'Tenure_Years' in df.columns else 0,
    "avg_skills_mean": round(float(df['Avg_Skills_Score'].mean()), 2) if 'Avg_Skills_Score' in df.columns else 0,
}

def serialize(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

with open(RPT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=serialize)

print(f"\n  [OK] Final dataset  -> {OUT_PATH}")
print(f"  [OK] Phase-2 report -> {RPT_PATH}")
print(f"\n  Final shape: {final_shape[0]} rows x {final_shape[1]} cols")
print(f"  Missing values remaining: {final_missing}")
