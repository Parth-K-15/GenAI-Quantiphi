"""
====================================================================
Employee Data Preprocessing Pipeline
====================================================================
Phase 1: Data Cleaning & Preprocessing
Project: GenAI-Quantiphi - Employee Performance & Skill Analytics

Steps:
  1. Load & Inspect Raw Data
  2. Remove Duplicate Rows
  3. Drop Garbage/Unnamed Columns
  4. Fix Date Columns (parse dates, handle Excel serial dates)
  5. Handle Missing Values
  6. Standardize Categorical Columns
  7. Fix Mixed Data Types
  8. Feature Engineering
  9. Final Validation & Export
====================================================================
"""

import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

# ──────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "employee_data.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "employee_data_cleaned.csv")
REPORT_PATH = os.path.join(BASE_DIR, "reports", "preprocessing_report.json")

# Ensure output directories exist
os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

report = {
    "pipeline": "Employee Data Preprocessing",
    "timestamp": datetime.now().isoformat(),
    "steps": []
}


def log_step(step_name, details):
    """Log a preprocessing step to the report."""
    entry = {"step": step_name, "details": details}
    report["steps"].append(entry)
    print(f"\n{'='*70}")
    print(f"  STEP: {step_name}")
    print(f"{'='*70}")
    for k, v in details.items():
        print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────────
# STEP 0: Load Raw Data
# ──────────────────────────────────────────────────────────────────
print("\n" + ">>" * 35)
print("  EMPLOYEE DATA PREPROCESSING PIPELINE")
print(">>" * 35)

df = pd.read_csv(RAW_DATA_PATH)
initial_shape = df.shape

log_step("0. Load Raw Data", {
    "file": RAW_DATA_PATH,
    "rows": initial_shape[0],
    "columns": initial_shape[1],
    "memory_usage_MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
})

# ──────────────────────────────────────────────────────────────────
# STEP 1: Inspect Data Quality Issues
# ──────────────────────────────────────────────────────────────────
print("\n" + "-"*70)
print("  DATA QUALITY INSPECTION")
print("-"*70)

# Check for duplicates
num_duplicates = df.duplicated().sum()
print(f"  [*] Duplicate Rows: {num_duplicates}")

# Check for unnamed/garbage columns
unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
print(f"  [*] Unnamed/Garbage Columns: {unnamed_cols}")

# Check for missing values
missing_summary = df.isnull().sum()
cols_with_missing = missing_summary[missing_summary > 0]
print(f"  [*] Columns with Missing Values: {len(cols_with_missing)}")
if len(cols_with_missing) > 0:
    for col, count in cols_with_missing.items():
        pct = round(count / len(df) * 100, 2)
        print(f"      - {col}: {count} missing ({pct}%)")

# Check data types
print(f"\n  [*] Data Types Summary:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"      - {dtype}: {count} columns")

# Identify date columns
date_cols_candidates = ['Hire_Date', 'Onboarding_Date', 'First_Project_Start_Date', 'Offer_Acceptance_Date']
date_cols_present = [col for col in date_cols_candidates if col in df.columns]
print(f"  [*] Date Columns Found: {date_cols_present}")

# Check Future Date column (likely Excel serial date)
if 'Future Date' in df.columns:
    print(f"  [*] 'Future Date' column: dtype={df['Future Date'].dtype}, sample values={df['Future Date'].head(3).tolist()}")

# Check mixed data types
print(f"\n  [*] Checking for mixed data types in object columns...")
mixed_type_cols = []
for col in df.select_dtypes(include='object').columns:
    # Check if column has numeric-like values mixed with strings
    numeric_count = pd.to_numeric(df[col], errors='coerce').notna().sum()
    total_non_null = df[col].notna().sum()
    if 0 < numeric_count < total_non_null and numeric_count > total_non_null * 0.1:
        mixed_type_cols.append(col)
        print(f"      - {col}: {numeric_count}/{total_non_null} values are numeric")

log_step("1. Data Quality Inspection", {
    "duplicate_rows": int(num_duplicates),
    "unnamed_columns": unnamed_cols,
    "columns_with_missing": len(cols_with_missing),
    "missing_details": {col: int(count) for col, count in cols_with_missing.items()},
    "date_columns": date_cols_present,
    "mixed_type_columns": mixed_type_cols
})

# ──────────────────────────────────────────────────────────────────
# STEP 2: Remove Duplicate Rows
# ──────────────────────────────────────────────────────────────────
before_dedup = len(df)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
after_dedup = len(df)
rows_removed = before_dedup - after_dedup

log_step("2. Remove Duplicate Rows", {
    "rows_before": before_dedup,
    "rows_after": after_dedup,
    "duplicates_removed": rows_removed
})

# ──────────────────────────────────────────────────────────────────
# STEP 3: Drop Garbage / Unnamed Columns
# ──────────────────────────────────────────────────────────────────
cols_before = list(df.columns)
garbage_cols = [col for col in df.columns if 'Unnamed' in str(col)]
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
cols_after = list(df.columns)

log_step("3. Drop Garbage Columns", {
    "columns_before": len(cols_before),
    "columns_after": len(cols_after),
    "dropped_columns": garbage_cols
})

# ──────────────────────────────────────────────────────────────────
# STEP 4: Fix Date Columns
# ──────────────────────────────────────────────────────────────────
date_parse_results = {}

# Parse standard date columns
standard_date_cols = ['Hire_Date', 'Onboarding_Date', 'First_Project_Start_Date', 'Offer_Acceptance_Date']
for col in standard_date_cols:
    if col in df.columns:
        before_dtype = str(df[col].dtype)
        df[col] = pd.to_datetime(df[col], errors='coerce')
        invalid_count = df[col].isna().sum()
        date_parse_results[col] = {
            "before_dtype": before_dtype,
            "after_dtype": str(df[col].dtype),
            "invalid_dates": int(invalid_count)
        }

# Handle 'Future Date' (Excel serial date format: days since 1899-12-30)
if 'Future Date' in df.columns:
    before_dtype = str(df['Future Date'].dtype)
    # Convert from Excel serial date number
    df['Future_Date'] = pd.to_timedelta(df['Future Date'], unit='D') + pd.Timestamp('1899-12-30')
    df.drop(columns=['Future Date'], inplace=True)
    date_parse_results['Future_Date'] = {
        "before_dtype": before_dtype,
        "after_dtype": str(df['Future_Date'].dtype),
        "note": "Converted from Excel serial date (origin=1899-12-30)",
        "sample_values": df['Future_Date'].head(3).astype(str).tolist()
    }

log_step("4. Fix Date Columns", date_parse_results)

# ──────────────────────────────────────────────────────────────────
# STEP 5: Handle Missing Values
# ──────────────────────────────────────────────────────────────────
missing_before = df.isnull().sum().sum()
missing_details = {}

# Strategy: 
#   - Numeric columns: fill with median
#   - Categorical columns: fill with mode (most frequent) or 'Unknown'
#   - Date columns: leave as NaT (don't impute dates)

# Numeric columns - fill with median
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in numeric_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        missing_details[col] = {
            "type": "numeric",
            "strategy": "median",
            "fill_value": float(median_val),
            "missing_count": int(missing)
        }

# Categorical columns - fill with mode or 'Unknown'
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    missing = df[col].isnull().sum()
    if missing > 0:
        # If less than 10% missing, fill with mode; otherwise 'Unknown'
        pct_missing = missing / len(df) * 100
        if pct_missing < 10:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            missing_details[col] = {
                "type": "categorical",
                "strategy": "mode",
                "fill_value": str(mode_val),
                "missing_count": int(missing),
                "pct_missing": round(pct_missing, 2)
            }
        else:
            df[col].fillna('Unknown', inplace=True)
            missing_details[col] = {
                "type": "categorical",
                "strategy": "fill_unknown",
                "fill_value": "Unknown",
                "missing_count": int(missing),
                "pct_missing": round(pct_missing, 2)
            }

missing_after = df.isnull().sum().sum()

log_step("5. Handle Missing Values", {
    "total_missing_before": int(missing_before),
    "total_missing_after": int(missing_after),
    "columns_imputed": len(missing_details),
    "imputation_details": missing_details
})

# ──────────────────────────────────────────────────────────────────
# STEP 6: Standardize Categorical Columns
# ──────────────────────────────────────────────────────────────────
standardized_cols = {}

# Key categorical columns to standardize
cats_to_standardize = [
    'Department', 'Job_Title', 'Training_Program', 'Highest_Education_Level',
    'Certifications', 'Hiring_Source', 'Project_Type', 'Project_Size',
    'Project_Complexity', 'Project_Role', 'Project_Outcome',
    'Employee_Resignation_Status', 'Supplier_Contact',
    'Career_Goals_Set', 'Career_Goals_Achievement_Status',
    'Innovation_Projects_Involvement', 'Leadership_Potential',
    'Development_Plan_Completion', 'Internship_Completion_Status',
    'Internship_Conversion_Status', 'Mentor_Experience_Level',
    'Employee_Health_Insurance_Coverage', 'Employee_Training_Certification_Status',
    'Work_Quality_Improvement_Plan'
]

for col in cats_to_standardize:
    if col in df.columns and df[col].dtype == 'object':
        before_unique = df[col].nunique()
        df[col] = df[col].str.strip().str.title()
        after_unique = df[col].nunique()
        if before_unique != after_unique:
            standardized_cols[col] = {
                "unique_before": int(before_unique),
                "unique_after": int(after_unique),
                "values_merged": int(before_unique - after_unique)
            }

log_step("6. Standardize Categories", {
    "columns_processed": len(cats_to_standardize),
    "columns_with_changes": len(standardized_cols),
    "details": standardized_cols
})

# ──────────────────────────────────────────────────────────────────
# STEP 7: Fix Mixed Data Types
# ──────────────────────────────────────────────────────────────────
type_fixes = {}

# Ensure rating columns are numeric
rating_cols = [col for col in df.columns if 'Rating' in col or 'Score' in col]
for col in rating_cols:
    if col in df.columns and df[col].dtype == 'object':
        before_dtype = str(df[col].dtype)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill any new NaN values from coercion
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
        type_fixes[col] = {"from": before_dtype, "to": str(df[col].dtype)}

# Ensure percentage columns are numeric
pct_cols = [col for col in df.columns if 'Percentage' in col or 'Rate' in col]
for col in pct_cols:
    if col in df.columns and df[col].dtype == 'object':
        before_dtype = str(df[col].dtype)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
        type_fixes[col] = {"from": before_dtype, "to": str(df[col].dtype)}

# Ensure count/number columns are numeric
count_cols = [col for col in df.columns if 'Number' in col or 'Hours' in col or 'Days' in col or 'Cost' in col]
for col in count_cols:
    if col in df.columns and df[col].dtype == 'object':
        before_dtype = str(df[col].dtype)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
        type_fixes[col] = {"from": before_dtype, "to": str(df[col].dtype)}

log_step("7. Fix Mixed Data Types", {
    "columns_fixed": len(type_fixes),
    "details": type_fixes
})

# ──────────────────────────────────────────────────────────────────
# STEP 8: Feature Engineering
# ──────────────────────────────────────────────────────────────────
new_features = {}

# 8.1 Onboarding Delay (days from Hire to Onboarding)
if 'Onboarding_Date' in df.columns and 'Hire_Date' in df.columns:
    df['Onboarding_Delay_Days'] = (df['Onboarding_Date'] - df['Hire_Date']).dt.days
    valid = df['Onboarding_Delay_Days'].notna().sum()
    new_features['Onboarding_Delay_Days'] = {
        "formula": "Onboarding_Date - Hire_Date",
        "valid_values": int(valid),
        "mean": round(float(df['Onboarding_Delay_Days'].mean()), 2),
        "median": round(float(df['Onboarding_Delay_Days'].median()), 2),
        "min": int(df['Onboarding_Delay_Days'].min()),
        "max": int(df['Onboarding_Delay_Days'].max())
    }

# 8.2 Project Start Delay (days from Onboarding to First Project)
if 'First_Project_Start_Date' in df.columns and 'Onboarding_Date' in df.columns:
    df['Project_Start_Delay_Days'] = (df['First_Project_Start_Date'] - df['Onboarding_Date']).dt.days
    valid = df['Project_Start_Delay_Days'].notna().sum()
    new_features['Project_Start_Delay_Days'] = {
        "formula": "First_Project_Start_Date - Onboarding_Date",
        "valid_values": int(valid),
        "mean": round(float(df['Project_Start_Delay_Days'].mean()), 2),
        "median": round(float(df['Project_Start_Delay_Days'].median()), 2),
        "min": int(df['Project_Start_Delay_Days'].min()),
        "max": int(df['Project_Start_Delay_Days'].max())
    }

# 8.3 Tenure (days from Hire to today)
if 'Hire_Date' in df.columns:
    df['Tenure_Days'] = (pd.Timestamp.now() - df['Hire_Date']).dt.days
    df['Tenure_Years'] = round(df['Tenure_Days'] / 365.25, 2)
    new_features['Tenure_Years'] = {
        "formula": "(Today - Hire_Date) / 365.25",
        "mean": round(float(df['Tenure_Years'].mean()), 2),
        "median": round(float(df['Tenure_Years'].median()), 2),
        "min": round(float(df['Tenure_Years'].min()), 2),
        "max": round(float(df['Tenure_Years'].max()), 2)
    }

# 8.4 Total Skills Score
skill_rating_cols = [
    'Technical_Skills_Rating', 'Communication_Skills_Rating',
    'Problem_Solving_Skills_Rating', 'Teamwork_Skills_Rating',
    'Leadership_Qualities_Rating'
]
skill_cols_present = [col for col in skill_rating_cols if col in df.columns]
if len(skill_cols_present) > 0:
    df['Total_Skills_Score'] = df[skill_cols_present].sum(axis=1)
    df['Avg_Skills_Score'] = round(df[skill_cols_present].mean(axis=1), 2)
    new_features['Total_Skills_Score'] = {
        "formula": f"Sum of {skill_cols_present}",
        "mean": round(float(df['Total_Skills_Score'].mean()), 2),
        "median": round(float(df['Total_Skills_Score'].median()), 2)
    }
    new_features['Avg_Skills_Score'] = {
        "formula": f"Mean of {skill_cols_present}",
        "mean": round(float(df['Avg_Skills_Score'].mean()), 2),
        "median": round(float(df['Avg_Skills_Score'].median()), 2)
    }

# 8.5 Soft Skills Score
soft_skill_cols = [
    'Initiative_Rating', 'Adaptability_Rating',
    'Creativity_Rating', 'Strategic_Thinking_Rating'
]
soft_cols_present = [col for col in soft_skill_cols if col in df.columns]
if len(soft_cols_present) > 0:
    df['Total_Soft_Skills_Score'] = df[soft_cols_present].sum(axis=1)
    df['Avg_Soft_Skills_Score'] = round(df[soft_cols_present].mean(axis=1), 2)
    new_features['Total_Soft_Skills_Score'] = {
        "formula": f"Sum of {soft_cols_present}",
        "mean": round(float(df['Total_Soft_Skills_Score'].mean()), 2)
    }

# 8.6 Performance Category
if 'Performance_Rating' in df.columns:
    bins = [0, 5, 10, 15, 20]
    labels = ['Low', 'Medium', 'High', 'Exceptional']
    df['Performance_Category'] = pd.cut(df['Performance_Rating'], bins=bins, labels=labels, include_lowest=True)
    new_features['Performance_Category'] = {
        "formula": "Binned Performance_Rating: 0-5=Low, 6-10=Medium, 11-15=High, 16-20=Exceptional",
        "distribution": df['Performance_Category'].value_counts().to_dict()
    }
    # Convert Category to string for CSV compatibility
    df['Performance_Category'] = df['Performance_Category'].astype(str)

log_step("8. Feature Engineering", {
    "new_features_created": len(new_features),
    "details": new_features
})

# ──────────────────────────────────────────────────────────────────
# STEP 9: Final Validation & Data Distribution
# ──────────────────────────────────────────────────────────────────
final_shape = df.shape
final_missing = df.isnull().sum().sum()
final_dtypes = df.dtypes.value_counts()

# Key distribution stats
validation = {
    "final_rows": final_shape[0],
    "final_columns": final_shape[1],
    "remaining_missing_values": int(final_missing),
    "data_types": {str(k): int(v) for k, v in final_dtypes.items()},
    "memory_usage_MB": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
}

# Performance Rating distribution
if 'Performance_Rating' in df.columns:
    validation["performance_rating_stats"] = {
        "mean": round(float(df['Performance_Rating'].mean()), 2),
        "median": round(float(df['Performance_Rating'].median()), 2),
        "std": round(float(df['Performance_Rating'].std()), 2),
        "min": int(df['Performance_Rating'].min()),
        "max": int(df['Performance_Rating'].max()),
        "distribution": df['Performance_Rating'].value_counts().sort_index().to_dict()
    }

# Department distribution
if 'Department' in df.columns:
    validation["department_distribution"] = df['Department'].value_counts().to_dict()

# Resignation status distribution
if 'Employee_Resignation_Status' in df.columns:
    validation["resignation_status"] = df['Employee_Resignation_Status'].value_counts().to_dict()

log_step("9. Final Validation", validation)

# ──────────────────────────────────────────────────────────────────
# SAVE OUTPUTS
# ──────────────────────────────────────────────────────────────────

# Save cleaned dataset
df.to_csv(PROCESSED_DATA_PATH, index=False)
print(f"\n{'='*70}")
print(f"  [OK] Cleaned data saved to: {PROCESSED_DATA_PATH}")

# Save preprocessing report
report["summary"] = {
    "initial_shape": list(initial_shape),
    "final_shape": list(final_shape),
    "rows_removed": initial_shape[0] - final_shape[0],
    "columns_removed": initial_shape[1] - final_shape[1],
    "new_features_added": len(new_features),
    "total_missing_resolved": int(missing_before - missing_after)
}

# Convert non-serializable objects
def make_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Categorical):
        return str(obj)
    return str(obj)

with open(REPORT_PATH, 'w') as f:
    json.dump(report, f, indent=2, default=make_serializable)

print(f"  [OK] Report saved to: {REPORT_PATH}")

# ──────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────
print(f"\n" + "==" * 35)
print("  PREPROCESSING COMPLETE - SUMMARY")
print("==" * 35)
print(f"  [DATA] Initial: {initial_shape[0]} rows x {initial_shape[1]} cols")
print(f"  [DATA] Final:   {final_shape[0]} rows x {final_shape[1]} cols")
print(f"  [DEL]  Rows removed (duplicates):  {initial_shape[0] - final_shape[0]}")
print(f"  [DEL]  Columns removed (garbage):  {initial_shape[1] - final_shape[1] + len(new_features)}")
print(f"  [NEW]  New features created:        {len(new_features)}")
print(f"  [FIX]  Missing values resolved:     {int(missing_before - missing_after)}")
print(f"  [MEM]  Memory: {round(df.memory_usage(deep=True).sum() / 1024**2, 2)} MB")
print()

# Print column listing
print("  [LIST] Final Columns:")
for i, col in enumerate(df.columns, 1):
    dtype = df[col].dtype
    print(f"      {i:3d}. {col} ({dtype})")
