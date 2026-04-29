"""Section 1 - Q2: High Performance but Low Leadership Potential"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, json, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF   = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT  = os.path.join(BASE, "reports", "section1", "q2_high_perf_low_leadership.json")

# ── Segment ─────────────────────────────────────────────────────
high_perf       = DF['Performance_Rating'] >= 12
low_lead        = DF['Leadership_Potential'].str.lower() == 'low'

segment         = DF[high_perf & low_lead].copy()
high_perf_only  = DF[high_perf & ~low_lead].copy()
total_high      = DF[high_perf].shape[0]

print(f"Total High Performers (>=12)   : {total_high}")
print(f"High Perf + Low Leadership     : {len(segment)}  ({round(len(segment)/total_high*100,1)}%)")

# ── Profile of segment ─────────────────────────────────────────
ctx_cols = ['Department','Job_Title','Project_Role','Project_Outcome',
            'Highest_Education_Level','Certifications','Training_Program',
            'Innovation_Projects_Involvement','Work_Quality_Improvement_Plan',
            'Career_Goals_Achievement_Status','Employee_Resignation_Status',
            'Mentor_Experience_Level','Internship_Conversion_Status']

num_cols = ['Performance_Rating','Leadership_Qualities_Rating','Technical_Skills_Rating',
            'Communication_Skills_Rating','Problem_Solving_Skills_Rating',
            'Employee_Engagement_Score','Employee_Job_Satisfaction_Score',
            'Overtime_Hours_Per_Week','Professional_Development_Hours',
            'Number_Of_Promotions','Conflict_Resolution_Cases',
            'Feedback_From_Supervisors','Feedback_From_Colleagues',
            'Avg_Skills_Score','Engagement_Index']

num_cols = [c for c in num_cols if c in DF.columns]

seg_means  = segment[num_cols].mean().round(2).to_dict()
rest_means = high_perf_only[num_cols].mean().round(2).to_dict()

print("\n=== Numeric Comparison (Segment vs High-Perf-with-Leadership) ===")
for c in num_cols:
    print(f"  {c}: segment={seg_means[c]}, others={rest_means.get(c,'N/A')}")

cat_profiles = {}
for col in ctx_cols:
    if col in segment.columns:
        cat_profiles[col] = segment[col].value_counts(normalize=True).round(3).mul(100).to_dict()

print("\n=== Categorical Distributions ===")
for k, v in cat_profiles.items():
    print(f"  {k}: {dict(list(v.items())[:3])}")

# ── Possible reasons (rule-based LLM-style reasoning) ─────────────
reasons = []
if seg_means.get('Conflict_Resolution_Cases', 0) < rest_means.get('Conflict_Resolution_Cases', 99):
    reasons.append("Lower conflict resolution cases suggest limited team leadership experience.")
if seg_means.get('Overtime_Hours_Per_Week', 0) > rest_means.get('Overtime_Hours_Per_Week', 0):
    reasons.append("Higher overtime may indicate individual contribution focus over team delegation.")
if seg_means.get('Professional_Development_Hours', 0) < rest_means.get('Professional_Development_Hours', 0):
    reasons.append("Fewer development hours may signal lack of investment in leadership-building activities.")
if seg_means.get('Employee_Engagement_Score', 0) < rest_means.get('Employee_Engagement_Score', 0):
    reasons.append("Lower engagement scores may correlate with reduced motivation to take on leadership roles.")
if seg_means.get('Number_Of_Promotions', 0) < rest_means.get('Number_Of_Promotions', 0):
    reasons.append("Fewer promotions despite high performance may reflect lack of visible leadership behaviours.")
reasons += [
    "Individual contributors often excel technically without needing to manage or influence others.",
    "Role type (Analyst/Developer) may naturally limit leadership exposure.",
    "High performance may be task/output driven rather than people/strategy driven."
]

print("\n=== LLM Reasoning: Possible Reasons ===")
for i, r in enumerate(reasons, 1):
    print(f"  {i}. {r}")

result = {
    "question": "Q2 - High Performance, Low Leadership Potential",
    "total_high_performers": int(total_high),
    "high_perf_low_leadership_count": int(len(segment)),
    "percentage_of_high_performers": round(len(segment)/total_high*100, 1),
    "segment_numeric_means": seg_means,
    "comparison_means": rest_means,
    "categorical_profiles": cat_profiles,
    "possible_reasons": reasons,
    "dept_breakdown": segment['Department'].value_counts().to_dict() if 'Department' in segment.columns else {},
    "role_breakdown": segment['Job_Title'].value_counts().to_dict() if 'Job_Title' in segment.columns else {},
    "project_role_breakdown": segment['Project_Role'].value_counts().to_dict() if 'Project_Role' in segment.columns else {},
    "outcome_breakdown": segment['Project_Outcome'].value_counts().to_dict() if 'Project_Outcome' in segment.columns else {},
    "resignation_breakdown": segment['Employee_Resignation_Status'].value_counts().to_dict() if 'Employee_Resignation_Status' in segment.columns else {},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n[OK] Report saved -> {OUT}")
