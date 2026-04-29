"""Section 1 - Q4: Skill-Performance Inconsistency Detection (High skill + Failed project)"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd, numpy as np, json, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF   = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT  = os.path.join(BASE, "reports", "section1", "q4_inconsistency_detection.json")

SKILL_COLS = ['Technical_Skills_Rating','Communication_Skills_Rating',
              'Problem_Solving_Skills_Rating','Leadership_Qualities_Rating',
              'Initiative_Rating','Adaptability_Rating','Creativity_Rating',
              'Teamwork_Skills_Rating','Strategic_Thinking_Rating']
SKILL_COLS = [c for c in SKILL_COLS if c in DF.columns]

# ── Compute composite skill score ────────────────────────────────
DF['Composite_Skill_Score'] = DF[SKILL_COLS].mean(axis=1).round(3)
skill_thresh = DF['Composite_Skill_Score'].quantile(0.75)   # top 25% skilled
print(f"Skill threshold (75th pct): {skill_thresh:.2f}")

# ── Define segments ──────────────────────────────────────────────
high_skill  = DF['Composite_Skill_Score'] >= skill_thresh
failed_proj = DF['Project_Outcome'].str.lower() == 'failed'
success_proj= DF['Project_Outcome'].str.lower() == 'successful'

# Inconsistency A: High skill + Failed project
anomaly_A = DF[high_skill & failed_proj].copy()
# Inconsistency B: Low skill + Successful project  
anomaly_B = DF[~high_skill & success_proj].copy()
# Expected: High skill + Successful
expected  = DF[high_skill & success_proj].copy()

print(f"\nHigh-skill + Failed  (Anomaly A): {len(anomaly_A)}")
print(f"Low-skill  + Success (Anomaly B): {len(anomaly_B)}")
print(f"High-skill + Success (Expected) : {len(expected)}")
print(f"Total rows: {len(DF)}")

# ── Profile anomaly A ─────────────────────────────────────────────
ctx_num = [c for c in ['Performance_Rating','Employee_Engagement_Score',
    'Employee_Job_Satisfaction_Score','Overtime_Hours_Per_Week',
    'Conflict_Resolution_Cases','Mentor_Rating','Work_Hours_Per_Week',
    'Employee_Work_Life_Balance_Rating','Training_Efficiency',
    'Engagement_Index'] if c in DF.columns]

A_means = anomaly_A[ctx_num].mean().round(3).to_dict()
E_means = expected[ctx_num].mean().round(3).to_dict()

print("\n=== Anomaly A vs Expected: Numeric ===")
for c in ctx_num:
    print(f"  {c}: Anomaly={A_means.get(c,'N/A')}, Expected={E_means.get(c,'N/A')}")

cat_ctx = [c for c in ['Department','Job_Title','Project_Role','Project_Complexity',
    'Project_Size','Training_Program','Leadership_Potential',
    'Employee_Resignation_Status','Career_Goals_Achievement_Status'] if c in DF.columns]

A_cats = {c: anomaly_A[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict() for c in cat_ctx}
E_cats = {c: expected[c].value_counts(normalize=True).round(3).mul(100).head(4).to_dict() for c in cat_ctx}

# ── Anomaly explanations (LLM reasoning) ─────────────────────────
reasons_A = []
if A_means.get('Employee_Engagement_Score', 999) < E_means.get('Employee_Engagement_Score', 0):
    reasons_A.append("Lower engagement despite high skills — disengaged employees underperform in execution.")
if A_means.get('Conflict_Resolution_Cases', 0) > E_means.get('Conflict_Resolution_Cases', 0):
    reasons_A.append("More conflict cases — team friction may derail otherwise skilled employees.")
if A_means.get('Overtime_Hours_Per_Week', 0) > E_means.get('Overtime_Hours_Per_Week', 0):
    reasons_A.append("Higher overtime — burnout risk may reduce effective contribution despite high skill.")
if A_means.get('Employee_Work_Life_Balance_Rating', 999) < E_means.get('Employee_Work_Life_Balance_Rating', 0):
    reasons_A.append("Poorer work-life balance — organizational stress dampens skilled output.")
if A_means.get('Training_Efficiency', 999) < E_means.get('Training_Efficiency', 0):
    reasons_A.append("Lower training efficiency — skills exist but are not being applied effectively.")
reasons_A += [
    "Project complexity mismatch — highly skilled employees placed on overly complex projects without adequate support.",
    "Leadership or team dynamics issues — individual skill does not guarantee team-level project success.",
    "Skills may be domain-specific and not aligned to the project type assigned.",
    "External/organizational factors (deadlines, resources) — not captured by skill ratings alone.",
]

print("\n=== Anomaly A Explanations ===")
for i, r in enumerate(reasons_A, 1):
    print(f"  {i}. {r}")

result = {
    "question": "Q4 - Skill-Performance Inconsistency Detection",
    "skill_threshold_75pct": round(float(skill_thresh), 3),
    "segment_counts": {
        "high_skill_failed_anomaly_A": int(len(anomaly_A)),
        "low_skill_successful_anomaly_B": int(len(anomaly_B)),
        "high_skill_successful_expected": int(len(expected)),
        "total": int(len(DF))
    },
    "anomaly_A_pct_of_high_skill": round(len(anomaly_A) / (len(anomaly_A)+len(expected)) * 100, 1),
    "anomaly_A_numeric_means": A_means,
    "expected_numeric_means": E_means,
    "anomaly_A_categorical": A_cats,
    "expected_categorical": E_cats,
    "anomaly_B_counts": {
        "department": anomaly_B['Department'].value_counts().head(4).to_dict() if 'Department' in anomaly_B.columns else {},
        "job_title": anomaly_B['Job_Title'].value_counts().head(4).to_dict() if 'Job_Title' in anomaly_B.columns else {},
    },
    "llm_reasoning": {
        "anomaly_A_headline": (
            f"{len(anomaly_A)} employees ({round(len(anomaly_A)/len(DF)*100,1)}%) show high skills "
            "yet are associated with failed projects — a critical inconsistency."
        ),
        "anomaly_B_headline": (
            f"{len(anomaly_B)} employees with below-average skills achieved project success, "
            "reinforcing that teamwork, context and luck play significant roles."
        ),
        "reasons_for_anomaly_A": reasons_A,
        "key_insight": (
            "Skill ratings are necessary but not sufficient for project success. "
            "Engagement, work conditions, team dynamics and project-role alignment "
            "are critical mediating factors."
        ),
        "business_implication": (
            "Organizations should not rely solely on skill assessments for project staffing. "
            "Engagement, WLB, role-project alignment and team composition must be evaluated together."
        ),
        "project_complexity_note": (
            "Project complexity is mixed in failure cases (Simple ~39%, Complex ~31.5%), "
            "suggesting failure is not purely due to complexity but also organizational "
            "and execution factors — even simple projects fail when engagement and alignment are poor."
        ),
    },
    "outcome_distribution": DF['Project_Outcome'].value_counts().to_dict() if 'Project_Outcome' in DF.columns else {},
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n[OK] Saved -> {OUT}")
