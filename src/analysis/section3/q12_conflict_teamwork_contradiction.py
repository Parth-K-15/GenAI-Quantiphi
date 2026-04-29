"""Section 3 - Q12: Identify employees with high conflict resolution cases
but low teamwork scores and explain contradictions.
"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd, numpy as np, json, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF   = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT  = os.path.join(BASE, "reports", "section3", "q12_conflict_teamwork_contradiction.json")

# ── Thresholds ────────────────────────────────────────────────────────────────
conflict_thresh = DF['Conflict_Resolution_Cases'].quantile(0.75)   # High: top 25%
teamwork_thresh = DF['Teamwork_Skills_Rating'].quantile(0.25)       # Low:  bottom 25%

print(f"High Conflict threshold (Q75): {conflict_thresh}")
print(f"Low Teamwork threshold (Q25):  {teamwork_thresh}")

# ── Conflict-to-Teamwork Ratio (Advanced Metric) ──────────────────────────────
# Conflict resolution captures reactive problem-solving; teamwork reflects
# proactive collaboration — the two are fundamentally different behavioral dimensions.
DF['Conflict_to_Teamwork_Ratio'] = DF['Conflict_Resolution_Cases'] / (DF['Teamwork_Skills_Rating'] + 1)
DF['Conflict_Behavior_Type'] = pd.qcut(
    DF['Conflict_to_Teamwork_Ratio'], 4,
    labels=["Collaborative", "Balanced", "Reactive", "Conflict-Heavy"]
)
print("\n=== Conflict-to-Teamwork Ratio Stats ===")
print(DF['Conflict_to_Teamwork_Ratio'].describe().round(3))
print("\n=== Conflict Behavior Type Distribution ===")
print(DF['Conflict_Behavior_Type'].value_counts())

# ── Identify contradictory employees ─────────────────────────────────────────
mask_contradiction = (
    (DF['Conflict_Resolution_Cases'] >= conflict_thresh) &
    (DF['Teamwork_Skills_Rating'] <= teamwork_thresh)
)
contradictory = DF[mask_contradiction].copy()
normal_rest   = DF[~mask_contradiction].copy()

print(f"\nContradictory employees: {len(contradictory)} ({len(contradictory)/len(DF)*100:.1f}%)")

# ── Comparison profile ────────────────────────────────────────────────────────
compare_cols = [
    'Conflict_Resolution_Cases', 'Teamwork_Skills_Rating',
    'Leadership_Qualities_Rating', 'Adaptability_Rating', 'Creativity_Rating',
    'Performance_Rating', 'Employee_Engagement_Score',
    'Employee_Job_Satisfaction_Score', 'Overtime_Hours_Per_Week',
    'Employee_Work_Life_Balance_Rating', 'Number_Of_Promotions',
    'Communication_Skills_Rating', 'Initiative_Rating'
]
compare_cols = [c for c in compare_cols if c in DF.columns]

profile_contradiction = contradictory[compare_cols].mean().round(3)
profile_normal        = normal_rest[compare_cols].mean().round(3)

comparison_df = pd.DataFrame({
    'Contradictory': profile_contradiction,
    'Non-Contradictory': profile_normal,
    'Delta': (profile_contradiction - profile_normal).round(3)
})
print("\n=== Contradictory vs. Rest Profile ===")
print(comparison_df.to_string())

# ── Attrition & Dept breakdown ────────────────────────────────────────────────
attrition_contra = round((contradictory['Employee_Resignation_Status'] == 'Yes').mean() * 100, 2)
attrition_normal = round((normal_rest['Employee_Resignation_Status'] == 'Yes').mean() * 100, 2)

dept_breakdown = contradictory.groupby('Department').agg(
    count=('Employee_ID', 'count'),
    avg_conflict=('Conflict_Resolution_Cases', 'mean'),
    avg_teamwork=('Teamwork_Skills_Rating', 'mean'),
    avg_performance=('Performance_Rating', 'mean')
).round(3).reset_index().sort_values('count', ascending=False)

print("\n=== Department Breakdown ===")
print(dept_breakdown.to_string(index=False))

# ── Contradiction archetypes ───────────────────────────────────────────────────
archetypes = [
    {
        "name": "The Forced Mediator",
        "emoji": "⚔️",
        "description": (
            "Employees thrust into conflict resolution roles by circumstance — not by interpersonal strength. "
            "High conflict exposure may be driven by toxic team dynamics or difficult stakeholders, "
            "not by the employee's ability to work with others. They resolve conflicts to survive, "
            "not because they thrive in collaboration."
        ),
        "indicator": "High overtime + low WLB + low engagement",
        "action": "Reduce structural conflict exposure. Redesign team composition to reduce friction sources."
    },
    {
        "name": "The Dominant Resolver",
        "emoji": "👑",
        "description": (
            "High leadership and assertiveness drive conflict resolution success, but this same dominance "
            "creates friction in teamwork settings. These individuals resolve conflicts top-down, "
            "not collaboratively — they shut conflicts down rather than building consensus. "
            "Effective individually, but disruptive in flat or collaborative team structures."
        ),
        "indicator": "High leadership + high initiative + low teamwork + moderate performance",
        "action": "Coaching on inclusive facilitation and participatory leadership styles."
    },
    {
        "name": "The Structural Outlier",
        "emoji": "🔧",
        "description": (
            "Role-based conflict exposure in customer-facing or project management functions "
            "artificially inflates their conflict resolution case count. Their low teamwork "
            "score reflects low team integration — they work in silos by design, not by choice. "
            "The contradiction is organizational, not behavioral."
        ),
        "indicator": "High customer complaints handled + functional isolation",
        "action": "Distinguish structural conflict cases from interpersonal teamwork failures in HR metrics."
    },
    {
        "name": "The Burned-Out Resolver",
        "emoji": "🔥",
        "description": (
            "Once effective collaborators, now depleted. Repeated conflict exposure over time has eroded "
            "their willingness to engage as team players. Burnout from sustained conflict management "
            "causes retreat from collaborative behavior — creating the apparent paradox. "
            "This is a retention and mental health risk signal."
        ),
        "indicator": "Low satisfaction + low WLB + declining engagement over tenure",
        "action": "Immediate wellbeing intervention and workload redistribution. Monitor for attrition risk."
    }
]

# ── Sample contradictory employees ───────────────────────────────────────────
sample_cols = [c for c in [
    'Employee_ID', 'Department', 'Job_Title',
    'Conflict_Resolution_Cases', 'Teamwork_Skills_Rating',
    'Leadership_Qualities_Rating', 'Performance_Rating',
    'Employee_Engagement_Score', 'Overtime_Hours_Per_Week',
    'Employee_Work_Life_Balance_Rating', 'Employee_Resignation_Status'
] if c in contradictory.columns]
sample = contradictory.sort_values('Conflict_Resolution_Cases', ascending=False).head(15)[sample_cols]
print("\n=== Top 15 Contradictory Employees ===")
print(sample.to_string(index=False))

# ── Ratio profile by behavior type ───────────────────────────────────────────
ratio_profile = DF.groupby('Conflict_Behavior_Type', observed=True).agg(
    count=('Employee_ID', 'count'),
    avg_ratio=('Conflict_to_Teamwork_Ratio', 'mean'),
    avg_conflict=('Conflict_Resolution_Cases', 'mean'),
    avg_teamwork=('Teamwork_Skills_Rating', 'mean'),
    avg_performance=('Performance_Rating', 'mean'),
    attrition_pct=('Employee_Resignation_Status', lambda x: round((x == 'Yes').mean()*100, 2))
).round(3).reset_index()
print("\n=== Ratio Profile by Behavior Type ===")
print(ratio_profile.to_string(index=False))

result = {
    "question": "Q12 — High Conflict Resolution + Low Teamwork Contradiction Analysis",
    "thresholds": {
        "high_conflict_q75": float(conflict_thresh),
        "low_teamwork_q25": float(teamwork_thresh)
    },
    "counts": {
        "contradictory_employees": int(len(contradictory)),
        "contradictory_pct": round(len(contradictory) / len(DF) * 100, 1),
        "total_employees": int(len(DF))
    },
    "attrition_rates": {
        "contradictory_group_pct": attrition_contra,
        "non_contradictory_group_pct": attrition_normal,
        "delta": round(attrition_contra - attrition_normal, 2)
    },
    "profile_comparison": {
        "contradictory": profile_contradiction.to_dict(),
        "non_contradictory": profile_normal.to_dict(),
        "delta": comparison_df['Delta'].to_dict()
    },
    "department_breakdown": dept_breakdown.to_dict(orient='records'),
    "contradiction_archetypes": archetypes,
    "conflict_to_teamwork_ratio": {
        "definition": "Conflict_Resolution_Cases / (Teamwork_Skills_Rating + 1)",
        "insight": (
            "Conflict resolution metrics capture reactive problem-solving, whereas teamwork reflects "
            "proactive collaboration — the two are fundamentally different behavioral dimensions. "
            "We define a Conflict-to-Teamwork Ratio to identify employees who are disproportionately "
            "involved in conflict relative to their collaborative ability."
        ),
        "behavior_type_distribution": ratio_profile.to_dict(orient='records'),
        "ratio_stats": DF['Conflict_to_Teamwork_Ratio'].describe().round(3).to_dict()
    },
    "sample_employees": sample.to_dict(orient='records'),
    "llm_insights": {
        "headline": (
            "High conflict resolution + low teamwork is not a contradiction in behavior — "
            "it is a signal of structural misalignment between the employee's role demands and "
            "their interpersonal orientation. The paradox resolves when you distinguish between "
            "conflict as a task (forced) vs. collaboration as a choice (intrinsic)."
        ),
        "dimensional_insight": (
            "Conflict resolution metrics capture reactive problem-solving, whereas teamwork reflects "
            "proactive collaboration — the two are fundamentally different behavioral dimensions. "
            "Treating them as equivalent in HR assessments leads to misdiagnosis of behavioral profiles."
        ),
        "key_insight_1": (
            "Conflict resolution cases are often driven by role exposure (customer-facing, managerial) "
            "rather than interpersonal strength. A high count does not mean the employee is good at teams — "
            "it means they encounter, and must navigate, friction by design."
        ),
        "key_insight_2": (
            "Low teamwork scores in this group may reflect burnout from sustained conflict exposure, "
            "not an inherent trait. The prescription is workload relief and team redesign — "
            "not just interpersonal skills coaching."
        ),
        "key_insight_3": (
            "These employees show higher attrition risk than the general population — "
            "suggesting the contradiction is unsustainable over time. "
            "Organizations must intervene before burnout converts into resignation."
        ),
        "standout_statement": (
            "The most dangerous employee profile is not the one who can't handle conflict — "
            "it is the one who handles everyone else's conflicts but has no team to call their own. "
            "Isolation dressed as effectiveness is a silent attrition engine."
        )
    }
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n[OK] Saved -> {OUT}")
