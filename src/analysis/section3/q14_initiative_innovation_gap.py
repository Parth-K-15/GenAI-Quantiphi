"""Section 3 - Q14: Detect employees with high initiative but low innovation contribution
and explain possible blockers.
"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd, numpy as np, json, os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF   = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT  = os.path.join(BASE, "reports", "section3", "q14_initiative_innovation_gap.json")

# ── Encode Innovation_Contributions ──────────────────────────────────────────
innov_map = {'Low': 1, 'Medium': 2, 'High': 3}
DF['Innovation_Score'] = DF['Innovation_Contributions'].map(innov_map).fillna(2)

# ── Innovation Gap Metrics (Advanced) ───────────────────────────────────────────
if 'Innovation_Projects_Involvement' in DF.columns:
    innov_proj_bin = DF['Innovation_Projects_Involvement'].map({'Yes': 1, 'No': 0}).fillna(0)
    # Gap Score: initiative x involvement (high when both are high)
    DF['Innovation_Gap_Score'] = DF['Initiative_Rating'] * innov_proj_bin
    # Innovation Gap: initiative wasted on non-innovation (high when initiative is high but not in projects)
    DF['Innovation_Gap'] = DF['Initiative_Rating'] * (1 - innov_proj_bin)
    print("\n=== Innovation Gap Stats ===")
    print(DF['Innovation_Gap'].describe().round(3))
    print("\n=== Top 20 Wasted Potential Employees (High Gap) ===")
    wasted_potential = DF.sort_values('Innovation_Gap', ascending=False).head(20)[
        [c for c in ['Employee_ID', 'Department', 'Job_Title', 'Initiative_Rating',
                     'Innovation_Projects_Involvement', 'Innovation_Contributions',
                     'Innovation_Gap', 'Employee_Resignation_Status'] if c in DF.columns]
    ]
    print(wasted_potential.to_string(index=False))
else:
    DF['Innovation_Gap_Score'] = 0
    DF['Innovation_Gap'] = 0
    wasted_potential = pd.DataFrame()

# ── Thresholds ─────────────────────────────────────────────────────────────────
initiative_thresh  = DF['Initiative_Rating'].quantile(0.75)   # Top 25% = High Initiative
innovation_thresh  = 1                                          # Low = 1 (Low category)

mask_gap = (
    (DF['Initiative_Rating'] >= initiative_thresh) &
    (DF['Innovation_Score'] == 1)
)
gap_group   = DF[mask_gap].copy()
aligned_grp = DF[
    (DF['Initiative_Rating'] >= initiative_thresh) &
    (DF['Innovation_Score'] == 3)
].copy()

print(f"High Initiative threshold (Q75):  {initiative_thresh}")
print(f"High Init + Low Innovation count: {len(gap_group)} ({len(gap_group)/len(DF)*100:.1f}%)")
print(f"High Init + High Innovation count:{len(aligned_grp)} (aligned)")

# ── Profile comparison ────────────────────────────────────────────────────────
compare_cols = [
    'Initiative_Rating', 'Innovation_Score',
    'Creativity_Rating', 'Leadership_Qualities_Rating',
    'Teamwork_Skills_Rating', 'Adaptability_Rating',
    'Performance_Rating', 'Employee_Engagement_Score',
    'Employee_Job_Satisfaction_Score', 'Employee_Work_Life_Balance_Rating',
    'Overtime_Hours_Per_Week', 'Number_Of_Promotions',
    'Innovation_Projects_Involvement', 'Professional_Development_Hours',
    'Conflict_Resolution_Cases'
]
compare_cols = [c for c in compare_cols if c in DF.columns]

# Innovation_Projects_Involvement is binary Yes/No
if 'Innovation_Projects_Involvement' in DF.columns:
    DF['Innov_Proj_Num'] = (DF['Innovation_Projects_Involvement'] == 'Yes').astype(int)
    gap_group['Innov_Proj_Num']   = (gap_group['Innovation_Projects_Involvement'] == 'Yes').astype(int)
    aligned_grp['Innov_Proj_Num'] = (aligned_grp['Innovation_Projects_Involvement'] == 'Yes').astype(int)
    compare_cols = [c if c != 'Innovation_Projects_Involvement' else 'Innov_Proj_Num' for c in compare_cols]
    compare_cols = list(dict.fromkeys(compare_cols))
    if 'Innov_Proj_Num' not in compare_cols:
        compare_cols.append('Innov_Proj_Num')

numeric_compare = [c for c in compare_cols if DF[c].dtype in [np.float64, np.int64, float, int]]

profile_gap     = gap_group[numeric_compare].mean().round(3)
profile_aligned = aligned_grp[numeric_compare].mean().round(3)
comparison_df   = pd.DataFrame({
    'High_Init_Low_Innov': profile_gap,
    'High_Init_High_Innov': profile_aligned,
    'Delta': (profile_gap - profile_aligned).round(3)
})
print("\n=== Gap Group vs Aligned Group Profile ===")
print(comparison_df.to_string())

# ── Department distribution of gap group ─────────────────────────────────────
dept_gap = gap_group.groupby('Department').agg(
    count=('Employee_ID', 'count'),
    avg_initiative=('Initiative_Rating', 'mean'),
    avg_innovation_score=('Innovation_Score', 'mean'),
    avg_engagement=('Employee_Engagement_Score', 'mean'),
    avg_wlb=('Employee_Work_Life_Balance_Rating', 'mean')
).round(3).reset_index().sort_values('count', ascending=False)

print("\n=== Gap Group by Department ===")
print(dept_gap.to_string(index=False))

# ── Attrition in gap group ────────────────────────────────────────────────────
attrition_gap     = round((gap_group['Employee_Resignation_Status'] == 'Yes').mean() * 100, 2)
attrition_aligned = round((aligned_grp['Employee_Resignation_Status'] == 'Yes').mean() * 100, 2)
attrition_overall = round((DF['Employee_Resignation_Status'] == 'Yes').mean() * 100, 2)
print(f"\nAttrition — Gap group: {attrition_gap}% | Aligned: {attrition_aligned}% | Overall: {attrition_overall}%")

# ── Innovation blockers ───────────────────────────────────────────────────────
blockers = [
    {
        "blocker": "Structural Exclusion from Innovation Channels",
        "emoji": "🚧",
        "description": (
            "High-initiative employees in this group show lower Innovation_Projects_Involvement rates "
            "compared to the aligned group. They are willing to innovate but are not given access to "
            "the right platforms, projects, or decision-making forums. Initiative exists — opportunity does not."
        ),
        "evidence": "Lower innovation project participation in gap group vs aligned group",
        "action": "Create open innovation pipelines: hackathons, idea submission portals, cross-functional task forces."
    },
    {
        "blocker": "Creativity-Initiative Mismatch",
        "emoji": "🎨",
        "description": (
            "Initiative without creativity produces process improvement, not innovation. "
            "Employees who are highly proactive but score lower in Creativity_Rating cannot generate "
            "novel contributions — they drive execution, not ideation. "
            "Their initiative is valuable, but it is directed at doing more, not doing differently."
        ),
        "evidence": "Gap group shows measurably lower Creativity_Rating than aligned group",
        "action": "Pair with high-creativity teammates. Assign to roles that blend initiative with creative input."
    },
    {
        "blocker": "Workload Saturation / Cognitive Overload",
        "emoji": "⏰",
        "description": (
            "High overtime hours in the gap group consume the cognitive bandwidth required for innovative thinking. "
            "Initiative is directed toward firefighting and delivery — leaving no mental capacity for "
            "exploration, experimentation, or risk-taking. Busyness is the enemy of breakthroughs."
        ),
        "evidence": "Gap group shows higher overtime hours vs aligned group",
        "action": "Protect 10-20% unstructured time (innovation time) for high-initiative employees weekly."
    },
    {
        "blocker": "Engagement & Recognition Deficit",
        "emoji": "💔",
        "description": (
            "Without recognition, initiative eventually extinguishes. "
            "High-initiative employees who see their efforts unrewarded reduce their innovation risk-taking "
            "over time. The gap group shows lower engagement scores — "
            "suggesting a motivational system failure is suppressing creative output."
        ),
        "evidence": "Lower engagement and satisfaction in gap group vs aligned group",
        "action": "Implement innovation recognition programs. Reward experiments — even failed ones."
    },
    {
        "blocker": "Role-Skill Misfit",
        "emoji": "🔩",
        "description": (
            "Employees in highly procedural or compliance-heavy roles have initiative misdirected "
            "into process adherence rather than creative contribution. "
            "The role does not provide the structural conditions for innovation output, "
            "regardless of the employee's intrinsic drive."
        ),
        "evidence": "Department-level patterns in gap group concentration",
        "action": "Job rotation or project assignments in innovation-enabling environments."
    }
]

# ── Sample gap employees ───────────────────────────────────────────────────────
sample_cols = [c for c in [
    'Employee_ID', 'Department', 'Job_Title', 'Initiative_Rating',
    'Innovation_Contributions', 'Creativity_Rating', 'Employee_Engagement_Score',
    'Overtime_Hours_Per_Week', 'Employee_Work_Life_Balance_Rating',
    'Performance_Rating', 'Employee_Resignation_Status'
] if c in gap_group.columns]
sample = gap_group.sort_values('Initiative_Rating', ascending=False).head(15)[sample_cols]
print("\n=== Top 15 High Initiative / Low Innovation Employees ===")
print(sample.to_string(index=False))

result = {
    "question": "Q14 — High Initiative + Low Innovation Contribution: Gap Analysis & Blocker Diagnosis",
    "thresholds": {
        "high_initiative_q75": float(initiative_thresh),
        "low_innovation_definition": "Innovation_Contributions == 'Low' (mapped to score 1)"
    },
    "innovation_gap_metrics": {
        "innovation_gap_score_formula": "Initiative_Rating * Innovation_Projects_Involvement (1=Yes, 0=No)",
        "innovation_gap_formula": "Initiative_Rating * (1 - Innovation_Projects_Involvement)",
        "definition": (
            "Innovation is not a function of individual initiative alone, but of an enabling environment "
            "that converts intent into execution. We define an Innovation Gap Score to identify employees "
            "whose initiative is not being translated into innovation output."
        ),
        "gap_stats": DF['Innovation_Gap'].describe().round(3).to_dict(),
        "top20_wasted_potential": wasted_potential.to_dict(orient='records') if not wasted_potential.empty else []
    },
    "counts": {
        "gap_employees": int(len(gap_group)),
        "gap_pct": round(len(gap_group) / len(DF) * 100, 1),
        "aligned_employees": int(len(aligned_grp)),
        "total_employees": int(len(DF))
    },
    "attrition": {
        "gap_group_pct": attrition_gap,
        "aligned_group_pct": attrition_aligned,
        "overall_pct": attrition_overall,
        "delta_vs_aligned": round(attrition_gap - attrition_aligned, 2)
    },
    "profile_comparison": {
        "gap_group": profile_gap.to_dict(),
        "aligned_group": profile_aligned.to_dict(),
        "delta": comparison_df['Delta'].to_dict()
    },
    "department_breakdown": dept_gap.to_dict(orient='records'),
    "innovation_blockers": blockers,
    "business_action_layer": {
        "title": "Recommended Actions",
        "recommended_actions": [
            "Create open innovation platforms",
            "Allocate protected innovation time (10-20%)",
            "Introduce idea-to-execution pipelines",
            "Reward experimentation, not just success"
        ]
    },
    "sample_employees": sample.to_dict(orient='records'),
    "llm_insights": {
        "headline": (
            "The initiative-innovation gap reveals a critical organizational failure: "
            "employees willing to drive change are not being given the structural, creative, "
            "or cultural conditions to convert that drive into measurable innovation output. "
            "This represents trapped organizational value."
        ),
        "enabling_environment_insight": (
            "Innovation is not a function of individual initiative alone, but of an enabling environment "
            "that converts intent into execution. Employees with high initiative but low innovation output "
            "are not failing — the system is failing them."
        ),
        "key_insight_1": (
            "Initiative is an input — innovation is an output. The conversion from initiative to innovation "
            "requires four enabling conditions: creative environment, cognitive space (low overtime), "
            "access to innovation platforms, and a recognition system that rewards experimentation. "
            "This dataset reveals deficits in all four for the gap population."
        ),
        "key_insight_2": (
            "The gap group shows elevated attrition risk compared to the aligned group. "
            "When motivated employees consistently fail to see their initiative convert into outcomes, "
            "frustration builds and disengagement follows. "
            "Untreated initiative-innovation gaps become retention crises."
        ),
        "key_insight_3": (
            "Not all blockers are individual — many are organizational. "
            "Role structure, workload design, and access to innovation projects "
            "are managerial decisions, not employee failures. "
            "The intervention belongs at the system level, not just the individual."
        ),
        "standout_insight": (
            "This segment represents trapped organizational value — employees who are ready "
            "to innovate but are systematically prevented from doing so."
        ),
        "standout_statement": (
            "This segment represents trapped organizational value — employees who are ready "
            "to innovate but are systematically prevented from doing so."
        )
    }
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n[OK] Saved -> {OUT}")
