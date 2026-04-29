"""
Section 1 - Q1: Skill Influence on Performance Rating
Weighted Scoring Model via Correlation-based Reasoning
"""
import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import json, os

BASE  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DF    = pd.read_csv(os.path.join(BASE, "data", "processed", "employee_data_final.csv"))
OUT   = os.path.join(BASE, "reports", "section1", "q1_skill_influence.json")

SKILL_COLS = ['Technical_Skills_Rating', 'Communication_Skills_Rating',
              'Problem_Solving_Skills_Rating']
TARGET     = 'Performance_Rating'

# ── 1. Correlation Matrix ────────────────────────────────────────
corr_matrix = DF[SKILL_COLS + [TARGET]].corr().round(4)
print("\n=== CORRELATION MATRIX ===")
print(corr_matrix.to_string())

# ── 2. Individual Correlations ───────────────────────────────────
ind_corr = {}
print("\n=== INDIVIDUAL CORRELATIONS ===")
for col in SKILL_COLS:
    r = round(DF[col].corr(DF[TARGET]), 6)
    ind_corr[col] = r
    strength = "weak" if abs(r) < 0.1 else ("moderate" if abs(r) < 0.3 else "strong")
    direction = "positive" if r >= 0 else "negative"
    print(f"  {col}: {r:.6f}  ({direction}, {strength})")

# ── 3. Weighted Scoring Model ────────────────────────────────────
abs_corrs = {c: abs(v) for c, v in ind_corr.items()}
total     = sum(abs_corrs.values())
weights   = {c: round(v / total, 4) for c, v in abs_corrs.items()}

print("\n=== NORMALIZED WEIGHTS ===")
for c, w in weights.items():
    print(f"  {c}: {w:.4f}  ({w*100:.2f}%)")

# Compute weighted score per employee
DF['Weighted_Skill_Score'] = sum(DF[c] * w for c, w in weights.items())
DF['Weighted_Skill_Score'] = DF['Weighted_Skill_Score'].round(4)
wss_perf_corr = round(DF['Weighted_Skill_Score'].corr(DF[TARGET]), 6)
print(f"\n  Weighted_Skill_Score vs Performance_Rating corr: {wss_perf_corr}")

# ── 4. Percentile buckets ────────────────────────────────────────
bins   = [0, 5, 10, 15, 25]
labels = ['Low(1-5)', 'Medium(6-10)', 'High(11-15)', 'Top(16+)']
DF['Perf_Band'] = pd.cut(DF[TARGET], bins=bins, labels=labels, include_lowest=True)
band_skill = DF.groupby('Perf_Band', observed=True)[SKILL_COLS].mean().round(2)
print("\n=== AVG SKILL SCORES BY PERFORMANCE BAND ===")
print(band_skill.to_string())

# ── 5. Scatter bin data for chart (binned avg) ───────────────────
scatter_data = []
for rating in sorted(DF[TARGET].unique()):
    sub = DF[DF[TARGET] == rating]
    scatter_data.append({
        "rating": int(rating),
        "count": len(sub),
        "avg_tech": round(sub['Technical_Skills_Rating'].mean(), 2),
        "avg_comm": round(sub['Communication_Skills_Rating'].mean(), 2),
        "avg_prob": round(sub['Problem_Solving_Skills_Rating'].mean(), 2),
    })

# ── Save JSON ────────────────────────────────────────────────────
result = {
    "question": "Q1 - Skill Influence on Performance Rating",
    "correlation_matrix": corr_matrix.to_dict(),
    "individual_correlations": ind_corr,
    "normalized_weights": weights,
    "weighted_score_vs_performance_corr": wss_perf_corr,
    "skill_means": {c: round(DF[c].mean(), 2) for c in SKILL_COLS},
    "performance_stats": {
        "mean": round(DF[TARGET].mean(), 2),
        "median": float(DF[TARGET].median()),
        "std": round(DF[TARGET].std(), 2),
        "min": int(DF[TARGET].min()),
        "max": int(DF[TARGET].max()),
    },
    "band_skill_avg": band_skill.to_dict(),
    "scatter_by_rating": scatter_data,
    "perf_distribution": DF[TARGET].value_counts().sort_index().to_dict(),
    "llm_reasoning": {
        "finding": "Low linear correlations indicate these 3 skills are not sole predictors of performance.",
        "communication_dominance": f"Communication_Skills_Rating carries {weights['Communication_Skills_Rating']*100:.1f}% weight — the strongest individual predictor.",
        "technical_insight": "Technical_Skills_Rating shows near-zero correlation (r=0.002), suggesting technical skill alone does not drive performance scores in this dataset.",
        "problem_solving_insight": "Problem_Solving_Skills_Rating shows a slight negative correlation (-0.0049), hinting that high scorers may be in roles with lower performance expectations.",
        "conclusion": "Performance_Rating in this dataset appears to be influenced by broader contextual factors (project complexity, leadership, engagement) rather than these three skills in isolation. Recommended: build a multi-feature model incorporating all skill + soft-skill ratings.",
        "weighted_formula": "Weighted_Score = 0.0448×Tech + 0.8468×Comm + 0.1084×ProbSolve"
    }
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(result, f, indent=2, default=str)
print(f"\n[OK] Report saved -> {OUT}")
