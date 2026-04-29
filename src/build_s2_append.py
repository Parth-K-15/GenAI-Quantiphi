"""Append Q9 + Q10 + Stand-Out Statement to section2_dashboard.html"""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html_path = os.path.join(BASE, "ui", "section2_dashboard.html")

content = open(html_path, 'r', encoding='utf-8').read().rstrip()

appendage = """

<!-- ============================================================ Q9 ============================================================ -->
<div class="section">
  <div class="section-header">
    <div class="section-icon icon-teal">🔀</div>
    <div>
      <div class="section-title">Q9 — Training Program Comparison: Basic vs Advanced</div>
      <div class="section-subtitle">Impact on performance &amp; career growth · Redefining success via Training_Effectiveness</div>
    </div>
  </div>
  <div class="insight-box teal">
    <div class="insight-label">🏆 Headline Finding</div>
    <div class="insight-text">The difference between Basic and Advanced training programs is <strong>smaller than expected</strong>. Training Effectiveness scores are nearly identical (Basic: 0.1976 · Advanced: 0.1983). <strong>The program label matters far less than how the training is applied.</strong> A Basic program with high ROI outperforms an Advanced program with low ROI — every time.</div>
  </div>
  <div class="kpi-row">
    <div class="kpi-card"><div class="kv">1,384</div><div class="kt">Basic Program Employees</div></div>
    <div class="kpi-card blue"><div class="kv">1,298</div><div class="kt">Advanced Program Employees</div></div>
    <div class="kpi-card orange"><div class="kv">0.1976</div><div class="kt">Basic Avg Effectiveness</div></div>
    <div class="kpi-card violet"><div class="kv">0.1983</div><div class="kt">Advanced Avg Effectiveness</div></div>
    <div class="kpi-card green"><div class="kv">+3.02%</div><div class="kt">Project Success Lift: Advanced vs Basic</div></div>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h4>📊 Performance &amp; Effectiveness: Basic vs Advanced</h4>
      <div class="chart-container"><canvas id="q9CompChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>🎯 Career Growth Metrics by Program Type (Radar)</h4>
      <div class="chart-container"><canvas id="q9CareerChart"></canvas></div>
    </div>
  </div>
  <table class="data-table">
    <thead><tr><th>Program</th><th>Count</th><th>Avg Performance</th><th>Avg Effectiveness</th><th>Avg Promotions</th><th>Career Achieved %</th><th>Resignation %</th><th>Project Success %</th></tr></thead>
    <tbody>
      <tr><td><span class="badge-pill pill-teal">🟢 Basic</span></td><td>1,384</td><td>9.897</td><td>0.1976</td><td>0.809</td><td>31.21%</td><td>49.57%</td><td>30.42%</td></tr>
      <tr><td><span class="badge-pill pill-blue">🔵 Advanced</span></td><td>1,298</td><td>9.868</td><td>0.1983</td><td>0.799</td><td>32.82%</td><td>48.77%</td><td>34.44%</td></tr>
    </tbody>
  </table>
  <div class="insight-box orange">
    <div class="insight-label">💡 System Proposal — Impact-Based Training Design</div>
    <div class="insight-text">Move from <strong>Basic/Advanced labels</strong> to <strong>Effectiveness Tier classification</strong>. Track Training_Effectiveness monthly per program cohort. Match employees using <strong>Training Readiness Score (Q10)</strong> — not job title or seniority. The best training program is not the most advanced — it is the one that produces the highest measurable impact per hour invested.</div>
  </div>
  <div class="findings-grid">
    <div class="finding-card"><h4>📌 Project Success Signal</h4><p>Advanced program employees show 34.44% project success rate vs 30.42% for Basic — stronger applied project impact even with similar overall performance.</p></div>
    <div class="finding-card"><h4>📌 ROI Distribution is Uniform</h4><p>Cross-segmentation (Program × ROI Level) shows ~25% distribution across all ROI tiers for both programs — program type alone does not determine training ROI.</p></div>
    <div class="finding-card"><h4>📌 Career Alignment Matters More</h4><p>Career goals achievement shows marginal differences (Basic: 31.21% vs Advanced: 32.82%) — career outcomes driven by readiness and context, not program level.</p></div>
  </div>
</div>

<!-- ============================================================ Q10 ============================================================ -->
<div class="section">
  <div class="section-header">
    <div class="section-icon icon-violet">🎯</div>
    <div>
      <div class="section-title">Q10 — Advanced Training Readiness Score: Predicting Who Benefits Most</div>
      <div class="section-subtitle">Reasoning-based composite prediction · No ML required · Fully interpretable by HR</div>
    </div>
  </div>
  <div class="insight-box violet">
    <div class="insight-label">🏆 Headline Finding</div>
    <div class="insight-text">Advanced training should <strong>not</strong> be assigned based on performance alone. A high-performer with low Training_ROI and high Mentorship_Dependency will <strong>NOT</strong> benefit — they are not ready to absorb and apply it independently. This model identifies the <em>right</em> employees using five principled, explainable features.</div>
  </div>
  <div class="kpi-row">
    <div class="kpi-card green"><div class="kv">1,000</div><div class="kt">Highly Ready Employees</div></div>
    <div class="kpi-card blue"><div class="kv">1,000</div><div class="kt">Ready Employees</div></div>
    <div class="kpi-card orange"><div class="kv">1,000</div><div class="kt">Developing</div></div>
    <div class="kpi-card red"><div class="kv">1,001</div><div class="kt">Not Ready</div></div>
    <div class="kpi-card"><div class="kv">0.586</div><div class="kt">Avg Readiness Score (max 0.90)</div></div>
  </div>
  <div class="formula-box">
    <code><span class="comment"># 🚀 Advanced Training Readiness Score — Composite Prediction Without ML</span>
Advanced_Training_Readiness =
  0.30 x Training_ROI                     <span class="comment"># Proven training-to-performance efficiency</span>
+ 0.20 x (Avg_Skills_Score / 10)          <span class="comment"># Foundation competence for advanced content</span>
+ 0.20 x (Avg_Soft_Skills_Score / 10)     <span class="comment"># Collaboration in advanced environments</span>
+ 0.15 x (WLB_Rating / 10)               <span class="comment"># Capacity to apply learning without burnout</span>
+ 0.15 x (1 / (Mentorship_Dependency+1)) <span class="comment"># Autonomous learner — self-directed growth</span></code>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h4>📊 Performance Profile by Readiness Tier</h4>
      <div class="chart-container"><canvas id="q10TierChart"></canvas></div>
    </div>
    <div class="chart-box">
      <h4>🏢 Readiness Score &amp; Highly-Ready % by Department</h4>
      <div class="chart-container"><canvas id="q10DeptChart"></canvas></div>
    </div>
  </div>
  <table class="data-table">
    <thead><tr><th>Readiness Tier</th><th>Count</th><th>Avg Performance</th><th>Avg Training ROI</th><th>Avg WLB</th><th>Avg Skills</th><th>Mentor Dependency</th></tr></thead>
    <tbody>
      <tr><td><span class="badge-pill pill-green">🌟 Highly Ready</span></td><td>1,000</td><td><span class="badge-pill pill-green">11.665</span></td><td>0.2388</td><td><strong>10.424</strong></td><td>8.602</td><td><span class="badge-pill pill-green">0.614 ✅</span></td></tr>
      <tr><td><span class="badge-pill pill-teal">✅ Ready</span></td><td>1,000</td><td>10.457</td><td>0.2084</td><td>8.375</td><td>8.152</td><td>0.717</td></tr>
      <tr><td><span class="badge-pill pill-orange">🔄 Developing</span></td><td>1,000</td><td>9.428</td><td>0.1863</td><td>7.229</td><td>7.816</td><td>0.815</td></tr>
      <tr><td><span class="badge-pill pill-red">⛔ Not Ready</span></td><td>1,001</td><td><span class="badge-pill pill-red">8.109</span></td><td>0.1584</td><td>5.784</td><td>7.304</td><td><span class="badge-pill pill-red">1.045 ⬆</span></td></tr>
    </tbody>
  </table>
  <h4 style="color:#1a2340;font-weight:700;font-size:1.1em;margin:24px 0 16px;">🗺️ Tier-Based Action Map</h4>
  <div class="findings-grid">
    <div class="finding-card" style="border-color:#ccfbf1;"><h4>🌟 Highly Ready</h4><p>Enroll immediately in Advanced Training. High ROI, strong skills, autonomous learner, excellent WLB — all conditions for maximum return are met.</p></div>
    <div class="finding-card" style="border-color:#dbeafe;"><h4>✅ Ready</h4><p>Eligible for Advanced Training with light mentorship support. Monitor Training_ROI post-enrollment to confirm impact trajectory.</p></div>
    <div class="finding-card" style="border-color:#fed7aa;"><h4>🔄 Developing</h4><p>Invest in readiness-building first: improve WLB conditions, reduce Mentorship_Dependency, confirm engagement before advancing.</p></div>
    <div class="finding-card" style="border-color:#fee2e2;"><h4>⛔ Not Ready</h4><p>Do not advance yet. Focus on Q8 interventions: engagement diagnosis, job-fit review, structured post-training support.</p></div>
  </div>
  <table class="data-table" style="margin-top:20px;">
    <thead><tr><th>Rank</th><th>Employee ID</th><th>Department</th><th>Job Title</th><th>Performance</th><th>Training ROI</th><th>Readiness Score</th></tr></thead>
    <tbody>
      <tr><td>🥇 1</td><td>2024/Marketing/0155</td><td>Marketing</td><td>Developer</td><td>15</td><td>0.3947</td><td><span class="badge-pill pill-green">0.9018</span></td></tr>
      <tr><td>🥈 2</td><td>2024/Marketing/3673</td><td>Finance</td><td>Manager</td><td>13</td><td>0.260</td><td><span class="badge-pill pill-green">0.8420</span></td></tr>
      <tr><td>🥉 3</td><td>2024/Marketing/3138</td><td>IT</td><td>Consultant</td><td>24</td><td>0.5217</td><td><span class="badge-pill pill-green">0.8267</span></td></tr>
      <tr><td>4</td><td>2024/Finance/2770</td><td>Marketing</td><td>Consultant</td><td>14</td><td>0.3590</td><td><span class="badge-pill pill-teal">0.8202</span></td></tr>
      <tr><td>5</td><td>2024/IT/0096</td><td>Sales</td><td>Manager</td><td>18</td><td>0.3600</td><td><span class="badge-pill pill-teal">0.8056</span></td></tr>
    </tbody>
  </table>
  <div class="insight-box dark" style="margin-top:28px;">
    <div class="insight-label">🔗 Connecting Q6 → Q7 → Q8 → Q9 → Q10</div>
    <div class="insight-text">This model synthesizes all Section 2 findings: <strong>Training_ROI (Q6/Q8)</strong>, <strong>Mentorship_Dependency (Q7)</strong>, <strong>Training_Effectiveness (Q9)</strong>, and contextual conditions into a single actionable readiness predictor — the final product of the entire training analytics pipeline.</div>
  </div>

  <!-- 🚀 Stand-Out Statement -->
  <div style="margin-top:32px;background:linear-gradient(135deg,#1a2340 0%,#0d9488 60%,#38bdf8 100%);border-radius:16px;padding:36px 40px;text-align:center;box-shadow:0 8px 32px rgba(13,148,136,0.35);position:relative;overflow:hidden;">
    <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;background:rgba(255,255,255,0.05);border-radius:50%;"></div>
    <div style="position:absolute;bottom:-50px;left:20%;width:250px;height:250px;background:rgba(255,255,255,0.03);border-radius:50%;"></div>
    <div style="font-size:2em;margin-bottom:12px;">🚀</div>
    <div style="font-size:0.78em;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#7dd3fc;margin-bottom:14px;">Stand-Out Statement</div>
    <div style="font-size:1.22em;font-weight:600;color:white;line-height:1.75;position:relative;font-style:italic;">"This model transforms training allocation from a subjective decision into a data-driven, explainable system, enabling organizations to maximize ROI on learning investments."</div>
  </div>
</div>

<footer style="text-align:center;padding:40px;color:#9ca3af;font-size:0.88em;">
  <p>Quantiphi HR Analytics &middot; Section 2 Complete — Q6 through Q10 &middot; <a href="section3_dashboard.html" style="color:#0d9488;font-weight:600;">Continue to Section 3 &rarr;</a></p>
</footer>

<script>
/* Q6 */
new Chart(document.getElementById('q6PerfChart'),{type:'bar',data:{labels:['Q1 Low','Q2 Med-Low','Q3 Med-High','Q4 High'],datasets:[{label:'Avg Performance',data:[9.998,9.921,9.918,9.805],backgroundColor:['#0d9488','#14b8a6','#2dd4bf','#ef4444'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:9.7,grid:{color:'#f1f5f9'}}}}});
new Chart(document.getElementById('q6ImpactChart'),{type:'line',data:{labels:['Q1 Low','Q2 Med-Low','Q3 Med-High','Q4 High'],datasets:[{label:'Training Impact',data:[0.237,0.203,0.184,0.162],borderColor:'#f97316',backgroundColor:'rgba(249,115,22,0.1)',tension:0.4,fill:true,pointBackgroundColor:'#f97316',pointRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{grid:{color:'#f1f5f9'}}}}});
/* Q7 */
new Chart(document.getElementById('q7ConvChart'),{type:'bar',data:{labels:['Q1 Low','Q2 Med-Low','Q3 Med-High','Q4 High'],datasets:[{label:'Conversion Rate %',data:[49.0,49.8,49.4,51.7],backgroundColor:['#e5e7eb','#7dd3fc','#38bdf8','#f97316'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:47,grid:{color:'#f1f5f9'}}}}});
new Chart(document.getElementById('q7ExpChart'),{type:'bar',data:{labels:['Associate','Junior','Mid-Senior','Senior'],datasets:[{label:'Avg Performance',data:[10.007,9.962,9.800,9.891],backgroundColor:['#0d9488','#f97316','#ef4444','#7c3aed'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{y:{min:9.6,grid:{color:'#f1f5f9'}}}}});
new Chart(document.getElementById('q7DepChart'),{type:'doughnut',data:{labels:['High Dep + Low Perf','Rest'],datasets:[{data:[602,3399],backgroundColor:['#ef4444','#e5e7eb'],borderWidth:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}}}});
/* Q8 */
new Chart(document.getElementById('q8ProfileChart'),{type:'bar',data:{labels:['Low ROI','Medium ROI','High ROI','Very High ROI'],datasets:[{label:'Avg Performance',data:[6.40,8.96,10.80,13.55],backgroundColor:['#ef4444','#f97316','#14b8a6','#0d9488'],borderRadius:6},{label:'Avg Dev Hours',data:[53.6,51.3,49.6,46.0],backgroundColor:['rgba(239,68,68,0.3)','rgba(249,115,22,0.3)','rgba(20,184,166,0.3)','rgba(13,148,136,0.3)'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}},scales:{y:{min:5,grid:{color:'#f1f5f9'}}}}});
new Chart(document.getElementById('q8DistChart'),{type:'doughnut',data:{labels:['Low ROI (1013)','Medium (989)','High (1005)','Very High (994)'],datasets:[{data:[1013,989,1005,994],backgroundColor:['#ef4444','#f97316','#14b8a6','#0d9488'],borderWidth:0}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}}}});
/* Q9 */
new Chart(document.getElementById('q9CompChart'),{type:'bar',data:{labels:['Basic','Advanced'],datasets:[{label:'Avg Performance',data:[9.897,9.868],backgroundColor:['#0d9488','#2563eb'],borderRadius:6},{label:'Effectiveness x100',data:[19.76,19.83],backgroundColor:['rgba(13,148,136,0.3)','rgba(37,99,235,0.3)'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}},scales:{y:{min:9,grid:{color:'#f1f5f9'}}}}});
new Chart(document.getElementById('q9CareerChart'),{type:'radar',data:{labels:['Promotions x10','Career Achieved %','Project Success %','Leadership High %','Engagement'],datasets:[{label:'Basic',data:[8.09,31.21,30.42,31.21,79.77],borderColor:'#0d9488',backgroundColor:'rgba(13,148,136,0.15)',pointBackgroundColor:'#0d9488'},{label:'Advanced',data:[7.99,32.82,34.44,32.97,80.25],borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,0.1)',pointBackgroundColor:'#2563eb'}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}},scales:{r:{beginAtZero:true}}}});
/* Q10 */
new Chart(document.getElementById('q10TierChart'),{type:'bar',data:{labels:['Not Ready','Developing','Ready','Highly Ready'],datasets:[{label:'Avg Performance',data:[8.109,9.428,10.457,11.665],backgroundColor:['#ef4444','#f97316','#14b8a6','#0d9488'],borderRadius:8},{label:'Training ROI x100',data:[15.84,18.63,20.84,23.88],backgroundColor:['rgba(239,68,68,0.2)','rgba(249,115,22,0.2)','rgba(20,184,166,0.2)','rgba(13,148,136,0.2)'],borderRadius:8}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}},scales:{y:{min:7,grid:{color:'#f1f5f9'}}}}});
new Chart(document.getElementById('q10DeptChart'),{type:'bar',data:{labels:['Finance','IT','Sales','HR','Marketing'],datasets:[{label:'Avg Readiness x100',data:[59.02,58.74,58.54,58.42,58.17],backgroundColor:['#0d9488','#2563eb','#f97316','#7c3aed','#ef4444'],borderRadius:6},{label:'Highly Ready %',data:[27.07,25.29,24.40,24.71,23.52],backgroundColor:['rgba(13,148,136,0.25)','rgba(37,99,235,0.25)','rgba(249,115,22,0.25)','rgba(124,58,237,0.25)','rgba(239,68,68,0.25)'],borderRadius:6}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'bottom'}},scales:{y:{min:20,grid:{color:'#f1f5f9'}}}}});
</script>
</body>
</html>
"""

with open(html_path, 'w', encoding='utf-8') as f:
    f.write(content + appendage)
print(f"Done. Total lines: {len((content+appendage).splitlines())}")
