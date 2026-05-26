# ⚙️ Quantiphi HR Intelligence Command Center

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Vercel-blue?style=for-the-badge&logo=vercel&logoColor=white)](https://hr-analytics-dashboard-three.vercel.app/index.html)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](#)
[![Gemini API](https://img.shields.io/badge/Gemini_API-2.5_Flash-8E75C2?style=flat-square&logo=google-gemini&logoColor=white)](#)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384?style=flat-square&logo=chartdotjs&logoColor=white)](#)
[![Licence](https://img.shields.io/badge/License-MIT-green?style=flat-square)](#)

An end-to-end HR analytics pipeline executing statistical reasoning, predictive modeling, and Gemini-powered LLM insights across `4,001` employee records. The system generates high-fidelity dashboards powered by a robust Python compilation build engine.

> [!IMPORTANT]  
> **Explore the Live Executive Command Center:**  
> 👉 **[https://hr-analytics-dashboard-three.vercel.app/index.html](https://hr-analytics-dashboard-three.vercel.app/index.html)**

---

## 🗺️ Project Scope & Phase Architecture

### 🧠 Section 1: Performance and Skills (Q1-Q5)
*   **Skill Influence**: Computes skill-to-performance correlation (revealing the *"illusion of competence"* saturation).
*   **Silent Contributors**: Filters high performance coupled with low leadership score and high flight risk.
*   **Band Profiling**: Multi-factor statistical comparisons across high, medium, and low performance cohorts.
*   **Anomaly Detection**: Flags high-skill individuals in failed projects (burnout and overtime mediators).
*   **Ideal Employee Profile**: Synthesizes optimal composite threshold variables into a reusable talent metric.

### 📚 Section 2: Training & L&D ROI (Q6-Q10)
*   **Training Gaps**: Dev hours compared to performance ratings and promotions (hours vs. effectiveness).
*   **Mentorship Impact**: Maps mentor rating/dependency (revealing mentorship as a compensatory buffer).
*   **ROI Analysis**: Segments low-ROI vs. high-ROI training groups.
*   **Program Comparison**: Rigorous $p$-value comparisons of Basic vs. Advanced training programs.
*   **Advanced Training Readiness**: Trains a multi-factor logic classifier to select target cohorts.

### 🗣️ Section 3: Soft Skills & Behavioral Clustering (Q11-Q14)
*   **Behavioral Clustering**: Uses **K-Means Clustering** to partition employees into 4 distinct behavioral archetypes.
*   **The Collaboration Paradox**: Resolves the conflict-teamwork variance.
*   **Engagement Signals**: Evaluates engagement levels against direct retention indicators.
*   **Initiative-Innovation Gap**: Spots high-initiative employees lacking active innovation opportunities.

### 🚀 Section 4: Project Execution Analytics (Q15-Q18)
*   **Complexity & Scope**: Evaluates whether project size or complexity independently predict failure.
*   **Department Outcomes**: Tracks success patterns (Finance success vs. Sales execution gaps).
*   **Success Predictor**: Random Forest model mapping environmental vs. skill factors.
*   **Role Profiling**: ANOVA calculations across Managers, Developers, and Analysts.

### 🚪 Section 5: Attrition & Retention (Q19-Q21)
*   **Resignation Drivers**: Logistic regression mapping dominant attrition drivers (WLB, overtime, delay).
*   **Flight Risk Profiling**: Pinpoints the organization's **top 201 high-value flight risk employees**.
*   **Resigned vs. Retained**: Conducts independent t-tests on critical retention features.

### 💵 Section 6: Compensation & Benefits (Q22-Q24)
*   **Pay-Performance Disconnect**: Uncovers correlation ($r \approx 0.0$) between compensation and performance.
*   **Underpaid Contributors**: Identifies high-performing employees positioned in low salary bands.
*   **Benefits Correlation**: Evaluates stock options, health, and retirement options on employee retention.

### 🔍 Section 7: Recruitment Impact (Q25)
*   **Talent Sourcing**: Multi-variable audit of time-to-hire, hiring costs, and performance by recruitment channel.

---

## 📂 Repository Structure

```text
├── data/
│   ├── raw/employee_data.csv                 # Canonical raw dataset (4,001 rows)
│   └── processed/
│       ├── employee_data_cleaned.csv         # Cleaned baseline dataset
│       └── employee_data_final.csv           # Final processed dataset with derived features
├── reports/
│   ├── preprocessing_report.json             # Cleaning pipeline validation
│   ├── preprocessing_phase2_report.json      # Feature engineering validation
│   └── section1..7/                          # Question-level JSON reports and target CSV extracts
├── src/
│   ├── preprocessing/                        # Cleaning and engineering scripts
│   ├── analysis/
│   │   └── section1..7/                      # Modular analysis pipelines
│   └── llm_section4_insights_generator.py    # Gemini inference orchestrator
└── ui/
    ├── index.html                            # Dynamic Command Center (Interactive Stepper)
    └── section1_dashboard.html..section7_dashboard.html  # Upgraded high-fidelity dashboards
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Setup
*   Requires Python `3.10+` (Recommended `3.11`)
*   Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Environment Configuration
Create a `.env` file in the repository root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-2.5-flash,gemini-2.5-flash-lite
GEMINI_MIN_REQUEST_GAP_SEC=13
GEMINI_OVERLOAD_RETRY_LIMIT=3
```

### 3. Run End-to-End Pipeline
Run the data cleaning, feature engineering, and statistical calculation routines:
```powershell
# 1. Preprocess & Feature Engineer
python src/preprocessing/preprocess.py
python src/preprocessing/preprocess_phase2.py

# 2. Execute Analytics Pipelines (with Gemini active)
python src/analysis/section1/run_q1_q5_pipeline.py --with-llm --llm-quiet
python src/analysis/section2/run_q6_q10_pipeline.py --with-llm --llm-quiet
python src/analysis/section3/run_q11_q14_pipeline.py --with-llm --llm-quiet
python src/analysis/section4/run_q15_q16_pipeline.py --with-llm --llm-quiet
python src/analysis/section5/run_q19_q20_pipeline.py --with-llm --llm-quiet
python src/analysis/section6/run_q22_q23_pipeline.py --with-llm --llm-quiet
python src/analysis/section7/run_q25_pipeline.py --with-llm --llm-quiet

# 3. Compile and Style Dashboards
python src/analysis/upgrade_ui.py
```

### 4. Serve Dashboards Locally
Start a lightweight Python HTTP server to view dashboards offline:
```powershell
python -m http.server 8000
```
Then navigate to: **`http://localhost:8000/ui/index.html`**

---

## ⚙️ Modular Pipelines Reference

Each section is powered by a dedicated pipeline runner, allowing target execution:

| Section | Target Domain | Command Pipeline |
| :--- | :--- | :--- |
| **Section 1** | Performance & Skills | `python src/analysis/section1/run_q1_q5_pipeline.py` |
| **Section 2** | Training & Mentorship | `python src/analysis/section2/run_q6_q10_pipeline.py` |
| **Section 3** | Soft Skills & Clustering | `python src/analysis/section3/run_q11_q14_pipeline.py` |
| **Section 4** | Project Execution | `python src/analysis/section4/run_q15_q16_pipeline.py` |
| **Section 5** | Attrition & Retention | `python src/analysis/section5/run_q19_q20_pipeline.py` |
| **Section 6** | Compensation & Benefits | `python src/analysis/section6/run_q22_q23_pipeline.py` |
| **Section 7** | Recruitment ROI | `python src/analysis/section7/run_q25_pipeline.py` |

---

## 💡 Gemini Narrative Integration Engine

*   **Inference Orchestrator**: Governed by `src/llm_section4_insights_generator.py`.
*   **Behavioral Principles**: Injects structured schemas directly into local report stores, verifying data grounding to avoid hallucination.
*   **Throttling & Fallbacks**: Automatically handles API quota limits (`429` / `503`) through exponential backoff throttling and automatic model fallbacks (`gemini-2.5-flash` $\rightarrow$ `gemini-2.5-flash-lite`).

---

## 🛠️ Troubleshooting & Local Build Tips

*   **Missing Report Warnings**: Ensure you run the individual section pipeline script first to generate the raw statistical JSON files before running `upgrade_ui.py`.
*   **CORS Local Browser Block**: Modern browsers block `fetch()` operations when HTML files are opened directly via `file://`. **Our compiler solves this** by injecting JSON payloads directly into comments (`<!-- Anchor_START -->`), allowing 100% offline usage without a local server.
*   **Vercel / Hosting Deployment**: To deploy to hosting, simply push the repository to GitHub and link Vercel to serve the `ui/` directory.

---

## 🔬 Reproducibility Guarantee

*   **Canon Data**: Always use `data/raw/employee_data.csv` as the baseline.
*   **Execution Order**: Run preprocessing files before initiating analytical modules to ensure correct calculations for derived features.
