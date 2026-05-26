# HR Analytics Command Center

End-to-end HR analytics project on `4,001` employee records, combining:
- statistical analysis (Sections 1-7, Q1-Q25)
- model-based reasoning
- Gemini-powered narrative insights
- static dashboard generation

The project outputs JSON reports, optional CSV extracts, and section dashboards under `ui/`.

## What This Project Covers

### Section 1: Performance and Skills (Q1-Q5)
- skill influence on performance
- high performer / low leadership signals
- behavioral comparisons across performance bands
- inconsistency and anomaly detection
- ideal employee profile

### Section 2: Training and Development (Q6-Q10)
- dev hours vs performance and promotions
- mentorship impact
- training ROI gaps
- basic vs advanced program comparison
- advanced training readiness scoring

### Section 3: Soft Skills and Leadership (Q11-Q14)
- soft-skill clustering
- conflict-teamwork contradictions
- engagement effects
- initiative-innovation gap analysis

### Section 4: Project Execution (Q15-Q18)
- project complexity and size impact
- successful vs failed project patterns
- success prediction model
- role-wise execution comparison

### Section 5: Attrition and Retention (Q19-Q21)
- multi-variable resignation drivers
- attrition risk profiling
- resigned vs retained comparisons

### Section 6: Compensation and Benefits (Q22-Q24)
- pay-performance relationships
- underpaid employee identification
- benefits impact on retention and satisfaction

### Section 7: Recruitment Impact (Q25)
- hiring source, time-to-hire, and cost impact on outcomes

## Repository Layout

```text
data/
  raw/employee_data.csv
  processed/employee_data_cleaned.csv
  processed/employee_data_final.csv
reports/
  preprocessing_report.json
  preprocessing_phase2_report.json
  section1..section7/*.json, *.csv
src/
  preprocessing/
  analysis/section1..section7/
  llm_section4_insights_generator.py
ui/
  index.html
  section1_dashboard.html ... section7_dashboard.html
```

## Prerequisites

- Python `3.10+` (recommended `3.11`)
- PowerShell (Windows) or any shell
- Gemini API key for LLM insight generation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Copy `.env.example` to `.env`.
2. Set your key:

```env
GEMINI_API_KEY=your_key_here
```

Optional settings:
- `GEMINI_MODEL_NAME` (comma-separated model candidates)
- `GEMINI_MIN_REQUEST_GAP_SEC` (default: `13`)
- `GEMINI_OVERLOAD_RETRY_LIMIT` (default: `2` or `3` depending on script defaults)

## Quick Start (End-to-End)

Run from repo root:

```powershell
python src/preprocessing/preprocess.py
python src/preprocessing/preprocess_phase2.py

python src/analysis/section1/run_q1_q5_pipeline.py --with-llm --llm-quiet
python src/analysis/section2/run_q6_q10_pipeline.py --with-llm --llm-quiet
python src/analysis/section3/run_q11_q14_pipeline.py --with-llm --llm-quiet
python src/analysis/section4/run_q15_q16_pipeline.py --with-llm --llm-quiet
python src/analysis/section5/run_q19_q20_pipeline.py --with-llm --llm-quiet
python src/analysis/section6/run_q22_q23_pipeline.py --with-llm --llm-quiet
python src/analysis/section7/run_q25_pipeline.py --with-llm --llm-quiet

python src/analysis/compile_dashboards.py
```

Then serve UI locally:

```powershell
python -m http.server 8000
```

Open:
- `http://localhost:8000/ui/index.html`

## Section Pipelines

Each section has a runner script that executes all question-level analyses and can optionally enrich results with Gemini.

| Section | Runner |
|---|---|
| 1 (Q1-Q5) | `src/analysis/section1/run_q1_q5_pipeline.py` |
| 2 (Q6-Q10) | `src/analysis/section2/run_q6_q10_pipeline.py` |
| 3 (Q11-Q14) | `src/analysis/section3/run_q11_q14_pipeline.py` |
| 4 (Q15-Q18) | `src/analysis/section4/run_q15_q16_pipeline.py` |
| 5 (Q19-Q21) | `src/analysis/section5/run_q19_q20_pipeline.py` |
| 6 (Q22-Q24) | `src/analysis/section6/run_q22_q23_pipeline.py` |
| 7 (Q25) | `src/analysis/section7/run_q25_pipeline.py` |

Common flags:
- `--with-llm` run Gemini insight generation
- `--llm-quiet` reduce logs
- `--llm-delay <seconds>` inter-file delay
- `--llm-model <model_name>` preferred model
- `--llm-strict-model` disable fallback candidates

## LLM Insight Generation

Primary generator:
- `src/llm_section4_insights_generator.py`

Current behavior:
- used by all section pipeline runners when `--with-llm` is enabled
- writes/overwrites `llm_insights` in report JSON files
- adds `llm_generation_meta` with model and timestamp
- includes retries, throttling, and model fallback handling

Legacy script:
- `src/llm_insights_generator.py` (older Section-3-focused flow)
- prefer the section pipeline runners or `llm_section4_insights_generator.py`

## Dashboard Build Notes

- `src/analysis/compile_dashboards.py` currently compiles dynamic content for `section1`, `section2`, and `section3` dashboards from JSON reports.
- `section4` to `section7` dashboards exist in `ui/`, but are not currently rebuilt by `compile_dashboards.py`.
- `src/analysis/upgrade_ui.py` applies UI restyling for sections 1-3 and then calls `compile_dashboards.py`.

## Output Artifacts

Generated artifacts include:
- preprocessed datasets in `data/processed/`
- statistical reports in `reports/section*/q*.json`
- targeted employee lists in selected CSVs (for example top candidates / underpaid / at-risk subsets)
- dashboard HTML in `ui/`

## Troubleshooting

### `GEMINI_API_KEY is not set`
- Ensure `.env` exists at repo root and contains a valid key.

### `ModuleNotFoundError`
- Reinstall dependencies:
  - `pip install -r requirements.txt`

### Gemini `429` or `503` errors
- Increase delay:
  - add `--llm-delay 15` or `--llm-delay 20`
- Use a lighter model:
  - `--llm-model gemini-2.5-flash-lite`

### Dashboard not reflecting updated reports
- Re-run:
  - `python src/analysis/compile_dashboards.py`
- Refresh browser cache.

### Missing report warnings during compile
- Run the corresponding section pipeline first to generate required JSON files.

## Reproducibility Notes

- Use `data/raw/employee_data.csv` as the canonical input.
- Run preprocessing phase 1 and phase 2 before section analyses for consistent derived features.
- Keep `.env` private (`.gitignore` already excludes it).

