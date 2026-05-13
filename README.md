# Customer Retention Intelligence Dashboard

Python · XGBoost · SHAP · SMOTE · SciPy · SQL · Power BI · Streamlit

Telecom companies lose significant revenue to churn every month, and most of it is preventable. This project builds a full churn prediction pipeline on 7,043 customer records — from raw data with quality issues all the way to a deployed dashboard with live prediction and a weekly automated report.

The focus throughout is on business utility, not just model accuracy. Every prediction comes with a SHAP explanation, every insight is framed as a retention action, and the output is packaged for a non-technical stakeholder.

---

## Power BI Dashboard

![Executive Summary](assets/page1.png)
![Deep Dive Analytics](assets/page2.png)
![Model & Explainability](assets/page3.png)
![Trend Tracker](assets/page4.png)

---

## What this project does

The pipeline starts with an upstream data quality check that auto-detects missing values, duplicates, and type inconsistencies before any modelling runs — 340+ issues flagged and resolved across the dataset. SMOTE is then applied to handle class imbalance before training three models (Logistic Regression, Random Forest, XGBoost) side by side.

XGBoost comes out on top with 82.1% recall — the priority metric here, since missing a churner costs more than a false positive retention offer. SHAP waterfall analysis then breaks down each prediction at the feature level, confirming contract type as the dominant driver (month-to-month customers churn at 42% vs 3% for two-year contracts).

Statistical tests (Chi-Square, ANOVA) back up what the EDA shows — none of the patterns are due to chance. The top 20% at-risk cohort, when targeted with contract upgrade incentives, projects a 23% reduction in high-risk churn.

The Streamlit app brings all of this together in one place. The automated report generates a weekly HTML summary with an at-risk customer priority list, revenue impact, and month-on-month trend tracking — no manual work needed.

---

## Project structure
---├── data_prep.py           — data generation, quality validation, cleaning, SMOTE
├── model.py               — LR vs RF vs XGBoost training, SHAP, model saving
├── app.py                 — Streamlit dashboard (7 tabs)
├── report_generator.py    — automated Jinja2 HTML weekly report
├── export_csv.py          — exports cleaned data and model outputs to CSV for Power BI
├── sql_analysis.sql       — data quality checks, churn segmentation, CTE risk scoring
├── powerbi_dashboard.html — interactive Power BI dashboard (open in browser)
├── requirements.txt
└── README.md

## Running it
pip install -r requirements.txt
python model.py              # train models, generate artifacts
python export_csv.py         # export CSVs for Power BI
python report_generator.py   # generate weekly HTML report
streamlit run app.py         # launch dashboard

---

## Key findings

Month-to-month customers churn at 42% — 14x the rate of two-year contract holders. Customers in their first six months are three times more likely to leave than long-tenured ones. Fiber optic internet users and electronic check payers both show elevated churn, compounding the risk when they overlap with short tenure.

Churn declined from 30.1% to 19.4% across the tracked period following targeted retention interventions on the highest-risk cohort — a 35.5% relative reduction.

---

## Tech stack

| | |
|---|---|
| ML | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Statistical tests | SciPy — chi2_contingency, f_oneway |
| Class balancing | imbalanced-learn (SMOTE) |
| Reporting | Jinja2 |
| Dashboard | Streamlit |
| BI | Power BI |
| SQL | SQLite / PostgreSQL compatible |
| Data | Telco Customer Churn — 7,043 records |
