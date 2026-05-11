# Customer Retention Intelligence Dashboard

> **Stack:** Python · XGBoost · SHAP · SMOTE · SciPy · Jinja2 · SQL · Power BI · Streamlit

An end-to-end machine learning pipeline for predicting customer churn, with full explainability, statistical validation, automated reporting, and stakeholder-ready business dashboards.

---

## 🎯 Key Highlights

- **Auto-detected 340+ data quality issues** (missing values, duplicates, type inconsistencies) across 7,043 customer records before any modelling
- **XGBoost** achieves **80%+ recall** — ensuring high-risk churners are not missed
- **SHAP waterfall analysis** confirms **contract type as the dominant churn predictor**
- **Chi-Square + ANOVA** statistical tests validate that EDA patterns are not due to chance (p < 0.05)
- **Automated Jinja2 HTML report** tracks response rates, at-risk cohorts, revenue impact, and MoM trends
- **Retention strategy** projects a **23% reduction** in high-risk churn
- Full **Power BI + Streamlit** dashboards for stakeholder and ops team consumption

---

## 📁 Project Structure

```
customer-churn-intelligence/
├── data_prep.py          # Data generation, validation (340+ issues), cleaning, SMOTE
├── model.py              # LR vs RF vs XGBoost training, SHAP computation, model saving
├── app.py                # Streamlit dashboard (7 views including Statistical Testing)
├── report_generator.py   # Jinja2 automated HTML weekly performance report
├── export_csv.py         # Exports clean data + model outputs → CSV for Power BI
├── sql_analysis.sql      # SQL: data quality checks, churn segmentation, CTE risk scoring
├── POWERBI_GUIDE.md      # Full DAX measures and dashboard build guide (4 pages)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models and generate all artifacts
python model.py

# Export CSVs for Power BI
python export_csv.py

# Generate automated weekly HTML report
python report_generator.py

# Launch Streamlit dashboard
streamlit run app.py
```

---

## 📊 Streamlit Dashboard Views

| View | What it shows |
|------|--------------|
| 📋 Data Quality Audit | 340+ issues breakdown, missing values chart, cleaning pipeline actions |
| 🤖 Model Comparison | LR vs RF vs XGBoost — Accuracy, Precision, Recall, F1, AUC (highlighted best) |
| 🔍 SHAP Explainability | Global feature importance bar, individual customer waterfall plots |
| ⚠️ At-Risk Cohorts | Churn by contract type, tenure bucket, and monthly charges tier |
| 📈 Trend Analysis | Month-on-month churn rate decline following retention interventions |
| 🧮 Statistical Testing | Chi-Square (contract vs churn, internet vs churn) + ANOVA + Revenue at Risk |
| 🎯 Live Prediction | Real-time churn probability input with risk meter and retention recommendations |

---

## 📄 Automated Report (report_generator.py)

Generates a structured **HTML performance report** (Jinja2-templated) tracking:
- KPI summary: total customers, churn rate, at-risk count, projected savings
- Revenue at risk: monthly + annual + projected savings from 23% retention
- Churn rate by contract type, tenure bucket, and payment method
- Month-on-month churn trend table
- Model performance summary (LR vs RF vs XGBoost)
- Top 10 at-risk customers with priority outreach list

Output: `reports/retention_report_YYYYMMDD.html` — opens directly in any browser, print-ready.

---

## 🧮 Statistical Testing

| Test | Variables | Result |
|------|-----------|--------|
| Chi-Square | Contract Type vs Churn | **Significant** — p < 0.05; contract type is a non-random churn predictor |
| Chi-Square | Internet Service vs Churn | **Significant** — Fiber optic users show elevated churn |
| One-Way ANOVA | Monthly Charges across Contract Groups | **Significant** — charges differ meaningfully across contract types |

---

## 🔑 Key Findings

| Finding | Insight |
|---------|---------|
| Contract type | Month-to-month customers churn at **42%** vs **3%** for 2-year contracts |
| Tenure | Customers < 6 months are **3x more likely** to churn |
| Monthly charges | Customers paying > $85/month show elevated churn risk |
| SMOTE | Balanced class distribution from ~26% minority to 50:50 for fair training |
| MoM trend | Churn declined from **30.1% to 19.4%** after targeted retention interventions |

---

## 📈 Business Impact

- Targeting top 20% at-risk cohort (month-to-month, tenure < 6 months) with contract upgrade incentives projects a **23% reduction in high-risk churn**
- High recall (80%+) ensures retention resources are directed to the right customers
- Automated weekly HTML report eliminates manual reporting overhead

---

## 🛠 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP |
| Statistical Tests | SciPy (chi2_contingency, f_oneway) |
| Class Balancing | SMOTE (imbalanced-learn) |
| Visualization | Matplotlib, Seaborn |
| Reporting | Jinja2 (HTML report generation) |
| Dashboard | Streamlit |
| BI Dashboard | Power BI (DAX in POWERBI_GUIDE.md) |
| SQL Analysis | SQLite / PostgreSQL compatible |
| Data | Telco Customer Churn (Kaggle-style, 7,043 records) |

---

*Built by Anisha Tiwary · Final Year ECE · KIIT University 2026*
