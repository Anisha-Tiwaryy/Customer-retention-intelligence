"""
export_csv.py
-------------
Exports the cleaned customer dataset and model outputs to CSV files
so Power BI can import them directly (as referenced in POWERBI_GUIDE.md).

Generates:
  - customer_data.csv       ← main dataset for Power BI
  - model_comparison.csv    ← LR vs RF vs XGBoost metrics
  - shap_feature_importance.csv ← SHAP global importance

Usage:
    python export_csv.py
"""

import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

from data_prep import generate_telco_data, validate_data, clean_data, feature_engineer, apply_smote, split_data
from model     import get_models, train_and_evaluate, compute_shap

OUTPUT_DIR = "powerbi_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_all():
    print("=" * 55)
    print("  CUSTOMER CHURN — CSV EXPORT FOR POWER BI")
    print("=" * 55)

    # ── Step 1: Generate and clean data ──
    raw    = generate_telco_data(n=7043)
    report = validate_data(raw)
    clean  = clean_data(raw)

    # Export clean customer data
    clean_export = clean.copy()
    clean_export["Churn_label"] = clean_export["Churn"].map({1: "Yes", 0: "No"})
    customer_path = os.path.join(OUTPUT_DIR, "customer_data.csv")
    clean_export.to_csv(customer_path, index=False)
    print(f"\n[EXPORTED] customer_data.csv → {customer_path}")
    print(f"  Rows: {len(clean_export)} | Columns: {len(clean_export.columns)}")

    # ── Step 2: Train models and export comparison ──
    X, y, scaler, features = feature_engineer(clean)
    X_res, y_res           = apply_smote(X, y)
    X_train, X_test, y_train, y_test = split_data(X_res, y_res)

    models  = get_models()
    trained, results_df = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    results_df["Model"] = results_df.index
    results_df = results_df[["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
    comp_path  = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    results_df.to_csv(comp_path, index=False)
    print(f"[EXPORTED] model_comparison.csv → {comp_path}")

    # ── Step 3: SHAP feature importance ──
    best_model = trained["XGBoost"]
    _, shap_values, shap_df = compute_shap(best_model, X_test, features, "XGBoost")
    shap_path  = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    shap_df.to_csv(shap_path, index=False)
    print(f"[EXPORTED] shap_feature_importance.csv → {shap_path}")

    # ── Step 4: At-risk customers export ──
    raw_scored = raw.copy()
    raw_scored["MonthlyCharges"] = pd.to_numeric(raw_scored["MonthlyCharges"], errors="coerce")
    raw_scored["tenure"]         = pd.to_numeric(raw_scored["tenure"], errors="coerce")
    raw_scored["risk_score"] = (
        raw_scored["Contract"].map({"Month-to-month": 50, "One year": 20, "Two year": 5}).fillna(0)
        + (raw_scored["tenure"] < 6).astype(int) * 30
        + (raw_scored["MonthlyCharges"] > 85).astype(int) * 10
        + (raw_scored["InternetService"] == "Fiber optic").astype(int) * 10
    )
    atrisk_path = os.path.join(OUTPUT_DIR, "at_risk_customers.csv")
    (raw_scored[raw_scored["risk_score"] >= 50]
     .sort_values("risk_score", ascending=False)
     .to_csv(atrisk_path, index=False))
    print(f"[EXPORTED] at_risk_customers.csv → {atrisk_path}")

    print(f"\n[ALL DONE] Files saved in /{OUTPUT_DIR}/")
    print("  → Open Power BI Desktop → Get Data → CSV → import these files")
    print("  → Follow POWERBI_GUIDE.md for DAX measures and dashboard layout")

    return {
        "customer_data":          customer_path,
        "model_comparison":       comp_path,
        "shap_feature_importance": shap_path,
        "at_risk_customers":      atrisk_path,
    }


if __name__ == "__main__":
    export_all()
