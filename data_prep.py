"""
data_prep.py
------------
Data loading, validation, and preprocessing for Customer Churn Intelligence Dashboard.
Covers:
  - Auto-detection of missing values, duplicates, type inconsistencies (340+ issues)
  - SMOTE for class imbalance
  - Feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


# ─── 1. SYNTHETIC DATA GENERATION (mirrors Telco Kaggle dataset structure) ───

def generate_telco_data(n=7043, seed=42):
    """Generate synthetic Telco-style churn dataset (7,043 records)."""
    np.random.seed(seed)
    n = n

    genders        = np.random.choice(["Male", "Female"], n)
    senior         = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner        = np.random.choice(["Yes", "No"], n)
    dependents     = np.random.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure         = np.random.randint(0, 72, n)
    phone_service  = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.random.choice(["Yes", "No", "No phone service"], n)
    internet       = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    online_sec     = np.random.choice(["Yes", "No", "No internet service"], n)
    online_backup  = np.random.choice(["Yes", "No", "No internet service"], n)
    device_prot    = np.random.choice(["Yes", "No", "No internet service"], n)
    tech_support   = np.random.choice(["Yes", "No", "No internet service"], n)
    streaming_tv   = np.random.choice(["Yes", "No", "No internet service"], n)
    streaming_mv   = np.random.choice(["Yes", "No", "No internet service"], n)
    contract       = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless      = np.random.choice(["Yes", "No"], n)
    payment        = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], n
    )
    monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges   = np.clip(total_charges, 0, None)

    # Churn influenced by contract type and tenure
    churn_prob = np.where(contract == "Month-to-month", 0.42,
                 np.where(contract == "One year", 0.11, 0.03))
    churn_prob -= tenure * 0.003
    churn_prob  = np.clip(churn_prob, 0.02, 0.85)
    churn       = np.array(["Yes" if np.random.rand() < p else "No" for p in churn_prob])

    df = pd.DataFrame({
        "customerID":        [f"CUST-{i:05d}" for i in range(n)],
        "gender":            genders,
        "SeniorCitizen":     senior,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone_service,
        "MultipleLines":     multiple_lines,
        "InternetService":   internet,
        "OnlineSecurity":    online_sec,
        "OnlineBackup":      online_backup,
        "DeviceProtection":  device_prot,
        "TechSupport":       tech_support,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_mv,
        "Contract":          contract,
        "PaperlessBilling":  paperless,
        "PaymentMethod":     payment,
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
        "Churn":             churn,
    })

    # Inject realistic data quality issues (340+ issues total)
    # Missing values (~180)
    for col in ["TotalCharges", "tenure", "MonthlyCharges"]:
        idx = np.random.choice(df.index, 60, replace=False)
        df.loc[idx, col] = np.nan
    # Duplicates (~80)
    dup_rows = df.sample(80, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)
    # Type inconsistencies: inject strings into numeric cols (~80)
    for col in ["SeniorCitizen"]:
        idx = np.random.choice(df.index, 80, replace=False)
        df.loc[idx, col] = "unknown"

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─── 2. DATA QUALITY VALIDATION ───

def validate_data(df: pd.DataFrame) -> dict:
    """Auto-detect and report data quality issues."""
    report = {}

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    report["missing_total"] = int(missing.sum())

    # Duplicates
    report["duplicate_rows"] = int(df.duplicated().sum())

    # Type inconsistencies (non-numeric values in numeric columns)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    type_issues  = {}
    for col in numeric_cols:
        if col in df.columns:
            non_numeric = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            if non_numeric.sum() > 0:
                type_issues[col] = int(non_numeric.sum())
    report["type_inconsistencies"] = type_issues
    report["type_issues_total"]    = sum(type_issues.values())

    report["total_issues"] = report["missing_total"] + report["duplicate_rows"] + report["type_issues_total"]

    print(f"[DATA VALIDATION REPORT]")
    print(f"  Missing values  : {report['missing_total']}")
    print(f"  Duplicate rows  : {report['duplicate_rows']}")
    print(f"  Type issues     : {report['type_issues_total']}")
    print(f"  ─────────────────────────────")
    print(f"  Total issues    : {report['total_issues']}")
    return report


# ─── 3. DATA CLEANING ───

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the dataset: fix types, drop duplicates, impute missing values."""
    df = df.copy()

    # Fix type inconsistencies: coerce numeric columns
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Impute missing numeric values
    for col in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    print(f"[CLEANED] Shape: {df.shape} | Churn rate: {df['Churn'].mean()*100:.1f}%")
    return df


# ─── 4. FEATURE ENGINEERING & ENCODING ───

def feature_engineer(df: pd.DataFrame):
    """Encode categoricals, scale numerics, split features/target."""
    df = df.copy()
    df.drop(columns=["customerID"], errors="ignore", inplace=True)

    le  = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, scaler, X.columns.tolist()


# ─── 5. SMOTE BALANCING ───

def apply_smote(X, y, seed=42):
    """Apply SMOTE to handle class imbalance."""
    print(f"[BEFORE SMOTE] Class distribution: {dict(y.value_counts())}")
    sm     = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"[AFTER SMOTE]  Class distribution: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res


# ─── 6. TRAIN/TEST SPLIT ───

def split_data(X, y, test_size=0.2, seed=42):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# ─── PIPELINE ENTRYPOINT ───

def run_pipeline():
    print("=" * 55)
    print("  CUSTOMER CHURN — DATA PREPARATION PIPELINE")
    print("=" * 55)

    raw_df  = generate_telco_data()
    report  = validate_data(raw_df)
    clean   = clean_data(raw_df)
    X, y, scaler, features = feature_engineer(clean)
    X_res, y_res = apply_smote(X, y)
    X_train, X_test, y_train, y_test = split_data(X_res, y_res)

    print(f"\n[PIPELINE COMPLETE]")
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, features, scaler, X, y


if __name__ == "__main__":
    run_pipeline()
