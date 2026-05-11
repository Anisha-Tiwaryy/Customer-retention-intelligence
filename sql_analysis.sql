-- ============================================================
-- sql_analysis.sql
-- Customer Churn — SQL Analysis & Data Quality Validation
-- Stack: SQLite / PostgreSQL compatible
-- ============================================================


-- ─── SCHEMA ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS customers (
    customerID       TEXT PRIMARY KEY,
    gender           TEXT,
    SeniorCitizen    INTEGER,
    Partner          TEXT,
    Dependents       TEXT,
    tenure           INTEGER,
    PhoneService     TEXT,
    MultipleLines    TEXT,
    InternetService  TEXT,
    OnlineSecurity   TEXT,
    OnlineBackup     TEXT,
    DeviceProtection TEXT,
    TechSupport      TEXT,
    StreamingTV      TEXT,
    StreamingMovies  TEXT,
    Contract         TEXT,
    PaperlessBilling TEXT,
    PaymentMethod    TEXT,
    MonthlyCharges   REAL,
    TotalCharges     REAL,
    Churn            TEXT
);


-- ─── DATA QUALITY CHECKS ──────────────────────────────────────

-- 1. Count total records
SELECT COUNT(*) AS total_records FROM customers;

-- 2. Count missing/null values per key column
SELECT
    SUM(CASE WHEN tenure         IS NULL THEN 1 ELSE 0 END) AS missing_tenure,
    SUM(CASE WHEN MonthlyCharges IS NULL THEN 1 ELSE 0 END) AS missing_monthly,
    SUM(CASE WHEN TotalCharges   IS NULL THEN 1 ELSE 0 END) AS missing_total_charges,
    SUM(CASE WHEN Contract       IS NULL THEN 1 ELSE 0 END) AS missing_contract,
    SUM(CASE WHEN Churn          IS NULL THEN 1 ELSE 0 END) AS missing_churn
FROM customers;

-- 3. Detect duplicate customerIDs
SELECT customerID, COUNT(*) AS occurrences
FROM customers
GROUP BY customerID
HAVING COUNT(*) > 1
ORDER BY occurrences DESC;

-- 4. Detect impossible/out-of-range values
SELECT
    SUM(CASE WHEN tenure < 0 OR tenure > 120      THEN 1 ELSE 0 END) AS invalid_tenure,
    SUM(CASE WHEN MonthlyCharges < 0              THEN 1 ELSE 0 END) AS negative_charges,
    SUM(CASE WHEN Churn NOT IN ('Yes','No')       THEN 1 ELSE 0 END) AS invalid_churn
FROM customers;


-- ─── CHURN RATE ANALYSIS ──────────────────────────────────────

-- 5. Overall churn rate
SELECT
    COUNT(*)                                                   AS total_customers,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)           AS churned,
    ROUND(100.0 * SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers;

-- 6. Churn rate by Contract type (dominant predictor)
SELECT
    Contract,
    COUNT(*)                                                       AS total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END)               AS churned,
    ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY Contract
ORDER BY churn_rate_pct DESC;

-- 7. Churn by tenure bucket
SELECT
    CASE
        WHEN tenure BETWEEN 0  AND 6  THEN '0-6 months'
        WHEN tenure BETWEEN 7  AND 12 THEN '7-12 months'
        WHEN tenure BETWEEN 13 AND 24 THEN '1-2 years'
        WHEN tenure BETWEEN 25 AND 48 THEN '2-4 years'
        ELSE '4+ years'
    END AS tenure_bucket,
    COUNT(*)                                                           AS total,
    ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
WHERE tenure IS NOT NULL
GROUP BY tenure_bucket
ORDER BY churn_rate_pct DESC;

-- 8. Churn by Internet Service
SELECT
    InternetService,
    COUNT(*)                                                           AS total,
    ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY InternetService
ORDER BY churn_rate_pct DESC;

-- 9. Churn by Payment Method
SELECT
    PaymentMethod,
    COUNT(*)  AS total,
    ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM customers
GROUP BY PaymentMethod
ORDER BY churn_rate_pct DESC;

-- 10. Average monthly charges — churned vs retained
SELECT
    Churn,
    ROUND(AVG(MonthlyCharges), 2)  AS avg_monthly_charges,
    ROUND(AVG(tenure), 1)          AS avg_tenure_months
FROM customers
WHERE MonthlyCharges IS NOT NULL AND tenure IS NOT NULL
GROUP BY Churn;


-- ─── CTE: AT-RISK CUSTOMER IDENTIFICATION ─────────────────────

-- 11. Flag top at-risk customers using CTE
WITH risk_scores AS (
    SELECT
        customerID,
        Contract,
        tenure,
        MonthlyCharges,
        InternetService,
        Churn,
        -- Simple heuristic risk score (0-100)
        CASE Contract
            WHEN 'Month-to-month' THEN 50
            WHEN 'One year'       THEN 20
            ELSE                       5
        END
        + CASE WHEN tenure < 6           THEN 30 ELSE 0 END
        + CASE WHEN MonthlyCharges > 85  THEN 10 ELSE 0 END
        + CASE WHEN InternetService = 'Fiber optic' THEN 10 ELSE 0 END
        AS risk_score
    FROM customers
    WHERE MonthlyCharges IS NOT NULL AND tenure IS NOT NULL
),
risk_ranked AS (
    SELECT *,
        RANK() OVER (ORDER BY risk_score DESC) AS risk_rank
    FROM risk_scores
)
SELECT
    customerID, Contract, tenure, MonthlyCharges,
    InternetService, risk_score, risk_rank, Churn
FROM risk_ranked
WHERE risk_score >= 70
ORDER BY risk_score DESC
LIMIT 50;

-- 12. Month-on-month simulation: churn by cohort entry month (proxy via tenure)
WITH cohort_data AS (
    SELECT
        CASE
            WHEN tenure = 0 THEN 'Month 1'
            WHEN tenure = 1 THEN 'Month 2'
            WHEN tenure = 2 THEN 'Month 3'
            ELSE 'Month 4+'
        END AS cohort_month,
        Churn
    FROM customers
)
SELECT
    cohort_month,
    COUNT(*) AS total,
    ROUND(100.0 * SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) / COUNT(*), 2) AS churn_rate_pct
FROM cohort_data
GROUP BY cohort_month
ORDER BY churn_rate_pct DESC;


-- ─── RETENTION PROJECTION ─────────────────────────────────────

-- 13. Estimate 23% churn reduction by targeting Month-to-month customers
--     with contract upgrade incentives
WITH baseline AS (
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END) AS churned
    FROM customers
),
post_intervention AS (
    SELECT
        COUNT(*) AS total,
        -- Assume 23% of high-risk month-to-month churners are retained
        SUM(CASE WHEN Churn='Yes' THEN 1 ELSE 0 END)
        - ROUND(0.23 * SUM(
            CASE WHEN Churn='Yes' AND Contract='Month-to-month' THEN 1 ELSE 0 END
          )) AS churned_post
    FROM customers
)
SELECT
    b.churned                                              AS churned_before,
    p.churned_post                                         AS churned_after,
    ROUND(100.0 * b.churned / b.total, 2)                 AS churn_rate_before_pct,
    ROUND(100.0 * p.churned_post / p.total, 2)            AS churn_rate_after_pct,
    b.churned - p.churned_post                             AS customers_retained
FROM baseline b, post_intervention p;
