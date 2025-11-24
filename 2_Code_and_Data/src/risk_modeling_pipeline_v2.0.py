import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import os

# =========================================================
# 0. Setup and Key Assumptions
# =========================================================

# Base project directory (script location)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input file
FILE_COMBINED_DATA = os.path.join(BASE_DIR, "..", "outputs", "combined_features_v3.csv")

# Output folders
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# Ensure output folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Output files
OUTPUT_SCENARIO_CSV = os.path.join(OUTPUT_DIR, "scenario_analysis_results_v3.csv")
OUTPUT_SEGMENT_CSV = os.path.join(OUTPUT_DIR, "segmentation_summary_v3.csv")
OUTPUT_CHART_SEGMENT_CSV = os.path.join(OUTPUT_DIR, "chart_segment_performance_v3.csv")
OUTPUT_CHART_COEFFICIENT_CSV = os.path.join(OUTPUT_DIR, "chart_feature_importance_v3.csv")

# Model save paths
RISK_MODEL_FILE = os.path.join(MODEL_DIR, "risk_model_v2.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler_v2.pkl")

# Financial assumptions
ACTIVE_BORROWERS = 20000
AVG_INTEREST_INCOME_MONTHLY = 8.0
EXPECTED_DEFAULT_RATE = 0.06
AVG_LOAN_PRINCIPAL = 200.0
LGD = 0.80
OPEX_PER_BORROWER = 2.0
INITIAL_INVESTMENT = 4_500_000.0
TERM_MONTHS = 36
ANNUAL_DISCOUNT_RATE = 0.10

# =========================================================
# 1. Financial Viability Function
# =========================================================

def calculate_viability(default_rate, monthly_interest_income, active_borrowers):
    monthly_revenue = active_borrowers * monthly_interest_income
    monthly_loss = active_borrowers * default_rate * AVG_LOAN_PRINCIPAL * LGD
    monthly_opex = active_borrowers * OPEX_PER_BORROWER
    monthly_profit = monthly_revenue - monthly_loss - monthly_opex

    monthly_discount_rate = (1 + ANNUAL_DISCOUNT_RATE) ** (1/12) - 1
    cashflows = np.array([monthly_profit] * TERM_MONTHS)
    npv = -INITIAL_INVESTMENT + np.sum(
        cashflows / (1 + monthly_discount_rate) ** np.arange(1, TERM_MONTHS + 1)
    )

    breakeven_rate = (monthly_interest_income - OPEX_PER_BORROWER) / (AVG_LOAN_PRINCIPAL * LGD)
    return monthly_profit, npv, breakeven_rate

# Base case
monthly_profit_base, npv_base, breakeven_dr = calculate_viability(
    EXPECTED_DEFAULT_RATE, AVG_INTEREST_INCOME_MONTHLY, ACTIVE_BORROWERS
)

print("\n--- 1. Chain Forward Profitability Analysis (Base Case) ---")
print(f"Monthly Profit: ${monthly_profit_base:,.2f}")
print(f"NPV (3 years): ${npv_base:,.2f}")
print(f"Break-even default rate: {breakeven_dr*100:.2f}%")
print("-" * 50)

# =========================================================
# 2. Load Data + Feature List
# =========================================================

print("\n--- 2. Load Combined Data ---")

TARGET_COLUMN = "Default"
RISK_FEATURES = [
    "Cashflow_Volatility_Ratio",
    "Partner_Risk_Score",
    "Avg_Payment_Speed_Days",
    "Days_Active_Platform",
    "Max_Days_in_Arrears_TRAD",
    "Max_Utilization_TRAD",
    "Unique_Loan_Count_TRAD"
]

try:
    df = pd.read_csv(FILE_COMBINED_DATA)
    df = df[RISK_FEATURES + [TARGET_COLUMN]].dropna()
    print(f"Loaded {len(df)} rows for modeling.")
except FileNotFoundError:
    print(f"ERROR: Missing file: {FILE_COMBINED_DATA}")
    raise

# =========================================================
# 3. Customer Risk Segmentation (KMeans)
# =========================================================

print("\n--- 3. Customer Risk Segmentation (KMeans) ---")

cluster_features = [
    "Cashflow_Volatility_Ratio",
    "Partner_Risk_Score",
    "Avg_Payment_Speed_Days",
    "Max_Days_in_Arrears_TRAD"
]

X_cluster = StandardScaler().fit_transform(df[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(X_cluster)

segment_summary = df.groupby("Segment").agg(
    Avg_Cashflow=("Cashflow_Volatility_Ratio", "mean"),
    Avg_Partner_Risk=("Partner_Risk_Score", "mean"),
    Avg_Payment_Speed=("Avg_Payment_Speed_Days", "mean"),
    Avg_Max_DPD=("Max_Days_in_Arrears_TRAD", "mean"),
    Default_Rate=(TARGET_COLUMN, "mean"),
    Count=("Segment", "count")
).sort_values(by="Default_Rate", ascending=False)

print(segment_summary)
segment_summary.to_csv(OUTPUT_SEGMENT_CSV)
print(f"Saved segmentation to {OUTPUT_SEGMENT_CSV}")

segment_chart = segment_summary.reset_index()
segment_chart["Default_Rate"] *= 100
segment_chart.to_csv(OUTPUT_CHART_SEGMENT_CSV, index=False)
print(f"Saved segment chart to {OUTPUT_CHART_SEGMENT_CSV}")
print("-" * 50)

# =========================================================
# 4. Default Prediction Model + Save Model/Scaler
# =========================================================

print("\n--- 4. Logistic Regression Model ---")

X = df[RISK_FEATURES]
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc:.4f}")

coeffs = pd.DataFrame({
    "Feature": RISK_FEATURES,
    "Coefficient": model.coef_[0]
}).sort_values(by="Coefficient", ascending=False)
coeffs.to_csv(OUTPUT_CHART_COEFFICIENT_CSV, index=False)
print(f"Saved coefficients to {OUTPUT_CHART_COEFFICIENT_CSV}")

# Save model and scaler
joblib.dump(model, RISK_MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)
print(f"Model saved:  {RISK_MODEL_FILE}")
print(f"Scaler saved: {SCALER_FILE}")
print("-" * 50)

# =========================================================
# 5. Scenario Analysis
# =========================================================

print("\n--- 5. Scenario Analysis ---")

scenarios = {
    "Optimistic": {"Default_Rate_Multiplier": 0.75, "Interest_Income_Multiplier": 1.10, "Borrower_Multiplier": 1.10},
    "Base Case": {"Default_Rate_Multiplier": 1.00, "Interest_Income_Multiplier": 1.00, "Borrower_Multiplier": 1.00},
    "Pessimistic": {"Default_Rate_Multiplier": 1.30, "Interest_Income_Multiplier": 0.90, "Borrower_Multiplier": 0.90}
}

results = []

for name, params in scenarios.items():
    dr = EXPECTED_DEFAULT_RATE * params["Default_Rate_Multiplier"]
    inc = AVG_INTEREST_INCOME_MONTHLY * params["Interest_Income_Multiplier"]
    bor = ACTIVE_BORROWERS * params["Borrower_Multiplier"]

    mp, npv, _ = calculate_viability(dr, inc, bor)

    results.append({
        "Scenario": name,
        "Active Borrowers": int(bor),
        "Default Rate (%)": dr * 100,
        "Monthly Profit ($)": mp,
        "NPV ($)": npv
    })

scenario_df = pd.DataFrame(results)
scenario_df.to_csv(OUTPUT_SCENARIO_CSV, index=False)
print(scenario_df)
print(f"Saved scenario analysis to {OUTPUT_SCENARIO_CSV}")
