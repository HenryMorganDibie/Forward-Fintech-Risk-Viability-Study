import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

# =========================================================
# 0. Setup and Key Assumptions
# =========================================================

# File Paths
FILE_COMBINED_DATA = "2_Code_and_Data/outputs/combined_risk_dataset_final_v4.csv"
OUTPUT_SCENARIO_CSV = "2_Code_and_Data/outputs/scenario_analysis_results.csv"
OUTPUT_SEGMENT_CSV = "2_Code_and_Data/outputs/segmentation_summary.csv" 
OUTPUT_CHART_SEGMENT_CSV = "2_Code_and_Data/outputs/chart_segment_performance.csv"
OUTPUT_CHART_COEFFICIENT_CSV = "2_Code_and_Data/outputs/chart_feature_importance.csv"

# Financial Assumptions
ACTIVE_BORROWERS = 20000
AVG_INTEREST_INCOME_MONTHLY = 8.0  # $8 USD per borrower
EXPECTED_DEFAULT_RATE = 0.06       # Base Case default rate
AVG_LOAN_PRINCIPAL = 200.0         # Assumed average principal
LGD = 0.80                          # Loss Given Default
OPEX_PER_BORROWER = 2.0            # Monthly operating cost per borrower
INITIAL_INVESTMENT = 4_500_000.0   # $4.5M upfront investment
TERM_MONTHS = 3 * 12                # 3 years
ANNUAL_DISCOUNT_RATE = 0.10         # 10% annual discount rate

# =========================================================
# 1. Financial Viability Function
# =========================================================

def calculate_viability(default_rate, monthly_interest_income, active_borrowers):
    """
    Calculates monthly profit, NPV, and break-even default rate.
    """
    monthly_revenue = active_borrowers * monthly_interest_income
    monthly_loss_expected = active_borrowers * default_rate * AVG_LOAN_PRINCIPAL * LGD
    monthly_opex = active_borrowers * OPEX_PER_BORROWER
    monthly_profit = monthly_revenue - monthly_loss_expected - monthly_opex

    # Discounted cash flows
    monthly_discount_rate = (1 + ANNUAL_DISCOUNT_RATE) ** (1/12) - 1
    monthly_cash_flows = np.array([monthly_profit] * TERM_MONTHS)
    npv = -INITIAL_INVESTMENT + np.sum(monthly_cash_flows / (1 + monthly_discount_rate) ** np.arange(1, TERM_MONTHS + 1))

    # Break-even default rate
    break_even_rate = (monthly_interest_income - OPEX_PER_BORROWER) / (AVG_LOAN_PRINCIPAL * LGD)
    
    return monthly_profit, npv, break_even_rate

# Run Base Case
monthly_profit_base, npv_base, breakeven_dr = calculate_viability(
    EXPECTED_DEFAULT_RATE, AVG_INTEREST_INCOME_MONTHLY, ACTIVE_BORROWERS
)

print("--- 1. Chain Forward Profitability Analysis (Base Case) ---")
print(f"**Monthly Net Profit (Base Case): ${monthly_profit_base:,.2f}**")
print(f"**Net Present Value (NPV) over 3 years: ${npv_base:,.2f}**")
print(f"**Break-Even Default Rate: {breakeven_dr*100:.2f}%**")
print("-" * 50)

# =========================================================
# 2. Load Combined Data and Define Features
# =========================================================

print("\n--- 2. Load Combined Data and Define Features ---")

TARGET_COLUMN = 'Default'
RISK_FEATURES = [
    # Alternative Features
    'Cashflow_Volatility_Ratio', 
    'Partner_Risk_Score', 
    'Avg_Payment_Speed_Days', 
    'Days_Active_Platform',
    # Traditional Features
    'Max_Days_in_Arrears_TRAD',
    'Max_Utilization_TRAD',
    'Unique_Loan_Count_TRAD'
]

try:
    df = pd.read_csv(FILE_COMBINED_DATA)
    df = df[RISK_FEATURES + [TARGET_COLUMN]].dropna()
    print(f"Loaded modeling data: {len(df)} rows. Features ready.")
except FileNotFoundError:
    print(f"CRITICAL ERROR: Run data_combination_pipeline.py first to create {FILE_COMBINED_DATA}")
    raise

# =========================================================
# 3. Customer Segmentation (K-Means)
# =========================================================

print("\n--- 3. Customer Segmentation by Risk (K-Means) ---")

features_cluster = ['Cashflow_Volatility_Ratio', 'Partner_Risk_Score', 'Avg_Payment_Speed_Days', 'Max_Days_in_Arrears_TRAD']
X_cluster = StandardScaler().fit_transform(df[features_cluster])

K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_cluster)

# Segment Analysis
segment_analysis = df.groupby('Segment').agg(
    Avg_Vol_Cashflow=('Cashflow_Volatility_Ratio', 'mean'),
    Avg_Partner_Risk=('Partner_Risk_Score', 'mean'),
    Avg_Payment_Speed=('Avg_Payment_Speed_Days', 'mean'),
    Avg_Max_DPD=('Max_Days_in_Arrears_TRAD', 'mean'),
    Actual_Default_Rate=(TARGET_COLUMN, 'mean'),
    Customer_Count=(TARGET_COLUMN, 'count')
).sort_values(by='Actual_Default_Rate', ascending=False)

print("K-Means Segment Analysis (Ordered by Default Rate):")
print(segment_analysis)
segment_analysis.to_csv(OUTPUT_SEGMENT_CSV)
print(f"Segment summary saved to: {OUTPUT_SEGMENT_CSV}")

# Chart Data for Segments
segment_chart_data = segment_analysis[['Actual_Default_Rate', 'Customer_Count']].reset_index()
segment_chart_data.rename(columns={'Actual_Default_Rate':'Default_Rate', 'Customer_Count':'Customer_Volume'}, inplace=True)
segment_chart_data['Default_Rate'] *= 100
segment_chart_data['Segment'] = segment_chart_data['Segment'].astype(str)
segment_chart_data.to_csv(OUTPUT_CHART_SEGMENT_CSV, index=False)
print(f"Chart data for Segmentation saved to: {OUTPUT_CHART_SEGMENT_CSV}")
print("-" * 50)

# =========================================================
# 4. Default Prediction Model (Logistic Regression)
# =========================================================

print("\n--- 4. Default Prediction Model (Logistic Regression) ---")

X = df[RISK_FEATURES]
y = df[TARGET_COLUMN]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)
y_pred_proba = model.predict_proba(X_test_scaled)[:,1]

# Evaluate
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC Score on Test Data: {roc_auc:.4f}")

# Coefficients / Feature Importance
coefficients = pd.DataFrame({'Feature': RISK_FEATURES, 'Coefficient': model.coef_[0]}).sort_values(by='Coefficient', ascending=False)
print("\nRisk Feature Weights (Positive Coefficient = Higher Risk):")
print(coefficients)
coefficients.to_csv(OUTPUT_CHART_COEFFICIENT_CSV, index=False)
print(f"Chart data for Feature Importance saved to: {OUTPUT_CHART_COEFFICIENT_CSV}")
print("-" * 50)

# =========================================================
# 5. Scenario Analysis
# =========================================================

print("\n--- 5. Scenario Analysis for Profitability ---")

scenarios = {
    'Optimistic': {'Default_Rate_Multiplier': 0.75, 'Interest_Income_Multiplier': 1.10, 'Borrower_Multiplier': 1.10},
    'Base Case': {'Default_Rate_Multiplier': 1.00, 'Interest_Income_Multiplier': 1.00, 'Borrower_Multiplier': 1.00},
    'Pessimistic': {'Default_Rate_Multiplier': 1.30, 'Interest_Income_Multiplier': 0.90, 'Borrower_Multiplier': 0.90}
}

results = []
for name, params in scenarios.items():
    adj_default_rate = EXPECTED_DEFAULT_RATE * params['Default_Rate_Multiplier']
    adj_interest_income = AVG_INTEREST_INCOME_MONTHLY * params['Interest_Income_Multiplier']
    adj_borrowers = ACTIVE_BORROWERS * params['Borrower_Multiplier']

    monthly_profit_scen, npv_scen, _ = calculate_viability(
        adj_default_rate, adj_interest_income, adj_borrowers
    )

    results.append({
        'Scenario': name,
        'Active Borrowers': int(adj_borrowers),
        'Default Rate (%)': adj_default_rate * 100,
        'Monthly Profit ($)': monthly_profit_scen,
        'NPV ($)': npv_scen
    })

scenario_df = pd.DataFrame(results)
scenario_df.set_index('Scenario', inplace=True)

# Format for Output
scenario_df_formatted = scenario_df.copy()
scenario_df_formatted['Monthly Profit ($)'] = scenario_df_formatted['Monthly Profit ($)'].map('${:,.2f}'.format)
scenario_df_formatted['NPV ($)'] = scenario_df_formatted['NPV ($)'].map('${:,.2f}'.format)

scenario_df_formatted.to_csv(OUTPUT_SCENARIO_CSV)
print("\n--- FINAL SCENARIO SUMMARY ---")
print(scenario_df_formatted)
print(f"Scenario analysis results saved to: {OUTPUT_SCENARIO_CSV}")
