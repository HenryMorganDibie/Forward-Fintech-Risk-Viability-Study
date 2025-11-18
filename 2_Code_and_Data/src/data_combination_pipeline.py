import pandas as pd
import numpy as np
import os

# --- 1. Define File Names & Parameters ---
FILE_SNAPSHOT_EXCEL = "2_Code_and_Data/data/Loan_Snapshot_Dataset.xlsx"
SHEET_NAME = 'loan_snapshot_60days'
FILE_SIMULATED_CSV = "2_Code_and_Data/data/simulated_msme_data.csv"
FILE_COMBINED_OUTPUT = "2_Code_and_Data/outputs/combined_risk_dataset_final_v4.csv"
FILE_SAMPLE_OUTPUT = "2_Code_and_Data/outputs/sample_combined_v4.csv"
N_MODELING_ROWS = 25000 

print("--- Starting Data Combination Script (v4.0 - Production Ready) ---")

# --- 2. Load and Aggregate Traditional Data (Excel Snapshot) ---
try:
    df_snapshot = pd.read_excel(FILE_SNAPSHOT_EXCEL, sheet_name=SHEET_NAME)
    print(f"Loaded snapshot data from Excel: {FILE_SNAPSHOT_EXCEL}")

    # Aggregation for traditional features
    df_traditional_agg = df_snapshot.groupby('customer_id').agg(
        Loan_Amount_TRAD=('loan_amount', 'first'),
        Max_Days_in_Arrears_TRAD=('days_in_arrears', 'max'),
        Max_Utilization_TRAD=('utilization (%)', 'max'),
        Unique_Loan_Count_TRAD=('loan_id', 'nunique')
    ).reset_index()

    # Create a sequential ID for temporary merging (Necessary because of ID incompatibility)
    df_traditional_agg['Sequential_Merge_ID'] = np.arange(1, len(df_traditional_agg) + 1)
    df_traditional_agg.set_index('Sequential_Merge_ID', inplace=True)
    
    print(f"Aggregated snapshot data to {len(df_traditional_agg)} unique rows.")

except Exception as e:
    print(f"CRITICAL ERROR: Failed to load or aggregate Excel file. Error: {e}")
    df_traditional_agg = pd.DataFrame()
    raise 


# --- 3. Load Alternative/Simulated Data ---
try:
    df_simulated = pd.read_csv(FILE_SIMULATED_CSV)
    print(f"Loaded simulated data: {FILE_SIMULATED_CSV} ({len(df_simulated)} rows)")
    
    # Create the same sequential ID to merge on.
    df_simulated['Sequential_Merge_ID'] = np.arange(1, len(df_simulated) + 1)
    df_simulated.set_index('Sequential_Merge_ID', inplace=True)

except FileNotFoundError:
    print(f"CRITICAL ERROR: Could not find the required simulated data {FILE_SIMULATED_CSV}.")
    raise 


# --- 4. Merge the Two Feature Sets ---

trad_cols = [col for col in df_traditional_agg.columns if col not in ['customer_id']]

df_combined = df_simulated.merge(
    df_traditional_agg[trad_cols],
    left_index=True, 
    right_index=True, 
    how='left'
)

# --- 5. Data Cleaning, Imputation, and Type Conversion (Robustness Check) ---

# Imputation
df_combined.fillna({
    'Loan_Amount_TRAD': df_combined['Loan_Amount_TRAD'].median(),
    'Max_Days_in_Arrears_TRAD': 0, 
    'Max_Utilization_TRAD': df_combined['Max_Utilization_TRAD'].median(),
    'Unique_Loan_Count_TRAD': 1
}, inplace=True)

# Explicit Type Conversion (Ensuring consistency)
numeric_cols = [
    'Loan_Amount_TRAD', 'Max_Days_in_Arrears_TRAD', 
    'Max_Utilization_TRAD', 'Unique_Loan_Count_TRAD'
]
for col in numeric_cols:
    if 'Days' in col or 'Count' in col:
        df_combined[col] = df_combined[col].astype(int) 
    else:
        df_combined[col] = df_combined[col].astype(float) 

# FIX: Utilization Percentage Conversion
# Check if utilization is likely a percentage (e.g., values > 1) and convert to decimal.
# Assuming the simulated utilization is already in decimal format (0 to 1).
if (df_combined['Max_Utilization_TRAD'].max() > 1.0):
    df_combined['Max_Utilization_TRAD'] = df_combined['Max_Utilization_TRAD'] / 100.0
    print("FIX: Converted Max_Utilization_TRAD from percentage (0-100) to decimal (0-1).")


# --- 6. Final Cleanup and ID Creation ---

df_combined.drop(columns=['customer_id'], inplace=True, errors='ignore')
df_combined.reset_index(drop=True, inplace=True)

# Add a clean, new sequential customer ID column
df_combined['final_customer_id'] = np.arange(1, len(df_combined) + 1)

# Reorder columns
final_columns = ['final_customer_id'] + [col for col in df_combined.columns if col != 'final_customer_id']
df_combined = df_combined[final_columns]


# --- 7. Final Output ---
print("\n--- Combined Dataset Final Review ---")
print(f"Total rows: {len(df_combined)}")

# Save the full and sample files
df_combined.to_csv(FILE_COMBINED_OUTPUT, index=False)
df_combined.head(1000).to_csv(FILE_SAMPLE_OUTPUT, index=False)
print(f"\n Full combined dataset saved to: {FILE_COMBINED_OUTPUT}")
print(f" Sample dataset saved to: {FILE_SAMPLE_OUTPUT} (1000 rows for quick checks)")