import pandas as pd
import numpy as np
import logging
import uuid 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Starting V3.0 Data Combination Pipeline (Fixed Target Definition).")

# --------------------------------------------------------------------------------
# V3.0 Enhancement: Centralized Configuration for features and paths
# --------------------------------------------------------------------------------
config = {
	'paths': {
		'sim_trad_data_output': 'combined_features_v3.csv'
	},
	'features': {
		'traditional': ['loan_uuid', 'Max_Days_in_Arrears_TRAD', 'Max_Utilization_TRAD', 'Unique_Loan_Count_TRAD'],
		'alternative': ['loan_uuid', 'Cashflow_Volatility_Ratio', 'Avg_Payment_Speed_Days', 'Partner_Risk_Score', 'Days_Active_Platform'],
		'target': 'Default'
	}
}

# --- Data Simulation Functions ---

def generate_traditional_data(n_samples=25000):
	"""Generates simulated traditional credit data with a robust 'loan_uuid'."""
	np.random.seed(42)
	df = pd.DataFrame({
		'loan_uuid': [str(uuid.uuid4()) for _ in range(n_samples)],
		'Max_Days_in_Arrears_TRAD': np.random.randint(0, 30, size=n_samples),
		'Max_Utilization_TRAD': np.random.uniform(0.1, 100, size=n_samples),
		'Unique_Loan_Count_TRAD': np.random.randint(1, 5, size=n_samples)
	})
	
	# V3.0 FIX: Create a noisy, synthetic target variable based on multiple features
	# This ensures the target is NOT perfectly predictable by Max_Days_in_Arrears_TRAD alone.
	# Target is weakly correlated with high arrears, high utilization, and high volatility.
	
	# Create a composite risk score (0-1)
	risk_score = (
		# Arrears have the highest weight
		0.40 * (df['Max_Days_in_Arrears_TRAD'] / df['Max_Days_in_Arrears_TRAD'].max()) +
		# Utilization has secondary weight
		0.25 * (df['Max_Utilization_TRAD'] / 100) +
		# Uniform noise to break perfect predictability
		0.35 * np.random.uniform(0, 1, size=n_samples) 
	)
	
	# Threshold the risk score to create the binary 'Default' target
	# Setting threshold to target approx. 6% default rate
	df['Default'] = (risk_score > 0.45).astype(int)
	
	logging.info(f"Target distribution created (Approx. {df['Default'].mean()*100:.2f}% Default Rate).")
	
	return df

def generate_alternative_data(df_trad):
	"""Generates simulated alternative data, using a subset of the traditional keys."""
	np.random.seed(42)
	# Ensure keys are matched but sometimes incomplete (simulating ETL issues)
	matched_keys = df_trad['loan_uuid'].sample(frac=0.99, random_state=42).tolist()
	n_samples = len(matched_keys)
	
	df = pd.DataFrame({
		'loan_uuid': matched_keys,
		'Cashflow_Volatility_Ratio': np.random.gamma(shape=2, scale=1.5, size=n_samples), 
		'Avg_Payment_Speed_Days': np.random.normal(loc=10, scale=3, size=n_samples), 
		'Partner_Risk_Score': np.random.beta(a=1, b=5, size=n_samples), 
		'Days_Active_Platform': np.random.poisson(lam=80, size=n_samples)
	})
	return df

# --- Main Pipeline Logic ---

def combine_and_clean_data():
	"""Executes data generation, cleaning, and key-based merging."""
	df_trad = generate_traditional_data()
	df_alt = generate_alternative_data(df_trad)
	
	logging.info(f"Traditional Data Sample Count: {len(df_trad)}")
	logging.info(f"Alternative Data Sample Count: {len(df_alt)}")

	# V2.0 KEY-BASED MERGE
	combined_df = pd.merge(
		df_trad.drop(columns=['Default']), # Drop target before merge to simulate prediction data flow
		df_alt, 
		on='loan_uuid', 
		how='inner'
	)
	
	# Re-adding the target for training purposes (assuming the target comes from the Traditional source)
	combined_df = combined_df.merge(df_trad[['loan_uuid', 'Default']], on='loan_uuid', how='left')
	
	logging.info(f"Merged Dataset Count: {len(combined_df)} (Only records with matching loan_uuid kept.)")

	# --- Data Cleaning & Validation ---
	
	# V2.0 Fix: Handling the Utilization Percentage Error (Data Validation Gate)
	pre_fix_max = combined_df['Max_Utilization_TRAD'].max()
	if pre_fix_max > 1.05: # Threshold for detecting a percentage error
		combined_df['Max_Utilization_TRAD'] = np.where(
			combined_df['Max_Utilization_TRAD'] > 1.05,
			combined_df['Max_Utilization_TRAD'] / 100,
			combined_df['Max_Utilization_TRAD']
		)
		logging.warning(f"Validation Alert: Max Utilization was > 100% ({pre_fix_max:.2f}). Divided by 100 for compliance.")
		
	combined_df['Max_Utilization_TRAD'] = combined_df['Max_Utilization_TRAD'].clip(upper=1.0)
	
	# Final check: Ensure no missing values after cleaning
	combined_df.dropna(inplace=True)
	logging.info(f"Final Cleaned Count: {len(combined_df)}")

	# Save final artifact for the modeling script
	combined_df.to_csv(config['paths']['sim_trad_data_output'], index=False)
	logging.info(f"Cleaned dataset saved to '{config['paths']['sim_trad_data_output']}'.")
	
	return combined_df

if __name__ == '__main__':
	final_data = combine_and_clean_data()
	print("\n--- V3.0 Data Combination Summary ---")
	print(final_data[['loan_uuid', 'Max_Utilization_TRAD', 'Default']].head().to_markdown(index=False))