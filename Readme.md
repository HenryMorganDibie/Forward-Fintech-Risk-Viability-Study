# Chain Forward Risk Viability Assessment — V3.0 Strategic Cut-Off & Viability Path

## Model Iterations & Validation: Documenting the Shift from Pricing to Hard Filtering

📌 **Project Overview**

This repository documents the full analytical journey—from the initial V1.0 feasibility study to the validated V3.0 model deployment strategy. The core objective remains achieving portfolio viability, but the strategic path has evolved from marginal pricing adjustments to implementing a definitive, high-confidence strategic cut-off.

It includes all core project artifacts, model code, and documentation across iterations.

---

### Folder Structure

<pre lang="markdown">
Chain_Forward_Risk_Assessment/
├── 1_Presentation/
│ ├── Achieving Profitability, Risk Model V3.0 Findings & Strategic Cut-Off Recommendation.pptx
│ ├── Chain Forward Profitability Analysis, Strategic Path to Viability.pbix
│ └── Chain Forward Risk Assessment & Profitability Study.pptx
│
├── 2_Code_and_Data/
│ ├── data/
│ │ ├── Loan_Snapshot_Dataset.xlsx
│ │ └── simulated_msme_data.csv
│ │
│ ├── outputs/
│ │ ├── chart_feature_importance.csv
│ │ ├── chart_feature_importance_v3.csv
│ │ ├── chart_segment_performance.csv
│ │ ├── chart_segment_performance_v3.csv
│ │ ├── combined_features_v3.csv
│ │ ├── combined_risk_dataset_final_v4.csv
│ │ ├── sample_combined_v4.csv
│ │ ├── scenario_analysis_results.csv
│ │ ├── scenario_analysis_results_v3.csv
│ │ ├── segmentation_summary.csv
│ │ ├── segmentation_summary_v3.csv
│ │ └── models/
│ │ ├── risk_model_v2.pkl
│ │ └── scaler_v2.pkl
│ │
│ └── src/
│ ├── chain_forward_risk_model.py
│ ├── data_combination_pipeline.py
│ ├── data_combination_pipeline_v2.py
│ └── risk_modeling_pipeline_v2.0.py
│
├── 3_Documentation/
│ ├── Chain Forward Profitability Analysis, Strategic Path to Viability.pdf
│ ├── Model Governance and Strategic Roadmap.pdf
│ └── Risk Management & Monitoring Framework (V3.0 - Post-Deployment).pdf
│
├── 4_Monitoring_Dashboard/
│ └── governance_dashboard.html
│
├── Readme.md
└── run_full_pipeline.ps1
</pre>



---

## 🎯 Core Finding (V3.0 Final)

The V3.0 model provides a high-confidence solution:  

### **A Hard Cut-Off (Filtering) of the most loss-concentrated segment (Segment 0: CRITICAL RISK) is required to reduce the portfolio-wide default rate below the 3.75% break-even point and achieve immediate positive NPV.**

---

## 📊 Key Financial Insights (Baseline)

| Metric | Result | Implication |
|--------|--------|-------------|
| **Net Present Value (NPV)** | **−$6,745,425 (3-year horizon)** | Product is not financially viable in current form |
| **Break-Even Default Rate** | **3.75%** | Portfolio must stay below this risk level |
| **Current Expected Default Rate** | **6.00%** | Product operates 2.25 percentage points above sustainability |

---

## 🔍 Analytical Findings: Evolution from V1.0 to V3.0

---

## **V1.0 Findings (Initial Hypothesis & Pricing Strategy)**

The initial modeling focused heavily on non-traditional, behavioral variables, finding the model's rejection power was weak (AUC < 0.6). This led to a strategy centered on marginal pricing adjustments.

### 🧭 V1.0 Methodology

V1.0 followed a structured four-pillar approach:  
- Portfolio Viability Modeling  
- Behavioral Risk Analytics  
- K-Means Segmentation  
- Logistic Regression for Default Prediction  

### 🔑 V1.0 Key Analytical Findings

| Metric | V1.0 Result | Strategic Recommendation |
|--------|-------------|---------------------------|
| **Dominant Risk Driver** | Cashflow Volatility Ratio (+0.3347) | Implement risk-based pricing premiums |
| **Model Limit** | ROC AUC = 0.5962 (Weak) | Pricing adjustments needed before stricter modeling |
| **V1.0 Segmentation** | Segment 1 (6.66% DR, 48.5% share) | Aggressive pricing required |

**Loss concentration:** Nearly half the borrowers belonged to a high-risk segment above the break-even threshold.

---

## **V3.0 Findings (Validated Strategy & Hard Filtering)**

Refined feature engineering and model tuning yielded a high-confidence predictive model (AUC 0.8820), validating traditional credit signals and enabling a decisive cut-off strategy.

| Metric | V3.0 Result | Strategic Implication |
|--------|-------------|-----------------------|
| **Model Performance** | ROC AUC = 0.8820 (Excellent) | High-confidence segmentation and rejection rules |
| **Top Risk Drivers** | Max Days in Arrears (+1.94), Max Utilization (+1.23) | Traditional credit signals dominate |
| **Loss Concentration** | Segment 0: 86.1% DR, 40% volume | Segment 0 must be eliminated |

### V3.0 Segmentation Summary

| Segment | Risk Classification | Default Rate | Portfolio Share |
|---------|---------------------|--------------|-----------------|
| **Segment 0** | CRITICAL RISK | **86.1%** | **40%** |
| **Segment 2** | High Risk | 61.6% | 20% |
| **Segment 1** | Base Risk | 35.8% | 40% |

---

## 🎯 Strategic Recommendations (V3.0 Path to Viability)

The V3.0 strategy is a 3-part plan to immediately reduce portfolio risk and then optimize pricing and retention.

---

### 📌 **Phase 1 — Immediate Strategic Filtering (The Primary Fix)**

**Objective:** Achieve positive NPV by forcing the portfolio DR below 3.75%.

| Action | Target Segment | Strategic Action |
|--------|----------------|------------------|
| **Filter (Hard Cut-Off)** | Segment 0 | Auto-decline based on arrears/utilization profile |
| **Price** | Segment 2 | Apply aggressive risk-based pricing |
| **Retain** | Segment 1 | Preferential pricing, core segment |

### Financial Impact Scenario

| Scenario | Hypothesis | Post-Filter DR | 3-Year NPV |
|----------|------------|----------------|-------------|
| **Optimistic** | Filter Segment 0 (40% volume) | 4.5% | **+$750,000** |
| **Base Case** | No filtering | 6.0% | −$6,745,425 |

---

### 📌 **Phase 2 & 3 — Deployment and Future Modeling**

- Deploy the V3.0 model and Segment 0 rejection rule into the lending platform  
- Setup Early Warning Indicators  
- Future exploration: XGBoost / LightGBM  

---

## 🔄 Automated Execution Pipeline

The repository includes a PowerShell script:

**`run_full_pipeline.ps1`**

Runs the entire workflow end-to-end.

### ▶️ How to Run

```bash
.\run_full_pipeline.ps1
```


### 📂 Pipeline Flow

<pre lang="markdown">
run_full_pipeline.ps1
│
├── STEP 1 → data_combination_pipeline.py
│       • Load data
│       • Clean & engineer features
│       • Export final combined dataset
│
└── STEP 2 → chain_forward_risk_model.py
        • K-means segmentation
        • Logistic regression modeling
        • Profitability & NPV analysis
        • Scenario stress-testing
        • Output generation
</pre>

## 🧭 Methodology: End-to-End Risk Assessment Framework

The analysis followed a structured, four-pillar quantitative approach:

### 1. Portfolio Viability Modeling
- Calculated monthly profit, expected loss, and NPV across multiple scenarios (base, optimistic, pessimistic).  
- Identified the exact financial gap and required default thresholds.

### 2. Behavioral Risk Analytics
Focused on:  
- Cashflow volatility  
- Payment speed  
- Partner risk score  
- Platform activity  

Used to understand drivers of MSME default in short-term, high-frequency lending.

### 3. Customer Segmentation (K-Means)
- Clustered MSMEs into 3 risk groups.  
- Identified where losses are concentrated.

### 4. Default Prediction (Logistic Regression)
- Measured feature importance and predicted risk.  
- Validated behavioral variables as superior risk indicators.

---

## 🔑 Key Analytical Findings

### 1. Loss Concentration Is the Root Cause

Nearly half of the borrowers belong to a **high-risk segment** that exceeds the break-even threshold.

| Segment | Default Rate | Portfolio Share | Insight |
|---------|--------------|----------------|--------|
| Segment 1 (High Risk) | 6.66% | 48.5% | Primary driver of the −$72k monthly loss. |
| Segment 0 (Medium Risk) | 5.29% | 51.4% | Still above break-even; requires moderate risk premium. |
| Segment 2 (Low Risk) | 4.0% | <1% | Small but stable; potential “anchor borrowers.” |

### 2. Cashflow Volatility Is the Dominant Risk Driver

- **Cashflow_Volatility_Ratio** has the strongest positive coefficient (+0.3347)  
- Payment speed also contributes to higher default probability  
- Traditional credit variables (arrears, utilization) add minimal predictive lift  

**Model Limit:**  
`ROC AUC = 0.5962` → Weak rejection capability.  
Pricing adjustments must occur before stricter modeling.

---

## 🎯 Strategic Recommendations (Path to Viability)

### 📌 Phase 1 — Immediate Pricing Adjustment
**Objective:** Achieve positive monthly profit
- Segment 1: Apply +35–45% interest premium  
- Segment 0: Apply +10–15% premium  
- Segment 2: Retain/offer preferential pricing to maintain quality borrowers

### 📌 Phase 2 — Upgrade Risk Modeling
**Objective:** Improve reject strategy
- Move from Logistic Regression → XGBoost or LightGBM  
- Target ROC AUC ≥ 0.70  
- Integrate time-series patterns for better cashflow interpretation

### 📌 Phase 3 — Operational Risk Controls
**Objective:** Prevent risk drift and systemic losses
- Deploy Cashflow Volatility Ratio as a primary decision rule  
- Build Early Warning Indicators (EWI) on payment delays & partner dependency  
- Monitor value-chain concentration to prevent cascade failures

---

## 📁 Repository Structure

| Folder | Description |
|--------|------------|
| 1_Presentation/ | Final PowerPoint presentation and Power BI visuals |
| 2_Code_and_Data/src/ | Core Python scripts for modeling and scenario analysis |
| 2_Code_and_Data/outputs/ | CSV outputs for segmentation, profitability, and feature weights |
| 3_Documentation/ | Supporting notes, assumptions, frameworks |

---

## 🛠 Technical Stack

| Tool | Purpose |
|------|--------|
| Python | Core analytics & modeling |
| Pandas / NumPy | Data engineering & financial calculations |
| Scikit-learn | Clustering + Logistic Regression modeling |
| Power BI | Executive-ready dashboard and visualization |
