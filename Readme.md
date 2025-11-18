# Chain Forward Risk Viability Assessment — Forward Fintech
## Strategic Risk-Based Pricing & Portfolio Viability Analysis

---

## 📌 Project Overview

This repository contains the full quantitative analysis, modeling code, and strategic recommendations for assessing the financial and risk viability of Forward Fintech’s proposed **Chain Forward short-term value-chain lending product**. It includes:

- A reproducible ETL and modeling pipeline

- Integrated snapshot and simulated MSME datasets

- Risk scoring, customer segmentation, and scenario simulations

- Executive-ready dashboards and documentation

- A clean, structured project directory

The project supports strategic decision-making regarding risk, profitability, and portfolio viability.

### Folder Structure

<pre lang="markdown">
Chain_Forward_Risk_Assessment/
│
├── 1_Presentation/
│   ├── Chain Forward Profitability Analysis, Strategic Path to Viability.pbix
│   └── Chain Forward Risk Assessment & Profitability Study.pptx
│
├── 2_Code_and_Data/
│   ├── data/
│   ├── outputs/
│   └── src/
│
├── 3_Documentation/
│   ├── Chain Forward Profitability Analysis, Strategic Path to Viability.pdf
│   └── Model Governance and Strategic Roadmap.pdf
│
├── Readme.md
└── run_full_pipeline.ps1
</pre>

**Core Finding:**

> The product is structurally unprofitable under current pricing and default expectations.  
> A shift to risk-based pricing and stronger risk segmentation is required to achieve profitability.

---

## 📊 Key Financial Insights

| Metric | Result | Implication |
|--------|--------|------------|
| Net Present Value (NPV) | −$6,745,425 (3-year horizon) | Product is not financially viable in its current form. |
| Break-Even Default Rate | 3.75% | Portfolio must stay below this risk level to break even. |
| Current Expected Default Rate | 6.00% | The product operates 2.25 percentage points above financial sustainability. |

---

## 🔄 Automated Execution Pipeline

To ensure reproducibility, the repository includes a PowerShell automation script:

### **`run_full_pipeline.ps1`**

This script executes the **entire Chain Forward Risk Assessment workflow** — from data preparation to full modeling and scenario analysis — in the correct order.

### ✅ What It Does

- Loads and validates paths to all Python scripts  
- Executes:
  1. **Data Combination & Feature Engineering**
  2. **Risk Modeling, Segmentation & Scenario Analysis**
- Automatically stops if any step fails  
- Saves all outputs to: 2_Code_and_Data/outputs

---

### ▶️ How to Run

From the project root:

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
