# run_full_pipeline.ps1
# Executes the Data Combination and Risk Model scripts in sequence.

# --- 1. Define Paths ---
$ScriptRoot = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$SRC_DIR = Join-Path -Path $ScriptRoot -ChildPath "2_Code_and_Data\src"
$OUTPUT_DIR = Join-Path -Path $ScriptRoot -ChildPath "2_Code_and_Data\outputs"

# Define Python Script Paths (assuming they are saved in the 'src' folder)
$DataPrepScript = Join-Path -Path $SRC_DIR -ChildPath "data_combination_pipeline.py"
$ModelingScript = Join-Path -Path $SRC_DIR -ChildPath "chain_forward_risk_model.py"

# --- 2. Execution Function ---

function Invoke-PythonScript {
    param(
        [Parameter(Mandatory=$true)]
        [string]$Path,
        [Parameter(Mandatory=$true)]
        [string]$StepName
    )
    
    Write-Host "`n--- Starting $StepName ---" -ForegroundColor Yellow
    
    if (-not (Test-Path $Path)) {
        Write-Host "CRITICAL ERROR: Python script not found at $Path" -ForegroundColor Red
        return $false
    }
    
    # Execute the Python script. Output goes directly to the console.
    & python $Path
    
    # Check the exit code (0 is success)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $StepName failed. Python script exited with code $LASTEXITCODE." -ForegroundColor Red
        return $false
    }
    
    Write-Host "SUCCESS: $StepName completed." -ForegroundColor Green
    return $true
}

# --- 3. Main Execution Flow ---

Write-Host "--- Starting Full Chain Forward Risk Assessment Pipeline ---" -ForegroundColor Cyan

# Ensure output directory exists
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -Path $OUTPUT_DIR -ItemType Directory | Out-Null
    Write-Host "Created output directory: $OUTPUT_DIR"
}

# 1. Run Data Preparation (Creates combined_risk_dataset_final_v4.csv)
if (Invoke-PythonScript -Path $DataPrepScript -StepName "STEP 1: Data Combination and Feature Prep") {
    
    # 2. Run Modeling and Analysis (Uses the output from Step 1)
    if (Invoke-PythonScript -Path $ModelingScript -StepName "STEP 2: Risk Modeling, Segmentation, and Scenario Analysis") {
        Write-Host "`n--- Pipeline Execution SUCCESSFUL! All reports saved to '2_Code_and_Data\outputs' ---" -ForegroundColor Green
    } else {
        Write-Host "`n--- Pipeline Execution HALTED due to failure in Step 2 ---" -ForegroundColor Red
    }
} else {
    Write-Host "`n--- Pipeline Execution HALTED due to failure in Step 1 ---" -ForegroundColor Red
}