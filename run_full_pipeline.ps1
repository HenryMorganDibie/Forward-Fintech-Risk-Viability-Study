# run_full_pipeline.ps1
# Executes multiple Python scripts in sequence with error handling.

# --- 1. Define Paths ---
$ScriptRoot = Split-Path -Path $MyInvocation.MyCommand.Definition -Parent
$SRC_DIR = Join-Path -Path $ScriptRoot -ChildPath "2_Code_and_Data\src"
$OUTPUT_DIR = Join-Path -Path $ScriptRoot -ChildPath "2_Code_and_Data\outputs"

# Ensure output directory exists
if (-not (Test-Path $OUTPUT_DIR)) {
    New-Item -Path $OUTPUT_DIR -ItemType Directory | Out-Null
    Write-Host "Created output directory: $OUTPUT_DIR"
}

# --- 2. Define Scripts in Order ---
$scripts = @(
    @{ Name = "STEP 1: Data Combination v1"; File = "data_combination_pipeline.py" },
    @{ Name = "STEP 2: Data Combination v2"; File = "data_combination_pipeline_v2.py" },
    @{ Name = "STEP 3: Risk Modeling v1"; File = "chain_forward_risk_model.py" },
    @{ Name = "STEP 4: Risk Modeling v2"; File = "risk_modeling_pipeline_v2.0.py" }
)

# --- 3. Function to Run Python Script ---
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

    # Run Python script
    & python $Path

    # Check for missing module errors (like tabulate)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $StepName failed. Exit code: $LASTEXITCODE" -ForegroundColor Red
        Write-Host "TIP: Ensure all required Python packages are installed in your virtual environment." -ForegroundColor Cyan
        return $false
    }

    Write-Host "SUCCESS: $StepName completed." -ForegroundColor Green
    return $true
}

# --- 4. Main Execution Loop ---
Write-Host "--- Starting Full Chain Forward Risk Assessment Pipeline ---" -ForegroundColor Cyan

foreach ($script in $scripts) {
    $fullPath = Join-Path -Path $SRC_DIR -ChildPath $script.File
    $success = Invoke-PythonScript -Path $fullPath -StepName $script.Name
    if (-not $success) {
        Write-Host "`n--- Pipeline HALTED at $($script.Name) ---" -ForegroundColor Red
        break
    }
}

Write-Host "`n--- Pipeline Execution Complete ---" -ForegroundColor Cyan
Write-Host "Check the '2_Code_and_Data\outputs' folder for all results." -ForegroundColor Cyan
