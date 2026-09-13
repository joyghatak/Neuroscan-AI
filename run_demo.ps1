# NeuroScan AI v3.0 — PowerShell Demo Launcher

Write-Host "NeuroScan AI v3.0 - Demo Launcher" -ForegroundColor Cyan
Write-Host "==================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting Streamlit application..." -ForegroundColor Green
Write-Host "The browser will open at http://localhost:8501"
Write-Host ""
Write-Host "To expose via ngrok (for mobile demo):" -ForegroundColor Yellow
Write-Host "  1. Open a NEW terminal window"
Write-Host "  2. Run: ngrok http 8501"
Write-Host "  3. Copy the https://xxxx.ngrok-free.app URL to your phone"
Write-Host ""
Write-Host "IMPORTANT: Use only synthetic or de-identified images for demonstrations." -ForegroundColor Red
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Activate virtual environment if it exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating virtual environment..." -ForegroundColor Green
    & ".venv\Scripts\Activate.ps1"
}

# Run Streamlit
streamlit run app.py
