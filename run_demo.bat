@echo off
echo NeuroScan AI v3.0 — Demo Launcher
echo ==================================
echo.
echo Starting Streamlit application...
echo The browser will open at http://localhost:8501
echo.
echo To expose via ngrok (for mobile demo):
echo   1. Open a NEW terminal window
echo   2. Run: ngrok http 8501
echo   3. Copy the https://xxxx.ngrok-free.app URL to your phone
echo.
echo IMPORTANT: Use only synthetic or de-identified images for demonstrations.
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated.
)

streamlit run app.py

pause
