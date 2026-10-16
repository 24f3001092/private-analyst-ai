@echo off
REM Start Private Analyst AI on this Windows computer, bound to localhost only.
cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PRIVATE_ANALYST_EDITION=desktop

if exist "%CD%\.venv\Scripts\python.exe" (
  set PY=%CD%\.venv\Scripts\python.exe
) else (
  set PY=python
)

echo Starting Private Analyst AI at http://127.0.0.1:8501
"%PY%" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
