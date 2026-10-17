@echo off
REM Private-server edition. Bind remains loopback unless you change config on purpose.
cd /d "%~dp0\.."
set PYTHONPATH=%CD%\src
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
set PRIVATE_ANALYST_EDITION=server

if exist "%CD%\.venv\Scripts\python.exe" (
  set PY=%CD%\.venv\Scripts\python.exe
) else (
  set PY=python
)

echo Server edition on http://127.0.0.1:8501
echo Put a reverse proxy with HTTPS in front before exposing this to an internal network.
"%PY%" -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
