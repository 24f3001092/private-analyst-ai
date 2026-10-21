@echo off
REM Install Private Analyst AI dependencies offline from local wheelhouse directory.
cd /d "%~dp0\.."

echo =================================================================
echo  Private Analyst AI — Offline Installation for Windows
echo =================================================================

if not exist "wheelhouse" (
    echo [ERROR] 'wheelhouse' directory not found!
    echo On an internet-connected computer, create the wheelhouse first:
    echo     python -m pip download -r requirements.txt -d wheelhouse
    echo Then copy the wheelhouse folder here on approved local media (USB).
    exit /b 1
)

if not exist ".venv" (
    echo [1/3] Creating Python virtual environment (.venv)...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment. Ensure Python 3.11+ is installed.
        exit /b 1
    )
)

echo [2/3] Installing dependencies offline (--no-index --find-links wheelhouse)...
"%CD%\.venv\Scripts\python.exe" -m pip install --no-index --find-links=wheelhouse -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Offline installation failed. Check wheelhouse packages.
    exit /b 1
)

echo [3/3] Executing offline acceptance test suite...
"%CD%\.venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 (
    echo [WARNING] Some tests reported issues. Please review test output.
) else (
    echo [SUCCESS] Offline verification test suite passed 100%%!
)

echo.
echo =================================================================
echo Offline Installation Complete!
echo Run desktop mode:  scripts\run_desktop.bat
echo Run server mode:   scripts\run_server.bat
echo =================================================================
