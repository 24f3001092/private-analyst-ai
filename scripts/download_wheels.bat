@echo off
REM Download Python wheels on an internet-connected machine for offline installation.
cd /d "%~dp0\.."

echo =================================================================
echo  Private Analyst AI — Prepare Offline Wheelhouse
echo =================================================================
echo Downloading compatible Windows wheels for Python 3.11/3.12/3.13...
echo.

if not exist "wheelhouse" mkdir wheelhouse

python -m pip download -r requirements.txt -d wheelhouse
if errorlevel 1 (
    echo [ERROR] Failed to download one or more packages. Check internet connection.
    exit /b 1
)

echo.
echo =================================================================
echo [SUCCESS] Offline wheelhouse populated in: %CD%\wheelhouse
echo Copy this folder alongside the application to your offline computer.
echo =================================================================
