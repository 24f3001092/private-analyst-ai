# Windows setup guide

This guide is for a single Windows laptop (desktop edition).

## 1. Check Python

Open PowerShell:

```powershell
python --version
```

You need Python 3.11, 3.12, or 3.13. This project was developed on Python 3.13.14.

## 2. Create an isolated environment

```powershell
cd "D:\Ai Model"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `Activate.ps1` is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Run tests

```powershell
$env:PYTHONPATH = "$pwd\src"
python -m pytest
```

All tests must pass before you rely on the app.

## 4. Start the app (localhost only)

Double-click `scripts\run_desktop.bat` or:

```powershell
$env:PYTHONPATH = "$pwd\src"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

The app listens on `127.0.0.1:8501` (this computer only).

## 5. Optional local model

1. Install Ollama from the vendor’s official Windows installer (customer downloads this themselves).
2. From a command prompt: `ollama pull llama3.2` (requires internet **once**, and you must accept that model’s licence).
3. Confirm `ollama list` shows the model.
4. Restart Private Analyst AI. Settings show the target `127.0.0.1:11434`.

Do not expose Ollama on a public interface.

## 6. Data location

By default: `%LOCALAPPDATA%\PrivateAnalystAI`

Override:

```powershell
$env:PRIVATE_ANALYST_DATA_DIR = "D:\PrivateAnalystData"
```

## 7. Uninstall / data retention

- Delete the project folder and `.venv` to remove the program.
- Delete `%LOCALAPPDATA%\PrivateAnalystAI` (or your custom data directory) to delete local config, SQLite, and backups.
- Session uploads are not written unless you export or create a backup.

## 8. Hardware

- CPU-only analytics: 8 GB RAM is usually enough for the sample files and moderate CSVs.
- Local 7B-class models typically want 8–16 GB RAM; larger models need more. The app still runs without a model.
