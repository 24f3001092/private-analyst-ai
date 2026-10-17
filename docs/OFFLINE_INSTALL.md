# Offline installation guide

Routine use after install must work with the network disconnected. Installation of Python packages and (optional) models usually needs a **one-time** download on a connected machine.

## Recommended approach

1. On a connected PC, download this source tree and Python wheels:

```powershell
python -m pip download -r requirements.txt -d wheelhouse
```

2. Copy `wheelhouse\`, `requirements.txt`, and the project folder to the offline PC (USB).
3. On the offline PC:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --no-index --find-links wheelhouse -r requirements.txt
python -m pytest
.\scripts\run_desktop.bat
```

4. Disconnect the network (see `OFFLINE_VERIFICATION.md`) and repeat import, dashboard, ask, and export.

## Models

Model weights are **not** in this repository. Copy an Ollama model blob from a machine that already pulled it, using Ollama’s documented local backup/restore process, only if the model licence allows that transfer.

Until a model is present, the analytics engine still runs fully offline.

## Rules

- Do not configure pip to use a vendor cloud index at runtime.
- Do not enable Streamlit sharing or cloud features.
- Do not set the model host to anything except `127.0.0.1` or `localhost`.
