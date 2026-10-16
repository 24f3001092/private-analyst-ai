# Private Analyst AI

Local, offline-first data analysis for organizations that must keep files, questions, and reports on their own computers.

This repository is a working MVP. It is designed so you can use it yourself and later license separate installations to companies. **Customer data is not sent to the software developer or to a cloud AI API.**

## What works in this version (tested)

- CSV and Excel import with size, type, and malformed-file checks
- Deterministic statistics (totals, mean, median, min, max, sample standard deviation)
- Data-quality report and Plotly charts
- In-memory read-only DuckDB SQL console with kernel-level sandbox (`enable_external_access=False`), AST validator, and output limits
- Advanced ML analytics (Decision Tree classification, Linear Regression, K-Means clustering) with evaluated holdout metrics and explicit methodological caveats
- Natural-language questions mapped to a **whitelist** of Python analytics (the model does not execute code and does not invent the numbers)
- Optional local Ollama model on `127.0.0.1` for explanations; if the model is absent, a deterministic explanation of the calculated JSON is shown
- Local PDF text extraction and in-memory TF-IDF document search with file name and page number
- Local user accounts, roles, organization isolation, and idle session management via `SessionManager`
- Application-level encryption at rest (Fernet envelope encryption with PBKDF2HMAC key derivation)
- Authenticated, encrypted data backups (`.zip.enc`) with fail-closed key and tampering verification
- CSV / Excel / Markdown / PDF export
- 187 automated tests passing (calculations, files, auth, isolation, ML metrics, SQL safety, encryption, backup, and network-blocking)

## What this version does not claim

- It is **not** a complete commercial-hardening release (Windows installer, HTTPS reverse proxy, OCR pack, and licence keys are documented, not fully packaged).
- A local model does **not** by itself equal “perfect privacy.” You still control disk encryption, who has login access, backups, and whether the PC is on a network.
- Offline licence checks cannot provide perfect piracy protection. See `docs/KNOWN_LIMITATIONS.md`.

## Quick start (Windows, beginner)

1. Install Python 3.13 from https://www.python.org or the Microsoft Store (you already have 3.13 on this laptop).
2. Open **PowerShell**, then run:

```powershell
cd "D:\Ai Model"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
.\scripts\run_desktop.bat
```

3. In the browser, open `http://127.0.0.1:8501` if it does not open automatically.
4. Upload `samples\sales_known.csv`. The **units** column total must be **50** and the mean **10**.

If PowerShell blocks scripts, run: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Streamlit usage statistics are disabled in `.streamlit/config.toml` and by the start script.

## Local AI model (optional)

Ollama is **not** bundled. Install it separately on the customer machine, pull a model you have the right to use, and keep it on localhost. The app never calls OpenAI, Anthropic, or other hosted APIs.

If Ollama is not running, analysis still works; only the narrative explanation falls back to a template that restates calculated values.

## Documentation

| File | Purpose |
| --- | --- |
| `docs/WINDOWS_SETUP.md` | Laptop install |
| `docs/OFFLINE_INSTALL.md` | Install without internet |
| `docs/PRIVATE_SERVER.md` | Multi-user private server |
| `docs/SECURITY_PRIVACY.md` | Security and privacy settings |
| `docs/OFFLINE_VERIFICATION.md` | Manual offline checklist |
| `docs/LICENCE_INVENTORY.md` | Third-party licences |
| `docs/TROUBLESHOOTING.md` | Common errors |
| `docs/RELEASE_CHECKLIST.md` | Shipping a version |
| `docs/KNOWN_LIMITATIONS.md` | Honest gaps |
| `docs/THREAT_MODEL.md` | Threat model |

## Privacy design (short)

- Uploads live in **memory** for the session unless the user exports or an admin saves a backup.
- The analytics engine is ordinary Python / pandas. The language model only explains JSON it is given.
- Model HTTP is allowed only to `127.0.0.1` / `localhost`.
- `telemetry_enabled` is forced off when configuration is loaded.
- Audit logs drop details that look like passwords or payloads.

## Project layout

```
app.py                      Streamlit entry
src/private_analyst/        Application modules
tests/                      pytest suite
samples/                    Synthetic files with known answers
scripts/                    Windows start helpers
docs/                       Guides
```
