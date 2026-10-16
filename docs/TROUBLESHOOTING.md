# Troubleshooting

## `python` is not recognized

Install Python 3.13 and tick “Add python.exe to PATH”, or call the full path shown by `where.exe python`.

## `Activate.ps1` cannot be loaded

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Streamlit opens then immediately exits

Run from the project root with `PYTHONPATH` set to `src`:

```powershell
$env:PYTHONPATH = "$pwd\src"
python -m streamlit run app.py --server.address 127.0.0.1
```

## Import error: `No module named private_analyst`

`src` is not on `PYTHONPATH`. Use `scripts\run_desktop.bat`.

## Tests fail on malformed CSV

pandas is strict (`on_bad_lines=error`). That is intentional.

## “Local OCR is unavailable”

Install Tesseract for Windows and ensure `tesseract.exe` is on PATH. Text-based PDFs do not need OCR.

## Model explanation is a JSON restatement

Ollama is not running or the model name in config does not match `ollama list`. Analytics numbers are still valid.

## Port 8501 in use

Close the other Streamlit process or change `server.port` in `.streamlit/config.toml`.

## Excel file will not open

Use `.xlsx` (Excel 2007+). Legacy `.xls` is not supported in this MVP.
