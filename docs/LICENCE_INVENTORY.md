# Third-party software and model licence inventory

Verify these terms yourself before you redistribute a commercial installer. Licences can change between versions. This file records versions used during development on 16 Oct 2026 (Windows, Python 3.13).

| Component | Version | Licence (as installed metadata / upstream) | Commercial use | Redistribution notes |
| --- | --- | --- | --- | --- |
| Python | 3.13.14 | PSF | Yes | Not bundled; customer installs Python |
| streamlit | 1.64.0 | Apache-2.0 | Yes | Trademark: do not imply you are Streamlit Inc. Disable usage stats. |
| pandas | 2.3.3 | BSD-3-Clause | Yes | Attribution in notices file |
| numpy | 2.5.2 | BSD-style | Yes | Attribution |
| openpyxl | 3.1.5 | MIT | Yes | |
| plotly | 7.1.0 | MIT | Yes | |
| PyPDF2 | 3.0.1 | BSD | Yes | Prefer pypdf going forward |
| pypdf | 6.20.1 | BSD | Yes | |
| reportlab | 5.0.1 | BSD | Yes | |
| bcrypt | 5.0.0 | Apache-2.0 | Yes | |
| scikit-learn | 1.9.0 | BSD-3-Clause | Yes | |
| cryptography | 50.0.0 | Apache-2.0 / BSD | Yes | |
| duckdb | 1.5.6 | MIT | Yes | In-memory analytical query engine |
| python-dateutil | 2.9.0.post0 | Dual: Apache-2.0 / BSD-3-Clause | Yes | Date parsing extension |
| pytesseract | 0.3.13 | Apache-2.0 | Yes | Wrapper only |
| pillow | 12.3.0 | HPND-ish / PIL | Yes | |
| Tesseract OCR engine | (system) | Apache-2.0 | Yes | Separate native install |
| Ollama runtime | (optional) | MIT (confirm current) | Yes | Not bundled |
| Open-weight LLMs | (optional) | **Model-specific** | **Check each card** | Do **not** assume Meta Llama, etc. allow your redistribution. Customer should pull models they are licensed to use. |
| sklearn English stop words | bundled | See scikit-learn | Yes | Used for local TF-IDF |

## Product licence (this source)

Private Analyst AI application code in this repository is intended for your commercial licensing to end-customer **installations**. Offer customers a written licence that:

- Grants use on their infrastructure
- Does not require them to send you their datasets
- States that third-party components remain under their own licences

A full lawyer-reviewed EULA is **not** included in this MVP.

## Models

This project does **not** ship model weights. Before you include any model in a paid installer, read that model’s licence (commercial use, attribution, acceptable use, and whether weights may be redistributed).
