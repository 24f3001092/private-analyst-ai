# Known Limitations and Production Boundaries

Be honest with customers. Version **0.1.0** is an enterprise-grade private MVP designed for offline, on-premise, and self-hosted environments.

---

## 1. Implemented in Current Build (241 Automated Tests Passing)

- **Hardened DuckDB Analytical Sandbox**: Kernel-level external access disabling (`enable_external_access=False`), configuration lock (`lock_configuration=True`), and conservative AST/token query validation restricting queries to registered in-memory tables and declared CTEs.
- **Strict Server Authentication Invariant**: Fail-closed configuration precedence unconditionally forcing `require_auth=True` in server mode, blocking desktop bypass, and integrating session token management with idle timeouts.
- **Desktop Mode Network Exposure Guard**: Hard constraint rejecting non-loopback bind hosts in unauthenticated desktop mode, requiring explicit switch to server mode or authentication.
- **Envelope Encryption & Encrypted Backups**: Authenticated encryption (`Fernet` AES-128-CBC + HMAC-SHA256) for backup archives (`.zip.enc`), key verification, tampering detection, and PBKDF2HMAC (600,000 iterations) key derivation via `SecureKeyStore`.
- **CSV Formula Injection Mitigation (DDE Guard)**: Automatic neutralization of dangerous spreadsheet formula prefixes (`=`, `@`, `+`, `-`, `\t`, `\r`) in exported CSV tables and headers.
- **Resilient Multi-Tenant Audit Logging**: Scoped audit event persistence with pre-storage sanitization, tamper-resistant deduplication, and non-blocking failure decoupling ensuring session teardown on logout/timeout.
- **Validated Offline Packaging**: Windows distribution builder bundling required `.streamlit` privacy/security settings while strictly excluding secrets, credentials, databases, and temporary caches.
- **Explainable Machine Learning Engine**: Decision tree classification, linear regression, and K-Means clustering with explicit statistical caveats and train/test evaluation metrics.
- **End-to-End Offline Acceptance Test Suite**: Verifying all tabular, calculation, document search, and report generation workflows in air-gapped environments.

---

## 2. Recommended Production Deployments & Boundaries

- **Web Server Reverse Proxy**: Streamlit provides the internal user interface. In server deployments, administrators should place an enterprise reverse proxy (e.g. Nginx, Caddy, or IIS) with TLS termination (HTTPS) in front of the Streamlit application port (8501).
- **Filesystem Security**: Active analytical sessions process datasets in memory. Data directory contents (`app.db`, `config.json`) must be protected by OS-level BitLocker and strict NTFS Access Control Lists (ACLs) restricting access to the designated service user.
- **Air-Gapped Operation**: Follow the OS firewall rules documented in `docs/OFFLINE_VERIFICATION.md` to ensure complete outbound packet blocking for Python and Ollama runtimes.

---

## 3. Not Implemented / Future Roadmap

- Native PyInstaller / `.exe` single-file self-contained bundle (wrapper batch scripts `run_desktop.bat` and `run_server.bat` are provided).
- Native Hardware Security Module (HSM) / PKCS#11 key storage integration.
- Direct external database connectors (PostgreSQL / SQL Server) with enforced read-only accounts.
- Automated software updates (all updates currently require manual customer extraction to prevent unconsented network access).
- Bundled Tesseract OCR binary installer (system-level installation required for image OCR).
