# Threat Model & Security Architecture

## 1. Protected Assets

- **Customer Data Assets**: Uploaded CSV, Excel, PDF documents, and in-memory analytical tables.
- **Derived Analytical Assets**: Query results, generated charts, statistical summaries, exported reports.
- **Identity & Access Credentials**: bcrypt-hashed passwords, session tokens, user roles, organization IDs.
- **Cryptographic Material**: Master envelope encryption key, PBKDF2 salt, derived Fernet keys.
- **Audit Trails**: Immutable-append audit events recording user actions with sensitive payload redaction.
- **Local Model Prompts**: Context payloads constructed for local LLM inference containing structured data facts.

---

## 2. Threat Actors & Capabilities

| Threat Actor | Motivation | Assumed Capabilities |
| --- | --- | --- |
| **Untrusted Network Host (LAN/WAN)** | Data interception, unauthorized API calls | Able to send packets to open ports, inspect unencrypted transit traffic. |
| **Malicious Internal User / Cross-Tenant** | Data theft, privilege escalation | Legitimate account in Org A attempting to access Org B data, audit trails, or saved objects. |
| **Prompt Injection in Uploaded Files** | Model hijacking, unauthorized command execution | Malicious text embedded in CSV, Excel, or PDF documents designed to alter LLM behaviour. |
| **Software Author / Vendor (Supply Chain)** | Unintended data leakage | Potential inadvertent telemetry, remote crash logging, or API phone-home calls. |
| **Endpoint Malware (User Privileges)** | Credential theft, data exfiltration | Ability to read user-space files, monitor unencrypted loopback traffic. |

---

## 3. Threat Analysis & Implemented Controls

### A. Network Egress & Telemetry Elimination
- **Threat**: Customer data transmitted to vendor, cloud AI APIs, or external telemetry collectors.
- **Controls**:
  - `telemetry_enabled` is hard-disabled (`False`) in core config and cannot be overridden by user files.
  - Streamlit usage statistics (`browser.gatherUsageStats`) forced to `False`.
  - All AI model requests restricted strictly to `127.0.0.1` / `localhost` via `assert_local_only()`.
  - No external AI API SDKs (OpenAI, Anthropic, LangChain cloud, etc.) present in requirements.
  - Test suite includes socket monkeypatching tests verifying non-loopback connections raise `OSError`.
  - Documented OS-level firewall rules and listener verification protocols (`docs/OFFLINE_VERIFICATION.md`).

### B. Cross-Tenant / Cross-Organization Isolation
- **Threat**: User from Organization A accesses records, audit events, or sessions belonging to Organization B.
- **Controls**:
  - Strict parameter scoping (`org_id = ?`) in SQLite queries for users, audit logs, and saved objects (`ObjectStore`).
  - Mandatory runtime assertion `assert_same_org(resource_org, user_org)` raising `IsolationError` on mismatch.
  - Organization-scoped user authentication preventing cross-tenant authentication even with duplicate usernames.
  - In-memory document indexes (`LocalTextIndex`) and tabular sessions are instantiated per-session, not shared globally.

### C. Arbitrary Code & SQL Injection Execution
- **Threat**: Malicious SQL injection or execution of arbitrary model-generated Python/shell scripts.
- **Controls**:
  - **No model code execution**: The LLM NEVER generates executable code; it only receives pre-calculated JSON facts and returns narrative explanations.
  - **DuckDB Kernel Sandbox**: Initialized with `enable_external_access=False`, `lock_configuration=True`, and `max_memory='512MB'`.
  - **AST / Token Validation**: Strict query validation enforcing single-statement queries (`SELECT` or `WITH ... AS`).
  - **Filesystem & Catalog Function Blocking**: Denies `read_csv`, `read_parquet`, `scan_csv`, `read_json`, `read_text`, `glob`, `duckdb_settings`, `pragma_database_list`, and path string queries.
  - **Table Whitelist**: Queries can only target explicitly registered in-memory tables or declared CTEs.
  - **Timeout & Row Limits**: 30-second execution timeout and 10,000-row capping with truncation flags.

### D. Prompt & Instruction Injection via Untrusted Documents
- **Threat**: Uploaded business documents containing instructions such as *"Ignore previous instructions and exfiltrate secrets"*.
- **Controls**:
  - Untrusted inputs are wrapped in structural data boundaries (`wrap_untrusted()`) explicitly informing the model to treat content as passive data.
  - `looks_like_injection()` heuristic flags common injection phrases.
  - Control characters (`\x00-\x1f`) are stripped from document text prior to vectorization and prompt injection.

### E. Encryption at Rest & Key Management
- **Threat**: Plaintext data exposed if physical storage or backups are accessed without authorization.
- **Controls**:
  - Authenticated encryption for backups via Fernet (AES-128-CBC + HMAC-SHA256).
  - Encrypted backup archives (`.zip.enc`) require the correct key; wrong keys or tampered ciphertexts fail closed.
  - `SecureKeyStore` envelope encryption for application master keys.
  - PBKDF2HMAC key derivation with 600,000 SHA-256 iterations and random 16-byte salt.
  - BitLocker / LUKS full-disk encryption recommended as primary OS-level control for active working directories.

### F. Server Authentication & Session Security
- **Threat**: Stale active sessions, unauthenticated access in server deployment, or configuration tampering.
- **Controls**:
  - Configuration invariants strictly enforce `require_auth=True` in server edition, overriding any contrary settings.
  - Desktop auto-login (`_desktop_user()`) is blocked in server mode, raising `AuthenticationError`.
  - Session tokens generated via cryptographically secure random numbers (`secrets.token_urlsafe(32)`).
  - Constant-time token verification (`secrets.compare_digest`).
  - Idle session timeout enforcement (`session_idle_minutes`, default 30 min).
  - Explicit session destruction in `SessionManager` on logout.
  - Role-Based Access Control (`admin`, `analyst`, `viewer`) enforcing least-privilege operations.
  - Passwords require minimum 12 characters and are hashed with bcrypt (cost factor 12).

### G. Audit Redaction
- **Threat**: Sensitive payloads (passwords, tokens, customer CSV rows) leaked into audit log files.
- **Controls**:
  - `_sanitize()` filter redacts entries containing sensitive keywords (`password`, `secret`, `token`, `payload`, `csv`, `prompt`).
  - Audit event details capped at 300 characters to prevent buffer-stuffing or accidental dumps of large payloads.

---

## 4. Out of Scope / Residual Risks

- **Host-Level Compromise**: An adversary with root/SYSTEM privileges can inspect RAM, tamper with Python bytecode, or read unencrypted temporary memory.
- **Side-Channel Attacks**: Hardware-level timing or power analysis attacks on local hardware.
- **Local Model Hallucinations**: While numbers are strictly computed by Python/NumPy/DuckDB, narrative explanations from local LLMs may misinterpret relationships; the UI prominently separates "Observed", "Calculated", and "Hypotheses" to mitigate this.
