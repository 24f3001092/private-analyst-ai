# Security and Privacy Architecture & Controls

This document details the security posture, cryptographic controls, authentication invariants, and privacy guarantees implemented in **Private Analyst AI v0.1.0**.

---

## 1. Non-Negotiable Core Invariants

The application applies strict configuration precedence and immutable security invariants enforced at application initialization:
- **Zero Telemetry**: `telemetry_enabled` is hard-coded to `False`. Streamlit's `browser.gatherUsageStats` is unconditionally forced to `False`.
- **Loopback-Only AI**: The AI inference host is restricted strictly to loopback addresses (`127.0.0.1`, `localhost`, `::1`). Any attempt to configure an external host raises a `ValidationError`.
- **No Cloud AI SDKs**: No external AI APIs, client libraries (OpenAI, Anthropic, LangChain cloud, Hugging Face Hub), or cloud vector databases exist in the codebase or dependencies.
- **No Arbitrary Code Execution**: The LLM is never invoked to generate executable Python, Bash, or shell scripts. Model output is consumed purely as descriptive text.

---

## 2. In-Memory DuckDB Analytical Sandbox

The interactive SQL console provides analyst querying directly over imported tables with dual-layer defense:

### Layer 1: DuckDB Kernel-Level Configuration
- `enable_external_access = False`: Disables DuckDB's C++ filesystem access, socket connections, and external readers.
- `lock_configuration = True`: Permanently locks DuckDB settings, preventing user queries from re-enabling external access via `SET`.
- `max_memory = '512MB'`: Prevents memory exhaustion and Denial-of-Service attacks.

### Layer 2: AST & Token Validation
- **Read-Only Enforced**: Only single-statement queries beginning with `SELECT` or `WITH` are permitted.
- **Keyword Blacklist**: Rejects all DDL, DML, and mutating statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `ATTACH`, `DETACH`, `LOAD`, `INSTALL`, `COPY`, `PRAGMA`, `TRUNCATE`, `MERGE`, `UPSERT`, `VACUUM`).
- **File & Function Blocking**: Denies file-reading functions (`read_csv`, `read_parquet`, `read_json`, `read_text`, `scan_csv`, `glob`) and direct string-literal path queries (`FROM 'file.csv'`).
- **Catalog Probing Prevention**: Rejects queries accessing internal DuckDB settings or catalog tables (`duckdb_settings`, `pragma_database_list`).
- **Table Whitelist**: Queries must only reference registered in-memory tables or explicitly defined Common Table Expressions (`WITH ... AS`).
- **Execution Safeguards**: Row limits (capped at 10,000 rows with a truncation flag) and query execution timeouts (default 30 seconds).

---

## 3. Server Authentication & Session Management

### Configuration Precedence & Mandatory Auth
Configuration is loaded using strict four-tier precedence:
1. Dataclass defaults
2. `config.json` file
3. Environment variables (`PRIVATE_ANALYST_*`)
4. **Mandatory Security Invariants (applied last)**:
   - If `edition == "server"`, `require_auth` is forced to `True`. Attempts in `config.json` or environment variables to disable authentication are unconditionally overridden.
   - Any unknown edition value immediately raises a `ValidationError` (fail closed).

### Access Boundaries
- **Server Edition**: Every page and action requires an authenticated session. The desktop auto-login bypass (`_desktop_user()`) is explicitly forbidden in server mode and raises an `AuthenticationError`.
- **Desktop Edition**: Uses implicit local ownership, relying on Windows OS user account controls and physical workstation security.

### Cryptographic Passwords & Sessions
- **Password Hashing**: Passwords require a minimum length of 12 characters and are hashed using `bcrypt` (cost factor 12) with unique salts.
- **Session Tokens**: Generated via `secrets.token_urlsafe(32)`.
- **Constant-Time Verification**: Tokens are compared using `secrets.compare_digest`.
- **Idle Expiration**: Inactivity exceeding `session_idle_minutes` (default 30 minutes) invalidates the session and destroys token state.
- **Logout Destruction**: Signing out immediately destroys the server session token in `SessionManager`.

---

## 4. Multi-Tenant Organization Isolation

Multi-tenant environments are strictly isolated:
- **Query Scoping**: All database operations against users, audit logs, and saved objects enforce parameter binding scoped by `org_id` (`WHERE org_id = ?`).
- **Isolated User Identifiers**: Unique constraint on `(org_id, username)` allows different organizations to maintain identical usernames without cross-contamination.
- **Fail-Closed Isolation Checks**: `assert_same_org(resource_org, user_org)` validates boundaries before object retrieval or deletion, raising `IsolationError` on mismatch.
- **Audit Redaction**: Sensitive attributes (`password`, `secret`, `token`, `payload`, `prompt`) are automatically replaced with `[redacted]` before audit event persistence.

---

## 5. Storage Security & Encryption at Rest

### Encrypted Backups
- **Authenticated Encryption**: `create_backup()` and `restore_backup()` support optional application-level envelope encryption using Fernet (AES-128-CBC + HMAC-SHA256).
- **Key Derivation**: Keys are derived from master passphrases using `PBKDF2HMAC` with SHA-256 and 600,000 iterations with a cryptographically random 16-byte salt (`SecureKeyStore`).
- **Fail Closed**: Encrypted backup archives (`.zip.enc`) require the correct key; wrong keys or tampered ciphertexts are rejected before any archive unpacking occurs.
- **Storage Boundaries**: Active tabular sessions are held in RAM and discarded upon browser exit unless exported. Persistent application files (`app.db`, `config.json`) rely on OS-level BitLocker and restricted file permissions (NTFS ACLs / 0600).

---

## 6. Prompt Injection Defenses

Documents ingested for local context search or AI explanation are wrapped as passive data payloads:
- Ingestion applies `wrap_untrusted()`, demarcating boundaries so the local LLM treats file contents as data rather than instructions.
- `looks_like_injection()` heuristic flags common prompt injection patterns.
- Control characters (`\x00-\x1f`) are stripped from extracted document text before indexing.
