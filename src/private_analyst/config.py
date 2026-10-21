"""Local configuration with strict precedence and non-negotiable security invariants."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from private_analyst.exceptions import ValidationError
from private_analyst.paths import default_data_dir, default_temp_dir

ALLOWED_EDITIONS = frozenset({"desktop", "server"})
ALLOWED_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


@dataclass
class Limits:
    max_upload_bytes: int = 50 * 1024 * 1024
    max_rows: int = 500_000
    max_columns: int = 200
    max_pdf_pages: int = 200
    max_query_seconds: int = 30
    preview_rows: int = 25


@dataclass
class Retention:
    keep_uploads: bool = False
    keep_reports_days: int = 30
    keep_audit_days: int = 90
    auto_delete_temp: bool = True


@dataclass
class AIConfig:
    runtime: str = "ollama"
    host: str = "127.0.0.1"
    port: int = 11434
    model: str = "llama3.2"
    timeout_seconds: int = 60
    allow_network_beyond_localhost: bool = False
    temperature: float = 0.1
    max_tokens: int = 800


@dataclass
class ServerConfig:
    edition: str = "desktop"  # desktop | server
    bind_host: str = "127.0.0.1"
    bind_port: int = 8501
    require_auth: bool = False
    require_https: bool = False
    session_idle_minutes: int = 30


@dataclass
class Branding:
    product_name: str = "Private Analyst AI"
    organization_name: str = "Local installation"
    support_contact: str = ""


@dataclass
class AppConfig:
    branding: Branding = field(default_factory=Branding)
    limits: Limits = field(default_factory=Limits)
    retention: Retention = field(default_factory=Retention)
    ai: AIConfig = field(default_factory=AIConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    data_dir: str = ""
    enable_ocr: bool = True
    enable_forecasting: bool = True
    enable_ml: bool = True
    telemetry_enabled: bool = False  # always false; field exists so audits can assert it

    def resolved_data_dir(self) -> Path:
        if self.data_dir:
            return Path(self.data_dir)
        return default_data_dir()


def _merge(dc: Any, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if not hasattr(dc, key):
            continue
        current = getattr(dc, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(dc, key, value)


def load_config(path: Path | None = None) -> AppConfig:
    """Load application configuration using strict precedence:
    
    1. Base defaults (dataclass initializers)
    2. Configuration file (config.json)
    3. Environment variables (PRIVATE_ANALYST_*)
    4. Mandatory security invariants (applied last, non-negotiable):
       - If server edition: require_auth MUST ALWAYS BE True
       - Edition must be 'desktop' or 'server'
       - Telemetry is hard-disabled (False)
       - AI host must be loopback (127.0.0.1 or localhost)
       - AI allow_network_beyond_localhost is False
    """
    cfg = AppConfig()

    # 1. Discover data directory override if specified in env
    data_dir_env = os.environ.get("PRIVATE_ANALYST_DATA_DIR", "").strip()
    if data_dir_env:
        cfg.data_dir = data_dir_env

    # 2. Merge config file if it exists
    config_path = path or (cfg.resolved_data_dir() / "config.json")
    if config_path.is_file():
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValidationError("Configuration file must contain a JSON object.")
            _merge(cfg, raw)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"Configuration file is not valid JSON: {exc}") from exc

    # 3. Environment variables take precedence over config file
    edition_env = os.environ.get("PRIVATE_ANALYST_EDITION", "").strip().lower()
    if edition_env:
        cfg.server.edition = edition_env

    if data_dir_env:
        cfg.data_dir = data_dir_env

    server_host_env = os.environ.get("PRIVATE_ANALYST_SERVER_HOST", "").strip()
    if server_host_env:
        cfg.server.bind_host = server_host_env

    server_port_env = os.environ.get("PRIVATE_ANALYST_SERVER_PORT", "").strip()
    if server_port_env:
        try:
            cfg.server.bind_port = int(server_port_env)
        except ValueError:
            raise ValidationError("PRIVATE_ANALYST_SERVER_PORT must be an integer.")

    ai_host_env = os.environ.get("PRIVATE_ANALYST_AI_HOST", "").strip()
    if ai_host_env:
        cfg.ai.host = ai_host_env

    ai_model_env = os.environ.get("PRIVATE_ANALYST_AI_MODEL", "").strip()
    if ai_model_env:
        cfg.ai.model = ai_model_env

    ai_port_env = os.environ.get("PRIVATE_ANALYST_AI_PORT", "").strip()
    if ai_port_env:
        try:
            cfg.ai.port = int(ai_port_env)
        except ValueError:
            raise ValidationError("PRIVATE_ANALYST_AI_PORT must be an integer.")

    # 4. Mandatory security invariants (applied LAST, cannot be bypassed)
    cfg.server.edition = cfg.server.edition.strip().lower()
    if cfg.server.edition not in ALLOWED_EDITIONS:
        raise ValidationError(
            f"Invalid edition: '{cfg.server.edition}'. Must be 'desktop' or 'server'."
        )

    # In server mode, authentication is NON-NEGOTIABLE.
    # It cannot be disabled by config file or environment.
    if cfg.server.edition == "server":
        cfg.server.require_auth = True
    elif not cfg.server.require_auth:
        # In desktop mode without authentication, bind_host MUST be loopback only
        if cfg.server.bind_host.strip().lower() not in ALLOWED_LOOPBACK_HOSTS:
            raise ValidationError(
                f"Non-loopback bind host '{cfg.server.bind_host}' is not allowed in desktop mode without authentication. "
                "Switch to server mode or enable require_auth."
            )

    # Telemetry and external AI network access are permanently disabled
    cfg.telemetry_enabled = False
    cfg.ai.allow_network_beyond_localhost = False

    # AI host must be loopback only
    if cfg.ai.host.lower() not in ALLOWED_LOOPBACK_HOSTS:
        raise ValidationError(
            f"AI inference host '{cfg.ai.host}' is not allowed. Must be a local loopback address."
        )

    # Validate model identifier format
    cfg.ai.model = cfg.ai.model.strip()
    if not cfg.ai.model:
        raise ValidationError("AI model name cannot be empty.")
    if any(c in cfg.ai.model for c in (" ", "\t", "\n", "\r", ";", "&", "|", "`", "$")):
        raise ValidationError(f"Invalid AI model name '{cfg.ai.model}'. Control characters and whitespace are not allowed.")

    os.makedirs(cfg.resolved_data_dir(), exist_ok=True)
    os.makedirs(default_temp_dir(), exist_ok=True)
    return cfg


def save_config(cfg: AppConfig, path: Path | None = None) -> Path:
    path = path or (cfg.resolved_data_dir() / "config.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    payload["telemetry_enabled"] = False
    payload["ai"]["allow_network_beyond_localhost"] = False
    if payload.get("server", {}).get("edition") == "server":
        payload["server"]["require_auth"] = True
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
