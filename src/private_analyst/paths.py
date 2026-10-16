from __future__ import annotations

import os
import tempfile
from pathlib import Path


def default_data_dir() -> Path:
    override = os.environ.get("PRIVATE_ANALYST_DATA_DIR")
    if override:
        return Path(override)
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if local:
        return Path(local) / "PrivateAnalystAI"
    return Path.home() / ".private_analyst_ai"


def default_temp_dir() -> Path:
    return Path(tempfile.gettempdir()) / "private_analyst_ai"


def org_root(data_dir: Path, org_id: str) -> Path:
    safe = "".join(ch for ch in org_id if ch.isalnum() or ch in ("-", "_"))
    if not safe:
        raise ValueError("Invalid organization id")
    return data_dir / "orgs" / safe
