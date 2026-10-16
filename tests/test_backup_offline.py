from __future__ import annotations

import json
from pathlib import Path

import pytest

from private_analyst.security.network import LocalOnlySocket, assert_local_only, dump_policy, is_local_host
from private_analyst.storage.backup import create_backup, restore_backup
from private_analyst.config import load_config, save_config


def test_backup_and_restore(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "note.txt").write_text("hello", encoding="utf-8")
    archive = create_backup(data, tmp_path / "backups")
    dest = tmp_path / "restored"
    restore_backup(archive, dest)
    assert (dest / "note.txt").read_text(encoding="utf-8") == "hello"


def test_restore_refuses_nonempty(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "note.txt").write_text("hello", encoding="utf-8")
    archive = create_backup(data, tmp_path / "backups")
    dest = tmp_path / "restored"
    dest.mkdir()
    (dest / "existing.txt").write_text("nope", encoding="utf-8")
    with pytest.raises(FileExistsError):
        restore_backup(archive, dest)


def test_local_host_policy() -> None:
    assert is_local_host("127.0.0.1")
    with pytest.raises(PermissionError):
        assert_local_only("api.openai.com")
    policy = json.loads(dump_policy())
    assert policy["cloud_ai"] is False
    assert policy["developer_telemetry"] is False


def test_config_forces_no_telemetry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIVATE_ANALYST_DATA_DIR", str(tmp_path))
    cfg = load_config()
    cfg.telemetry_enabled = True
    save_config(cfg, tmp_path / "config.json")
    loaded = load_config(tmp_path / "config.json")
    assert loaded.telemetry_enabled is False
    assert loaded.ai.allow_network_beyond_localhost is False


@pytest.mark.network
def test_non_local_connect_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    import socket

    monkeypatch.setattr(socket, "socket", LocalOnlySocket)
    with pytest.raises(OSError):
        socket.create_connection(("93.184.216.34", 80), timeout=1)
