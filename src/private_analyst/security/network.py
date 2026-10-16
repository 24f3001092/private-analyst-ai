"""Helpers for verifying that customer data is not sent off-box."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass

ALLOWED_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass
class NetworkPolicy:
    allow_loopback: bool = True
    allow_outbound: bool = False


def is_local_host(host: str) -> bool:
    return host.lower() in ALLOWED_LOCAL_HOSTS


def assert_local_only(host: str) -> None:
    if not is_local_host(host):
        raise PermissionError("Outbound network destinations are not allowed for AI inference.")


class LocalOnlySocket(socket.socket):
    """Test helper that blocks non-loopback TCP connects."""

    def connect(self, address):  # type: ignore[override]
        host = address[0] if isinstance(address, tuple) else str(address)
        if not is_local_host(str(host)):
            raise OSError("blocked non-local connect")
        return super().connect(address)


def streamlit_telemetry_disabled() -> dict:
    return {
        "browser.gatherUsageStats": False,
        "telemetry_enabled": False,
        "crash_reporter": False,
    }


def dump_policy() -> str:
    return json.dumps(
        {
            "customer_data_destinations": "local_disk_and_optional_localhost_model",
            "developer_telemetry": False,
            "cloud_ai": False,
            "external_vector_db": False,
        },
        indent=2,
    )
