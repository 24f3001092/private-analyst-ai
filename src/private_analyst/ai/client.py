"""Local inference client. Talks only to a configured localhost runtime."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from private_analyst.config import AIConfig
from private_analyst.exceptions import ModelResponseError, ModelUnavailableError
from private_analyst.security.network import assert_local_only


@dataclass
class ModelStatus:
    available: bool
    runtime: str
    host: str
    model: str
    detail: str
    port: int = 11434

    @property
    def is_ready(self) -> bool:
        return self.available and self.detail == "runtime_reachable"

    @property
    def friendly_status(self) -> str:
        if self.available:
            if self.detail == "runtime_reachable":
                return f"Online & Ready ({self.runtime} with '{self.model}')"
            return f"Runtime Reachable, but model '{self.model}' is not installed"
        return f"Offline / Unreachable (http://{self.host}:{self.port})"

    @property
    def troubleshooting_steps(self) -> list[str]:
        if self.available and self.detail == "runtime_reachable":
            return []
        if self.available and self.detail == "runtime_reachable_model_not_listed":
            return [
                f"Model '{self.model}' is not yet downloaded in your local Ollama environment.",
                f"Open your terminal and run: `ollama pull {self.model}`",
                "Once download completes, return here and test the connection.",
            ]
        return [
            f"Local {self.runtime} runtime is not currently running on loopback (http://{self.host}:{self.port}).",
            "1. Start Ollama locally: Open a terminal and run `ollama serve`.",
            f"2. Pull the model if needed: Run `ollama pull {self.model}`.",
            "3. Click 'Test Model Connection' to verify connectivity.",
            "Privacy guarantee: All AI processing runs strictly on your machine loopback. No data is sent over the internet.",
        ]


def check_local_model(cfg: AIConfig) -> ModelStatus:
    """Helper to check local model runtime connection without catching errors manually."""
    try:
        return LocalModelClient(cfg).status()
    except Exception as exc:
        return ModelStatus(False, cfg.runtime, cfg.host, cfg.model, type(exc).__name__, cfg.port)


def verify_local_model(cfg: AIConfig, model_name: str | None = None) -> dict[str, Any]:
    """Helper to verify local model metadata, architecture, parameters, and license."""
    try:
        return LocalModelClient(cfg).verify_model(model_name)
    except Exception as exc:
        return {
            "verified": False,
            "error": str(exc),
            "detail": type(exc).__name__,
        }


class LocalModelClient:
    def __init__(self, cfg: AIConfig):
        try:
            assert_local_only(cfg.host)
        except PermissionError as exc:
            raise ModelUnavailableError("The model host must be a local loopback address.") from exc
        if cfg.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ModelUnavailableError("The model host must be a local loopback address.")
        self.cfg = cfg
        self.base = f"http://{cfg.host}:{cfg.port}"

    def status(self) -> ModelStatus:
        try:
            with urllib.request.urlopen(f"{self.base}/api/tags", timeout=2) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            names = [str(item.get("name", "")) for item in payload.get("models", [])]
            present = any(self.cfg.model in name for name in names) if names else False
            detail = "runtime_reachable" if present else "runtime_reachable_model_not_listed"
            return ModelStatus(True, self.cfg.runtime, self.cfg.host, self.cfg.model, detail, self.cfg.port)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return ModelStatus(False, self.cfg.runtime, self.cfg.host, self.cfg.model, type(exc).__name__, self.cfg.port)

    def verify_model(self, model_name: str | None = None) -> dict[str, Any]:
        """Verify local model metadata, architecture, quantization, and license via loopback API."""
        target = model_name or self.cfg.model
        body = json.dumps({"name": target}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}/api/show",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=min(self.cfg.timeout_seconds, 5)) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return {
                "verified": False,
                "model": target,
                "error": f"Local model runtime unreachable: {exc}",
                "detail": type(exc).__name__,
            }
        except json.JSONDecodeError as exc:
            return {
                "verified": False,
                "model": target,
                "error": f"Invalid JSON response: {exc}",
                "detail": "JSONDecodeError",
            }

        details = payload.get("details", {})
        return {
            "verified": True,
            "model": target,
            "format": details.get("format", "unknown"),
            "family": details.get("family", "unknown"),
            "parameter_size": details.get("parameter_size", "unknown"),
            "quantization_level": details.get("quantization_level", "unknown"),
            "modified_at": payload.get("modified_at", ""),
            "license": payload.get("license", "Local open-weight license (refer to model card)"),
            "parameters": payload.get("parameters", ""),
        }

    def generate(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.cfg.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.cfg.temperature,
                    "num_predict": self.cfg.max_tokens,
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.cfg.timeout_seconds) as resp:
                payload: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ModelUnavailableError("The local model runtime did not respond.") from exc
        except json.JSONDecodeError as exc:
            raise ModelResponseError("The local model returned data that was not valid JSON.") from exc
        text = payload.get("response")
        if not isinstance(text, str) or not text.strip():
            raise ModelResponseError("The local model returned an empty or malformed response.")
        return text.strip()
