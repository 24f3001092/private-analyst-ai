"""Local inference client. Talks only to a configured localhost runtime."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from private_analyst.config import AIConfig
from private_analyst.exceptions import ModelResponseError, ModelUnavailableError


@dataclass
class ModelStatus:
    available: bool
    runtime: str
    host: str
    model: str
    detail: str


class LocalModelClient:
    def __init__(self, cfg: AIConfig):
        if cfg.host not in {"127.0.0.1", "localhost"}:
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
            return ModelStatus(True, self.cfg.runtime, self.cfg.host, self.cfg.model, detail)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            return ModelStatus(False, self.cfg.runtime, self.cfg.host, self.cfg.model, type(exc).__name__)

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
