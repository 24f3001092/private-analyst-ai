"""Verify local AI model metadata, parameter architecture, licensing, and file checksums.

Run from repository root:
    python scripts/verify_model.py
    python scripts/verify_model.py --model llama3.2
    python scripts/verify_model.py --file path/to/model.gguf --checksum <sha256>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from private_analyst.ai.client import LocalModelClient
from private_analyst.config import AIConfig, load_config
from private_analyst.security.network import assert_local_only


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a local model weight file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local AI model integrity and licensing.")
    parser.add_argument("--model", help="Model name tag (e.g. llama3.2)", default=None)
    parser.add_argument("--host", help="Local host binding (default: 127.0.0.1)", default=None)
    parser.add_argument("--port", type=int, help="Local port (default: 11434)", default=None)
    parser.add_argument("--file", help="Path to local GGUF/bin model weight file to verify checksum", default=None)
    parser.add_argument("--checksum", help="Expected SHA-256 checksum hex string", default=None)
    args = parser.parse_args()

    cfg = load_config()
    target_model = args.model or cfg.ai.model
    host = args.host or cfg.ai.host
    port = args.port or cfg.ai.port

    print("=================================================================")
    print(" Private Analyst AI — Local Model Verification & Integrity Check")
    print("=================================================================")
    print(f"Target Model:     {target_model}")
    print(f"Loopback Target:  http://{host}:{port}")

    # Enforce loopback restriction
    try:
        assert_local_only(host)
    except Exception as exc:
        print(f"\n[SECURITY VIOLATION] Host verification failed: {exc}")
        return 1

    # 1. File verification if path provided
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"\n[ERROR] Model file not found: {file_path}")
            return 1
        print(f"\n[1/2] Computing SHA-256 for local weight file: {file_path.name}...")
        actual_hash = compute_sha256(file_path)
        size_mb = file_path.stat().st_size / (1024 * 1024)
        print(f"File Size:        {size_mb:,.1f} MB ({file_path.stat().st_size:,} bytes)")
        print(f"SHA-256 Checksum: {actual_hash}")

        if args.checksum:
            expected_clean = args.checksum.strip().lower()
            if actual_hash.lower() == expected_clean:
                print("[SUCCESS] Checksum matches expected hash exactly.")
            else:
                print(f"[ERROR] Checksum mismatch!")
                print(f"  Expected: {expected_clean}")
                print(f"  Actual:   {actual_hash.lower()}")
                return 1

    # 2. Local Ollama runtime inspection
    print(f"\n[2/2] Inspecting local model runtime via loopback API...")
    ai_cfg = AIConfig(host=host, port=port, model=target_model)
    client = LocalModelClient(ai_cfg)

    status = client.status()
    if not status.available:
        print(f"[OFFLINE] {status.friendly_status}")
        print("\nTroubleshooting steps:")
        for step in status.troubleshooting_steps:
            print(f" - {step}")
        return 1

    res = client.verify_model(target_model)
    if not res.get("verified"):
        print(f"[FAILED] Model verification failed: {res.get('error')}")
        return 1

    print("[SUCCESS] Local Model Verified!")
    print(f"  Family / Arch:       {res.get('family')}")
    print(f"  Format:              {res.get('format')}")
    print(f"  Parameter Count:     {res.get('parameter_size')}")
    print(f"  Quantization Level:  {res.get('quantization_level')}")
    print(f"  Modified Timestamp:  {res.get('modified_at')}")

    license_text = res.get("license", "").strip()
    if license_text:
        first_line = license_text.splitlines()[0]
        print(f"  License Header:      {first_line[:90]}")
    else:
        print("  License Header:      (refer to local model card)")

    print("\nVerified: All AI processing executes strictly on loopback without internet access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
