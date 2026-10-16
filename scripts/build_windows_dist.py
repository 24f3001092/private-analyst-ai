"""Packaging and release distribution script for Private Analyst AI.

Prepares a clean, commercial distribution bundle with all tests verified.
Run from repository root: python scripts/build_windows_dist.py
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from private_analyst.version import PRODUCT_NAME, __version__


def run_tests() -> bool:
    print(f"\n[1/4] Running full automated test suite for {PRODUCT_NAME} v{__version__}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    res = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT), env=env)
    return res.returncode == 0


def build_distribution() -> Path:
    dist_root = ROOT / "dist"
    bundle_name = f"PrivateAnalystAI-v{__version__}"
    bundle_dir = dist_root / bundle_name

    print(f"\n[2/4] Cleaning previous builds and staging bundle at {bundle_dir}...")
    if dist_root.exists():
        shutil.rmtree(dist_root, ignore_errors=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Folders to copy
    folders = ["src", "scripts", "deploy", "docs", "samples"]
    for folder in folders:
        src = ROOT / folder
        if src.is_dir():
            shutil.copytree(
                src,
                bundle_dir / folder,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache"),
            )

    # Root files to copy
    files = ["app.py", "pyproject.toml", "requirements.txt", "README.md", "LICENSE", "pytest.ini"]
    for file in files:
        src = ROOT / file
        if src.is_file():
            shutil.copy2(src, bundle_dir / file)

    print(f"\n[3/4] Creating distribution zip archive...")
    archive_base = dist_root / f"{bundle_name}-windows-offline"
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=dist_root, base_dir=bundle_name))

    print(f"\n[4/4] Release build completed successfully!")
    print(f"Archive path: {archive_path}")
    print(f"Archive size: {archive_path.stat().st_size:,} bytes")
    return archive_path


def main() -> None:
    if not run_tests():
        print("\nERROR: Cannot build distribution bundle because test suite failed!")
        sys.exit(1)
    archive = build_distribution()
    print(f"\n[SUCCESS] Commercial distribution package created at: {archive}")


if __name__ == "__main__":
    main()
