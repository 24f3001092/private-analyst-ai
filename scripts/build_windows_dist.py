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


DIST_IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".git*",
    ".env*",
    "*.db",
    "*.sqlite",
    "*.sqlite3",
    "*.key",
    "master.key",
    "*.log",
    "*.tmp",
    ".coverage",
    ".venv",
    "venv",
)


def run_tests() -> bool:
    print(f"\n[1/4] Running full automated test suite for {PRODUCT_NAME} v{__version__}...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    res = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=str(ROOT), env=env)
    return res.returncode == 0


def build_distribution(
    output_dir: Path | None = None,
    archive_name: str | None = None,
    include_tests: bool = True,
) -> Path:
    dist_root = Path(output_dir) if output_dir else (ROOT / "dist")
    bundle_name = archive_name or f"PrivateAnalystAI-v{__version__}"
    bundle_dir = dist_root / bundle_name

    print(f"\n[2/4] Staging distribution bundle at {bundle_dir}...")
    dist_root.mkdir(parents=True, exist_ok=True)
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Required folders to copy
    folders = ["src", "scripts", "deploy", "docs", "samples", ".streamlit"]
    if include_tests:
        folders.append("tests")

    for folder in folders:
        src = ROOT / folder
        if src.is_dir():
            shutil.copytree(
                src,
                bundle_dir / folder,
                ignore=shutil.ignore_patterns(*DIST_IGNORE_PATTERNS),
            )
        elif folder == ".streamlit":
            raise FileNotFoundError(f"Required configuration directory '{src}' not found.")

    # Required root files to copy
    files = ["app.py", "pyproject.toml", "requirements.txt", "README.md", "LICENSE", "pytest.ini"]
    for file in files:
        src = ROOT / file
        if src.is_file():
            shutil.copy2(src, bundle_dir / file)
        else:
            raise FileNotFoundError(f"Required release file '{src}' not found.")

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
