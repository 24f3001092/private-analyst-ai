"""Milestone 1 helper: print local runtime facts. Does not send data anywhere."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from private_analyst.version import PRODUCT_NAME, __version__


def main() -> None:
    print(f"{PRODUCT_NAME} {__version__}")
    print("python", sys.version)
    print("executable", sys.executable)
    print("platform", platform.platform())
    print("cwd", os.getcwd())
    for name in (
        "pandas",
        "numpy",
        "openpyxl",
        "plotly",
        "streamlit",
        "pytest",
        "bcrypt",
        "sklearn",
        "duckdb",
        "cryptography",
        "pypdf",
        "reportlab",
    ):
        try:
            mod = __import__(name)
            print(f"{name}: {getattr(mod, '__version__', 'unknown')}")
        except ImportError:
            print(f"{name}: NOT INSTALLED")
    print("telemetry_env", os.environ.get("STREAMLIT_BROWSER_GATHER_USAGE_STATS"))


if __name__ == "__main__":
    main()
