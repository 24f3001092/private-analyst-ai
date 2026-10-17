"""Automated tests for the Windows offline distribution packaging builder."""

from __future__ import annotations

import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_windows_dist import (
    DIST_IGNORE_PATTERNS,
    ROOT,
    build_distribution,
)


def test_build_distribution_layout_and_contents(tmp_path: Path) -> None:
    """Verify that the packaging builder produces a valid archive with all required runtime files."""
    out_dir = tmp_path / "dist_test"
    archive = build_distribution(output_dir=out_dir, archive_name="TestReleaseBundle", include_tests=True)

    assert archive.is_file()
    assert archive.name.endswith(".zip")
    assert archive.stat().st_size > 10_000

    with zipfile.ZipFile(archive, "r") as z:
        names = z.namelist()

        # 1. Critical .streamlit configuration MUST be present (SEC-STG3-02)
        assert any(n.endswith(".streamlit/config.toml") for n in names)

        # 2. Application entrypoint and descriptors
        assert any(n.endswith("app.py") for n in names)
        assert any(n.endswith("requirements.txt") for n in names)
        assert any(n.endswith("pyproject.toml") for n in names)
        assert any(n.endswith("README.md") for n in names)
        assert any(n.endswith("LICENSE") for n in names)
        assert any(n.endswith("pytest.ini") for n in names)

        # 3. Core source code and modules
        assert any("src/private_analyst/version.py" in n for n in names)
        assert any("src/private_analyst/analytics/sql_console.py" in n for n in names)
        assert any("src/private_analyst/audit/logger.py" in n for n in names)

        # 4. Windows runner scripts
        assert any(n.endswith("scripts/run_desktop.bat") for n in names)
        assert any(n.endswith("scripts/run_server.bat") for n in names)

        # 5. Documentation and sample data
        assert any(n.endswith("docs/OFFLINE_INSTALL.md") for n in names)
        assert any(n.endswith("docs/SECURITY_PRIVACY.md") for n in names)
        assert any(n.endswith("samples/sales_known.csv") for n in names)

        # 6. Verification tests included for on-premise validation
        assert any(n.endswith("tests/test_reports.py") for n in names)


def test_build_distribution_excludes_sensitive_files(tmp_path: Path) -> None:
    """Verify that the packaging builder strictly excludes secrets, databases, keys, and caches."""
    out_dir = tmp_path / "dist_test"
    archive = build_distribution(output_dir=out_dir, archive_name="TestExclusionBundle", include_tests=False)

    with zipfile.ZipFile(archive, "r") as z:
        names = z.namelist()

        # Must not contain version control internals
        assert not any(".git/" in n or n.endswith(".git") for n in names)

        # Must not contain environment files or secrets
        assert not any(".env" in n for n in names)

        # Must not contain SQLite databases, DuckDB databases, or key stores
        assert not any(n.endswith(".db") or n.endswith(".sqlite") or n.endswith(".sqlite3") or n.endswith(".duckdb") for n in names)
        assert not any(n.endswith(".key") or "master.key" in n for n in names)

        # Must not contain bytecode caches or test caches
        assert not any("__pycache__" in n or n.endswith(".pyc") or ".pytest_cache" in n for n in names)


def test_build_distribution_fails_on_missing_required_file(tmp_path: Path) -> None:
    """Verify that packaging fails safely if a required release file is missing."""
    out_dir = tmp_path / "dist_test"
    with patch("scripts.build_windows_dist.ROOT", tmp_path):
        # tmp_path has no files; must raise FileNotFoundError rather than producing broken archive
        with pytest.raises(FileNotFoundError):
            build_distribution(output_dir=out_dir, archive_name="BrokenBundle")


def test_clean_extraction_and_config_validation(tmp_path: Path) -> None:
    """Verify that the generated archive can be cleanly extracted and its configuration read."""
    dist_dir = tmp_path / "dist"
    extract_dir = tmp_path / "extracted"

    archive = build_distribution(output_dir=dist_dir, archive_name="CleanExtractBundle")

    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(extract_dir)

    bundle_root = extract_dir / "CleanExtractBundle"
    assert bundle_root.is_dir()
    assert (bundle_root / "app.py").is_file()
    assert (bundle_root / "requirements.txt").is_file()

    # Verify extracted .streamlit/config.toml properties
    config_file = bundle_root / ".streamlit" / "config.toml"
    assert config_file.is_file()
    content = config_file.read_text(encoding="utf-8")
    assert "gatherUsageStats = false" in content
    assert 'address = "127.0.0.1"' in content
    assert "enableXsrfProtection = true" in content
