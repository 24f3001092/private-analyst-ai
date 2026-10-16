from __future__ import annotations

from pathlib import Path

from private_analyst.config import Limits
from private_analyst.exceptions import FileTooLargeError, UnsupportedFormatError, ValidationError

ALLOWED_TABULAR = {".csv", ".xlsx"}
ALLOWED_DOCUMENTS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
ALLOWED_ALL = ALLOWED_TABULAR | ALLOWED_DOCUMENTS


def _suffix(name: str) -> str:
    return Path(name).suffix.lower()


def validate_upload(
    filename: str,
    size_bytes: int,
    limits: Limits,
    *,
    kind: str = "any",
) -> str:
    if not filename or filename.strip() != filename:
        raise ValidationError("File name is missing or contains leading/trailing spaces.")
    if ".." in Path(filename).parts or filename.startswith(("/", "\\")):
        raise ValidationError("File name is not allowed.")
    suffix = _suffix(filename)
    allowed = ALLOWED_ALL if kind == "any" else (
        ALLOWED_TABULAR if kind == "tabular" else ALLOWED_DOCUMENTS
    )
    if suffix not in allowed:
        raise UnsupportedFormatError(f"Unsupported file type: {suffix or '(none)'}")
    if size_bytes < 0:
        raise ValidationError("Invalid file size.")
    if size_bytes > limits.max_upload_bytes:
        raise FileTooLargeError(
            f"File exceeds the configured size limit of {limits.max_upload_bytes} bytes."
        )
    if size_bytes == 0:
        raise ValidationError("File is empty.")
    return suffix
