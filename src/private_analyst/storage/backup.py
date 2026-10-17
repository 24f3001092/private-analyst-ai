import secrets
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from private_analyst.exceptions import PrivateAnalystError
from private_analyst.security.encryption import decrypt_file, encrypt_file


def create_backup(data_dir: Path, dest_dir: Path, key: bytes | None = None) -> Path:
    """Create a backup of data_dir into dest_dir, optionally encrypted with a Fernet key.

    Args:
        data_dir: Source directory to back up.
        dest_dir: Destination directory where the backup file will be created.
        key: Optional Fernet key. If provided, the backup is encrypted as a .zip.enc file.

    Returns:
        Path to created backup file (.zip or .zip.enc).
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_base = dest_dir / f"private-analyst-backup-{stamp}"
    raw_archive = Path(shutil.make_archive(str(archive_base), "zip", root_dir=data_dir))

    if key is None:
        return raw_archive

    enc_path = dest_dir / f"{raw_archive.name}.enc"
    try:
        encrypt_file(raw_archive, enc_path, key)
    finally:
        # Securely remove unencrypted temporary zip archive
        if raw_archive.is_file():
            raw_archive.unlink()

    return enc_path


def restore_backup(archive: Path, dest_dir: Path, key: bytes | None = None) -> Path:
    """Restore a backup archive into dest_dir, decrypting first if encrypted.

    Args:
        archive: Path to backup file (.zip or .zip.enc).
        dest_dir: Destination directory to unpack into (must be empty or not yet exist).
        key: Optional Fernet key. Required if archive is encrypted (.enc).

    Returns:
        Path to dest_dir.

    Raises:
        FileNotFoundError: If archive does not exist.
        FileExistsError: If dest_dir exists and is non-empty.
        PrivateAnalystError: If decryption key is missing or invalid, or data is corrupted.
    """
    archive_path = Path(archive)
    dest_dir = Path(dest_dir)

    if not archive_path.is_file():
        raise FileNotFoundError(f"Backup archive not found: {archive_path}")

    if dest_dir.exists() and any(dest_dir.iterdir()):
        raise FileExistsError("Restore destination is not empty.")

    is_encrypted = archive_path.name.endswith(".enc")

    if is_encrypted:
        if key is None:
            raise PrivateAnalystError("Encryption key is required to restore an encrypted backup archive.")

        # Decrypt to a temporary directory
        temp_dir = dest_dir.parent / f"_temp_restore_{secrets.token_hex(8)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        # Strip .enc from filename
        decrypted_zip = temp_dir / archive_path.stem
        try:
            decrypt_file(archive_path, decrypted_zip, key)
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.unpack_archive(str(decrypted_zip), extract_dir=dest_dir)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive_path), extract_dir=dest_dir)
    return dest_dir

