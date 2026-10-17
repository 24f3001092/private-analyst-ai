"""Regression tests for encrypted storage, backups, and key handling."""

from __future__ import annotations

from pathlib import Path
import pytest

from private_analyst.exceptions import PrivateAnalystError
from private_analyst.security.encryption import (
    SecureKeyStore,
    decrypt_bytes,
    decrypt_file,
    derive_key_from_password,
    encrypt_bytes,
    encrypt_file,
    generate_key,
)
from private_analyst.storage.backup import create_backup, restore_backup


def test_encrypted_backup_roundtrip(tmp_path: Path):
    """Encrypted backup creates a .zip.enc file and restores all files accurately with valid key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "sample.csv").write_text("id,val\n1,100\n2,200\n", encoding="utf-8")
    (data_dir / "nested").mkdir()
    (data_dir / "nested" / "doc.txt").write_text("confidential business notes", encoding="utf-8")

    dest_dir = tmp_path / "backups"
    key = generate_key()

    archive_path = create_backup(data_dir, dest_dir, key=key)
    assert archive_path.is_file()
    assert archive_path.name.endswith(".zip.enc")

    # Verify content is encrypted (does not contain zip magic bytes PK\x03\x04)
    raw_bytes = archive_path.read_bytes()
    assert not raw_bytes.startswith(b"PK\x03\x04")
    assert b"confidential" not in raw_bytes

    restore_dir = tmp_path / "restored"
    restore_backup(archive_path, restore_dir, key=key)

    assert (restore_dir / "sample.csv").read_text(encoding="utf-8") == "id,val\n1,100\n2,200\n"
    assert (restore_dir / "nested" / "doc.txt").read_text(encoding="utf-8") == "confidential business notes"


def test_encrypted_backup_wrong_key_fails(tmp_path: Path):
    """Restoring an encrypted backup with an incorrect key must fail closed with PrivateAnalystError."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("secret contents", encoding="utf-8")

    key1 = generate_key()
    key2 = generate_key()

    dest_dir = tmp_path / "backups"
    archive_path = create_backup(data_dir, dest_dir, key=key1)

    restore_dir = tmp_path / "restored"
    with pytest.raises(PrivateAnalystError, match="Decryption failed"):
        restore_backup(archive_path, restore_dir, key=key2)

    # Ensure no decrypted data was unpacked
    assert not (restore_dir / "file.txt").exists()


def test_encrypted_backup_tampering_detected(tmp_path: Path):
    """Corrupting/tampering with the ciphertext backup must be detected and rejected."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("secret contents", encoding="utf-8")

    key = generate_key()
    dest_dir = tmp_path / "backups"
    archive_path = create_backup(data_dir, dest_dir, key=key)

    # Tamper with ciphertext by altering bytes in the payload
    raw = bytearray(archive_path.read_bytes())
    raw[30] ^= 0xFF
    archive_path.write_bytes(bytes(raw))

    restore_dir = tmp_path / "restored"
    with pytest.raises(PrivateAnalystError, match="Decryption failed"):
        restore_backup(archive_path, restore_dir, key=key)


def test_encrypted_backup_requires_key(tmp_path: Path):
    """Attempting to restore an encrypted archive without providing a key must raise PrivateAnalystError."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "file.txt").write_text("secret contents", encoding="utf-8")

    key = generate_key()
    dest_dir = tmp_path / "backups"
    archive_path = create_backup(data_dir, dest_dir, key=key)

    restore_dir = tmp_path / "restored"
    with pytest.raises(PrivateAnalystError, match="Encryption key is required"):
        restore_backup(archive_path, restore_dir, key=None)


def test_unencrypted_backup_backwards_compatibility(tmp_path: Path):
    """Backups created without a key remain standard zip files and restore properly."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "records.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    dest_dir = tmp_path / "backups"
    archive_path = create_backup(data_dir, dest_dir, key=None)

    assert archive_path.is_file()
    assert archive_path.name.endswith(".zip")
    assert archive_path.read_bytes().startswith(b"PK\x03\x04")

    restore_dir = tmp_path / "restored"
    restore_backup(archive_path, restore_dir)
    assert (restore_dir / "records.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_restore_nonexistent_archive_raises(tmp_path: Path):
    """Restoring from a non-existent archive raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        restore_backup(tmp_path / "missing.zip", tmp_path / "dest")


def test_restore_nonempty_destination_raises(tmp_path: Path):
    """Restoring into a non-empty directory raises FileExistsError to prevent accidental overwrites."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "item.txt").write_text("hello", encoding="utf-8")

    archive_path = create_backup(data_dir, tmp_path / "backups")

    dest_dir = tmp_path / "dest"
    dest_dir.mkdir()
    (dest_dir / "pre_existing.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError):
        restore_backup(archive_path, dest_dir)


def test_encrypt_decrypt_file_error_handling(tmp_path: Path):
    """File encryption and decryption utilities fail cleanly on missing or bad paths."""
    key = generate_key()
    with pytest.raises(PrivateAnalystError, match="Source file does not exist"):
        encrypt_file(tmp_path / "nonexistent.txt", tmp_path / "out.enc", key)

    with pytest.raises(PrivateAnalystError, match="Source file does not exist"):
        decrypt_file(tmp_path / "nonexistent.enc", tmp_path / "out.txt", key)


def test_keystore_derived_key_integration(tmp_path: Path):
    """SecureKeyStore can be unlocked with master password and its key used for encrypted backups."""
    key_file = tmp_path / "keystore.json"
    keystore = SecureKeyStore(key_file)
    assert not keystore.is_initialized()

    password = "MasterPasswordSecure2026!"
    keystore.initialize(password)
    assert keystore.is_initialized()

    # Unlocking with correct password succeeds
    master_key = keystore.unlock(password)
    assert len(master_key) == 44  # base64 encoded 32 bytes

    # Unlocking with wrong password fails
    with pytest.raises(PrivateAnalystError, match="Decryption failed"):
        keystore.unlock("WrongPassword123!")

    # Master key can encrypt and restore backups
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.txt").write_text("vault content", encoding="utf-8")

    archive = create_backup(data_dir, tmp_path / "enc_backups", key=master_key)
    restored = restore_backup(archive, tmp_path / "vault_restored", key=master_key)
    assert (restored / "test.txt").read_text(encoding="utf-8") == "vault content"
