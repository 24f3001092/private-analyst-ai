from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time

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
from private_analyst.security.sessions import SessionInfo, SessionManager


def test_encrypt_decrypt_roundtrip() -> None:
    """Verify data survives encrypt/decrypt roundtrip."""
    key = generate_key()
    original_data = b"Confidential business metrics for Q3: revenue=$1,200,000"
    token = encrypt_bytes(original_data, key)
    assert token != original_data

    decrypted = decrypt_bytes(token, key)
    assert decrypted == original_data

    # Also test empty bytes
    empty_token = encrypt_bytes(b"", key)
    assert decrypt_bytes(empty_token, key) == b""


def test_wrong_key_fails() -> None:
    """Verify wrong key raises PrivateAnalystError."""
    key1 = generate_key()
    key2 = generate_key()
    data = b"Sensitive customer payload"

    token = encrypt_bytes(data, key1)

    with pytest.raises(PrivateAnalystError):
        decrypt_bytes(token, key2)

    # Corrupted key
    with pytest.raises(PrivateAnalystError):
        decrypt_bytes(token, b"invalid-key-bytes")


def test_file_encrypt_decrypt(tmp_path: Path) -> None:
    """Verify file encryption and decryption roundtrip."""
    source_file = tmp_path / "plaintext.csv"
    encrypted_file = tmp_path / "encrypted.bin"
    decrypted_file = tmp_path / "decrypted.csv"

    file_content = b"col1,col2,col3\nval1,val2,val3\n100,200,300\n"
    source_file.write_bytes(file_content)

    key = generate_key()
    result_enc = encrypt_file(source_file, encrypted_file, key)
    assert result_enc == encrypted_file
    assert encrypted_file.is_file()
    assert encrypted_file.read_bytes() != file_content

    result_dec = decrypt_file(encrypted_file, decrypted_file, key)
    assert result_dec == decrypted_file
    assert decrypted_file.is_file()
    assert decrypted_file.read_bytes() == file_content


def test_key_derivation_same_salt() -> None:
    """Verify same password + salt produces identical key."""
    password = "CorrectHorseBatteryStaple123!"
    key1, salt1 = derive_key_from_password(password)
    key2, salt2 = derive_key_from_password(password, salt=salt1)

    assert salt1 == salt2
    assert key1 == key2


def test_key_derivation_different_salt() -> None:
    """Verify different salt produces different key."""
    password = "CorrectHorseBatteryStaple123!"
    key1, salt1 = derive_key_from_password(password)
    key2, salt2 = derive_key_from_password(password)

    assert salt1 != salt2
    assert key1 != key2


def test_secure_key_store_roundtrip(tmp_path: Path) -> None:
    """Initialize, unlock, and verify same master key across unlock operations."""
    key_file = tmp_path / "subfolder" / "master.keystore"
    store = SecureKeyStore(key_file)

    assert not store.is_initialized()

    password = "MasterVaultPassword999!"
    store.initialize(password)

    assert store.is_initialized()
    assert key_file.is_file()

    master_key_1 = store.unlock(password)
    master_key_2 = store.unlock(password)

    assert master_key_1 == master_key_2
    assert isinstance(master_key_1, bytes)

    # Verify that the unlocked key is a valid Fernet key that works
    test_payload = b"Master key encrypted content"
    enc = encrypt_bytes(test_payload, master_key_1)
    dec = decrypt_bytes(enc, master_key_2)
    assert dec == test_payload


def test_secure_key_store_wrong_password(tmp_path: Path) -> None:
    """Verify wrong password raises error when unlocking key store."""
    key_file = tmp_path / "master.keystore"
    store = SecureKeyStore(key_file)
    store.initialize("CorrectMasterPassword123!")

    with pytest.raises(PrivateAnalystError):
        store.unlock("IncorrectPassword")


def test_session_create_validate() -> None:
    """Create session and validate it."""
    mgr = SessionManager(idle_timeout_minutes=30)
    token = mgr.create_session(user_id=101, org_id="acme-corp")

    assert isinstance(token, str)
    assert len(token) > 20

    session = mgr.validate_session(token)
    assert session is not None
    assert isinstance(session, SessionInfo)
    assert session.user_id == 101
    assert session.org_id == "acme-corp"
    assert session.token == token
    assert session.created_at <= datetime.now(timezone.utc)
    assert session.last_activity <= datetime.now(timezone.utc)


def test_session_expiry() -> None:
    """Create with 0-minute timeout, verify expired immediately."""
    mgr = SessionManager(idle_timeout_minutes=0)
    token = mgr.create_session(user_id=1, org_id="acme")

    assert mgr.validate_session(token) is None


def test_session_destroy() -> None:
    """Create then destroy, verify validation returns None."""
    mgr = SessionManager(idle_timeout_minutes=30)
    token = mgr.create_session(user_id=202, org_id="beta-corp")

    assert mgr.validate_session(token) is not None

    mgr.destroy_session(token)
    assert mgr.validate_session(token) is None

    # Destroying non-existent session should not raise
    mgr.destroy_session("non-existent-token")


def test_session_cleanup() -> None:
    """Create expired sessions, verify cleanup removes them."""
    mgr = SessionManager(idle_timeout_minutes=0)
    t1 = mgr.create_session(user_id=1, org_id="org-a")
    t2 = mgr.create_session(user_id=2, org_id="org-b")

    removed_count = mgr.cleanup_expired()
    assert removed_count == 2
    assert mgr.validate_session(t1) is None
    assert mgr.validate_session(t2) is None

    # Further cleanup should return 0
    assert mgr.cleanup_expired() == 0


def test_session_touch() -> None:
    """Verify touching an active session updates last_activity timestamp."""
    mgr = SessionManager(idle_timeout_minutes=30)
    token = mgr.create_session(user_id=303, org_id="gamma-corp")
    initial_session = mgr.validate_session(token)
    assert initial_session is not None
    initial_last_activity = initial_session.last_activity

    time.sleep(0.01)
    mgr.touch_session(token)
    updated_session = mgr.validate_session(token)
    assert updated_session is not None
    assert updated_session.last_activity >= initial_last_activity


def test_secure_key_store_uninitialized_error(tmp_path: Path) -> None:
    """Verify unlocking an uninitialized key store raises PrivateAnalystError."""
    store = SecureKeyStore(tmp_path / "nonexistent.key")
    with pytest.raises(PrivateAnalystError):
        store.unlock("any-password")


def test_file_missing_source_error(tmp_path: Path) -> None:
    """Verify encrypting or decrypting a missing file raises PrivateAnalystError."""
    key = generate_key()
    missing_source = tmp_path / "does_not_exist.txt"
    dest = tmp_path / "output.bin"

    with pytest.raises(PrivateAnalystError):
        encrypt_file(missing_source, dest, key)

    with pytest.raises(PrivateAnalystError):
        decrypt_file(missing_source, dest, key)
