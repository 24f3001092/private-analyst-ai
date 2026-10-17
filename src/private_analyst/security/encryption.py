from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from private_analyst.exceptions import PrivateAnalystError


def generate_key() -> bytes:
    """Generate a random Fernet key.

    Returns:
        32-byte URL-safe base64-encoded key as bytes.
    """
    return Fernet.generate_key()


def derive_key_from_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """Derive a Fernet-compatible encryption key from a password using PBKDF2HMAC.

    Uses SHA256 with 600,000 iterations to derive a 32-byte key, which is then
    URL-safe base64-encoded for Fernet compatibility.

    Args:
        password: User-provided plaintext password.
        salt: Optional 16-byte salt. If None, a cryptographically secure random salt is generated.

    Returns:
        tuple[bytes, bytes]: (derived_fernet_key, salt)
    """
    if salt is None:
        salt = os.urandom(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=600_000,
    )
    derived_raw = kdf.derive(password.encode("utf-8"))
    key = base64.urlsafe_b64encode(derived_raw)
    return key, salt


def encrypt_bytes(data: bytes, key: bytes) -> bytes:
    """Encrypt binary data using a Fernet key.

    Args:
        data: Plaintext data bytes to encrypt.
        key: Fernet key bytes.

    Returns:
        Encrypted ciphertext token as bytes.

    Raises:
        PrivateAnalystError: If encryption fails due to an invalid key.
    """
    try:
        fernet = Fernet(key)
        return fernet.encrypt(data)
    except Exception as exc:
        raise PrivateAnalystError("Encryption failed.") from exc


def decrypt_bytes(token: bytes, key: bytes) -> bytes:
    """Decrypt a Fernet token back into plaintext bytes.

    Args:
        token: Fernet ciphertext token bytes.
        key: Fernet key bytes.

    Returns:
        Decrypted plaintext data bytes.

    Raises:
        PrivateAnalystError: If the key or token is invalid, corrupted, or tampered with.
    """
    try:
        fernet = Fernet(key)
        return fernet.decrypt(token)
    except (InvalidToken, Exception) as exc:
        raise PrivateAnalystError("Decryption failed: invalid key or corrupted data.") from exc


def encrypt_file(source: Path, dest: Path, key: bytes) -> Path:
    """Encrypt contents of source file and write to dest.

    Args:
        source: Path to source plaintext file.
        dest: Path to destination ciphertext file.
        key: Fernet key bytes.

    Returns:
        Destination Path.

    Raises:
        PrivateAnalystError: If source cannot be read or encryption fails.
    """
    source_path = Path(source)
    dest_path = Path(dest)
    if not source_path.is_file():
        raise PrivateAnalystError("Source file does not exist.")
    try:
        data = source_path.read_bytes()
    except OSError as exc:
        raise PrivateAnalystError("Failed to read source file.") from exc

    encrypted = encrypt_bytes(data, key)
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(encrypted)
    except OSError as exc:
        raise PrivateAnalystError("Failed to write encrypted destination file.") from exc

    return dest_path


def decrypt_file(source: Path, dest: Path, key: bytes) -> Path:
    """Decrypt contents of source file and write to dest.

    Args:
        source: Path to source ciphertext file.
        dest: Path to destination plaintext file.
        key: Fernet key bytes.

    Returns:
        Destination Path.

    Raises:
        PrivateAnalystError: If source cannot be read or decryption fails.
    """
    source_path = Path(source)
    dest_path = Path(dest)
    if not source_path.is_file():
        raise PrivateAnalystError("Source file does not exist.")
    try:
        data = source_path.read_bytes()
    except OSError as exc:
        raise PrivateAnalystError("Failed to read source file.") from exc

    decrypted = decrypt_bytes(data, key)
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(decrypted)
    except OSError as exc:
        raise PrivateAnalystError("Failed to write decrypted destination file.") from exc

    return dest_path


class SecureKeyStore:
    """Encrypted key storage for managing application master keys.

    Security Notes:
    - Never logs or prints sensitive keys in repr or error messages.
    - Master key is encrypted with a PBKDF2HMAC-derived key using SHA-256 (600,000 iterations).
    - Salt is stored alongside the encrypted key in the key file.
    - IMPORTANT: The key file must be protected by OS-level permissions (e.g. chmod 0600
      on POSIX systems, or restricted NTFS ACLs on Windows) so only the authorized
      service account has read/write access.
    """

    def __init__(self, key_file: Path) -> None:
        self.key_file = Path(key_file)

    def is_initialized(self) -> bool:
        """Check if the key store file exists and is a regular file."""
        return self.key_file.is_file()

    def initialize(self, master_password: str) -> None:
        """Generate a new master key, encrypt it with master_password, and save to key_file.

        Args:
            master_password: Passphrase used to derive the encryption key.

        Raises:
            PrivateAnalystError: If the key store cannot be saved.
        """
        master_key = generate_key()
        derived_key, salt = derive_key_from_password(master_password)
        encrypted_master_key = encrypt_bytes(master_key, derived_key)

        payload = {
            "version": 1,
            "salt": salt.hex(),
            "encrypted_master_key": encrypted_master_key.decode("ascii"),
        }

        try:
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            self.key_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            try:
                os.chmod(self.key_file, 0o600)
            except OSError:
                pass
        except OSError as exc:
            raise PrivateAnalystError("Failed to write key store file.") from exc

    def unlock(self, master_password: str) -> bytes:
        """Load and decrypt the master key using master_password.

        Args:
            master_password: The passphrase used during initialization.

        Returns:
            Decrypted master key bytes.

        Raises:
            PrivateAnalystError: If key store is uninitialized, corrupted, or password is wrong.
        """
        if not self.is_initialized():
            raise PrivateAnalystError("Key store is not initialized.")

        try:
            content = self.key_file.read_text(encoding="utf-8")
            payload = json.loads(content)
            salt = bytes.fromhex(payload["salt"])
            encrypted_master_key = payload["encrypted_master_key"].encode("ascii")
        except Exception as exc:
            raise PrivateAnalystError("Key store file is corrupted or unreadable.") from exc

        derived_key, _ = derive_key_from_password(master_password, salt=salt)
        return decrypt_bytes(encrypted_master_key, derived_key)

    def __repr__(self) -> str:
        # Never expose keys or sensitive data
        return f"SecureKeyStore(key_file={self.key_file!r})"
