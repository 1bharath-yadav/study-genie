import os
import base64
from typing import List, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# --- Pure functions for key derivation etc. ---

def derive_key(secret: str, salt: bytes, iterations: int = 100_000, length: int = 32) -> bytes:
    """Derive a cryptographic key from a secret using PBKDF2."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=iterations,
    )
    return base64.urlsafe_b64encode(kdf.derive(secret.encode()))

def fernet_from_key(key_bytes: bytes) -> Fernet:
    """Get a Fernet instance from a derived key."""
    return Fernet(key_bytes)

def b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode()

def b64_decode(data_str: str) -> bytes:
    return base64.urlsafe_b64decode(data_str.encode())

# --- Encryption / Decryption functions --- 

def encrypt_with_fernet(fernet: Fernet, plaintext: str) -> str:
    """Encrypt plaintext and return base64‐encoded ciphertext."""
    token = fernet.encrypt(plaintext.encode())
    return b64_encode(token)

def decrypt_with_fernet(fernet: Fernet, ciphertext_b64: str) -> str:
    """Decrypt base64‐encoded ciphertext using given Fernet, or raise."""
    token = b64_decode(ciphertext_b64)
    plaintext_bytes = fernet.decrypt(token)
    return plaintext_bytes.decode()

# --- Public API with key variants passed in ---

def encrypt_data(
    plaintext: str,
    *,
    secret_key: Optional[str] = None,
    salt: bytes = b"studygenie-salt-2024",
) -> str:
    """Encrypt given plaintext using a derived key from secret_key (or env)."""
    sk = secret_key or os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
    key = derive_key(sk, salt)
    f = fernet_from_key(key)
    return encrypt_with_fernet(f, plaintext)

def decrypt_data(
    ciphertext_b64: str,
    *,
    secret_keys: List[str],  # list of candidates in order of preference, for migration
    salt: bytes = b"studygenie-salt-2024",
) -> str:
    """Attempt decryption using multiple secret keys; raise if all fail."""
    for sk in secret_keys:
        try:
            key = derive_key(sk, salt)
            f = fernet_from_key(key)
            return decrypt_with_fernet(f, ciphertext_b64)
        except InvalidToken:
            continue
    # If not caught by InvalidToken, let other exceptions bubble up
    raise ValueError("Could not decrypt using any provided secret key")

# --- Convenience wrappers using environment configuration --- 

def encrypt_api_key(api_key: str) -> str:
    return encrypt_data(api_key)

def decrypt_api_key(encrypted_api_key: str) -> str:
    # Define the keys to try: current, old, default
    current = os.getenv("SECRET_KEY", "default-secret-key-change-in-production")
    old = "bharath"
    default = "default-secret-key"
    return decrypt_data(encrypted_api_key, secret_keys=[current, old, default])
