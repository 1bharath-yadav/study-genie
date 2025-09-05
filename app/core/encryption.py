"""
Encryption utilities for StudyGenie
Handles encryption and decryption of sensitive data like API keys
"""
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EncryptionService:
    """Service for encrypting and decrypting sensitive data"""

    def __init__(self, secret_key: Optional[str] = None):
        """Initialize encryption service with secret key"""
        if secret_key is None:
            secret_key = os.getenv(
                "SECRET_KEY", "default-secret-key-change-in-production")

        # Generate a key from the secret
        self._fernet = self._create_fernet(secret_key)

    def _create_fernet(self, secret_key: str) -> Fernet:
        """Create Fernet instance from secret key"""
        # Use a fixed salt for consistency (in production, you might want to store this securely)
        salt = b'studygenie-salt-2024'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
        return Fernet(key)

    def encrypt(self, data: str) -> str:
        """Encrypt a string and return base64 encoded result"""
        try:
            encrypted_bytes = self._fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt base64 encoded encrypted data with fallback for migration"""
        try:
            # Try with current key first
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self._fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Decryption failed with current key: {e}")

            # Try with old key for migration
            try:
                old_fernet = self._create_fernet("bharath")
                encrypted_bytes = base64.urlsafe_b64decode(
                    encrypted_data.encode())
                decrypted_bytes = old_fernet.decrypt(encrypted_bytes)
                logger.info(
                    "Successfully decrypted with old key - consider re-encrypting")
                return decrypted_bytes.decode()
            except Exception as e2:
                logger.error(f"Decryption failed with old key too: {e2}")
                # Try default key as last resort
                try:
                    default_fernet = self._create_fernet("default-secret-key")
                    encrypted_bytes = base64.urlsafe_b64decode(
                        encrypted_data.encode())
                    decrypted_bytes = default_fernet.decrypt(encrypted_bytes)
                    logger.info(
                        "Successfully decrypted with default key - consider re-encrypting")
                    return decrypted_bytes.decode()
                except Exception as e3:
                    logger.error(f"All decryption attempts failed: {e3}")
                    raise Exception(
                        "Could not decrypt data with any available key")


# Global encryption service instance
_encryption_service: Optional[EncryptionService] = None


def get_encryption_service() -> EncryptionService:
    """Get or create global encryption service instance"""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def encrypt_api_key(api_key: str) -> str:
    """Convenience function to encrypt an API key"""
    return get_encryption_service().encrypt(api_key)


def decrypt_api_key(encrypted_api_key: str) -> str:
    """Convenience function to decrypt an API key"""
    return get_encryption_service().decrypt(encrypted_api_key)
