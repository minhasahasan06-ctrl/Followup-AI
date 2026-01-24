"""
GCP Cloud KMS Encryption Service
HIPAA-compliant encryption for sensitive database fields.

Encrypts:
- medical_license_number
- stripe_account_id
- Other PII/PHI fields that need at-rest encryption
"""

import os
import base64
import logging
from typing import Optional
from functools import lru_cache

logger = logging.getLogger(__name__)

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_KMS_LOCATION = os.getenv("GCP_KMS_LOCATION", "us-central1")
GCP_KMS_KEY_RING_ID = os.getenv("GCP_KMS_KEY_RING_ID")
GCP_KMS_CRYPTO_KEY_ID = os.getenv("GCP_KMS_CRYPTO_KEY_ID")


class GCPKMSService:
    """
    Envelope encryption service using GCP Cloud KMS.
    
    In production, uses GCP KMS to encrypt/decrypt sensitive data.
    In development, uses a local fallback (base64 encoding for testing).
    """
    
    def __init__(self):
        self._client = None
        self._key_name = None
        self._is_configured = False
        
        if all([GCP_PROJECT_ID, GCP_KMS_KEY_RING_ID, GCP_KMS_CRYPTO_KEY_ID]):
            try:
                from google.cloud import kms
                self._client = kms.KeyManagementServiceClient()
                self._key_name = self._client.crypto_key_path(
                    GCP_PROJECT_ID,
                    GCP_KMS_LOCATION,
                    GCP_KMS_KEY_RING_ID,
                    GCP_KMS_CRYPTO_KEY_ID
                )
                self._is_configured = True
                logger.info(f"[GCP KMS] Initialized with key: {GCP_KMS_CRYPTO_KEY_ID}")
            except Exception as e:
                logger.warning(f"[GCP KMS] Failed to initialize: {e}")
                logger.info("[GCP KMS] Using local fallback for development")
        else:
            logger.info("[GCP KMS] Not configured, using local fallback")
    
    @property
    def is_configured(self) -> bool:
        return self._is_configured
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.
        
        Returns:
            Base64-encoded ciphertext with 'gcp:' prefix (production)
            or 'dev:' prefix (development fallback)
        """
        if not plaintext:
            return ""
        
        if self._is_configured and self._client:
            try:
                response = self._client.encrypt(
                    request={
                        "name": self._key_name,
                        "plaintext": plaintext.encode("utf-8"),
                    }
                )
                ciphertext = base64.b64encode(response.ciphertext).decode("utf-8")
                return f"gcp:{ciphertext}"
            except Exception as e:
                logger.error(f"[GCP KMS] Encryption failed: {e}")
                raise ValueError(f"Encryption failed: {e}")
        else:
            encoded = base64.b64encode(plaintext.encode("utf-8")).decode("utf-8")
            return f"dev:{encoded}"
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt a ciphertext string.
        
        Expects:
            'gcp:' prefix for GCP KMS encrypted data
            'dev:' prefix for development fallback data
        """
        if not ciphertext:
            return ""
        
        if ciphertext.startswith("gcp:"):
            if not self._is_configured or not self._client:
                raise ValueError("GCP KMS not configured but encrypted data requires it")
            
            try:
                encrypted_data = base64.b64decode(ciphertext[4:])
                response = self._client.decrypt(
                    request={
                        "name": self._key_name,
                        "ciphertext": encrypted_data,
                    }
                )
                return response.plaintext.decode("utf-8")
            except Exception as e:
                logger.error(f"[GCP KMS] Decryption failed: {e}")
                raise ValueError(f"Decryption failed: {e}")
        
        elif ciphertext.startswith("dev:"):
            decoded = base64.b64decode(ciphertext[4:]).decode("utf-8")
            return decoded
        
        else:
            return ciphertext
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value is encrypted."""
        return value.startswith("gcp:") or value.startswith("dev:")


@lru_cache()
def get_kms_service() -> GCPKMSService:
    """Get the singleton KMS service instance."""
    return GCPKMSService()


def encrypt_sensitive_field(value: str) -> str:
    """Helper to encrypt a sensitive field value."""
    kms = get_kms_service()
    return kms.encrypt(value)


def decrypt_sensitive_field(value: str) -> str:
    """Helper to decrypt a sensitive field value."""
    kms = get_kms_service()
    return kms.decrypt(value)
