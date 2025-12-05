"""
AWS KMS Encryption Service - HIPAA-Compliant Field-Level Encryption
Provides envelope encryption for PHI at rest using AWS KMS

CRITICAL SECURITY REQUIREMENTS:
- All PHI fields MUST be encrypted before storage
- Encryption keys are managed by AWS KMS (never stored in application)
- Supports key rotation and multiple encryption contexts
- Audit logging for all encryption/decryption operations
"""

import boto3
import json
import base64
from typing import Optional, Dict, Any, Union
from botocore.exceptions import ClientError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import logging

from app.config import settings
from app.services.audit_logger import AuditLogger, AuditEvent

logger = logging.getLogger(__name__)


class KMSEncryptionService:
    """
    HIPAA-compliant encryption service using AWS KMS
    Implements envelope encryption pattern for efficient field-level encryption
    """
    
    def __init__(self):
        self.kms_client = None
        self.kms_key_id = os.getenv("AWS_KMS_KEY_ID")
        self.region = settings.AWS_REGION
        
        # Initialize KMS client if credentials available
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            try:
                self.kms_client = boto3.client(
                    'kms',
                    region_name=self.region,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                )
                logger.info(f"✅ KMS client initialized for region: {self.region}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize KMS client: {e}")
                self.kms_client = None
        else:
            logger.warning("⚠️  AWS credentials not configured - KMS encryption disabled")
    
    def _generate_data_key(self, encryption_context: Optional[Dict[str, str]] = None) -> tuple[bytes, bytes]:
        """
        Generate a data encryption key (DEK) from KMS
        Returns (plaintext_key, encrypted_key) tuple
        
        Args:
            encryption_context: Additional context for key derivation (e.g., patient_id, field_name)
        """
        if not self.kms_client or not self.kms_key_id:
            raise ValueError("KMS client not configured. Cannot generate encryption keys.")
        
        try:
            response = self.kms_client.generate_data_key(
                KeyId=self.kms_key_id,
                KeySpec='AES_256',
                EncryptionContext=encryption_context or {}
            )
            
            plaintext_key = response['Plaintext']
            encrypted_key = response['CiphertextBlob']
            
            return plaintext_key, encrypted_key
        except ClientError as e:
            logger.error(f"KMS generate_data_key failed: {e}")
            raise
    
    def _decrypt_data_key(self, encrypted_key: bytes, encryption_context: Optional[Dict[str, str]] = None) -> bytes:
        """
        Decrypt a data encryption key using KMS
        
        Args:
            encrypted_key: Encrypted data key blob from KMS
            encryption_context: Same context used during encryption
        """
        if not self.kms_client:
            raise ValueError("KMS client not configured. Cannot decrypt keys.")
        
        try:
            response = self.kms_client.decrypt(
                CiphertextBlob=encrypted_key,
                EncryptionContext=encryption_context or {}
            )
            return response['Plaintext']
        except ClientError as e:
            logger.error(f"KMS decrypt failed: {e}")
            raise
    
    def encrypt_phi(
        self,
        plaintext: Union[str, bytes],
        encryption_context: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        field_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Encrypt PHI using envelope encryption pattern
        
        Process:
        1. Generate data key from KMS
        2. Encrypt data with data key using Fernet (AES-128)
        3. Store encrypted data + encrypted data key
        
        Args:
            plaintext: PHI data to encrypt (string or bytes)
            encryption_context: Additional context for key derivation
            user_id: User performing encryption (for audit)
            field_name: Name of field being encrypted (for audit)
        
        Returns:
            Dictionary with encrypted_data and encrypted_key
        """
        if not plaintext:
            return {"encrypted_data": None, "encrypted_key": None}
        
        # Convert to bytes if string
        if isinstance(plaintext, str):
            plaintext_bytes = plaintext.encode('utf-8')
        else:
            plaintext_bytes = plaintext
        
        # Build encryption context
        context = encryption_context or {}
        if field_name:
            context['field_name'] = field_name
        if user_id:
            context['user_id'] = user_id
        
        try:
            # Generate data key from KMS
            plaintext_key, encrypted_key = self._generate_data_key(context)
            
            # Encrypt data with Fernet (AES-128 in CBC mode)
            fernet = Fernet(base64.urlsafe_b64encode(plaintext_key[:32]))
            encrypted_data = fernet.encrypt(plaintext_bytes)
            
            # Audit log encryption
            if user_id:
                AuditLogger.log_event(
                    event_type=AuditEvent.KMS_ENCRYPTION,
                    user_id=user_id,
                    resource_type="phi_field",
                    resource_id=field_name,
                    action="encrypt",
                    status="success",
                    metadata={
                        "field_name": field_name,
                        "data_length": len(plaintext_bytes),
                        "kms_key_id": self.kms_key_id
                    }
                )
            
            return {
                "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8'),
                "encrypted_key": base64.b64encode(encrypted_key).decode('utf-8'),
                "encryption_context": context,
                "algorithm": "AES-256-KMS-FERNET"
            }
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            if user_id:
                AuditLogger.log_event(
                    event_type=AuditEvent.KMS_ENCRYPTION,
                    user_id=user_id,
                    resource_type="phi_field",
                    resource_id=field_name,
                    action="encrypt",
                    status="failure",
                    metadata={"error": str(e)}
                )
            raise
    
    def decrypt_phi(
        self,
        encrypted_data: str,
        encrypted_key: str,
        encryption_context: Optional[Dict[str, str]] = None,
        user_id: Optional[str] = None,
        field_name: Optional[str] = None
    ) -> Union[str, bytes]:
        """
        Decrypt PHI using envelope encryption pattern
        
        Process:
        1. Decrypt data key using KMS
        2. Decrypt data with data key using Fernet
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            encrypted_key: Base64-encoded encrypted data key from KMS
            encryption_context: Same context used during encryption
            user_id: User performing decryption (for audit)
            field_name: Name of field being decrypted (for audit)
        
        Returns:
            Decrypted plaintext (string if original was string, bytes otherwise)
        """
        if not encrypted_data or not encrypted_key:
            return None
        
        # Build encryption context
        context = encryption_context or {}
        if field_name:
            context['field_name'] = field_name
        if user_id:
            context['user_id'] = user_id
        
        try:
            # Decode base64
            encrypted_data_bytes = base64.b64decode(encrypted_data)
            encrypted_key_bytes = base64.b64decode(encrypted_key)
            
            # Decrypt data key from KMS
            plaintext_key = self._decrypt_data_key(encrypted_key_bytes, context)
            
            # Decrypt data with Fernet
            fernet = Fernet(base64.urlsafe_b64encode(plaintext_key[:32]))
            decrypted_bytes = fernet.decrypt(encrypted_data_bytes)
            
            # Audit log decryption
            if user_id:
                AuditLogger.log_event(
                    event_type=AuditEvent.KMS_DECRYPTION,
                    user_id=user_id,
                    resource_type="phi_field",
                    resource_id=field_name,
                    action="decrypt",
                    status="success",
                    metadata={
                        "field_name": field_name,
                        "data_length": len(decrypted_bytes),
                        "kms_key_id": self.kms_key_id
                    }
                )
            
            # Try to decode as UTF-8 string, return bytes if fails
            try:
                return decrypted_bytes.decode('utf-8')
            except UnicodeDecodeError:
                return decrypted_bytes
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            if user_id:
                AuditLogger.log_event(
                    event_type=AuditEvent.KMS_DECRYPTION,
                    user_id=user_id,
                    resource_type="phi_field",
                    resource_id=field_name,
                    action="decrypt",
                    status="failure",
                    metadata={"error": str(e)}
                )
            raise
    
    def encrypt_phi_field(
        self,
        value: Optional[Union[str, bytes]],
        patient_id: str,
        field_name: str,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Convenience method to encrypt a PHI field with standard context
        
        Args:
            value: Field value to encrypt
            patient_id: Patient ID for encryption context
            field_name: Name of the field
            user_id: User performing encryption
        
        Returns:
            Encryption result dict or None if value is None
        """
        if value is None:
            return None
        
        context = {
            "patient_id": patient_id,
            "field_name": field_name,
            "purpose": "phi_storage"
        }
        
        return self.encrypt_phi(
            plaintext=value,
            encryption_context=context,
            user_id=user_id,
            field_name=field_name
        )
    
    def decrypt_phi_field(
        self,
        encrypted_value: Optional[Dict[str, Any]],
        patient_id: str,
        field_name: str,
        user_id: Optional[str] = None
    ) -> Optional[Union[str, bytes]]:
        """
        Convenience method to decrypt a PHI field with standard context
        
        Args:
            encrypted_value: Encrypted value dict from encrypt_phi_field
            patient_id: Patient ID for decryption context
            field_name: Name of the field
            user_id: User performing decryption
        
        Returns:
            Decrypted value or None
        """
        if not encrypted_value or not isinstance(encrypted_value, dict):
            return None
        
        context = {
            "patient_id": patient_id,
            "field_name": field_name,
            "purpose": "phi_storage"
        }
        
        return self.decrypt_phi(
            encrypted_data=encrypted_value.get("encrypted_data"),
            encrypted_key=encrypted_value.get("encrypted_key"),
            encryption_context=context,
            user_id=user_id,
            field_name=field_name
        )
    
    def is_configured(self) -> bool:
        """Check if KMS encryption is properly configured"""
        return self.kms_client is not None and self.kms_key_id is not None


# Global singleton instance
_kms_service: Optional[KMSEncryptionService] = None


def get_kms_service() -> KMSEncryptionService:
    """Get or create KMS encryption service singleton"""
    global _kms_service
    if _kms_service is None:
        _kms_service = KMSEncryptionService()
    return _kms_service
