"""
AWS Secrets Manager Service - HIPAA-Compliant Secrets Management
Centralized secrets management using AWS Secrets Manager

CRITICAL: Never store secrets in environment variables or code
All secrets MUST be retrieved from AWS Secrets Manager at runtime
"""

import boto3
import json
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError
import logging
import os

from app.config import settings

logger = logging.getLogger(__name__)


class SecretsManagerService:
    """
    HIPAA-compliant secrets management using AWS Secrets Manager
    Provides secure retrieval and caching of secrets
    """
    
    def __init__(self):
        self.secrets_client = None
        self.region = settings.AWS_REGION
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 3600  # Cache secrets for 1 hour
        
        # Initialize Secrets Manager client if credentials available
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            try:
                self.secrets_client = boto3.client(
                    'secretsmanager',
                    region_name=self.region,
                    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
                )
                logger.info(f"✅ Secrets Manager client initialized for region: {self.region}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Secrets Manager client: {e}")
                self.secrets_client = None
        else:
            logger.warning("⚠️  AWS credentials not configured - Secrets Manager disabled")
    
    def get_secret(self, secret_name: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Retrieve secret from AWS Secrets Manager
        
        Args:
            secret_name: Name or ARN of the secret
            use_cache: Whether to use cached value if available
        
        Returns:
            Secret value as dictionary, or None if not found
        """
        # Check cache first
        if use_cache and secret_name in self._cache:
            cached_value, cached_time = self._cache[secret_name]
            import time
            if time.time() - cached_time < self._cache_ttl:
                logger.debug(f"Using cached secret: {secret_name}")
                return cached_value
        
        if not self.secrets_client:
            logger.warning(f"Secrets Manager not configured, falling back to environment for: {secret_name}")
            return None
        
        try:
            response = self.secrets_client.get_secret_value(SecretId=secret_name)
            
            # Parse secret string (can be JSON or plain text)
            secret_string = response['SecretString']
            try:
                secret_value = json.loads(secret_string)
            except json.JSONDecodeError:
                # Not JSON, return as plain string wrapped in dict
                secret_value = {"value": secret_string}
            
            # Cache the secret
            import time
            self._cache[secret_name] = (secret_value, time.time())
            
            logger.info(f"✅ Retrieved secret: {secret_name}")
            return secret_value
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'ResourceNotFoundException':
                logger.error(f"Secret not found: {secret_name}")
            elif error_code == 'InvalidParameterException':
                logger.error(f"Invalid secret name: {secret_name}")
            elif error_code == 'InvalidRequestException':
                logger.error(f"Invalid request for secret: {secret_name}")
            elif error_code == 'DecryptionFailureException':
                logger.error(f"Failed to decrypt secret: {secret_name}")
            else:
                logger.error(f"Error retrieving secret {secret_name}: {e}")
            return None
    
    def get_database_credentials(self) -> Optional[Dict[str, str]]:
        """Get database credentials from Secrets Manager"""
        secret_name = os.getenv("DB_SECRET_NAME", "followup-ai/database-credentials")
        secret = self.get_secret(secret_name)
        
        if secret:
            return {
                "username": secret.get("username"),
                "password": secret.get("password"),
                "host": secret.get("host"),
                "port": secret.get("port", 5432),
                "database": secret.get("database")
            }
        return None
    
    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from Secrets Manager"""
        secret_name = os.getenv("OPENAI_SECRET_NAME", "followup-ai/openai-api-key")
        secret = self.get_secret(secret_name)
        
        if secret:
            return secret.get("api_key") or secret.get("value")
        return None
    
    def get_cognito_credentials(self) -> Optional[Dict[str, str]]:
        """Get AWS Cognito credentials from Secrets Manager"""
        secret_name = os.getenv("COGNITO_SECRET_NAME", "followup-ai/cognito-credentials")
        secret = self.get_secret(secret_name)
        
        if secret:
            return {
                "user_pool_id": secret.get("user_pool_id"),
                "client_id": secret.get("client_id"),
                "client_secret": secret.get("client_secret"),
                "region": secret.get("region", settings.AWS_REGION)
            }
        return None
    
    def get_twilio_credentials(self) -> Optional[Dict[str, str]]:
        """Get Twilio credentials from Secrets Manager"""
        secret_name = os.getenv("TWILIO_SECRET_NAME", "followup-ai/twilio-credentials")
        secret = self.get_secret(secret_name)
        
        if secret:
            return {
                "account_sid": secret.get("account_sid"),
                "auth_token": secret.get("auth_token"),
                "phone_number": secret.get("phone_number")
            }
        return None
    
    def invalidate_cache(self, secret_name: Optional[str] = None):
        """Invalidate cached secrets"""
        if secret_name:
            self._cache.pop(secret_name, None)
            logger.info(f"Invalidated cache for secret: {secret_name}")
        else:
            self._cache.clear()
            logger.info("Invalidated all cached secrets")
    
    def is_configured(self) -> bool:
        """Check if Secrets Manager is properly configured"""
        return self.secrets_client is not None


# Global singleton instance
_secrets_service: Optional[SecretsManagerService] = None


def get_secrets_service() -> SecretsManagerService:
    """Get or create Secrets Manager service singleton"""
    global _secrets_service
    if _secrets_service is None:
        _secrets_service = SecretsManagerService()
    return _secrets_service
