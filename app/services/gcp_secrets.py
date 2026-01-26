"""
GCP Secret Manager Integration
==============================
Production-grade secret management for Cloud Run deployment.
Fetches secrets from GCP Secret Manager with local environment fallback.

HIPAA Compliance:
- Secrets never logged or exposed in error messages
- Access audit trail maintained by GCP
- Automatic secret rotation support

Usage:
    from app.services.gcp_secrets import get_secret, SecretManager
    
    # Single secret
    api_key = get_secret("OPENAI_API_KEY")
    
    # Batch initialization
    secrets = SecretManager()
    secrets.load_all()
"""

import os
import logging
from functools import lru_cache
from typing import Optional, Dict

logger = logging.getLogger(__name__)

_SECRET_CACHE: Dict[str, str] = {}
_GCP_CLIENT = None


def _get_gcp_client():
    """Lazy-load GCP Secret Manager client."""
    global _GCP_CLIENT
    if _GCP_CLIENT is None:
        try:
            from google.cloud import secretmanager
            _GCP_CLIENT = secretmanager.SecretManagerServiceClient()
            logger.info("[GCP] Secret Manager client initialized")
        except Exception as e:
            logger.warning(f"[GCP] Secret Manager unavailable: {e}")
            _GCP_CLIENT = False
    return _GCP_CLIENT if _GCP_CLIENT else None


def _get_project_id() -> Optional[str]:
    """Get GCP project ID from environment or metadata server."""
    project_id = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id
    
    try:
        import requests
        response = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/project/project-id",
            headers={"Metadata-Flavor": "Google"},
            timeout=2
        )
        if response.ok:
            return response.text
    except Exception:
        pass
    
    return None


def get_secret(
    secret_name: str,
    version: str = "latest",
    required: bool = False
) -> Optional[str]:
    """
    Retrieve a secret from GCP Secret Manager or environment.
    
    Priority:
    1. Memory cache (fastest)
    2. Environment variable (local dev)
    3. GCP Secret Manager (production)
    
    Args:
        secret_name: Name of the secret (e.g., "STYTCH_SECRET")
        version: Secret version (default: "latest")
        required: If True, raises ValueError when secret not found
    
    Returns:
        Secret value or None if not found
    
    Raises:
        ValueError: If required=True and secret not found
    """
    if secret_name in _SECRET_CACHE:
        return _SECRET_CACHE[secret_name]
    
    env_value = os.getenv(secret_name)
    if env_value:
        _SECRET_CACHE[secret_name] = env_value
        return env_value
    
    client = _get_gcp_client()
    project_id = _get_project_id()
    
    if client and project_id:
        try:
            secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
            response = client.access_secret_version(request={"name": secret_path})
            secret_value = response.payload.data.decode("UTF-8")
            _SECRET_CACHE[secret_name] = secret_value
            logger.info(f"[GCP] Loaded secret: {secret_name}")
            return secret_value
        except Exception as e:
            logger.warning(f"[GCP] Failed to load secret {secret_name}: {type(e).__name__}")
    
    if required:
        raise ValueError(f"Required secret not found: {secret_name}")
    
    return None


def clear_cache():
    """Clear the secret cache. Use for testing or secret rotation."""
    global _SECRET_CACHE
    _SECRET_CACHE = {}
    logger.info("[GCP] Secret cache cleared")


class SecretManager:
    """
    Centralized secret management for the application.
    Loads all required secrets on startup with validation.
    """
    
    REQUIRED_SECRETS = [
        "DATABASE_URL",
    ]
    
    OPTIONAL_SECRETS = [
        "STYTCH_PROJECT_ID",
        "STYTCH_SECRET",
        "STYTCH_PUBLIC_TOKEN",
        "OPENAI_API_KEY",
        "SESSION_SECRET",
        "DEV_MODE_SECRET",
        "STRIPE_API_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "SENTRY_DSN",
        "REDIS_URL",
        "DAILY_API_KEY",
    ]
    
    def __init__(self):
        self._secrets: Dict[str, Optional[str]] = {}
        self._loaded = False
    
    def load_all(self, validate: bool = True) -> Dict[str, Optional[str]]:
        """
        Load all secrets from GCP Secret Manager or environment.
        
        Args:
            validate: If True, raises error if required secrets missing
        
        Returns:
            Dictionary of secret names to values
        """
        for secret_name in self.REQUIRED_SECRETS:
            self._secrets[secret_name] = get_secret(secret_name, required=validate)
        
        for secret_name in self.OPTIONAL_SECRETS:
            self._secrets[secret_name] = get_secret(secret_name, required=False)
        
        self._loaded = True
        loaded_count = sum(1 for v in self._secrets.values() if v is not None)
        logger.info(f"[GCP] Loaded {loaded_count}/{len(self._secrets)} secrets")
        
        return self._secrets
    
    def get(self, secret_name: str) -> Optional[str]:
        """Get a loaded secret by name."""
        if not self._loaded:
            self.load_all(validate=False)
        return self._secrets.get(secret_name) or get_secret(secret_name)
    
    @property
    def database_url(self) -> str:
        """Get database URL (required)."""
        url = self.get("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL is required")
        return url
    
    @property
    def stytch_project_id(self) -> Optional[str]:
        """Get Stytch project ID."""
        return self.get("STYTCH_PROJECT_ID")
    
    @property
    def stytch_secret(self) -> Optional[str]:
        """Get Stytch secret."""
        return self.get("STYTCH_SECRET")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key."""
        return self.get("OPENAI_API_KEY")
    
    def is_stytch_configured(self) -> bool:
        """Check if Stytch authentication is configured."""
        return bool(self.stytch_project_id and self.stytch_secret)
    
    def export_to_env(self):
        """Export loaded secrets to environment variables."""
        for name, value in self._secrets.items():
            if value and name not in os.environ:
                os.environ[name] = value
        logger.info("[GCP] Secrets exported to environment")


@lru_cache(maxsize=1)
def get_secret_manager() -> SecretManager:
    """Get singleton SecretManager instance."""
    manager = SecretManager()
    manager.load_all(validate=False)
    return manager


def init_secrets():
    """
    Initialize secrets on application startup.
    Call this from FastAPI lifespan or startup event.
    
    Returns:
        SecretManager instance with loaded secrets
    """
    manager = get_secret_manager()
    manager.export_to_env()
    return manager
