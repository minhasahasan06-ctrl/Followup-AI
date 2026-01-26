"""
Google Cloud Platform Client Initialization (Python)

Centralized initialization of all GCP clients for Python services.
Supports GCS_SERVICE_ACCOUNT_KEY env var (JSON string) or GOOGLE_APPLICATION_CREDENTIALS file.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

from app.config.gcp_constants import GCP_CONFIG, is_gcp_configured, is_kms_configured

logger = logging.getLogger(__name__)

_storage_client = None
_kms_client = None
_initialized = False
_project_id = None


def _parse_gcs_credentials() -> Optional[Dict[str, Any]]:
    """Parse GCS credentials from GCS_SERVICE_ACCOUNT_KEY environment variable."""
    credentials_json = os.getenv("GCS_SERVICE_ACCOUNT_KEY")
    if not credentials_json:
        return None
    
    try:
        last_brace = credentials_json.rfind('}')
        if last_brace != -1 and last_brace < len(credentials_json) - 1:
            credentials_json = credentials_json[:last_brace + 1]
        
        credentials = json.loads(credentials_json)
        project_id = credentials.get("project_id")
        logger.info(f"[GCP] Credentials loaded from GCS_SERVICE_ACCOUNT_KEY for project: {project_id}")
        return credentials
    except json.JSONDecodeError as e:
        logger.error(f"[GCP] Failed to parse GCS_SERVICE_ACCOUNT_KEY: {e}")
        return None


def _initialize_clients():
    """Initialize GCP clients if credentials are available."""
    global _storage_client, _kms_client, _initialized, _project_id
    
    if _initialized:
        return
    
    parsed_credentials = _parse_gcs_credentials()
    
    if parsed_credentials:
        try:
            from google.cloud import storage
            from google.oauth2 import service_account
            
            _project_id = parsed_credentials.get("project_id")
            credentials = service_account.Credentials.from_service_account_info(parsed_credentials)
            
            _storage_client = storage.Client(project=_project_id, credentials=credentials)
            logger.info("[GCP] Storage client initialized with service account credentials")
            
            if is_kms_configured():
                try:
                    from google.cloud import kms
                    _kms_client = kms.KeyManagementServiceClient(credentials=credentials)
                    logger.info("[GCP] KMS client initialized")
                except Exception as e:
                    logger.error(f"[GCP] Failed to initialize KMS client: {e}")
            
            _initialized = True
            logger.info(f"[GCP] All clients initialized for project: {_project_id}")
            return
        except Exception as e:
            logger.error(f"[GCP] Failed to initialize with service account: {e}")
    
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    project_id = GCP_CONFIG.PROJECT_ID
    
    if not credentials_path and not project_id:
        logger.warning("[GCP] GOOGLE_APPLICATION_CREDENTIALS or GCP_PROJECT_ID not set. Using local fallback.")
        _initialized = True
        return
    
    try:
        from google.cloud import storage
        _storage_client = storage.Client(project=project_id if project_id else None)
        _project_id = project_id
        logger.info("[GCP] Storage client initialized with ADC")
    except Exception as e:
        logger.error(f"[GCP] Failed to initialize Storage client: {e}")
    
    if is_kms_configured():
        try:
            from google.cloud import kms
            _kms_client = kms.KeyManagementServiceClient()
            logger.info("[GCP] KMS client initialized")
        except Exception as e:
            logger.error(f"[GCP] Failed to initialize KMS client: {e}")
    
    _initialized = True
    logger.info(f"[GCP] Clients initialized for project: {project_id or 'default'}")


def get_storage_client():
    """Get the GCS storage client."""
    _initialize_clients()
    if _storage_client is None:
        raise RuntimeError("GCP Storage client not initialized. Check GOOGLE_APPLICATION_CREDENTIALS.")
    return _storage_client


def get_kms_client():
    """Get the Cloud KMS client."""
    _initialize_clients()
    if _kms_client is None:
        raise RuntimeError("GCP KMS client not initialized. Check GCP_KMS_* environment variables.")
    return _kms_client


def is_storage_available() -> bool:
    """Check if GCS storage is available."""
    _initialize_clients()
    return _storage_client is not None


def is_kms_available() -> bool:
    """Check if Cloud KMS is available."""
    _initialize_clients()
    return _kms_client is not None


def is_gcp_available() -> bool:
    """Check if any GCP services are available."""
    _initialize_clients()
    return _storage_client is not None or _kms_client is not None


@lru_cache(maxsize=1)
def get_bucket():
    """Get the default GCS bucket."""
    client = get_storage_client()
    return client.bucket(GCP_CONFIG.STORAGE.BUCKET_NAME)


def get_kms_key_path() -> str:
    """Get the full KMS key resource path."""
    return (
        f"projects/{GCP_CONFIG.PROJECT_ID}/locations/{GCP_CONFIG.KMS.LOCATION}/"
        f"keyRings/{GCP_CONFIG.KMS.KEY_RING}/cryptoKeys/{GCP_CONFIG.KMS.CRYPTO_KEY}"
    )
