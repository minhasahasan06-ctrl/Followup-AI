"""
Google Cloud Storage Service (Python)

Production-grade file storage replacing AWS S3:
- Signed URL generation for secure access
- Cloud KMS encryption for HIPAA compliance
- Local filesystem fallback for development
- Async operations to prevent blocking
"""

import os
import asyncio
import logging
import uuid
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from pathlib import Path

from app.config.gcp_constants import GCP_CONFIG, HIPAA_AUDIT_ACTIONS
from app.gcp_config import is_storage_available, get_bucket

logger = logging.getLogger(__name__)


class GCSService:
    """
    Environment-aware GCS service with local filesystem fallback.
    
    Automatically detects if GCP credentials are available:
    - If available: Uses GCS with encryption
    - If not available: Falls back to local filesystem storage
    """
    
    def __init__(self):
        self.use_gcs = self._should_use_gcs()
        self.local_storage_dir = "tmp/storage"
        self.bucket = None
        
        if self.use_gcs:
            try:
                self.bucket = get_bucket()
                logger.info(f"[GCS] Service initialized with bucket: {GCP_CONFIG.STORAGE.BUCKET_NAME}")
            except Exception as e:
                logger.warning(f"[GCS] Failed to initialize bucket: {e}. Using local fallback.")
                self.use_gcs = False
        
        if not self.use_gcs:
            logger.warning("[GCS] Using LOCAL STORAGE fallback - GCP credentials not available")
            self._ensure_local_storage_dir()
    
    def _should_use_gcs(self) -> bool:
        """Check if GCS credentials and bucket are available."""
        try:
            return is_storage_available() and bool(GCP_CONFIG.STORAGE.BUCKET_NAME)
        except Exception:
            return False
    
    def _ensure_local_storage_dir(self):
        """Ensure local storage directory exists."""
        os.makedirs(self.local_storage_dir, exist_ok=True)
    
    def _audit_log(self, action: str, resource_id: str, success: bool, metadata: Optional[Dict] = None):
        """Log HIPAA audit entry."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "resource_id": resource_id,
            "success": success,
            "metadata": metadata or {},
        }
        logger.info(f"[HIPAA Audit] {log_entry}")
    
    async def upload_file(
        self,
        file_data: bytes,
        key: str,
        content_type: str = "application/octet-stream",
        metadata: Optional[Dict[str, str]] = None,
        folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload file to GCS or local storage (async).
        
        Args:
            file_data: File bytes
            key: Object key path
            content_type: MIME type
            metadata: Optional metadata dict
            folder: Optional folder prefix
        
        Returns:
            Dict with bucket, key, url, uri, size, encrypted
        """
        final_key = f"{folder}/{key}" if folder else key
        
        self._audit_log(
            HIPAA_AUDIT_ACTIONS["FILE_UPLOAD"],
            final_key,
            True,
            {"content_type": content_type, "size": len(file_data)},
        )
        
        if self.use_gcs and self.bucket:
            return await self._upload_to_gcs(file_data, final_key, content_type, metadata)
        else:
            return await self._upload_to_local(file_data, final_key, content_type)
    
    async def _upload_to_gcs(
        self,
        file_data: bytes,
        key: str,
        content_type: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Upload to Google Cloud Storage."""
        def _upload():
            blob = self.bucket.blob(key)
            blob.content_type = content_type
            
            if metadata:
                blob.metadata = metadata
            
            kms_key = os.getenv("GCP_KMS_KEY_ID")
            if kms_key:
                blob.kms_key_name = kms_key
            
            blob.upload_from_string(file_data, content_type=content_type)
            return blob
        
        blob = await asyncio.to_thread(_upload)
        
        signed_url = await self.generate_signed_url(key, "read", GCP_CONFIG.STORAGE.SIGNED_URL_EXPIRATION)
        uri = f"gs://{GCP_CONFIG.STORAGE.BUCKET_NAME}/{key}"
        
        return {
            "bucket": GCP_CONFIG.STORAGE.BUCKET_NAME,
            "key": key,
            "url": signed_url or uri,
            "uri": uri,
            "size": len(file_data),
            "content_type": content_type,
            "encrypted": bool(os.getenv("GCP_KMS_KEY_ID")),
        }
    
    async def _upload_to_local(
        self,
        file_data: bytes,
        key: str,
        content_type: str,
    ) -> Dict[str, Any]:
        """Upload to local filesystem."""
        local_filename = key.replace("/", "_")
        local_path = os.path.join(self.local_storage_dir, local_filename)
        
        def _write():
            self._ensure_local_storage_dir()
            with open(local_path, "wb") as f:
                f.write(file_data)
        
        await asyncio.to_thread(_write)
        
        absolute_path = os.path.abspath(local_path)
        uri = f"file://{absolute_path}"
        
        return {
            "bucket": "local",
            "key": key,
            "url": uri,
            "uri": uri,
            "size": len(file_data),
            "content_type": content_type,
            "encrypted": False,
        }
    
    async def download_file(self, uri: str) -> bytes:
        """
        Download file from GCS or local storage (async).
        
        Args:
            uri: GCS URI (gs://...) or file URI (file://...)
        
        Returns:
            File bytes
        """
        self._audit_log(HIPAA_AUDIT_ACTIONS["FILE_DOWNLOAD"], uri, True)
        
        if uri.startswith("gs://"):
            return await self._download_from_gcs(uri)
        elif uri.startswith("file://"):
            return await self._download_from_local(uri)
        elif uri.startswith("s3://"):
            logger.warning(f"[GCS] Legacy S3 URI detected: {uri}. Attempting local fallback.")
            local_key = uri.split("/")[-1]
            local_path = os.path.join(self.local_storage_dir, local_key)
            if os.path.exists(local_path):
                return await self._download_from_local(f"file://{local_path}")
            raise ValueError(f"S3 URIs are no longer supported. URI: {uri}")
        else:
            raise ValueError(f"Invalid URI format: {uri}")
    
    async def _download_from_gcs(self, uri: str) -> bytes:
        """Download from Google Cloud Storage."""
        match = uri.replace("gs://", "").split("/", 1)
        if len(match) != 2:
            raise ValueError(f"Invalid GCS URI: {uri}")
        
        bucket_name, key = match
        
        def _download():
            from app.gcp_config import get_storage_client
            client = get_storage_client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(key)
            return blob.download_as_bytes()
        
        return await asyncio.to_thread(_download)
    
    async def _download_from_local(self, uri: str) -> bytes:
        """Download from local filesystem."""
        local_path = uri.replace("file://", "")
        
        def _read():
            with open(local_path, "rb") as f:
                return f.read()
        
        return await asyncio.to_thread(_read)
    
    async def generate_signed_url(
        self,
        key: str,
        action: str = "read",
        expiration_seconds: int = 3600,
    ) -> Optional[str]:
        """
        Generate signed URL for GCS object.
        Returns None for local storage.
        """
        if not self.use_gcs or not self.bucket:
            logger.warning("[GCS] Signed URLs not available in local storage mode")
            return None
        
        try:
            def _generate():
                blob = self.bucket.blob(key)
                method = "GET" if action == "read" else "PUT"
                return blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expiration_seconds),
                    method=method,
                )
            
            return await asyncio.to_thread(_generate)
        except Exception as e:
            logger.error(f"[GCS] Failed to generate signed URL: {e}")
            return None
    
    async def delete_file(self, uri: str) -> bool:
        """Delete file from storage."""
        self._audit_log(HIPAA_AUDIT_ACTIONS["FILE_DELETE"], uri, True)
        
        try:
            if uri.startswith("gs://"):
                match = uri.replace("gs://", "").split("/", 1)
                if len(match) != 2:
                    return False
                
                bucket_name, key = match
                
                def _delete():
                    from app.gcp_config import get_storage_client
                    client = get_storage_client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(key)
                    blob.delete()
                
                await asyncio.to_thread(_delete)
                return True
            
            elif uri.startswith("file://"):
                local_path = uri.replace("file://", "")
                
                def _delete_local():
                    os.remove(local_path)
                
                await asyncio.to_thread(_delete_local)
                return True
            
            return False
        except Exception as e:
            logger.error(f"[GCS] Delete failed: {e}")
            return False
    
    async def file_exists(self, uri: str) -> bool:
        """Check if file exists."""
        try:
            if uri.startswith("gs://"):
                match = uri.replace("gs://", "").split("/", 1)
                if len(match) != 2:
                    return False
                
                bucket_name, key = match
                
                def _exists():
                    from app.gcp_config import get_storage_client
                    client = get_storage_client()
                    bucket = client.bucket(bucket_name)
                    blob = bucket.blob(key)
                    return blob.exists()
                
                return await asyncio.to_thread(_exists)
            
            elif uri.startswith("file://"):
                local_path = uri.replace("file://", "")
                return os.path.exists(local_path)
            
            return False
        except Exception:
            return False
    
    async def list_files(self, prefix: str) -> list:
        """List files with given prefix."""
        if self.use_gcs and self.bucket:
            def _list():
                blobs = self.bucket.list_blobs(prefix=prefix)
                return [f"gs://{GCP_CONFIG.STORAGE.BUCKET_NAME}/{blob.name}" for blob in blobs]
            
            return await asyncio.to_thread(_list)
        else:
            try:
                files = os.listdir(self.local_storage_dir)
                local_prefix = prefix.replace("/", "_")
                matching = [f for f in files if f.startswith(local_prefix)]
                return [f"file://{os.path.abspath(os.path.join(self.local_storage_dir, f))}" for f in matching]
            except Exception:
                return []
    
    def is_using_gcs(self) -> bool:
        """Check if using GCS or local fallback."""
        return self.use_gcs


_gcs_service_instance = None


def get_gcs_service() -> GCSService:
    """Get the GCS service singleton with lazy initialization."""
    global _gcs_service_instance
    if _gcs_service_instance is None:
        _gcs_service_instance = GCSService()
    return _gcs_service_instance


class _LazyGCSService:
    """Lazy proxy for GCS service to support both attribute and call access."""
    
    def __getattr__(self, name):
        return getattr(get_gcs_service(), name)
    
    def __call__(self):
        return get_gcs_service()


gcs_service = _LazyGCSService()


async def upload_symptom_image(
    file_data: bytes,
    patient_id: str,
    body_area: str,
    content_type: str,
) -> Dict:
    """
    Upload symptom image to GCS with encryption.
    Replaces S3 upload function.
    """
    file_extension = content_type.split("/")[-1]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    key = f"{timestamp}_{unique_id}.{file_extension}"
    folder = f"{GCP_CONFIG.STORAGE.SYMPTOM_IMAGES_PREFIX}/{patient_id}/{body_area}"
    
    metadata = {
        "patient-id": patient_id,
        "body-area": body_area,
        "upload-timestamp": timestamp,
    }
    
    result = await get_gcs_service().upload_file(
        file_data,
        key,
        content_type,
        metadata,
        folder,
    )
    
    return {
        "bucket": result["bucket"],
        "key": result["key"],
        "url": result["url"],
    }


async def upload_file_to_storage(
    file_data: bytes,
    folder: str,
    filename: str,
    content_type: str,
) -> Dict:
    """
    Generic file upload for exam coach and other features.
    Replaces S3 upload_file_to_s3 function.
    """
    result = await get_gcs_service().upload_file(
        file_data,
        filename,
        content_type,
        folder=folder,
    )
    
    return {
        "bucket": result["bucket"],
        "key": result["key"],
        "url": result["url"],
    }


def generate_presigned_url(bucket: str, key: str, expiration: int = 3600) -> Optional[str]:
    """
    Generate a presigned URL for secure file access.
    Synchronous wrapper for backward compatibility.
    """
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, get_gcs_service().generate_signed_url(key, "read", expiration))
                return future.result()
        else:
            return asyncio.run(get_gcs_service().generate_signed_url(key, "read", expiration))
    except Exception as e:
        logger.error(f"[GCS] Error generating presigned URL: {e}")
        return None
