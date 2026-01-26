"""
Video Session Storage Service
HIPAA-compliant storage for video exam frames and recordings.

NOTE: AWS S3/boto3 integration has been disabled. All storage operations
use local filesystem fallback or return mock data.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from enum import Enum

from app.config import settings
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)

# STUB: boto3 has been removed - S3 operations are disabled
logger.warning("AWS S3 integration disabled - video storage using local filesystem fallback")


class StorageType(str, Enum):
    EXAM_FRAME = "exam-frame"
    EXAM_RECORDING = "exam-recording"
    COMBINED_SESSION = "combined-session"


class VideoSessionStorageService:
    """
    HIPAA-compliant video session storage service.
    
    NOTE: S3 integration is disabled. All operations use local storage fallback.
    
    Handles:
    - Frame uploads from guided exams (eyes, palm, tongue, lips, skin)
    - Video segment recordings during video consultations
    - Combined session archives for long-term storage
    
    All operations enforce strict RBAC:
    - Patients can only access their own session data
    - Doctors can access any patient's session data
    - All access is logged via HIPAAAuditLogger
    """
    
    SIGNED_URL_TTL_SECONDS = 900  # 15-minute TTL for signed URLs
    RETENTION_DAYS = {
        StorageType.EXAM_FRAME: 2555,      # 7 years for medical records
        StorageType.EXAM_RECORDING: 2555,   # 7 years
        StorageType.COMBINED_SESSION: 2555, # 7 years
    }
    
    @staticmethod
    def _authorize_access(
        user_id: str,
        user_role: str,
        patient_id: str,
        action: str,
        client_ip: Optional[str] = None
    ) -> bool:
        """
        Enforce RBAC for session storage access.
        
        - Patients can only access their own data
        - Doctors can access any patient's data
        - All denied access is logged
        
        Returns True if authorized, raises PermissionError if denied.
        """
        if user_role == "doctor" or user_role == "admin":
            return True
        
        if user_role == "patient":
            if user_id == patient_id:
                return True
            else:
                HIPAAAuditLogger.log_phi_access(
                    action=f"denied_{action}",
                    resource_type="video_session",
                    resource_id=patient_id,
                    user_id=user_id,
                    patient_id=patient_id,
                    access_context={
                        "reason": "patient_attempting_cross_patient_access",
                        "denied": True
                    },
                    client_ip=client_ip
                )
                raise PermissionError(f"Patient {user_id} cannot access data for patient {patient_id}")
        
        HIPAAAuditLogger.log_phi_access(
            action=f"denied_{action}",
            resource_type="video_session",
            resource_id=patient_id,
            user_id=user_id,
            patient_id=patient_id,
            access_context={"reason": "unknown_role", "role": user_role, "denied": True},
            client_ip=client_ip
        )
        raise PermissionError(f"Role {user_role} cannot access video session data")
    
    def __init__(self):
        # STUB: S3 is disabled - always use local storage
        self.use_s3 = False
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME', 'local-storage')
        self.kms_key_id = None
        self.region = 'us-east-1'
        self.s3_client = None  # STUB: No S3 client available
        
        self.local_storage_dir = 'tmp/video_sessions'
        os.makedirs(self.local_storage_dir, exist_ok=True)
        logger.warning("VideoSessionStorageService: S3 disabled, using LOCAL storage fallback")
    
    def _should_use_s3(self) -> bool:
        """Check if S3 credentials are available - STUB: always returns False"""
        return False
    
    @staticmethod
    def _parse_region(region_str: str) -> str:
        """Extract actual region code from potentially formatted string"""
        if not region_str:
            return 'us-east-1'
        parts = region_str.split()
        for part in parts:
            if '-' in part and len(part) > 5:
                return part
        return region_str.strip()
    
    def _generate_s3_key(
        self,
        storage_type: StorageType,
        patient_id: str,
        session_id: str,
        file_extension: str = "jpg",
        stage: Optional[str] = None,
        segment_order: Optional[int] = None
    ) -> str:
        """
        Generate a deterministic S3 key for video session content.
        
        Pattern: video-exams/{patient_id}/{session_id}/{type}/{stage_or_segment}.{ext}
        """
        base_path = f"video-exams/{patient_id}/{session_id}"
        
        if storage_type == StorageType.EXAM_FRAME:
            return f"{base_path}/frames/{stage}.{file_extension}"
        elif storage_type == StorageType.EXAM_RECORDING:
            segment_suffix = f"segment-{segment_order:03d}" if segment_order else stage
            return f"{base_path}/recordings/{segment_suffix}.{file_extension}"
        else:
            return f"{base_path}/combined/session.{file_extension}"
    
    async def generate_upload_url(
        self,
        patient_id: str,
        session_id: str,
        storage_type: StorageType,
        content_type: str,
        file_extension: str = "jpg",
        stage: Optional[str] = None,
        segment_order: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> Dict:
        """
        Generate a URL for upload. STUB: Returns local file path since S3 is disabled.
        
        Enforces RBAC: patients can only upload to their own sessions.
        
        Returns:
            Dict with upload_url, s3_key, expires_at, method
        """
        if user_id and user_role:
            self._authorize_access(user_id, user_role, patient_id, "upload", client_ip)
        
        s3_key = self._generate_s3_key(
            storage_type=storage_type,
            patient_id=patient_id,
            session_id=session_id,
            file_extension=file_extension,
            stage=stage,
            segment_order=segment_order
        )
        
        expires_at = datetime.utcnow() + timedelta(seconds=self.SIGNED_URL_TTL_SECONDS)
        
        # STUB: S3 disabled - return local file path
        local_path = os.path.join(self.local_storage_dir, s3_key.replace('/', '_'))
        logger.warning(f"S3 disabled: returning local upload path for {s3_key}")
        
        HIPAAAuditLogger.log_phi_access(
            action="generate_upload_url",
            resource_type="video_session_frame",
            resource_id=session_id,
            user_id=user_id or "system",
            patient_id=patient_id,
            access_context={"storage": "local_stub", "s3_key": s3_key},
            client_ip=client_ip
        )
        
        return {
            "upload_url": f"file://{os.path.abspath(local_path)}",
            "s3_key": s3_key,
            "bucket": "local",
            "expires_at": expires_at.isoformat(),
            "method": "PUT",
            "storage_mode": "local_stub",
            "warning": "AWS S3 integration disabled - using local storage"
        }
    
    async def generate_download_url(
        self,
        s3_key: str,
        patient_id: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Dict:
        """
        Generate a URL for download. STUB: Returns local file path since S3 is disabled.
        
        Enforces RBAC: patients can only download their own session data.
        """
        if user_id and user_role:
            self._authorize_access(user_id, user_role, patient_id, "download", client_ip)
        
        expires_at = datetime.utcnow() + timedelta(seconds=self.SIGNED_URL_TTL_SECONDS)
        
        # STUB: S3 disabled - return local file path
        local_path = os.path.join(self.local_storage_dir, s3_key.replace('/', '_'))
        
        if not os.path.exists(local_path):
            logger.warning(f"S3 disabled: Local file not found: {local_path}")
        
        HIPAAAuditLogger.log_phi_access(
            action="generate_download_url",
            resource_type="video_session_frame",
            resource_id=s3_key,
            user_id=user_id or "system",
            patient_id=patient_id,
            access_context={"storage": "local_stub"},
            client_ip=client_ip
        )
        
        return {
            "download_url": f"file://{os.path.abspath(local_path)}",
            "s3_key": s3_key,
            "expires_at": expires_at.isoformat(),
            "storage_mode": "local_stub",
            "warning": "AWS S3 integration disabled - using local storage"
        }
    
    async def upload_frame_direct(
        self,
        file_data: bytes,
        patient_id: str,
        session_id: str,
        stage: str,
        content_type: str = "image/jpeg",
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> Dict:
        """
        Upload a frame directly. STUB: Saves to local filesystem since S3 is disabled.
        Enforces RBAC: patients can only upload to their own sessions.
        """
        if user_id and user_role:
            self._authorize_access(user_id, user_role, patient_id, "upload_frame", client_ip)
        
        file_extension = "jpg" if "jpeg" in content_type else content_type.split("/")[-1]
        s3_key = self._generate_s3_key(
            storage_type=StorageType.EXAM_FRAME,
            patient_id=patient_id,
            session_id=session_id,
            file_extension=file_extension,
            stage=stage
        )
        
        # STUB: S3 disabled - save to local filesystem
        local_path = os.path.join(self.local_storage_dir, s3_key.replace('/', '_'))
        os.makedirs(os.path.dirname(local_path) if '/' in s3_key else self.local_storage_dir, exist_ok=True)
        
        def _write_file():
            with open(local_path, 'wb') as f:
                f.write(file_data)
        
        await asyncio.to_thread(_write_file)
        
        logger.warning(f"S3 disabled: Frame saved to local path {local_path}")
        
        HIPAAAuditLogger.log_phi_access(
            action="upload_frame",
            resource_type="video_session_frame",
            resource_id=session_id,
            user_id=user_id or "system",
            patient_id=patient_id,
            access_context={"stage": stage, "storage": "local_stub", "size_bytes": len(file_data)},
            client_ip=client_ip
        )
        
        return {
            "s3_key": s3_key,
            "s3_uri": f"file://{os.path.abspath(local_path)}",
            "bucket": "local",
            "size_bytes": len(file_data),
            "storage_mode": "local_stub",
            "warning": "AWS S3 integration disabled - using local storage"
        }
    
    async def delete_session_content(
        self,
        patient_id: str,
        session_id: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> Dict:
        """
        Delete all content for a video exam session.
        Used for HIPAA-compliant data deletion upon request.
        
        Note: Only doctors/admins can delete session content.
        """
        if user_role and user_role not in ("doctor", "admin"):
            HIPAAAuditLogger.log_phi_access(
                action="denied_delete",
                resource_type="video_session",
                resource_id=session_id,
                user_id=user_id or "unknown",
                patient_id=patient_id,
                access_context={"reason": "only_doctors_can_delete", "denied": True},
                client_ip=client_ip
            )
            raise PermissionError("Only doctors and admins can delete session content")
        
        # STUB: S3 disabled - delete local files
        import glob
        pattern = os.path.join(self.local_storage_dir, f"video-exams_{patient_id}_{session_id}_*")
        deleted_count = 0
        for f in glob.glob(pattern):
            os.remove(f)
            deleted_count += 1
        
        logger.warning(f"S3 disabled: Deleted {deleted_count} local files for session {session_id}")
        
        HIPAAAuditLogger.log_phi_access(
            action="delete_session_content",
            resource_type="video_session",
            resource_id=session_id,
            user_id=user_id or "system",
            patient_id=patient_id,
            access_context={"deleted_count": deleted_count, "reason": "HIPAA_deletion_request", "storage": "local_stub"},
            client_ip=client_ip
        )
        
        return {
            "session_id": session_id,
            "deleted_objects": deleted_count,
            "deleted_at": datetime.utcnow().isoformat(),
            "warning": "AWS S3 integration disabled - deleted from local storage"
        }
    
    async def get_session_manifest(
        self,
        patient_id: str,
        session_id: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        client_ip: Optional[str] = None
    ) -> Dict:
        """
        Get a manifest of all files for a session.
        STUB: Returns local files since S3 is disabled.
        """
        if user_id and user_role:
            self._authorize_access(user_id, user_role, patient_id, "get_manifest", client_ip)
        
        # STUB: S3 disabled - list local files
        import glob
        pattern = os.path.join(self.local_storage_dir, f"video-exams_{patient_id}_{session_id}_*")
        files = []
        
        for f in glob.glob(pattern):
            files.append({
                "key": f,
                "size_bytes": os.path.getsize(f),
                "last_modified": datetime.fromtimestamp(os.path.getmtime(f)).isoformat()
            })
        
        manifest_files = []
        for f in files:
            url_info = await self.generate_download_url(
                s3_key=f["key"],
                patient_id=patient_id,
                user_id=user_id,
                user_role=user_role,
                client_ip=client_ip
            )
            manifest_files.append({
                **f,
                "download_url": url_info["download_url"],
                "url_expires_at": url_info["expires_at"]
            })
        
        HIPAAAuditLogger.log_phi_access(
            action="get_session_manifest",
            resource_type="video_session",
            resource_id=session_id,
            user_id=user_id or "system",
            patient_id=patient_id,
            access_context={"file_count": len(files), "storage": "local_stub"},
            client_ip=client_ip
        )
        
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "files": manifest_files,
            "total_size_bytes": sum(f.get("size_bytes", 0) for f in files),
            "generated_at": datetime.utcnow().isoformat(),
            "warning": "AWS S3 integration disabled - using local storage"
        }
    
    def configure_lifecycle_policy(self) -> Dict:
        """
        Configure S3 lifecycle policy for HIPAA-compliant retention.
        STUB: S3 is disabled - returns error.
        """
        logger.warning("S3 disabled: Cannot configure lifecycle policy")
        return {
            "success": False,
            "error": "AWS S3 integration disabled - lifecycle policy not available",
            "storage_mode": "local_stub"
        }


# Singleton instance
video_session_storage_service = VideoSessionStorageService()
