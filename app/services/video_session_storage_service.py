"""
Video Session Storage Service
HIPAA-compliant storage for video exam frames and recordings with:
- S3 server-side encryption (SSE-S3 or SSE-KMS)
- Pre-signed URLs for secure upload/download (15-minute TTL)
- Lifecycle policies for automatic retention management
- Comprehensive HIPAA audit logging
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from enum import Enum

import boto3
from botocore.exceptions import ClientError
from botocore.config import Config as BotoConfig

from app.config import settings
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)


class StorageType(str, Enum):
    EXAM_FRAME = "exam-frame"
    EXAM_RECORDING = "exam-recording"
    COMBINED_SESSION = "combined-session"


class VideoSessionStorageService:
    """
    HIPAA-compliant video session storage service.
    
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
        self.use_s3 = self._should_use_s3()
        self.bucket_name = os.getenv('AWS_S3_BUCKET_NAME')
        self.kms_key_id = os.getenv('AWS_KMS_KEY_ID')
        self.region = self._parse_region(os.getenv('AWS_REGION', 'us-east-1'))
        
        if self.use_s3:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=self.region,
                config=BotoConfig(signature_version='s3v4')
            )
            logger.info(f"VideoSessionStorageService initialized with S3 (bucket: {self.bucket_name})")
        else:
            self.s3_client = None
            self.local_storage_dir = 'tmp/video_sessions'
            os.makedirs(self.local_storage_dir, exist_ok=True)
            logger.warning("VideoSessionStorageService using LOCAL storage fallback")
    
    def _should_use_s3(self) -> bool:
        """Check if S3 credentials are available"""
        return all([
            os.getenv('AWS_ACCESS_KEY_ID'),
            os.getenv('AWS_SECRET_ACCESS_KEY'),
            os.getenv('AWS_S3_BUCKET_NAME')
        ])
    
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
        Generate a pre-signed URL for secure direct upload to S3.
        
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
        
        if not self.use_s3:
            local_path = os.path.join(self.local_storage_dir, s3_key.replace('/', '_'))
            HIPAAAuditLogger.log_phi_access(
                action="generate_upload_url",
                resource_type="video_session_frame",
                resource_id=session_id,
                user_id=user_id or "system",
                patient_id=patient_id,
                access_context={"storage": "local", "s3_key": s3_key},
                client_ip=client_ip
            )
            return {
                "upload_url": f"file://{os.path.abspath(local_path)}",
                "s3_key": s3_key,
                "bucket": "local",
                "expires_at": expires_at.isoformat(),
                "method": "PUT",
                "storage_mode": "local"
            }
        
        try:
            presigned_params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'ContentType': content_type,
            }
            
            conditions = [
                {"bucket": self.bucket_name},
                {"key": s3_key},
                {"Content-Type": content_type},
                {"x-amz-server-side-encryption": "AES256"},
            ]
            
            if self.kms_key_id:
                conditions[-1] = {"x-amz-server-side-encryption": "aws:kms"}
                conditions.append({"x-amz-server-side-encryption-aws-kms-key-id": self.kms_key_id})
            
            if file_size_bytes:
                conditions.append(["content-length-range", file_size_bytes - 1000, file_size_bytes + 1000])
            
            presigned_url = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                'put_object',
                Params={
                    **presigned_params,
                    'ServerSideEncryption': 'aws:kms' if self.kms_key_id else 'AES256',
                    **({"SSEKMSKeyId": self.kms_key_id} if self.kms_key_id else {})
                },
                ExpiresIn=self.SIGNED_URL_TTL_SECONDS
            )
            
            HIPAAAuditLogger.log_phi_access(
                action="generate_upload_url",
                resource_type="video_session_frame",
                resource_id=session_id,
                user_id=user_id or "system",
                patient_id=patient_id,
                access_context={
                    "s3_key": s3_key,
                    "storage_type": storage_type.value,
                    "stage": stage,
                    "expires_at": expires_at.isoformat(),
                    "encrypted": True,
                    "kms_enabled": bool(self.kms_key_id)
                },
                client_ip=client_ip
            )
            
            return {
                "upload_url": presigned_url,
                "s3_key": s3_key,
                "bucket": self.bucket_name,
                "expires_at": expires_at.isoformat(),
                "method": "PUT",
                "storage_mode": "s3",
                "encryption": "aws:kms" if self.kms_key_id else "AES256"
            }
            
        except ClientError as e:
            logger.error(f"Failed to generate upload URL: {e}")
            raise
    
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
        Generate a pre-signed URL for secure download from S3.
        
        Enforces RBAC: patients can only download their own session data.
        Always generates fresh URLs (never stores) - 15-minute TTL.
        """
        if user_id and user_role:
            self._authorize_access(user_id, user_role, patient_id, "download", client_ip)
        
        expires_at = datetime.utcnow() + timedelta(seconds=self.SIGNED_URL_TTL_SECONDS)
        
        if not self.use_s3:
            local_path = os.path.join(self.local_storage_dir, s3_key.replace('/', '_'))
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            HIPAAAuditLogger.log_phi_access(
                action="generate_download_url",
                resource_type="video_session_frame",
                resource_id=s3_key,
                user_id=user_id or "system",
                patient_id=patient_id,
                access_context={"storage": "local"},
                client_ip=client_ip
            )
            
            return {
                "download_url": f"file://{os.path.abspath(local_path)}",
                "s3_key": s3_key,
                "expires_at": expires_at.isoformat(),
                "storage_mode": "local"
            }
        
        try:
            presigned_url = await asyncio.to_thread(
                self.s3_client.generate_presigned_url,
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=self.SIGNED_URL_TTL_SECONDS
            )
            
            HIPAAAuditLogger.log_phi_access(
                action="generate_download_url",
                resource_type="video_session_frame",
                resource_id=s3_key,
                user_id=user_id or "system",
                patient_id=patient_id,
                access_context={
                    "url_expiry": expires_at.isoformat(),
                    "user_agent": user_agent
                },
                client_ip=client_ip
            )
            
            return {
                "download_url": presigned_url,
                "s3_key": s3_key,
                "expires_at": expires_at.isoformat(),
                "storage_mode": "s3"
            }
            
        except ClientError as e:
            logger.error(f"Failed to generate download URL: {e}")
            raise
    
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
        Upload a frame directly (server-side upload for backend processing).
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
        
        if not self.use_s3:
            local_path = os.path.join(self.local_storage_dir, s3_key.replace('/', '_'))
            os.makedirs(os.path.dirname(local_path) if '/' in s3_key else self.local_storage_dir, exist_ok=True)
            
            def _write_file():
                with open(local_path, 'wb') as f:
                    f.write(file_data)
            
            await asyncio.to_thread(_write_file)
            
            HIPAAAuditLogger.log_phi_access(
                action="upload_frame",
                resource_type="video_session_frame",
                resource_id=session_id,
                user_id=user_id or "system",
                patient_id=patient_id,
                access_context={"stage": stage, "storage": "local", "size_bytes": len(file_data)},
                client_ip=client_ip
            )
            
            return {
                "s3_key": s3_key,
                "s3_uri": f"file://{os.path.abspath(local_path)}",
                "bucket": "local",
                "size_bytes": len(file_data),
                "storage_mode": "local"
            }
        
        try:
            upload_params = {
                'Bucket': self.bucket_name,
                'Key': s3_key,
                'Body': file_data,
                'ContentType': content_type,
                'ServerSideEncryption': 'AES256'
            }
            
            if self.kms_key_id:
                upload_params['ServerSideEncryption'] = 'aws:kms'
                upload_params['SSEKMSKeyId'] = self.kms_key_id
            
            await asyncio.to_thread(self.s3_client.put_object, **upload_params)
            
            HIPAAAuditLogger.log_phi_access(
                action="upload_frame",
                resource_type="video_session_frame",
                resource_id=session_id,
                user_id=user_id or "system",
                patient_id=patient_id,
                access_context={
                    "stage": stage,
                    "s3_key": s3_key,
                    "size_bytes": len(file_data),
                    "encrypted": True
                },
                client_ip=client_ip
            )
            
            return {
                "s3_key": s3_key,
                "s3_uri": f"s3://{self.bucket_name}/{s3_key}",
                "bucket": self.bucket_name,
                "size_bytes": len(file_data),
                "storage_mode": "s3",
                "kms_key_id": self.kms_key_id
            }
            
        except ClientError as e:
            logger.error(f"Failed to upload frame: {e}")
            raise
    
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
        prefix = f"video-exams/{patient_id}/{session_id}/"
        deleted_count = 0
        
        if not self.use_s3:
            import glob
            pattern = os.path.join(self.local_storage_dir, f"video-exams_{patient_id}_{session_id}_*")
            for f in glob.glob(pattern):
                os.remove(f)
                deleted_count += 1
        else:
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                objects_to_delete = []
                for page in pages:
                    for obj in page.get('Contents', []):
                        objects_to_delete.append({'Key': obj['Key']})
                
                if objects_to_delete:
                    await asyncio.to_thread(
                        self.s3_client.delete_objects,
                        Bucket=self.bucket_name,
                        Delete={'Objects': objects_to_delete}
                    )
                    deleted_count = len(objects_to_delete)
                    
            except ClientError as e:
                logger.error(f"Failed to delete session content: {e}")
                raise
        
        HIPAAAuditLogger.log_phi_access(
            action="delete_session_content",
            resource_type="video_session",
            resource_id=session_id,
            user_id=user_id or "system",
            patient_id=patient_id,
            access_context={"deleted_count": deleted_count, "reason": "HIPAA_deletion_request"},
            client_ip=client_ip
        )
        
        return {
            "session_id": session_id,
            "deleted_objects": deleted_count,
            "deleted_at": datetime.utcnow().isoformat()
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
        Get a manifest of all files for a session with fresh signed URLs.
        Enforces RBAC: patients can only view their own session manifests.
        """
        if user_id and user_role:
            self._authorize_access(user_id, user_role, patient_id, "get_manifest", client_ip)
        prefix = f"video-exams/{patient_id}/{session_id}/"
        files = []
        
        if not self.use_s3:
            import glob
            pattern = os.path.join(self.local_storage_dir, f"video-exams_{patient_id}_{session_id}_*")
            for f in glob.glob(pattern):
                files.append({
                    "key": f,
                    "size_bytes": os.path.getsize(f),
                    "last_modified": datetime.fromtimestamp(os.path.getmtime(f)).isoformat()
                })
        else:
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)
                
                for page in pages:
                    for obj in page.get('Contents', []):
                        files.append({
                            "key": obj['Key'],
                            "size_bytes": obj['Size'],
                            "last_modified": obj['LastModified'].isoformat(),
                            "etag": obj.get('ETag', '').strip('"')
                        })
                        
            except ClientError as e:
                logger.error(f"Failed to get session manifest: {e}")
                raise
        
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
            access_context={"file_count": len(files)},
            client_ip=client_ip
        )
        
        return {
            "session_id": session_id,
            "patient_id": patient_id,
            "files": manifest_files,
            "total_size_bytes": sum(f.get("size_bytes", 0) for f in files),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def configure_lifecycle_policy(self) -> Dict:
        """
        Configure S3 lifecycle policy for HIPAA-compliant retention.
        Should be called once during bucket setup.
        
        Note: This modifies bucket policy - requires appropriate IAM permissions.
        """
        if not self.use_s3:
            return {"status": "skipped", "reason": "local_storage"}
        
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'VideoExamRetention7Years',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'video-exams/'},
                    'Expiration': {'Days': 2555},
                    'NoncurrentVersionExpiration': {'NoncurrentDays': 2555},
                    'AbortIncompleteMultipartUpload': {'DaysAfterInitiation': 7}
                },
                {
                    'ID': 'TransitionToGlacierAfter1Year',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'video-exams/'},
                    'Transitions': [
                        {'Days': 365, 'StorageClass': 'GLACIER'}
                    ]
                }
            ]
        }
        
        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=self.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            logger.info(f"Configured lifecycle policy for bucket {self.bucket_name}")
            return {"status": "configured", "rules": len(lifecycle_config['Rules'])}
        except ClientError as e:
            logger.error(f"Failed to configure lifecycle policy: {e}")
            return {"status": "error", "error": str(e)}


video_session_storage_service = VideoSessionStorageService()
