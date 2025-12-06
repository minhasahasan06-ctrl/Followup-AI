"""
HIPAA-compliant audit logging service
Tracks all PHI access and critical system events
"""

from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any
import json

from app.database import get_db
from app.models import User


class AuditEvent:
    """Audit event types for HIPAA compliance"""
    # Video examination events
    CAMERA_ACCESS_REQUESTED = "camera_access_requested"
    CAMERA_ACCESS_GRANTED = "camera_access_granted"
    CAMERA_ACCESS_DENIED = "camera_access_denied"
    
    VIDEO_EXAM_SESSION_STARTED = "video_exam_session_started"
    VIDEO_SEGMENT_CAPTURED = "video_segment_captured"
    VIDEO_SEGMENT_UPLOADED = "video_segment_uploaded"
    VIDEO_EXAM_SESSION_COMPLETED = "video_exam_session_completed"
    
    # PHI access events
    PHI_ACCESSED = "phi_accessed"
    PHI_MODIFIED = "phi_modified"
    PHI_EXPORTED = "phi_exported"
    
    # Authentication events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    AUTH_FAILED = "auth_failed"
    
    # System events
    S3_UPLOAD = "s3_upload"
    S3_DOWNLOAD = "s3_download"
    KMS_ENCRYPTION = "kms_encryption"
    KMS_DECRYPTION = "kms_decryption"


class AuditLogger:
    """
    HIPAA-compliant audit logger
    Logs all critical events with user, timestamp, and metadata
    """
    
    @staticmethod
    def log_event(
        event_type: str,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """
        Log an audit event to database and optionally to CloudWatch
        
        Args:
            event_type: Type of event (use AuditEvent constants)
            user_id: ID of user performing action
            resource_type: Type of resource accessed (e.g., 'video_exam_segment')
            resource_id: ID of resource
            action: Action performed (e.g., 'upload', 'view', 'delete')
            status: Result status ('success', 'failure', 'denied')
            metadata: Additional context data
            ip_address: Client IP address
            user_agent: Client user agent
        """
        
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "action": action,
            "status": status,
            "metadata": metadata or {},
            "ip_address": ip_address,
            "user_agent": user_agent,
        }
        
        # Log to stdout (CloudWatch will capture this)
        # Use secure logging instead of print to prevent information leakage
        import logging
        logger = logging.getLogger("audit")
        logger.info(f"[AUDIT] {json.dumps(audit_entry)}")
        
        # TODO: Store in dedicated audit_logs table for long-term retention
        # TODO: Send to AWS CloudWatch Logs for centralized monitoring
        
        return audit_entry
    
    @staticmethod
    def log_video_exam_session_started(
        user_id: str,
        session_id: str,
        ip_address: Optional[str] = None
    ):
        """Log when a user starts a video examination session"""
        return AuditLogger.log_event(
            event_type=AuditEvent.VIDEO_EXAM_SESSION_STARTED,
            user_id=user_id,
            resource_type="video_exam_session",
            resource_id=session_id,
            action="create",
            status="success",
            metadata={"session_type": "guided_examination"},
            ip_address=ip_address
        )
    
    @staticmethod
    def log_video_segment_uploaded(
        user_id: str,
        session_id: str,
        segment_id: str,
        exam_type: str,
        s3_key: str,
        file_size_bytes: int,
        encrypted: bool = True,
        ip_address: Optional[str] = None
    ):
        """Log when a video segment is uploaded to S3"""
        return AuditLogger.log_event(
            event_type=AuditEvent.VIDEO_SEGMENT_UPLOADED,
            user_id=user_id,
            resource_type="video_exam_segment",
            resource_id=segment_id,
            action="upload",
            status="success",
            metadata={
                "session_id": session_id,
                "exam_type": exam_type,
                "s3_key": s3_key,
                "file_size_bytes": file_size_bytes,
                "encrypted": encrypted
            },
            ip_address=ip_address
        )
    
    @staticmethod
    def log_video_exam_session_completed(
        user_id: str,
        session_id: str,
        completed_segments: int,
        skipped_segments: int,
        total_duration_seconds: int,
        ip_address: Optional[str] = None
    ):
        """Log when a video examination session is completed"""
        return AuditLogger.log_event(
            event_type=AuditEvent.VIDEO_EXAM_SESSION_COMPLETED,
            user_id=user_id,
            resource_type="video_exam_session",
            resource_id=session_id,
            action="complete",
            status="success",
            metadata={
                "completed_segments": completed_segments,
                "skipped_segments": skipped_segments,
                "total_duration_seconds": total_duration_seconds
            },
            ip_address=ip_address
        )
    
    @staticmethod
    def log_camera_access(
        user_id: str,
        status: str,  # 'granted' or 'denied'
        exam_type: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log camera access attempts"""
        event_type = (
            AuditEvent.CAMERA_ACCESS_GRANTED 
            if status == "granted" 
            else AuditEvent.CAMERA_ACCESS_DENIED
        )
        
        return AuditLogger.log_event(
            event_type=event_type,
            user_id=user_id,
            resource_type="camera_device",
            resource_id=None,
            action="access",
            status=status,
            metadata={"exam_type": exam_type},
            ip_address=ip_address
        )
    
    @staticmethod
    def log_s3_operation(
        user_id: str,
        operation: str,  # 'upload' or 'download'
        s3_key: str,
        bucket: str,
        encrypted: bool,
        kms_key_id: Optional[str] = None,
        status: str = "success",
        ip_address: Optional[str] = None
    ):
        """Log S3 operations for HIPAA compliance"""
        return AuditLogger.log_event(
            event_type=AuditEvent.S3_UPLOAD if operation == "upload" else AuditEvent.S3_DOWNLOAD,
            user_id=user_id,
            resource_type="s3_object",
            resource_id=s3_key,
            action=operation,
            status=status,
            metadata={
                "bucket": bucket,
                "encrypted": encrypted,
                "kms_key_id": kms_key_id
            },
            ip_address=ip_address
        )
