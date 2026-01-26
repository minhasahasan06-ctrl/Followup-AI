"""
HIPAA-compliant audit logging service
Tracks all PHI access and critical system events
Persists to database for long-term retention
"""

from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import logging
import uuid

from app.database import get_db, SessionLocal

logger = logging.getLogger(__name__)


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
    
    # Agent communication events
    AGENT_MESSAGE_SENT = "agent_message_sent"
    AGENT_MESSAGE_RECEIVED = "agent_message_received"
    AGENT_MESSAGE_READ = "agent_message_read"
    AGENT_CONVERSATION_STARTED = "agent_conversation_started"
    
    # Tool execution events
    TOOL_CALLED = "tool_called"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    TOOL_CONSENT_VERIFIED = "tool_consent_verified"
    TOOL_CONSENT_DENIED = "tool_consent_denied"
    
    # Approval workflow events
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_MODIFIED = "approval_modified"
    APPROVAL_EXPIRED = "approval_expired"
    
    # Consent events
    CONSENT_GRANTED = "consent_granted"
    CONSENT_REVOKED = "consent_revoked"
    CONSENT_VERIFIED = "consent_verified"


class AuditLogger:
    """
    HIPAA-compliant audit logger
    Logs all critical events with user, timestamp, and metadata
    Persists to database for long-term retention and compliance
    """
    
    # Map event types to action categories
    EVENT_CATEGORY_MAP = {
        AuditEvent.CAMERA_ACCESS_REQUESTED: "system",
        AuditEvent.CAMERA_ACCESS_GRANTED: "system",
        AuditEvent.CAMERA_ACCESS_DENIED: "system",
        AuditEvent.VIDEO_EXAM_SESSION_STARTED: "clinical_action",
        AuditEvent.VIDEO_SEGMENT_CAPTURED: "clinical_action",
        AuditEvent.VIDEO_SEGMENT_UPLOADED: "data_access",
        AuditEvent.VIDEO_EXAM_SESSION_COMPLETED: "clinical_action",
        AuditEvent.PHI_ACCESSED: "data_access",
        AuditEvent.PHI_MODIFIED: "data_access",
        AuditEvent.PHI_EXPORTED: "data_access",
        AuditEvent.USER_LOGIN: "system",
        AuditEvent.USER_LOGOUT: "system",
        AuditEvent.AUTH_FAILED: "system",
        AuditEvent.S3_UPLOAD: "data_access",
        AuditEvent.S3_DOWNLOAD: "data_access",
        AuditEvent.KMS_ENCRYPTION: "system",
        AuditEvent.KMS_DECRYPTION: "system",
        # Agent events
        AuditEvent.AGENT_MESSAGE_SENT: "communication",
        AuditEvent.AGENT_MESSAGE_RECEIVED: "communication",
        AuditEvent.AGENT_MESSAGE_READ: "communication",
        AuditEvent.AGENT_CONVERSATION_STARTED: "communication",
        # Tool events
        AuditEvent.TOOL_CALLED: "clinical_action",
        AuditEvent.TOOL_COMPLETED: "clinical_action",
        AuditEvent.TOOL_FAILED: "clinical_action",
        AuditEvent.TOOL_CONSENT_VERIFIED: "data_access",
        AuditEvent.TOOL_CONSENT_DENIED: "data_access",
        # Approval events
        AuditEvent.APPROVAL_REQUESTED: "clinical_action",
        AuditEvent.APPROVAL_GRANTED: "clinical_action",
        AuditEvent.APPROVAL_DENIED: "clinical_action",
        AuditEvent.APPROVAL_MODIFIED: "clinical_action",
        AuditEvent.APPROVAL_EXPIRED: "clinical_action",
        # Consent events
        AuditEvent.CONSENT_GRANTED: "data_access",
        AuditEvent.CONSENT_REVOKED: "data_access",
        AuditEvent.CONSENT_VERIFIED: "data_access",
    }
    
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
        user_agent: Optional[str] = None,
        patient_id: Optional[str] = None,
        phi_accessed: bool = False,
        phi_categories: Optional[List[str]] = None,
        access_reason: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None
    ):
        """
        Log an audit event to database for HIPAA compliance
        
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
            patient_id: ID of patient if action involves patient data
            phi_accessed: Whether PHI was accessed
            phi_categories: Types of PHI accessed (e.g., ['medical_records', 'medications'])
            access_reason: Reason for accessing PHI (for HIPAA compliance)
            session_id: Browser/app session ID
            conversation_id: Agent conversation ID if applicable
            message_id: Agent message ID if applicable
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
            "patient_id": patient_id,
            "phi_accessed": phi_accessed,
            "phi_categories": phi_categories,
        }
        
        # Log to stdout (CloudWatch will capture this in production)
        log_msg = f"[AUDIT] {json.dumps(audit_entry)}"
        if phi_accessed:
            logger.warning(log_msg)  # Higher severity for PHI access
        else:
            logger.info(log_msg)
        
        # Persist to database for long-term retention
        try:
            AuditLogger._persist_to_database(
                event_type=event_type,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                status=status,
                metadata=metadata,
                ip_address=ip_address,
                user_agent=user_agent,
                patient_id=patient_id,
                phi_accessed=phi_accessed,
                phi_categories=phi_categories,
                access_reason=access_reason,
                session_id=session_id,
                conversation_id=conversation_id,
                message_id=message_id
            )
        except Exception as e:
            # Never let audit logging failure break the main flow
            logger.error(f"Failed to persist audit log to database: {e}")
        
        return audit_entry
    
    @staticmethod
    def _persist_to_database(
        event_type: str,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str],
        action: str,
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        patient_id: Optional[str] = None,
        phi_accessed: bool = False,
        phi_categories: Optional[List[str]] = None,
        access_reason: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None
    ):
        """Persist audit event to agent_audit_logs table"""
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            # Determine action category
            action_category = AuditLogger.EVENT_CATEGORY_MAP.get(event_type, "system")
            
            # Determine actor type from metadata or default to 'user'
            actor_type = "user"
            if metadata and "actor_type" in metadata:
                actor_type = metadata["actor_type"]
            elif not user_id:
                actor_type = "system"
            
            # Determine actor role from metadata
            actor_role = None
            if metadata and "actor_role" in metadata:
                actor_role = metadata["actor_role"]
            
            # Success flag from status
            success = status == "success"
            error_code = None
            error_message = None
            if status == "failure":
                error_code = metadata.get("error_code") if metadata else None
                error_message = metadata.get("error_message") if metadata else None
            
            query = text("""
                INSERT INTO agent_audit_logs (
                    id, actor_type, actor_id, actor_role, action, action_category,
                    object_type, object_id, patient_id, conversation_id, message_id,
                    phi_accessed, phi_categories, access_reason, details,
                    ip_address, user_agent, session_id, success, error_code, error_message,
                    timestamp, created_at
                ) VALUES (
                    :id, :actor_type, :actor_id, :actor_role, :action, :action_category,
                    :object_type, :object_id, :patient_id, :conversation_id, :message_id,
                    :phi_accessed, :phi_categories, :access_reason, :details,
                    :ip_address, :user_agent, :session_id, :success, :error_code, :error_message,
                    NOW(), NOW()
                )
            """)
            
            db.execute(query, {
                "id": str(uuid.uuid4()),
                "actor_type": actor_type,
                "actor_id": user_id or "system",
                "actor_role": actor_role,
                "action": f"{event_type}:{action}",
                "action_category": action_category,
                "object_type": resource_type,
                "object_id": resource_id or "unknown",
                "patient_id": patient_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "phi_accessed": phi_accessed,
                "phi_categories": json.dumps(phi_categories) if phi_categories else None,
                "access_reason": access_reason,
                "details": json.dumps(metadata) if metadata else None,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "session_id": session_id,
                "success": success,
                "error_code": error_code,
                "error_message": error_message
            })
            db.commit()
            
        except Exception as e:
            db.rollback()
            logger.error(f"Database error persisting audit log: {e}")
            raise
        finally:
            db.close()
    
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
    
    @staticmethod
    def log_tool_execution(
        user_id: str,
        tool_name: str,
        tool_id: str,
        patient_id: Optional[str] = None,
        action: str = "execute",
        status: str = "success",
        phi_accessed: bool = False,
        phi_categories: Optional[List[str]] = None,
        access_reason: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None
    ):
        """Log tool execution for HIPAA compliance"""
        event_type = AuditEvent.TOOL_COMPLETED if status == "success" else AuditEvent.TOOL_FAILED
        
        return AuditLogger.log_event(
            event_type=event_type,
            user_id=user_id,
            resource_type="tool",
            resource_id=tool_id,
            action=action,
            status=status,
            metadata={
                "tool_name": tool_name,
                "execution_time_ms": execution_time_ms,
                "error_message": error_message,
                "actor_type": "agent"
            },
            ip_address=ip_address,
            patient_id=patient_id,
            phi_accessed=phi_accessed,
            phi_categories=phi_categories,
            access_reason=access_reason,
            conversation_id=conversation_id,
            message_id=message_id
        )
    
    @staticmethod
    def log_phi_access(
        user_id: str,
        patient_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        phi_categories: List[str],
        access_reason: str,
        status: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        conversation_id: Optional[str] = None
    ):
        """Log PHI access for HIPAA compliance"""
        return AuditLogger.log_event(
            event_type=AuditEvent.PHI_ACCESSED,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            status=status,
            metadata={
                "actor_type": "user",
            },
            ip_address=ip_address,
            user_agent=user_agent,
            patient_id=patient_id,
            phi_accessed=True,
            phi_categories=phi_categories,
            access_reason=access_reason,
            conversation_id=conversation_id
        )
    
    @staticmethod
    def log_approval_action(
        doctor_id: str,
        approval_id: str,
        patient_id: Optional[str],
        action: str,  # 'requested', 'granted', 'denied', 'modified'
        tool_name: Optional[str] = None,
        notes: Optional[str] = None,
        ip_address: Optional[str] = None,
        conversation_id: Optional[str] = None
    ):
        """Log approval workflow actions"""
        event_map = {
            "requested": AuditEvent.APPROVAL_REQUESTED,
            "granted": AuditEvent.APPROVAL_GRANTED,
            "denied": AuditEvent.APPROVAL_DENIED,
            "modified": AuditEvent.APPROVAL_MODIFIED,
            "expired": AuditEvent.APPROVAL_EXPIRED,
        }
        event_type = event_map.get(action, AuditEvent.APPROVAL_REQUESTED)
        
        return AuditLogger.log_event(
            event_type=event_type,
            user_id=doctor_id,
            resource_type="approval",
            resource_id=approval_id,
            action=action,
            status="success",
            metadata={
                "tool_name": tool_name,
                "notes": notes,
                "actor_type": "user",
                "actor_role": "doctor"
            },
            ip_address=ip_address,
            patient_id=patient_id,
            phi_accessed=False,
            conversation_id=conversation_id
        )
    
    @staticmethod
    def log_consent_verification(
        user_id: str,
        doctor_id: str,
        patient_id: str,
        consent_verified: bool,
        access_scope: Optional[str] = None,
        tool_name: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """Log consent verification for doctor-patient relationships"""
        event_type = AuditEvent.CONSENT_VERIFIED if consent_verified else AuditEvent.TOOL_CONSENT_DENIED
        
        return AuditLogger.log_event(
            event_type=event_type,
            user_id=user_id,
            resource_type="consent",
            resource_id=f"{doctor_id}:{patient_id}",
            action="verify",
            status="success" if consent_verified else "denied",
            metadata={
                "doctor_id": doctor_id,
                "access_scope": access_scope,
                "tool_name": tool_name,
                "actor_type": "system"
            },
            ip_address=ip_address,
            patient_id=patient_id,
            phi_accessed=False
        )
    
    @staticmethod
    def log_agent_message(
        user_id: str,
        agent_id: str,
        message_id: str,
        conversation_id: str,
        action: str,  # 'sent', 'received', 'read'
        message_type: str = "chat",
        ip_address: Optional[str] = None
    ):
        """Log agent communication events"""
        event_map = {
            "sent": AuditEvent.AGENT_MESSAGE_SENT,
            "received": AuditEvent.AGENT_MESSAGE_RECEIVED,
            "read": AuditEvent.AGENT_MESSAGE_READ,
        }
        event_type = event_map.get(action, AuditEvent.AGENT_MESSAGE_SENT)
        
        return AuditLogger.log_event(
            event_type=event_type,
            user_id=user_id,
            resource_type="agent_message",
            resource_id=message_id,
            action=action,
            status="success",
            metadata={
                "agent_id": agent_id,
                "message_type": message_type,
                "actor_type": "user"
            },
            ip_address=ip_address,
            conversation_id=conversation_id,
            message_id=message_id
        )
