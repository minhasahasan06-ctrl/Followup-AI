"""
Training Security Module
========================
Production-grade security for ML training infrastructure:
- JWT-based admin authentication
- Database-backed audit logging
- Structured failure types

HIPAA-compliant with comprehensive access control.
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from enum import Enum
from functools import wraps

import jwt
import psycopg2
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TrainingAuditAction(str, Enum):
    """Actions that can be audited"""
    JOB_CREATED = "ml_training_job_created"
    JOB_STARTED = "ml_training_job_started"
    JOB_COMPLETED = "ml_training_job_completed"
    JOB_FAILED = "ml_training_job_failed"
    JOB_CANCELLED = "ml_training_job_cancelled"
    JOB_RETRIED = "ml_training_job_retried"
    CONSENT_VERIFIED = "ml_training_consent_verified"
    CONSENT_FAILED = "ml_training_consent_failed"
    GOVERNANCE_VERIFIED = "ml_training_governance_verified"
    GOVERNANCE_FAILED = "ml_training_governance_failed"
    MODEL_DEPLOYED = "ml_model_deployed"
    MODEL_ARCHIVED = "ml_model_archived"
    ARTIFACT_STORED = "ml_artifact_stored"
    WORKER_STARTED = "ml_worker_started"
    WORKER_STOPPED = "ml_worker_stopped"
    API_ACCESS = "ml_training_api_access"
    AUTH_FAILED = "ml_training_auth_failed"


class FailureType(str, Enum):
    """Structured failure types for training jobs"""
    CONSENT_NOT_FOUND = "consent_not_found"
    CONSENT_INSUFFICIENT = "consent_insufficient"
    CONSENT_DB_ERROR = "consent_database_error"
    GOVERNANCE_REQUIRED = "governance_approval_required"
    GOVERNANCE_DENIED = "governance_denied"
    GOVERNANCE_EXPIRED = "governance_expired"
    DATA_INSUFFICIENT = "training_data_insufficient"
    MODEL_ERROR = "model_training_error"
    ARTIFACT_ERROR = "artifact_storage_error"
    TIMEOUT = "training_timeout"
    UNKNOWN = "unknown_error"


class TrainingFailure(Exception):
    """Structured exception for training failures"""
    
    def __init__(
        self,
        failure_type: FailureType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        retryable: bool = False
    ):
        self.failure_type = failure_type
        self.message = message
        self.details = details or {}
        self.retryable = retryable
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_type": self.failure_type.value,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable
        }


class AdminUser(BaseModel):
    """Authenticated admin user"""
    user_id: str
    email: str
    role: str


def get_db_connection():
    """Get database connection for audit logging"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        return None
    try:
        return psycopg2.connect(database_url)
    except Exception as e:
        logger.error(f"Failed to connect to database for audit logging: {e}")
        return None


def verify_admin_token(token: str) -> Optional[AdminUser]:
    """
    Verify JWT token and ensure user has admin/doctor role.
    Uses same JWT verification as main Python backend.
    """
    try:
        # Get secret from environment (matches app/utils/security.py)
        secret = os.environ.get('DEV_MODE_SECRET') or os.environ.get('SESSION_SECRET', 'dev-secret')
        
        payload = jwt.decode(token, secret, algorithms=['HS256'])
        
        user_id = payload.get('sub')
        email = payload.get('email', '')
        role = payload.get('role', '')
        
        if not user_id:
            logger.warning("Token missing user ID")
            return None
        
        # Require admin or doctor role for ML training access
        if role not in ['admin', 'doctor']:
            logger.warning(f"User {user_id} has insufficient role: {role}")
            return None
        
        return AdminUser(user_id=user_id, email=email, role=role)
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None


class TrainingAuditLogger:
    """
    Database-backed audit logger for ML training operations.
    Writes to autopilot_audit_logs table for HIPAA compliance.
    """
    
    def __init__(self):
        self._conn = None
    
    def _get_connection(self):
        """Get or create database connection"""
        if self._conn is None or self._conn.closed:
            self._conn = get_db_connection()
        return self._conn
    
    def log(
        self,
        action: TrainingAuditAction,
        entity_type: str,
        entity_id: Optional[str] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        old_values: Optional[Dict[str, Any]] = None,
        new_values: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> bool:
        """
        Log an audit entry to the database.
        
        Args:
            action: The action being performed
            entity_type: Type of entity (job, model, artifact)
            entity_id: ID of the entity
            user_id: ID of user performing action
            patient_id: Patient ID if applicable
            old_values: Previous state (for updates)
            new_values: New state (for updates/creates)
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            True if logged successfully, False otherwise
        """
        conn = self._get_connection()
        if not conn:
            logger.warning(f"Audit log to database failed - no connection. Action: {action.value}")
            return False
        
        try:
            import json
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autopilot_audit_logs 
                    (id, action, entity_type, entity_id, user_id, patient_id, 
                     old_values, new_values, ip_address, user_agent, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    str(uuid.uuid4()),
                    action.value,
                    entity_type,
                    entity_id,
                    user_id,
                    patient_id,
                    json.dumps(old_values) if old_values else None,
                    json.dumps(new_values) if new_values else None,
                    ip_address,
                    user_agent,
                    datetime.now(timezone.utc)
                ))
                conn.commit()
                logger.debug(f"Audit logged: {action.value} on {entity_type}/{entity_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
            try:
                conn.rollback()
            except:
                pass
            return False
    
    def log_job_event(
        self,
        job_id: str,
        action: TrainingAuditAction,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Convenience method for job-related audit logs"""
        return self.log(
            action=action,
            entity_type="training_job",
            entity_id=job_id,
            user_id=user_id,
            new_values=details
        )
    
    def log_consent_result(
        self,
        job_id: str,
        success: bool,
        patient_count: int = 0,
        categories_verified: Optional[list] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Log consent verification result"""
        action = TrainingAuditAction.CONSENT_VERIFIED if success else TrainingAuditAction.CONSENT_FAILED
        return self.log(
            action=action,
            entity_type="training_job",
            entity_id=job_id,
            new_values={
                "success": success,
                "patient_count": patient_count,
                "categories_verified": categories_verified or [],
                "error_message": error_message
            }
        )
    
    def log_governance_result(
        self,
        job_id: str,
        success: bool,
        approval_id: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Log governance verification result"""
        action = TrainingAuditAction.GOVERNANCE_VERIFIED if success else TrainingAuditAction.GOVERNANCE_FAILED
        return self.log(
            action=action,
            entity_type="training_job",
            entity_id=job_id,
            new_values={
                "success": success,
                "approval_id": approval_id,
                "error_message": error_message
            }
        )
    
    def log_api_access(
        self,
        endpoint: str,
        user: Optional[AdminUser] = None,
        success: bool = True,
        ip_address: Optional[str] = None
    ) -> bool:
        """Log API access attempt"""
        action = TrainingAuditAction.API_ACCESS if success else TrainingAuditAction.AUTH_FAILED
        return self.log(
            action=action,
            entity_type="api_endpoint",
            entity_id=endpoint,
            user_id=user.user_id if user else None,
            ip_address=ip_address,
            new_values={"success": success, "role": user.role if user else None}
        )
    
    def close(self):
        """Close database connection"""
        if self._conn and not self._conn.closed:
            self._conn.close()


# Global audit logger instance
_audit_logger: Optional[TrainingAuditLogger] = None


def get_audit_logger() -> TrainingAuditLogger:
    """Get global audit logger instance"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = TrainingAuditLogger()
    return _audit_logger


def require_admin_auth(f):
    """
    Decorator to require admin authentication on API endpoints.
    Must be used on FastAPI route handlers that have request parameter.
    """
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        from fastapi import Request, HTTPException
        
        # Get request from kwargs or args
        request = kwargs.get('request')
        if request is None:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
        
        if request is None:
            raise HTTPException(status_code=500, detail="Internal server error")
        
        # Extract token from Authorization header
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            audit_logger = get_audit_logger()
            audit_logger.log_api_access(
                endpoint=str(request.url.path),
                success=False,
                ip_address=request.client.host if request.client else None
            )
            raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        user = verify_admin_token(token)
        
        if user is None:
            audit_logger = get_audit_logger()
            audit_logger.log_api_access(
                endpoint=str(request.url.path),
                success=False,
                ip_address=request.client.host if request.client else None
            )
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        # Store user in request state for endpoint to use
        request.state.admin_user = user
        
        # Log successful access
        audit_logger = get_audit_logger()
        audit_logger.log_api_access(
            endpoint=str(request.url.path),
            user=user,
            success=True,
            ip_address=request.client.host if request.client else None
        )
        
        return await f(*args, **kwargs)
    
    return decorated_function
