"""
Approval Queue Repository
Provides HIPAA-compliant data access for approval queue and tool executions
Uses SQLAlchemy ORM to prevent SQL injection vulnerabilities

IMPORTANT: Column names must match the Drizzle schema in shared/schema.ts
SECURITY: All queries enforce consent verification for PHI access
"""

import uuid
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_, text

from app.models.approval_models import ApprovalQueue, ToolExecution
from app.services.audit_logger import AuditLogger, AuditEvent
from app.services.consent_service import ConsentService

logger = logging.getLogger(__name__)

# Singleton consent service for verification
_consent_service: Optional[ConsentService] = None

def _get_consent_service() -> ConsentService:
    """Get or create consent service singleton"""
    global _consent_service
    if _consent_service is None:
        _consent_service = ConsentService()
    return _consent_service


def _str(value: Any) -> Optional[str]:
    """Safely convert SQLAlchemy column to string"""
    if value is None:
        return None
    return str(value)


def _bool(value: Any) -> bool:
    """Safely convert SQLAlchemy column to bool"""
    return bool(value) if value else False


def _list(value: Any) -> Optional[List[str]]:
    """Safely convert SQLAlchemy column to list"""
    if value is None:
        return None
    return list(value) if value else None


class ApprovalRepository:
    """
    Repository for approval queue operations with HIPAA-compliant audit logging
    Uses approver_id, request_type, requester_id, etc. matching Drizzle schema
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def list_pending_approvals(
        self,
        approver_id: str,
        patient_id: Optional[str] = None,
        urgency: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List pending approvals for an approver (doctor) with consent verification.
        
        SECURITY: Only returns approvals where:
        1. Doctor is explicitly assigned as approver, OR
        2. Doctor has active consent relationship with the patient
        
        This prevents doctors from seeing PHI for patients they don't have consent for.
        """
        consent_service = _get_consent_service()
        
        # First get all pending approvals that could potentially be for this doctor
        base_query = self.db.query(ApprovalQueue).filter(
            ApprovalQueue.status == "pending"
        )
        
        if patient_id:
            base_query = base_query.filter(ApprovalQueue.patient_id == patient_id)
        
        if urgency:
            base_query = base_query.filter(ApprovalQueue.urgency == urgency)
        
        # Get candidates - doctor is either explicit approver or approval is for "doctor" role
        base_query = base_query.filter(
            or_(
                ApprovalQueue.approver_id == approver_id,
                ApprovalQueue.approver_role == "doctor"
            )
        )
        
        candidates = base_query.order_by(
            ApprovalQueue.urgency.desc(),
            ApprovalQueue.created_at.desc()
        ).all()
        
        # Filter by consent - only include approvals where doctor has consent for patient
        consented_approvals = []
        for approval in candidates:
            patient = _str(approval.patient_id)
            
            # If no patient_id, allow (system-level approvals)
            if not patient:
                consented_approvals.append(approval)
                continue
            
            # If doctor is explicitly assigned, allow
            if _str(approval.approver_id) == approver_id:
                consented_approvals.append(approval)
                continue
            
            # Otherwise verify consent
            is_connected, _ = consent_service.verify_connection(
                doctor_id=approver_id,
                patient_id=patient,
                require_consent=True
            )
            
            if is_connected:
                consented_approvals.append(approval)
            else:
                logger.debug(f"Filtering approval {approval.id} - no consent for patient {patient}")
        
        # Apply pagination after consent filtering
        total = len(consented_approvals)
        paginated = consented_approvals[offset:offset + limit]
        
        AuditLogger.log_event(
            event_type=AuditEvent.PHI_ACCESSED,
            user_id=approver_id,
            resource_type="approval_queue",
            resource_id="list",
            action="read",
            status="success",
            metadata={
                "actor_type": "user",
                "actor_role": "doctor",
                "count": len(paginated),
                "total_before_consent_filter": len(candidates),
                "total_after_consent_filter": total,
                "filters": {"patient_id": patient_id, "urgency": urgency}
            },
            ip_address=ip_address,
            phi_accessed=True,
            phi_categories=["medications", "treatment"]
        )
        
        return {
            "approvals": [self._approval_to_dict(a) for a in paginated],
            "total": total,
            "hasMore": offset + limit < total
        }
    
    def get_approval_by_id(
        self,
        approval_id: str,
        approver_id: str,
        ip_address: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get single approval by ID, verifying approver access
        """
        approval = self.db.query(ApprovalQueue).filter(
            ApprovalQueue.id == approval_id,
            or_(
                ApprovalQueue.approver_id == approver_id,
                ApprovalQueue.approver_role == "doctor"
            )
        ).first()
        
        if not approval:
            return None
        
        AuditLogger.log_event(
            event_type=AuditEvent.PHI_ACCESSED,
            user_id=approver_id,
            resource_type="approval_queue",
            resource_id=approval_id,
            action="read",
            status="success",
            metadata={
                "actor_type": "user",
                "actor_role": "doctor",
                "tool_name": _str(approval.tool_name),
                "patient_id": _str(approval.patient_id),
                "request_type": _str(approval.request_type)
            },
            ip_address=ip_address,
            patient_id=_str(approval.patient_id),
            phi_accessed=True,
            phi_categories=["medications", "treatment"]
        )
        
        return self._approval_to_dict(approval)
    
    def create_approval(
        self,
        request_type: str,
        requester_id: str,
        requester_type: str,
        request_payload: Dict[str, Any],
        approver_id: Optional[str] = None,
        approver_role: str = "doctor",
        patient_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        tool_execution_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        request_summary: Optional[str] = None,
        urgency: str = "normal",
        risk_level: Optional[str] = None,
        risk_factors: Optional[List[str]] = None,
        expires_hours: int = 24,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new approval request matching Drizzle schema
        """
        approval_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
        
        approval = ApprovalQueue(
            id=approval_id,
            request_type=request_type,
            requester_id=requester_id,
            requester_type=requester_type,
            approver_id=approver_id,
            approver_role=approver_role,
            patient_id=patient_id,
            conversation_id=conversation_id,
            message_id=message_id,
            tool_execution_id=tool_execution_id,
            tool_name=tool_name,
            request_payload=request_payload,
            request_summary=request_summary,
            urgency=urgency,
            risk_level=risk_level,
            risk_factors=risk_factors,
            status="pending",
            expires_at=expires_at
        )
        
        self.db.add(approval)
        self.db.commit()
        self.db.refresh(approval)
        
        AuditLogger.log_approval_action(
            user_id=requester_id,
            approval_id=approval_id,
            action="create",
            status="success",
            tool_name=tool_name,
            patient_id=patient_id,
            urgency=urgency,
            ip_address=ip_address
        )
        
        return self._approval_to_dict(approval)
    
    def process_decision(
        self,
        approval_id: str,
        approver_id: str,
        decision: str,
        notes: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        modified_payload: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Process an approval decision (approve/reject/modify)
        Uses decision_by, decision_at, decision_notes matching Drizzle schema
        """
        approval = self.db.query(ApprovalQueue).filter(
            ApprovalQueue.id == approval_id,
            ApprovalQueue.status == "pending"
        ).first()
        
        if not approval:
            return None
        
        valid_decisions = ["approved", "rejected", "modified"]
        if decision not in valid_decisions:
            raise ValueError(f"Invalid decision: {decision}. Must be one of {valid_decisions}")
        
        approval.status = decision  # type: ignore[assignment]
        approval.decision = decision  # type: ignore[assignment]
        approval.decision_by = approver_id  # type: ignore[assignment]
        approval.decision_at = datetime.utcnow()  # type: ignore[assignment]
        
        if notes:
            approval.decision_notes = notes  # type: ignore[assignment]
        
        if decision == "rejected" and rejection_reason:
            if not notes:
                approval.decision_notes = rejection_reason  # type: ignore[assignment]
        
        if decision == "modified" and modified_payload:
            approval.modified_payload = modified_payload  # type: ignore[assignment]
        
        self.db.commit()
        self.db.refresh(approval)
        
        AuditLogger.log_approval_action(
            user_id=approver_id,
            approval_id=approval_id,
            action=decision,
            status="success",
            tool_name=_str(approval.tool_name),
            patient_id=_str(approval.patient_id),
            urgency=_str(approval.urgency),
            ip_address=ip_address
        )
        
        if _str(approval.tool_execution_id):
            self._sync_execution_status(
                _str(approval.tool_execution_id),
                decision
            )
        
        return self._approval_to_dict(approval)
    
    def _sync_execution_status(self, execution_id: Optional[str], decision: str):
        """Sync tool execution status based on approval decision"""
        if not execution_id:
            return
            
        status_map = {
            "approved": "approved",
            "rejected": "rejected",
            "modified": "approved"
        }
        
        execution = self.db.query(ToolExecution).filter(
            ToolExecution.id == execution_id
        ).first()
        
        if execution:
            execution.status = status_map.get(decision, decision)  # type: ignore[assignment]
            self.db.commit()
    
    def _approval_to_dict(self, approval: ApprovalQueue) -> Dict[str, Any]:
        """Convert ApprovalQueue model to dictionary matching API response format"""
        decision_at = getattr(approval, 'decision_at', None)
        expires_at = getattr(approval, 'expires_at', None)
        created_at = getattr(approval, 'created_at', None)
        updated_at = getattr(approval, 'updated_at', None)
        executed_at = getattr(approval, 'executed_at', None)
        
        return {
            "id": _str(approval.id),
            "requestType": _str(approval.request_type),
            "requesterId": _str(approval.requester_id),
            "requesterType": _str(approval.requester_type),
            "approverId": _str(approval.approver_id),
            "approverRole": _str(approval.approver_role),
            "patientId": _str(approval.patient_id),
            "conversationId": _str(approval.conversation_id),
            "messageId": _str(approval.message_id),
            "toolExecutionId": _str(approval.tool_execution_id),
            "toolName": _str(approval.tool_name),
            "requestPayload": approval.request_payload,
            "requestSummary": _str(approval.request_summary),
            "urgency": _str(approval.urgency),
            "riskLevel": _str(approval.risk_level),
            "riskFactors": _list(approval.risk_factors),
            "status": _str(approval.status),
            "decision": _str(approval.decision),
            "decisionBy": _str(approval.decision_by),
            "decisionAt": decision_at.isoformat() if decision_at else None,
            "decisionNotes": _str(approval.decision_notes),
            "modifiedPayload": approval.modified_payload,
            "expiresAt": expires_at.isoformat() if expires_at else None,
            "executionResult": approval.execution_result,
            "executedAt": executed_at.isoformat() if executed_at else None,
            "createdAt": created_at.isoformat() if created_at else None,
            "updatedAt": updated_at.isoformat() if updated_at else None
        }


class ToolExecutionRepository:
    """
    Repository for tool execution operations with HIPAA-compliant audit logging
    Uses agent_id, user_id matching Drizzle schema
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_execution(
        self,
        tool_name: str,
        agent_id: str,
        user_id: str,
        input_parameters: Dict[str, Any],
        patient_id: Optional[str] = None,
        doctor_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        message_id: Optional[str] = None,
        tool_version: int = 1,
        phi_accessed: bool = False,
        phi_categories: Optional[List[str]] = None,
        ip_address: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new tool execution record matching Drizzle schema
        """
        execution_id = str(uuid.uuid4())
        
        execution = ToolExecution(
            id=execution_id,
            agent_id=agent_id,
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            tool_name=tool_name,
            tool_version=tool_version,
            input_parameters=input_parameters,
            status="pending",
            started_at=datetime.utcnow(),
            patient_id=patient_id,
            doctor_id=doctor_id,
            phi_accessed=phi_accessed,
            phi_categories=phi_categories
        )
        
        self.db.add(execution)
        self.db.commit()
        self.db.refresh(execution)
        
        AuditLogger.log_tool_execution(
            user_id=user_id,
            tool_name=tool_name,
            tool_id=execution_id,
            patient_id=patient_id,
            action="create",
            status="pending",
            phi_accessed=phi_accessed,
            phi_categories=phi_categories,
            ip_address=ip_address,
            conversation_id=conversation_id,
            message_id=message_id
        )
        
        return self._execution_to_dict(execution)
    
    def update_execution_status(
        self,
        execution_id: str,
        status: str,
        output_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        error_code: Optional[str] = None,
        execution_time_ms: Optional[int] = None,
        ip_address: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update tool execution status and result
        """
        execution = self.db.query(ToolExecution).filter(
            ToolExecution.id == execution_id
        ).first()
        
        if not execution:
            return None
        
        execution.status = status  # type: ignore[assignment]
        
        if output_result is not None:
            execution.output_result = output_result  # type: ignore[assignment]
        
        if error_message is not None:
            execution.error_message = error_message  # type: ignore[assignment]
        
        if error_code is not None:
            execution.error_code = error_code  # type: ignore[assignment]
        
        if execution_time_ms is not None:
            execution.execution_time_ms = execution_time_ms  # type: ignore[assignment]
        
        if status in ["completed", "failed"]:
            execution.completed_at = datetime.utcnow()  # type: ignore[assignment]
        
        self.db.commit()
        self.db.refresh(execution)
        
        actor_id = _str(execution.user_id) or _str(execution.agent_id) or "system"
        AuditLogger.log_tool_execution(
            user_id=actor_id,
            tool_name=_str(execution.tool_name) or "unknown",
            tool_id=execution_id,
            patient_id=_str(execution.patient_id),
            action="complete" if status == "completed" else status,
            status=status,
            phi_accessed=_bool(execution.phi_accessed),
            phi_categories=_list(execution.phi_categories),
            execution_time_ms=execution_time_ms,
            error_message=error_message,
            ip_address=ip_address,
            conversation_id=_str(execution.conversation_id),
            message_id=_str(execution.message_id)
        )
        
        return self._execution_to_dict(execution)
    
    def get_execution_by_id(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get execution by ID"""
        execution = self.db.query(ToolExecution).filter(
            ToolExecution.id == execution_id
        ).first()
        
        if not execution:
            return None
        
        return self._execution_to_dict(execution)
    
    def list_executions(
        self,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List executions with optional filters"""
        query = self.db.query(ToolExecution)
        
        if agent_id:
            query = query.filter(ToolExecution.agent_id == agent_id)
        
        if user_id:
            query = query.filter(ToolExecution.user_id == user_id)
        
        if patient_id:
            query = query.filter(ToolExecution.patient_id == patient_id)
        
        if tool_name:
            query = query.filter(ToolExecution.tool_name == tool_name)
        
        if status:
            query = query.filter(ToolExecution.status == status)
        
        total = query.count()
        
        executions = query.order_by(
            desc(ToolExecution.created_at)
        ).offset(offset).limit(limit).all()
        
        return {
            "executions": [self._execution_to_dict(e) for e in executions],
            "total": total,
            "hasMore": offset + limit < total
        }
    
    def _execution_to_dict(self, execution: ToolExecution) -> Dict[str, Any]:
        """Convert ToolExecution model to dictionary"""
        started_at = getattr(execution, 'started_at', None)
        completed_at = getattr(execution, 'completed_at', None)
        created_at = getattr(execution, 'created_at', None)
        updated_at = getattr(execution, 'updated_at', None)
        
        return {
            "id": _str(execution.id),
            "agentId": _str(execution.agent_id),
            "userId": _str(execution.user_id),
            "conversationId": _str(execution.conversation_id),
            "messageId": _str(execution.message_id),
            "toolName": _str(execution.tool_name),
            "toolVersion": execution.tool_version,
            "inputParameters": execution.input_parameters,
            "outputResult": execution.output_result,
            "status": _str(execution.status),
            "errorMessage": _str(execution.error_message),
            "errorCode": _str(execution.error_code),
            "executionTimeMs": execution.execution_time_ms,
            "startedAt": started_at.isoformat() if started_at else None,
            "completedAt": completed_at.isoformat() if completed_at else None,
            "patientId": _str(execution.patient_id),
            "doctorId": _str(execution.doctor_id),
            "phiAccessed": _bool(execution.phi_accessed),
            "phiCategories": _list(execution.phi_categories),
            "createdAt": created_at.isoformat() if created_at else None,
            "updatedAt": updated_at.isoformat() if updated_at else None
        }
