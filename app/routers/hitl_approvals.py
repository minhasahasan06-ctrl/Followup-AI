"""
Human-in-the-Loop (HITL) Approvals API
Doctor-facing endpoints for reviewing and acting on AI-generated recommendations.

HIPAA-compliant with comprehensive audit logging.
Integrates with Assistant Lysa dashboard.
"""

import logging
import uuid
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body, Header
from sqlalchemy.orm import Session
from sqlalchemy import text, desc
from pydantic import BaseModel, Field

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User

from app.models.followup_autopilot_models import (
    AutopilotPendingApproval,
    AutopilotApprovalHistory,
    AutopilotTriggerEvent,
    AutopilotPatientState,
    AutopilotNotification,
    AutopilotFollowupTask
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/hitl", tags=["hitl-approvals"])

def verify_internal_service_token(x_service_token: Optional[str] = Header(None, alias="X-Service-Token")) -> str:
    """
    Verify internal service token for protected internal endpoints.
    Token must be set via AUTOPILOT_SERVICE_TOKEN environment variable.
    """
    expected_token = os.getenv("AUTOPILOT_SERVICE_TOKEN")
    
    if not expected_token:
        # In development mode without token configured, allow internal localhost calls only
        # In production, token MUST be set
        if os.getenv("NODE_ENV") == "production":
            logger.error("AUTOPILOT_SERVICE_TOKEN not configured in production")
            raise HTTPException(status_code=500, detail="Service misconfiguration")
        logger.warning("AUTOPILOT_SERVICE_TOKEN not set, internal endpoint accessible in dev mode")
        return "dev-mode"
    
    if not x_service_token or x_service_token != expected_token:
        logger.warning("Invalid or missing service token attempt on internal endpoint")
        raise HTTPException(status_code=401, detail="Invalid or missing service token")
    
    return x_service_token


class ApprovalDecision(BaseModel):
    action: str = Field(..., pattern="^(approve|modify|reject)$")
    notes: Optional[str] = Field(None, max_length=1000)
    modified_action: Optional[Dict[str, Any]] = None
    rejection_reason: Optional[str] = Field(None, max_length=500)


class CreateApprovalRequest(BaseModel):
    patient_id: str
    doctor_id: str
    action_type: str
    title: str
    ai_recommendation: str
    ai_reasoning: Optional[str] = None
    confidence_score: float = Field(default=0.5, ge=0, le=1)
    priority: str = Field(default="medium", pattern="^(low|medium|high|critical)$")
    patient_context: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    risk_state: Optional[str] = None
    trigger_event_id: Optional[str] = None
    expires_hours: int = Field(default=48, ge=1, le=168)


class PatientSummary(BaseModel):
    id: str
    name: str
    email: Optional[str] = None
    risk_score: Optional[float] = None
    risk_state: Optional[str] = None
    last_checkin: Optional[str] = None
    active_medications_count: int = 0
    recent_symptoms: List[str] = []


def audit_log(db: Session, user_id: str, action: str, resource_type: str,
              resource_id: str, details: Optional[Dict[str, Any]] = None):
    """HIPAA-compliant audit logging"""
    try:
        db.execute(
            text("""
                INSERT INTO hipaa_audit_logs (id, user_id, action, resource_type, resource_id, details, created_at)
                VALUES (:id, :user_id, :action, :resource_type, :resource_id, :details, NOW())
            """),
            {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": json.dumps(details if details else {})
            }
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")


def get_patient_summary(db: Session, patient_id: str) -> Optional[Dict[str, Any]]:
    """Get patient summary for approval context"""
    result = db.execute(
        text("""
            SELECT u.id, u.name, u.email,
                   aps.risk_score, aps.risk_state, aps.last_checkin_at,
                   (SELECT COUNT(*) FROM medications m WHERE m.patient_id = u.id AND m.status = 'active') as med_count
            FROM users u
            LEFT JOIN autopilot_patient_states aps ON aps.patient_id = u.id
            WHERE u.id = :patient_id
        """),
        {"patient_id": patient_id}
    )
    row = result.fetchone()
    if not row:
        return None
    
    return {
        "id": row[0],
        "name": row[1] or "Unknown Patient",
        "email": row[2],
        "risk_score": row[3],
        "risk_state": row[4],
        "last_checkin": row[5].isoformat() if row[5] else None,
        "active_medications_count": row[6] or 0,
    }


def verify_doctor_patient_relationship(db: Session, doctor_id: str, patient_id: str) -> bool:
    """Verify doctor has access to patient"""
    result = db.execute(
        text("""
            SELECT 1 FROM doctor_patient_assignments
            WHERE doctor_id = :doctor_id AND patient_id = :patient_id AND status = 'active'
            UNION
            SELECT 1 FROM users WHERE id = :patient_id AND assigned_doctor_id = :doctor_id
            LIMIT 1
        """),
        {"doctor_id": doctor_id, "patient_id": patient_id}
    )
    return result.fetchone() is not None


@router.get("/pending")
def get_pending_approvals(
    status: str = Query(default="pending", pattern="^(pending|all)$"),
    priority: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get pending approvals for the current doctor"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view approvals")
    
    query = db.query(AutopilotPendingApproval).filter(
        AutopilotPendingApproval.doctor_id == current_user.id
    )
    
    if status == "pending":
        query = query.filter(AutopilotPendingApproval.status == "pending")
        query = query.filter(
            (AutopilotPendingApproval.expires_at == None) |
            (AutopilotPendingApproval.expires_at > datetime.now(timezone.utc))
        )
    
    if priority:
        query = query.filter(AutopilotPendingApproval.priority == priority)
    
    query = query.order_by(
        desc(AutopilotPendingApproval.priority == "critical"),
        desc(AutopilotPendingApproval.priority == "high"),
        desc(AutopilotPendingApproval.created_at)
    ).limit(limit)
    
    approvals = query.all()
    
    result = []
    for approval in approvals:
        patient_summary = get_patient_summary(db, approval.patient_id)
        result.append({
            "id": str(approval.id),
            "patient_id": approval.patient_id,
            "patient": patient_summary,
            "action_type": approval.action_type,
            "status": approval.status,
            "priority": approval.priority,
            "title": approval.title,
            "ai_recommendation": approval.ai_recommendation,
            "ai_reasoning": approval.ai_reasoning,
            "confidence_score": approval.confidence_score,
            "risk_score": approval.risk_score,
            "risk_state": approval.risk_state,
            "patient_context": approval.patient_context,
            "expires_at": approval.expires_at.isoformat() if approval.expires_at else None,
            "created_at": approval.created_at.isoformat() if approval.created_at else None,
        })
    
    audit_log(db, current_user.id, "view_pending_approvals", "hitl_approval", 
              "list", {"count": len(result), "status_filter": status})
    
    return {
        "approvals": result,
        "count": len(result),
        "has_critical": any(a["priority"] == "critical" for a in result),
        "has_high": any(a["priority"] == "high" for a in result),
    }


@router.get("/pending/count")
def get_pending_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get count of pending approvals for notification badge"""
    if current_user.role != "doctor":
        return {"count": 0, "critical": 0, "high": 0}
    
    now = datetime.now(timezone.utc)
    
    total = db.query(AutopilotPendingApproval).filter(
        AutopilotPendingApproval.doctor_id == current_user.id,
        AutopilotPendingApproval.status == "pending",
        (AutopilotPendingApproval.expires_at == None) |
        (AutopilotPendingApproval.expires_at > now)
    ).count()
    
    critical = db.query(AutopilotPendingApproval).filter(
        AutopilotPendingApproval.doctor_id == current_user.id,
        AutopilotPendingApproval.status == "pending",
        AutopilotPendingApproval.priority == "critical",
        (AutopilotPendingApproval.expires_at == None) |
        (AutopilotPendingApproval.expires_at > now)
    ).count()
    
    high = db.query(AutopilotPendingApproval).filter(
        AutopilotPendingApproval.doctor_id == current_user.id,
        AutopilotPendingApproval.status == "pending",
        AutopilotPendingApproval.priority == "high",
        (AutopilotPendingApproval.expires_at == None) |
        (AutopilotPendingApproval.expires_at > now)
    ).count()
    
    return {"count": total, "critical": critical, "high": high}


@router.get("/approvals/{approval_id}")
def get_approval_detail(
    approval_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed approval information"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view approvals")
    
    approval = db.query(AutopilotPendingApproval).filter(
        AutopilotPendingApproval.id == approval_id
    ).first()
    
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")
    
    if approval.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have access to this approval")
    
    patient_summary = get_patient_summary(db, approval.patient_id)
    
    audit_log(db, current_user.id, "view_approval_detail", "hitl_approval",
              approval_id, {"patient_id": approval.patient_id})
    
    return {
        "id": str(approval.id),
        "patient_id": approval.patient_id,
        "patient": patient_summary,
        "action_type": approval.action_type,
        "status": approval.status,
        "priority": approval.priority,
        "title": approval.title,
        "ai_recommendation": approval.ai_recommendation,
        "ai_reasoning": approval.ai_reasoning,
        "confidence_score": approval.confidence_score,
        "risk_score": approval.risk_score,
        "risk_state": approval.risk_state,
        "patient_context": approval.patient_context,
        "doctor_notes": approval.doctor_notes,
        "modified_action": approval.modified_action,
        "rejection_reason": approval.rejection_reason,
        "expires_at": approval.expires_at.isoformat() if approval.expires_at else None,
        "reviewed_at": approval.reviewed_at.isoformat() if approval.reviewed_at else None,
        "created_at": approval.created_at.isoformat() if approval.created_at else None,
    }


@router.post("/approvals/{approval_id}/decide")
def decide_approval(
    approval_id: str,
    decision: ApprovalDecision,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Make a decision on a pending approval"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can decide on approvals")
    
    approval = db.query(AutopilotPendingApproval).filter(
        AutopilotPendingApproval.id == approval_id
    ).first()
    
    if not approval:
        raise HTTPException(status_code=404, detail="Approval not found")
    
    if approval.doctor_id != current_user.id:
        raise HTTPException(status_code=403, detail="You don't have access to this approval")
    
    if approval.status != "pending":
        raise HTTPException(status_code=400, detail=f"Approval already {approval.status}")
    
    if approval.expires_at and approval.expires_at < datetime.now(timezone.utc):
        approval.status = "expired"
        db.commit()
        raise HTTPException(status_code=400, detail="Approval has expired")
    
    time_to_decision = None
    if approval.created_at:
        time_to_decision = int((datetime.now(timezone.utc) - approval.created_at.replace(tzinfo=timezone.utc)).total_seconds())
    
    status_mapping = {
        "approve": "approved",
        "modify": "modified",
        "reject": "rejected"
    }
    approval.status = status_mapping[decision.action]
    approval.reviewed_at = datetime.now(timezone.utc)
    approval.doctor_notes = decision.notes
    
    if decision.action == "modify" and decision.modified_action:
        approval.modified_action = decision.modified_action
    elif decision.action == "reject":
        approval.rejection_reason = decision.rejection_reason
    
    history = AutopilotApprovalHistory(
        approval_id=approval.id,
        doctor_id=current_user.id,
        patient_id=approval.patient_id,
        action_taken=decision.action,
        original_recommendation=approval.ai_recommendation,
        final_action=decision.modified_action if decision.action == "modify" else None,
        doctor_notes=decision.notes,
        time_to_decision_seconds=time_to_decision
    )
    db.add(history)
    
    if decision.action in ("approve", "modify"):
        execute_approved_action(db, approval, decision, current_user.id)
    
    db.commit()
    
    audit_log(db, current_user.id, f"approval_{decision.action}", "hitl_approval",
              approval_id, {
                  "patient_id": approval.patient_id,
                  "action_type": approval.action_type,
                  "time_to_decision_seconds": time_to_decision
              })
    
    return {
        "success": True,
        "status": approval.status,
        "message": f"Approval {approval.status} successfully"
    }


def execute_approved_action(db: Session, approval: AutopilotPendingApproval,
                           decision: ApprovalDecision, doctor_id: str):
    """Execute the approved action"""
    action_type = approval.action_type
    action_config = decision.modified_action if decision.action == "modify" else None
    
    if action_type == "schedule_followup":
        task = AutopilotFollowupTask(
            patient_id=approval.patient_id,
            task_type="doctor_approved_followup",
            priority=approval.priority,
            status="pending",
            due_at=datetime.now(timezone.utc) + timedelta(hours=24),
            created_by=f"doctor:{doctor_id}",
            reason=f"Doctor-approved: {approval.title}",
            task_metadata={"approval_id": str(approval.id), "notes": decision.notes}
        )
        db.add(task)
    
    elif action_type == "send_reminder":
        notification = AutopilotNotification(
            patient_id=approval.patient_id,
            channel="in_app",
            title=action_config.get("title", "Reminder from your care team") if action_config else "Reminder from your care team",
            body=action_config.get("body", approval.ai_recommendation) if action_config else approval.ai_recommendation,
            priority=approval.priority,
            status="pending"
        )
        db.add(notification)
    
    elif action_type == "request_checkin":
        task = AutopilotFollowupTask(
            patient_id=approval.patient_id,
            task_type="daily_checkin",
            priority="high",
            status="pending",
            due_at=datetime.now(timezone.utc),
            created_by=f"doctor:{doctor_id}",
            reason="Your doctor has requested a check-in",
            ui_tab_target="daily-followup"
        )
        db.add(task)
        
        notification = AutopilotNotification(
            patient_id=approval.patient_id,
            channel="in_app",
            title="Check-in Requested",
            body="Your doctor has requested you complete a health check-in today.",
            priority="high",
            status="pending"
        )
        db.add(notification)
    
    elif action_type == "notify_patient":
        notification = AutopilotNotification(
            patient_id=approval.patient_id,
            channel="in_app",
            title=action_config.get("title", "Message from your care team") if action_config else "Message from your care team",
            body=action_config.get("body", approval.ai_recommendation) if action_config else approval.ai_recommendation,
            priority=approval.priority,
            status="pending"
        )
        db.add(notification)


@router.get("/patients/{patient_id}/profile-summary")
def get_patient_profile_summary(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive patient profile for doctor review"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view patient profiles")
    
    if not verify_doctor_patient_relationship(db, current_user.id, patient_id):
        raise HTTPException(status_code=403, detail="You don't have access to this patient")
    
    patient_result = db.execute(
        text("""
            SELECT id, name, email, date_of_birth, phone, created_at
            FROM users WHERE id = :patient_id
        """),
        {"patient_id": patient_id}
    )
    patient_row = patient_result.fetchone()
    
    if not patient_row:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    state = db.query(AutopilotPatientState).filter(
        AutopilotPatientState.patient_id == patient_id
    ).first()
    
    meds_result = db.execute(
        text("""
            SELECT id, name, dosage, frequency, status
            FROM medications
            WHERE patient_id = :patient_id AND status = 'active'
            ORDER BY created_at DESC
            LIMIT 10
        """),
        {"patient_id": patient_id}
    )
    medications = [{"id": r[0], "name": r[1], "dosage": r[2], "frequency": r[3], "status": r[4]} 
                   for r in meds_result.fetchall()]
    
    symptoms_result = db.execute(
        text("""
            SELECT symptom_type, severity, recorded_at
            FROM symptom_entries
            WHERE patient_id = :patient_id
            ORDER BY recorded_at DESC
            LIMIT 10
        """),
        {"patient_id": patient_id}
    )
    recent_symptoms = [{"type": r[0], "severity": r[1], "date": r[2].isoformat() if r[2] else None}
                       for r in symptoms_result.fetchall()]
    
    mh_result = db.execute(
        text("""
            SELECT questionnaire_type, total_score, severity_level, created_at
            FROM mental_health_questionnaires
            WHERE patient_id = :patient_id
            ORDER BY created_at DESC
            LIMIT 5
        """),
        {"patient_id": patient_id}
    )
    mental_health = [{"type": r[0], "score": r[1], "severity": r[2], "date": r[3].isoformat() if r[3] else None}
                     for r in mh_result.fetchall()]
    
    adherence_result = db.execute(
        text("""
            SELECT 
                COUNT(*) FILTER (WHERE status = 'taken') as taken,
                COUNT(*) as total
            FROM medication_adherence
            WHERE patient_id = :patient_id
            AND scheduled_time > NOW() - INTERVAL '7 days'
        """),
        {"patient_id": patient_id}
    )
    adherence_row = adherence_result.fetchone()
    adherence_rate = (adherence_row[0] / adherence_row[1] * 100) if adherence_row and adherence_row[1] > 0 else None
    
    audit_log(db, current_user.id, "view_patient_profile_summary", "patient",
              patient_id, {"from_hitl": True})
    
    return {
        "patient": {
            "id": patient_row[0],
            "name": patient_row[1],
            "email": patient_row[2],
            "date_of_birth": patient_row[3].isoformat() if patient_row[3] else None,
            "phone": patient_row[4],
            "member_since": patient_row[5].isoformat() if patient_row[5] else None,
        },
        "autopilot_state": {
            "risk_score": state.risk_score if state else None,
            "risk_state": state.risk_state if state else None,
            "risk_components": state.risk_components if state else {},
            "last_checkin": state.last_checkin_at.isoformat() if state and state.last_checkin_at else None,
            "next_followup": state.next_followup_at.isoformat() if state and state.next_followup_at else None,
        },
        "medications": medications,
        "recent_symptoms": recent_symptoms,
        "mental_health": mental_health,
        "adherence_rate_7d": adherence_rate,
    }


@router.get("/history")
def get_approval_history(
    days: int = Query(default=30, ge=1, le=90),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get approval decision history for the doctor"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view history")
    
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    
    history = db.query(AutopilotApprovalHistory).filter(
        AutopilotApprovalHistory.doctor_id == current_user.id,
        AutopilotApprovalHistory.created_at >= cutoff
    ).order_by(desc(AutopilotApprovalHistory.created_at)).limit(limit).all()
    
    result = []
    for h in history:
        patient_summary = get_patient_summary(db, h.patient_id)
        result.append({
            "id": str(h.id),
            "approval_id": str(h.approval_id),
            "patient_id": h.patient_id,
            "patient": patient_summary,
            "action_taken": h.action_taken,
            "original_recommendation": h.original_recommendation,
            "doctor_notes": h.doctor_notes,
            "time_to_decision_seconds": h.time_to_decision_seconds,
            "created_at": h.created_at.isoformat() if h.created_at else None,
        })
    
    audit_log(db, current_user.id, "view_approval_history", "hitl_history",
              "list", {"days": days, "count": len(result)})
    
    return {
        "history": result,
        "count": len(result),
        "period_days": days,
    }


@router.post("/create")
def create_approval(
    request: CreateApprovalRequest,
    _token: str = Depends(verify_internal_service_token),
    db: Session = Depends(get_db)
):
    """
    Create a new pending approval (internal API for Autopilot trigger engine).
    Should be called when high-risk events are detected.
    
    SECURITY: Protected by internal service token. External calls will be rejected.
    """
    trigger_event_uuid = None
    if request.trigger_event_id:
        try:
            trigger_event_uuid = uuid.UUID(request.trigger_event_id)
        except ValueError:
            pass
    
    approval = AutopilotPendingApproval(
        patient_id=request.patient_id,
        doctor_id=request.doctor_id,
        trigger_event_id=trigger_event_uuid,
        action_type=request.action_type,
        status="pending",
        priority=request.priority,
        title=request.title,
        ai_recommendation=request.ai_recommendation,
        ai_reasoning=request.ai_reasoning,
        confidence_score=request.confidence_score,
        patient_context=request.patient_context or {},
        risk_score=request.risk_score,
        risk_state=request.risk_state,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=request.expires_hours)
    )
    
    db.add(approval)
    db.commit()
    db.refresh(approval)
    
    logger.info(f"Created HITL approval {approval.id} for patient {request.patient_id}")
    
    return {
        "id": str(approval.id),
        "status": "created",
        "expires_at": approval.expires_at.isoformat()
    }
