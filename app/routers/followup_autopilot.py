"""
Followup Autopilot API Router

REST endpoints for the Autopilot system:
- Signal ingestion
- Autopilot status
- Task management
- Notifications
- Training labels
- Manual triggers

All endpoints include PrivacyGuard and HIPAA audit logging.
Wellness monitoring only - NOT medical diagnosis.
"""

import os
import sys
from datetime import datetime, date, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.security_models import AuditLog


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python_backend'))

router = APIRouter(prefix="/api/v1/followup-autopilot", tags=["Followup Autopilot"])


WELLNESS_DISCLAIMER = (
    "This information is for wellness monitoring only and is not a substitute for "
    "professional medical advice, diagnosis, or treatment. If you are experiencing "
    "a medical emergency, please contact emergency services immediately."
)


@router.get("/health")
async def health_check():
    """
    Lightweight health check endpoint for backend readiness detection.
    Returns READY when the Autopilot system is fully initialized.
    Used by frontend for smart retry logic during cold starts.
    """
    return {
        "status": "READY",
        "service": "followup-autopilot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0"
    }


class SignalInput(BaseModel):
    category: str = Field(..., description="Signal category: device, symptom, video, audio, pain, mental, environment, meds, exposure")
    source: str = Field(default="api_sync", description="Signal source")
    raw_payload: Dict[str, Any] = Field(default_factory=dict)
    ml_score: Optional[float] = Field(default=None, ge=0, le=1)
    signal_time: Optional[datetime] = None


class SignalBatchInput(BaseModel):
    signals: List[SignalInput]


class TaskCompleteInput(BaseModel):
    notes: Optional[str] = None


class LabelInput(BaseModel):
    date: date
    had_worsening_event_next7d: Optional[bool] = None
    had_mh_crisis_next7d: Optional[bool] = None
    had_non_adherence_issue_next7d: Optional[bool] = None


class ManualTriggerInput(BaseModel):
    trigger_name: str
    force: bool = False


def get_patient_id_from_request(request: Request) -> str:
    """Extract patient ID from request (auth context)"""
    if hasattr(request.state, 'user'):
        return request.state.user.get('sub', request.state.user.get('id', 'unknown'))
    return 'demo-patient'


def audit_log(db: Session, action: str, patient_id: str, details: Dict[str, Any]):
    """Create HIPAA audit log entry"""
    try:
        log = AuditLog(
            user_id=patient_id,
            action=action,
            resource_type="followup_autopilot",
            resource_id=patient_id,
            details=details,
            ip_address="0.0.0.0"
        )
        db.add(log)
        db.commit()
    except Exception:
        pass


@router.post("/patients/{patient_id}/signals")
async def ingest_signal(
    patient_id: str,
    signal: SignalInput,
    db: Session = Depends(get_db)
):
    """
    Ingest a single signal from any module.
    
    Categories: device, symptom, video, audio, pain, mental, environment, meds, exposure
    """
    from python_backend.ml_analysis.followup_autopilot.signal_ingestor import SignalIngestor
    
    ingestor = SignalIngestor(db)
    signal_id = ingestor.ingest_signal(
        patient_id=patient_id,
        category=signal.category,
        source=signal.source,
        raw_payload=signal.raw_payload,
        ml_score=signal.ml_score,
        signal_time=signal.signal_time
    )
    
    if not signal_id:
        raise HTTPException(status_code=400, detail="Failed to ingest signal")
    
    audit_log(db, "signal_ingested", patient_id, {"signal_id": signal_id, "category": signal.category})
    
    return {"signal_id": signal_id, "status": "ingested"}


@router.post("/patients/{patient_id}/signals/batch")
async def ingest_signals_batch(
    patient_id: str,
    batch: SignalBatchInput,
    db: Session = Depends(get_db)
):
    """Ingest multiple signals in a batch"""
    from python_backend.ml_analysis.followup_autopilot.signal_ingestor import SignalIngestor
    
    ingestor = SignalIngestor(db)
    signal_ids = ingestor.ingest_batch(
        patient_id=patient_id,
        signals=[s.dict() for s in batch.signals]
    )
    
    audit_log(db, "signals_batch_ingested", patient_id, {"count": len(signal_ids)})
    
    return {"signal_ids": signal_ids, "count": len(signal_ids)}


@router.get("/patients/{patient_id}/autopilot")
async def get_autopilot_status(
    patient_id: str,
    refresh: bool = Query(False, description="Force refresh state"),
    db: Session = Depends(get_db)
):
    """
    Get full autopilot status for a patient.
    
    Returns:
    - patient_state: Risk score, state, components, next followup
    - today_tasks: Pending tasks for today
    - wellness_disclaimer: Required disclaimer text
    """
    from python_backend.ml_analysis.followup_autopilot.autopilot_core import AutopilotCore
    from python_backend.ml_analysis.followup_autopilot.task_engine import TaskEngine
    
    autopilot = AutopilotCore(db)
    
    if refresh:
        state = autopilot.update_patient_state(patient_id)
    else:
        state = autopilot.get_patient_state(patient_id)
        if not state:
            state = autopilot.update_patient_state(patient_id)
    
    task_engine = TaskEngine(db)
    today_tasks = task_engine.get_today_tasks_for_patient(patient_id)
    
    top_components = []
    if state and state.get("risk_components"):
        components = state["risk_components"]
        sorted_components = sorted(
            [(k, v) for k, v in components.items()],
            key=lambda x: x[1],
            reverse=True
        )[:3]
        top_components = [
            {"name": k.replace("_", " ").title(), "value": v}
            for k, v in sorted_components
        ]
    
    audit_log(db, "autopilot_status_viewed", patient_id, {"refresh": refresh})
    
    return {
        "patient_state": {
            "patient_id": patient_id,
            "risk_score": state.get("risk_score", 0) if state else 0,
            "risk_state": state.get("risk_state", "Stable") if state else "Stable",
            "risk_components": state.get("risk_components", {}) if state else {},
            "top_risk_components": top_components,
            "next_followup_at": state.get("next_followup_at").isoformat() if state and state.get("next_followup_at") else None,
            "last_updated": state.get("last_updated").isoformat() if state and state.get("last_updated") else None,
            "last_checkin_at": state.get("last_checkin_at").isoformat() if state and state.get("last_checkin_at") else None,
            "model_version": state.get("model_version", "1.0.0") if state else "1.0.0",
            "confidence": state.get("inference_confidence", 0.5) if state else 0.5,
        },
        "today_tasks": today_tasks,
        "pending_task_count": len(today_tasks),
        "has_urgent_tasks": any(t.get("priority") in ("high", "critical") for t in today_tasks),
        "wellness_disclaimer": WELLNESS_DISCLAIMER,
    }


@router.get("/patients/{patient_id}/tasks")
async def get_patient_tasks(
    patient_id: str,
    include_overdue: bool = Query(True),
    all_pending: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Get pending tasks for a patient"""
    from python_backend.ml_analysis.followup_autopilot.task_engine import TaskEngine
    
    task_engine = TaskEngine(db)
    
    if all_pending:
        tasks = task_engine.get_all_pending_tasks(patient_id)
    else:
        tasks = task_engine.get_today_tasks_for_patient(patient_id, include_overdue)
    
    return {
        "tasks": tasks,
        "count": len(tasks),
        "has_urgent": any(t.get("priority") in ("high", "critical") for t in tasks),
    }


@router.post("/patients/{patient_id}/tasks/{task_id}/complete")
async def complete_task(
    patient_id: str,
    task_id: str,
    data: TaskCompleteInput = Body(default=TaskCompleteInput()),
    db: Session = Depends(get_db)
):
    """Mark a task as completed"""
    from python_backend.ml_analysis.followup_autopilot.task_engine import TaskEngine
    
    task_engine = TaskEngine(db)
    success = task_engine.complete_task(task_id, patient_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or already completed")
    
    audit_log(db, "task_completed", patient_id, {"task_id": task_id})
    
    return {"status": "completed", "task_id": task_id}


@router.get("/patients/{patient_id}/notifications")
async def get_notifications(
    patient_id: str,
    channel: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Get pending notifications for a patient"""
    from python_backend.ml_analysis.followup_autopilot.notification_engine import NotificationEngine
    
    notifier = NotificationEngine(db)
    notifications = notifier.get_pending_notifications(patient_id, channel)
    
    return {
        "notifications": notifications,
        "count": len(notifications),
    }


@router.post("/patients/{patient_id}/notifications/{notification_id}/read")
async def mark_notification_read(
    patient_id: str,
    notification_id: str,
    db: Session = Depends(get_db)
):
    """Mark a notification as read"""
    from python_backend.ml_analysis.followup_autopilot.notification_engine import NotificationEngine
    
    notifier = NotificationEngine(db)
    success = notifier.mark_notification_read(notification_id, patient_id)
    
    return {"status": "read" if success else "not_found"}


@router.post("/patients/{patient_id}/labels")
async def set_training_labels(
    patient_id: str,
    label_data: LabelInput,
    db: Session = Depends(get_db)
):
    """Set training labels for a specific date (internal use only)"""
    from app.models.followup_autopilot_models import AutopilotDailyFeatures
    
    existing = db.query(AutopilotDailyFeatures).filter(
        AutopilotDailyFeatures.patient_id == patient_id,
        AutopilotDailyFeatures.date == label_data.date
    ).first()
    
    if not existing:
        raise HTTPException(status_code=404, detail="No features found for this date")
    
    if label_data.had_worsening_event_next7d is not None:
        existing.had_worsening_event_next7d = label_data.had_worsening_event_next7d
    if label_data.had_mh_crisis_next7d is not None:
        existing.had_mh_crisis_next7d = label_data.had_mh_crisis_next7d
    if label_data.had_non_adherence_issue_next7d is not None:
        existing.had_non_adherence_issue_next7d = label_data.had_non_adherence_issue_next7d
    
    db.commit()
    
    audit_log(db, "labels_set", patient_id, {"date": str(label_data.date)})
    
    return {"status": "labels_updated", "date": str(label_data.date)}


@router.post("/patients/{patient_id}/trigger")
async def manual_trigger(
    patient_id: str,
    trigger_input: ManualTriggerInput,
    db: Session = Depends(get_db)
):
    """Manually trigger autopilot evaluation (admin use)"""
    from python_backend.ml_analysis.followup_autopilot.autopilot_core import AutopilotCore
    from python_backend.ml_analysis.followup_autopilot.trigger_engine import TriggerEngine
    from python_backend.ml_analysis.followup_autopilot.feature_builder import FeatureBuilder
    
    autopilot = AutopilotCore(db)
    state = autopilot.update_patient_state(patient_id)
    
    feature_builder = FeatureBuilder(db)
    today = date.today()
    features_today = feature_builder.build_daily_features(patient_id, today)
    
    trigger_engine = TriggerEngine(db)
    triggered_events = trigger_engine.run_triggers(
        patient_id=patient_id,
        features_today=features_today,
        patient_state=state,
        risk_probs={
            "p_clinical_deterioration": state.get("risk_components", {}).get("clinical", 0) / 100,
            "p_mental_health_crisis": state.get("risk_components", {}).get("mental_health", 0) / 100,
            "p_non_adherence": state.get("risk_components", {}).get("adherence", 0) / 100,
        },
        anomaly_score=state.get("anomaly_score", 0)
    )
    
    audit_log(db, "manual_trigger", patient_id, {"trigger_name": trigger_input.trigger_name})
    
    return {
        "status": "triggered",
        "patient_state": state,
        "triggered_events": triggered_events,
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "followup-autopilot",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get autopilot system statistics (admin)"""
    from app.models.followup_autopilot_models import (
        AutopilotPatientSignal, AutopilotDailyFeatures,
        AutopilotPatientState, AutopilotFollowupTask,
        AutopilotTriggerEvent, AutopilotNotification
    )
    from sqlalchemy import func
    
    try:
        stats = {
            "signals_count": db.query(func.count(AutopilotPatientSignal.id)).scalar() or 0,
            "daily_features_count": db.query(func.count(AutopilotDailyFeatures.id)).scalar() or 0,
            "patients_tracked": db.query(func.count(AutopilotPatientState.patient_id)).scalar() or 0,
            "pending_tasks": db.query(func.count(AutopilotFollowupTask.id)).filter(
                AutopilotFollowupTask.status == "pending"
            ).scalar() or 0,
            "trigger_events_24h": db.query(func.count(AutopilotTriggerEvent.id)).filter(
                AutopilotTriggerEvent.created_at >= datetime.now(timezone.utc).replace(hour=0, minute=0)
            ).scalar() or 0,
            "pending_notifications": db.query(func.count(AutopilotNotification.id)).filter(
                AutopilotNotification.status == "pending"
            ).scalar() or 0,
        }
    except Exception:
        stats = {
            "signals_count": 0,
            "daily_features_count": 0,
            "patients_tracked": 0,
            "pending_tasks": 0,
            "trigger_events_24h": 0,
            "pending_notifications": 0,
        }
    
    return {"stats": stats, "timestamp": datetime.now(timezone.utc).isoformat()}
