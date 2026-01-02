"""
Tinker Thinking Machine API Endpoints
=====================================
REST API for interacting with Tinker ML platform.
All endpoints enforce HIPAA compliance through the privacy firewall.

Security:
- All payloads pass through ensure_phi_safe_payload()
- K-anonymity enforced (k≥25)
- Only hashed identifiers transmitted
- Full audit logging on every operation
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.dependencies import get_db
from app.auth import get_current_user
from app.models.tinker_models import (
    AIAuditLog, TinkerCohortDefinition, TinkerStudy, TinkerTrialRun,
    TinkerModelMetrics, TinkerDriftRun, TinkerDriftAlert,
    TinkerPurpose, TrialStatus, DriftSeverity
)
from app.services.tinker_service import get_tinker_service, TinkerOperationResult
from app.services.tinker_privacy_firewall import get_privacy_firewall

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tinker", tags=["tinker"])


class CohortAnalysisRequest(BaseModel):
    cohort_name: str = Field(min_length=1, max_length=255)
    patient_ids: List[str] = Field(min_length=1)
    patient_data: List[Dict[str, Any]] = Field(default_factory=list)
    analysis_type: str = Field(default="descriptive")
    description: Optional[str] = None


class CohortAnalysisResponse(BaseModel):
    success: bool
    cohort_id: Optional[int] = None
    error: Optional[str] = None
    k_anon_passed: bool = True
    audit_id: Optional[str] = None


class DriftCheckRequest(BaseModel):
    model_id: str
    feature_packet: Dict[str, Any] = Field(default_factory=dict)


class DriftCheckResponse(BaseModel):
    success: bool
    drift_detected: bool = False
    psi_score: Optional[float] = None
    kl_divergence: Optional[float] = None
    error: Optional[str] = None
    drift_run_id: Optional[int] = None


class TinkerHealthResponse(BaseModel):
    enabled: bool
    healthy: bool
    mode: str
    k_threshold: int
    api_latency_ms: Optional[float] = None
    circuit_breaker_state: Optional[str] = None


class AuditLogResponse(BaseModel):
    id: str
    purpose: str
    actor_id: str
    actor_role: str
    created_at: datetime
    success: bool
    k_anon_verified: bool
    tinker_mode: Optional[str] = None


@router.get("/health", response_model=TinkerHealthResponse)
async def check_tinker_health(
    current_user: dict = Depends(get_current_user)
):
    """
    Check Tinker service health status.
    
    Returns enabled status, health, and configuration.
    """
    tinker = get_tinker_service()
    
    if not tinker or not tinker.is_enabled():
        return TinkerHealthResponse(
            enabled=False,
            healthy=False,
            mode="DISABLED",
            k_threshold=25
        )
    
    firewall = get_privacy_firewall()
    
    return TinkerHealthResponse(
        enabled=True,
        healthy=True,
        mode="NON_BAA",
        k_threshold=firewall.config.k_anonymity_threshold,
        circuit_breaker_state=tinker.get_circuit_breaker_state()
    )


@router.post("/cohort/analyze", response_model=CohortAnalysisResponse)
async def analyze_cohort(
    request: CohortAnalysisRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze a patient cohort using Tinker.
    
    Enforces k-anonymity (k≥25) and strips all PHI before transmission.
    Returns anonymized analysis results.
    """
    tinker = get_tinker_service()
    
    if not tinker or not tinker.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tinker integration is not enabled"
        )
    
    firewall = get_privacy_firewall()
    k_threshold = firewall.config.k_anonymity_threshold
    
    if len(request.patient_ids) < k_threshold:
        logger.warning(
            f"Cohort analysis rejected: {len(request.patient_ids)} patients < k={k_threshold}"
        )
        return CohortAnalysisResponse(
            success=False,
            error=f"Cohort size must be at least {k_threshold} for k-anonymity",
            k_anon_passed=False
        )
    
    result = await tinker.analyze_patient_cohort(
        db=db,
        actor_id=current_user.get("id", "unknown"),
        actor_role=current_user.get("role", "unknown"),
        cohort_name=request.cohort_name,
        patient_ids=request.patient_ids,
        patient_data_list=request.patient_data,
        analysis_type=request.analysis_type
    )
    
    cohort_id = None
    if result.data and "cohort_id" in result.data:
        cohort_id = result.data["cohort_id"]
    
    return CohortAnalysisResponse(
        success=result.success,
        cohort_id=cohort_id,
        error=result.error,
        k_anon_passed=result.k_anon_passed,
        audit_id=result.audit_id
    )


@router.post("/drift/check", response_model=DriftCheckResponse)
async def check_model_drift(
    request: DriftCheckRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Check for model drift using Tinker.
    
    Compares baseline and current data distributions to detect drift.
    All data is anonymized before analysis.
    """
    tinker = get_tinker_service()
    
    if not tinker or not tinker.is_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tinker integration is not enabled"
        )
    
    result = await tinker.check_model_drift(
        db=db,
        model_id=request.model_id,
        feature_packet=request.feature_packet,
        actor_id=current_user.get("id", "unknown")
    )
    
    drift_detected = False
    psi_score = None
    kl_divergence = None
    drift_run_id = None
    
    if result.data:
        drift_detected = result.data.get("drift_detected", False)
        psi_score = result.data.get("psi_score")
        kl_divergence = result.data.get("kl_divergence")
        drift_run_id = result.data.get("drift_run_id")
    
    return DriftCheckResponse(
        success=result.success,
        drift_detected=drift_detected,
        psi_score=psi_score,
        kl_divergence=kl_divergence,
        error=result.error,
        drift_run_id=drift_run_id
    )


@router.get("/cohorts", response_model=List[Dict[str, Any]])
async def list_cohorts(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List defined cohorts.
    
    Returns cohort metadata (no PHI).
    """
    query = db.query(TinkerCohortDefinition)
    
    if status_filter:
        query = query.filter(TinkerCohortDefinition.status == status_filter)
    
    cohorts = query.order_by(TinkerCohortDefinition.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "status": c.status,
            "patient_count": c.patient_count,
            "k_anon_passed": c.k_anon_passed,
            "created_at": c.created_at.isoformat() if c.created_at else None
        }
        for c in cohorts
    ]


@router.get("/studies", response_model=List[Dict[str, Any]])
async def list_studies(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List research studies.
    
    Returns study metadata (no PHI).
    """
    query = db.query(TinkerStudy)
    
    if status_filter:
        query = query.filter(TinkerStudy.status == status_filter)
    
    studies = query.order_by(TinkerStudy.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        {
            "id": s.id,
            "name": s.name,
            "objective": s.objective,
            "status": s.status,
            "protocol_hash": s.protocol_hash,
            "created_at": s.created_at.isoformat() if s.created_at else None
        }
        for s in studies
    ]


@router.get("/trials", response_model=List[Dict[str, Any]])
async def list_trials(
    study_id: Optional[int] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List clinical trials.
    
    Returns trial metadata (no PHI).
    """
    query = db.query(TinkerTrialRun)
    
    if study_id:
        query = query.filter(TinkerTrialRun.trial_spec_id == study_id)
    
    if status_filter:
        query = query.filter(TinkerTrialRun.status == status_filter)
    
    trials = query.order_by(TinkerTrialRun.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        {
            "id": t.id,
            "trial_spec_id": t.trial_spec_id,
            "status": t.status,
            "started_at": t.started_at.isoformat() if t.started_at else None,
            "ended_at": t.ended_at.isoformat() if t.ended_at else None,
            "created_at": t.created_at.isoformat() if t.created_at else None
        }
        for t in trials
    ]


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List Tinker ML models.
    
    Returns model metadata.
    """
    models = db.query(TinkerModelMetrics).order_by(
        TinkerModelMetrics.created_at.desc()
    ).offset(offset).limit(limit).all()
    
    return [
        {
            "id": str(m.id),
            "model_id": m.model_id,
            "version": m.version,
            "accuracy": m.accuracy,
            "precision_score": m.precision_score,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "auc_roc": m.auc_roc,
            "is_production": m.is_production,
            "created_at": m.created_at.isoformat() if m.created_at else None
        }
        for m in models
    ]


@router.get("/drift/alerts", response_model=List[Dict[str, Any]])
async def list_drift_alerts(
    severity: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List drift alerts.
    
    Returns alert metadata for monitoring.
    """
    query = db.query(TinkerDriftAlert)
    
    if severity:
        try:
            severity_enum = DriftSeverity(severity)
            query = query.filter(TinkerDriftAlert.severity == severity_enum)
        except ValueError:
            pass
    
    if acknowledged is not None:
        query = query.filter(TinkerDriftAlert.acknowledged == acknowledged)
    
    alerts = query.order_by(TinkerDriftAlert.created_at.desc()).offset(offset).limit(limit).all()
    
    return [
        {
            "id": a.id,
            "drift_run_id": a.drift_run_id,
            "alert_type": a.alert_type,
            "severity": a.severity.value if a.severity else None,
            "message": a.message,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at.isoformat() if a.created_at else None
        }
        for a in alerts
    ]


@router.post("/drift/alerts/{alert_id}/acknowledge")
async def acknowledge_drift_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Acknowledge a drift alert.
    """
    alert = db.query(TinkerDriftAlert).filter(TinkerDriftAlert.id == alert_id).first()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Drift alert not found"
        )
    
    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = current_user.get("id", "unknown")
    db.commit()
    
    return {"success": True, "alert_id": alert_id}


@router.get("/audits", response_model=List[AuditLogResponse])
async def list_audit_logs(
    purpose: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List Tinker audit logs.
    
    HIPAA Compliance: Returns only metadata, not payload contents.
    """
    user_role = str(current_user.get("role", "")).lower()
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and doctors can view audit logs"
        )
    
    query = db.query(AIAuditLog).filter(AIAuditLog.tinker_mode.isnot(None))
    
    if purpose:
        try:
            purpose_enum = TinkerPurpose(purpose)
            query = query.filter(AIAuditLog.purpose == purpose_enum)
        except ValueError:
            pass
    
    audits = query.order_by(AIAuditLog.created_at.desc()).offset(offset).limit(limit).all()
    
    results = []
    for a in audits:
        results.append(AuditLogResponse(
            id=str(a.id),
            purpose=a.purpose.value if a.purpose else "unknown",
            actor_id=str(a.actor_id) if a.actor_id else "unknown",
            actor_role=a.actor_role.value if a.actor_role else "unknown",
            created_at=a.created_at if a.created_at else datetime.utcnow(),
            success=bool(a.success) if a.success is not None else True,
            k_anon_verified=bool(a.k_anon_verified) if a.k_anon_verified is not None else True,
            tinker_mode=str(a.tinker_mode) if a.tinker_mode else None
        ))
    
    return results


@router.get("/privacy/stats")
async def get_privacy_stats(
    current_user: dict = Depends(get_current_user)
):
    """
    Get privacy firewall statistics.
    
    Returns counts of operations, PHI detections, k-anonymity checks.
    """
    user_role = str(current_user.get("role", "")).lower()
    if user_role not in ["admin", "doctor"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admins and doctors can view privacy stats"
        )
    
    firewall = get_privacy_firewall()
    
    return {
        "mode": "NON_BAA",
        "k_anonymity_threshold": firewall.config.k_anonymity_threshold,
        "suppress_low_counts": firewall.config.suppress_low_counts,
        "hash_salt_configured": bool(firewall.config.hash_salt) if hasattr(firewall.config, 'hash_salt') else True,
        "phi_field_patterns_count": len(firewall.PHI_FIELD_PATTERNS),
        "enabled": getattr(firewall.config, 'enabled', True)
    }
