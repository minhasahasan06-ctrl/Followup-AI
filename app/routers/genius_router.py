"""
Genius Features Router (Phase E)
================================
REST API endpoints for Research, Clinical, and Patient Genius features.
Provides production-ready integration with Tinker and existing services.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response
from pydantic import BaseModel, Field, field_validator

from app.dependencies import get_current_user
from app.models.user import User
from app.services.access_control import HIPAAAuditLogger


PHI_PATTERNS = [
    r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
    r'\b\d{10,}\b',  # Phone numbers
    r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # Dates
    r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names (First Last pattern)
]


def is_valid_hash(value: str) -> bool:
    """Check if value is a valid SHA256 hash (64 hex characters)"""
    return bool(re.match(r'^[a-f0-9]{64}$', value.lower()))


def contains_phi(value: str) -> bool:
    """Check if value contains common PHI patterns"""
    for pattern in PHI_PATTERNS:
        if re.search(pattern, value, re.IGNORECASE):
            return True
    return False


def ensure_hashed_id(value: str, salt: str = "followup-ai-genius") -> str:
    """
    Ensure an ID is properly hashed. If it's already a valid SHA256 hash,
    return it as-is. Otherwise, hash it server-side.
    """
    if is_valid_hash(value):
        return value.lower()
    
    if contains_phi(value):
        raise ValueError("Patient identifier contains PHI patterns - must be pre-hashed")
    
    salted = f"{salt}:{value}"
    return hashlib.sha256(salted.encode()).hexdigest()


from app.services.genius_research import (
    get_genius_research_service,
    BiasWarning,
    SensitivityCheck,
)
from app.services.genius_clinical import (
    get_genius_clinical_service,
    AlertBudget,
    FairnessReport,
    DynamicThreshold,
)
from app.services.genius_patient import (
    get_genius_patient_service,
    PatientStability,
    EngagementBucket,
    TrendDirection,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/genius", tags=["genius"])


def require_doctor(user: User = Depends(get_current_user)) -> User:
    """Dependency to require doctor role"""
    if user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access this Genius feature"
        )
    return user


def require_user(user: User = Depends(get_current_user)) -> User:
    """Dependency to require any authenticated user"""
    if user.role not in ["patient", "doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authentication required"
        )
    return user


class ProtocolInput(BaseModel):
    """Input for preregistration generation"""
    objective: str
    schema_summary: Dict[str, Any] = Field(default_factory=dict)
    cohort_size_range: str = "50-200"
    analysis_types: List[str] = Field(default_factory=list)


class CohortDefinitionInput(BaseModel):
    """Input for bias/sensitivity analysis"""
    name: str
    filters: List[Dict[str, Any]] = Field(default_factory=list)


class BiasChecklistInput(BaseModel):
    """Input for bias checklist"""
    protocol: ProtocolInput
    cohort: CohortDefinitionInput


class SensitivityInput(BaseModel):
    """Input for sensitivity suite"""
    protocol: ProtocolInput
    cohort: CohortDefinitionInput


class StudyBundleInput(BaseModel):
    """Input for study bundle creation"""
    study_id: str
    study_name: str
    cohort_dsl: Dict[str, Any]
    protocol: Dict[str, Any]
    trial_spec: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None


class AlertBudgetInput(BaseModel):
    """Input for alert budget configuration"""
    clinic_id: str
    daily_limit: Optional[int] = None
    weekly_limit: Optional[int] = None
    priority_distribution: Optional[Dict[str, float]] = None


class FairnessInput(BaseModel):
    """Input for fairness analysis"""
    clinic_id: str
    alerts_by_subgroup: Dict[str, List[Dict[str, Any]]]
    subgroup_sizes: Dict[str, int]
    analysis_period_days: int = 30


class WorkloadInput(BaseModel):
    """Input for dynamic threshold calculation"""
    clinic_id: str
    base_threshold: float
    active_patients: int = 0
    pending_alerts: int = 0
    staff_available: int = 1
    avg_response_time_mins: float = 15


class CheckinInput(BaseModel):
    """Input for effort-aware check-in generation"""
    patient_id_hash: str
    stability_level: str = "stable"
    consecutive_stable_days: int = 0

    @field_validator('patient_id_hash')
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        """Ensure patient ID is hashed and contains no PHI"""
        return ensure_hashed_id(v)


class MicroHabitInput(BaseModel):
    """Input for micro-habit suggestions"""
    patient_id_hash: str
    engagement_bucket: str = "moderate"
    current_context: str = "general"
    max_suggestions: int = 3

    @field_validator('patient_id_hash')
    @classmethod
    def validate_patient_id(cls, v: str) -> str:
        """Ensure patient ID is hashed and contains no PHI"""
        return ensure_hashed_id(v)


class TrendInput(BaseModel):
    """Input for trend explanation"""
    metric_name: str
    trend_direction: str = "stable"
    timeframe_days: int = 7


@router.post("/research/preregistration")
async def generate_preregistration(
    input_data: ProtocolInput,
    user: User = Depends(require_doctor),
):
    """
    E.5: Generate preregistration outline from protocol.
    Doctor/Admin only.
    """
    HIPAAAuditLogger.log_phi_access(
        user_id=str(user.id),
        action="genius_preregistration",
        resource_type="research",
        details={"objective": input_data.objective[:50]}
    )
    
    service = get_genius_research_service()
    
    protocol = {
        "objective": input_data.objective,
        "schema_summary": input_data.schema_summary,
        "cohort_size_range": input_data.cohort_size_range,
        "analysis_types": input_data.analysis_types,
    }
    
    prereg, success = await service.generate_preregistration(
        protocol=protocol,
        actor_role=user.role
    )
    
    return {
        "success": success,
        "preregistration": {
            "title": prereg.title,
            "hypotheses": prereg.hypotheses,
            "study_design": prereg.study_design,
            "sample_size_justification": prereg.sample_size_justification,
            "primary_outcomes": prereg.primary_outcomes,
            "secondary_outcomes": prereg.secondary_outcomes,
            "analysis_plan": prereg.analysis_plan,
            "exclusion_criteria": prereg.exclusion_criteria,
            "timeline": prereg.timeline,
            "generated_at": prereg.generated_at,
            "protocol_hash": prereg.protocol_hash,
        }
    }


@router.post("/research/bias-checklist")
async def generate_bias_checklist(
    input_data: BiasChecklistInput,
    user: User = Depends(require_doctor),
):
    """
    E.6: Generate bias checklist for study design.
    Doctor/Admin only.
    """
    service = get_genius_research_service()
    
    protocol_dict = {
        "objective": input_data.protocol.objective,
        "schema_summary": input_data.protocol.schema_summary,
    }
    
    cohort_dict = {
        "name": input_data.cohort.name,
        "filters": input_data.cohort.filters,
    }
    
    warnings = service.generate_bias_checklist(protocol_dict, cohort_dict)
    
    return {
        "warnings": [
            {
                "bias_type": w.bias_type.value,
                "severity": w.severity,
                "description": w.description,
                "affected_variables": w.affected_variables,
                "mitigation_strategies": w.mitigation_strategies,
            }
            for w in warnings
        ],
        "warning_count": len(warnings),
    }


@router.post("/research/sensitivity-suite")
async def generate_sensitivity_suite(
    input_data: SensitivityInput,
    user: User = Depends(require_doctor),
):
    """
    E.7: Generate sensitivity analysis recommendations.
    Doctor/Admin only.
    """
    service = get_genius_research_service()
    
    protocol_dict = {
        "objective": input_data.protocol.objective,
        "schema_summary": input_data.protocol.schema_summary,
    }
    
    cohort_dict = {
        "name": input_data.cohort.name,
        "filters": input_data.cohort.filters,
    }
    
    checks = service.generate_sensitivity_suite(protocol_dict, cohort_dict)
    
    return {
        "sensitivity_checks": [
            {
                "check_type": c.check_type.value,
                "name": c.name,
                "description": c.description,
                "suggested_variables": c.suggested_variables,
                "rationale": c.rationale,
            }
            for c in checks
        ],
        "check_count": len(checks),
    }


@router.post("/research/study-bundle")
async def create_study_bundle(
    input_data: StudyBundleInput,
    user: User = Depends(require_doctor),
):
    """
    E.8: Create exportable study bundle (ZIP).
    Doctor/Admin only.
    """
    HIPAAAuditLogger.log_phi_access(
        user_id=str(user.id),
        action="genius_study_bundle",
        resource_type="research",
        details={"study_id": input_data.study_id}
    )
    
    service = get_genius_research_service()
    
    zip_bytes, bundle = service.create_study_bundle(
        study_id=input_data.study_id,
        study_name=input_data.study_name,
        cohort_dsl=input_data.cohort_dsl,
        protocol=input_data.protocol,
        trial_spec=input_data.trial_spec,
        metrics=input_data.metrics,
    )
    
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{input_data.study_name.replace(" ", "_")}_bundle.zip"'
        }
    )


@router.post("/clinical/alert-budget")
async def configure_alert_budget(
    input_data: AlertBudgetInput,
    user: User = Depends(require_doctor),
):
    """
    E.9: Configure alert budget for clinic.
    Doctor/Admin only.
    """
    service = get_genius_clinical_service()
    
    budget = service.configure_alert_budget(
        clinic_id=input_data.clinic_id,
        daily_limit=input_data.daily_limit,
        weekly_limit=input_data.weekly_limit,
        priority_distribution=input_data.priority_distribution,
    )
    
    return {
        "clinic_id": budget.clinic_id,
        "daily_alert_limit": budget.daily_alert_limit,
        "weekly_alert_limit": budget.weekly_alert_limit,
        "priority_distribution": budget.priority_distribution,
        "auto_escalation_threshold": budget.auto_escalation_threshold,
        "created_at": budget.created_at,
    }


@router.get("/clinical/alert-budget/{clinic_id}")
async def get_alert_budget(
    clinic_id: str,
    user: User = Depends(require_doctor),
):
    """Get alert budget for clinic."""
    service = get_genius_clinical_service()
    budget = service.get_alert_budget(clinic_id)
    
    if not budget:
        raise HTTPException(status_code=404, detail="Alert budget not found")
    
    return {
        "clinic_id": budget.clinic_id,
        "daily_alert_limit": budget.daily_alert_limit,
        "current_daily_count": budget.current_daily_count,
        "weekly_alert_limit": budget.weekly_alert_limit,
        "current_weekly_count": budget.current_weekly_count,
        "priority_distribution": budget.priority_distribution,
    }


@router.post("/clinical/fairness-analysis")
async def analyze_alert_fairness(
    input_data: FairnessInput,
    user: User = Depends(require_doctor),
):
    """
    E.10: Analyze alert burden fairness.
    Doctor/Admin only.
    """
    service = get_genius_clinical_service()
    
    report = service.analyze_alert_fairness(
        clinic_id=input_data.clinic_id,
        alerts_by_subgroup=input_data.alerts_by_subgroup,
        subgroup_sizes=input_data.subgroup_sizes,
        analysis_period_days=input_data.analysis_period_days,
    )
    
    return {
        "clinic_id": report.clinic_id,
        "analysis_period_days": report.analysis_period_days,
        "total_alerts": report.total_alerts,
        "fairness_score": report.fairness_score,
        "subgroup_distributions": [
            {
                "subgroup_name": d.subgroup_name,
                "subgroup_size": d.subgroup_size,
                "alert_count": d.alert_count,
                "alert_rate": d.alert_rate,
                "deviation": d.deviation,
                "is_over_represented": d.is_over_represented,
                "is_under_represented": d.is_under_represented,
            }
            for d in report.subgroup_distributions
        ],
        "warnings": report.warnings,
        "recommendations": report.recommendations,
        "generated_at": report.generated_at,
    }


@router.post("/clinical/dynamic-threshold")
async def calculate_dynamic_threshold(
    input_data: WorkloadInput,
    user: User = Depends(require_doctor),
):
    """
    E.11: Calculate dynamic alert threshold by workload.
    Doctor/Admin only.
    """
    service = get_genius_clinical_service()
    
    workload = {
        "active_patients": input_data.active_patients,
        "pending_alerts": input_data.pending_alerts,
        "staff_available": input_data.staff_available,
        "avg_response_time_mins": input_data.avg_response_time_mins,
    }
    
    threshold = service.calculate_dynamic_threshold(
        clinic_id=input_data.clinic_id,
        base_threshold=input_data.base_threshold,
        current_workload=workload,
    )
    
    return {
        "clinic_id": threshold.clinic_id,
        "base_threshold": threshold.base_threshold,
        "adjusted_threshold": threshold.adjusted_threshold,
        "workload_level": threshold.workload_level.value,
        "adjustment_factor": threshold.adjustment_factor,
        "reason": threshold.reason,
        "valid_until": threshold.valid_until,
    }


@router.post("/patient/effort-aware-checkin")
async def generate_effort_aware_checkin(
    input_data: CheckinInput,
    user: User = Depends(require_user),
):
    """
    E.12: Generate effort-aware check-in (minimal questions when stable).
    Available to patients and doctors.
    """
    service = get_genius_patient_service()
    
    try:
        stability = PatientStability(input_data.stability_level)
    except ValueError:
        stability = PatientStability.STABLE
    
    checkin = service.generate_effort_aware_checkin(
        patient_id_hash=input_data.patient_id_hash,
        stability_level=stability,
        consecutive_stable_days=input_data.consecutive_stable_days,
    )
    
    return {
        "patient_id_hash": checkin.patient_id_hash,
        "stability_level": checkin.stability_level.value,
        "question_count": checkin.question_count,
        "question_ids": checkin.question_ids,
        "estimated_time_seconds": checkin.estimated_time_seconds,
        "skip_allowed": checkin.skip_allowed,
        "reason": checkin.reason,
    }


@router.post("/patient/micro-habits")
async def get_micro_habit_suggestions(
    input_data: MicroHabitInput,
    user: User = Depends(require_user),
):
    """
    E.13: Get just-in-time micro-habit suggestions.
    Available to patients and doctors.
    """
    service = get_genius_patient_service()
    
    try:
        engagement = EngagementBucket(input_data.engagement_bucket)
    except ValueError:
        engagement = EngagementBucket.MODERATE
    
    suggestions = service.get_micro_habit_suggestions(
        patient_id_hash=input_data.patient_id_hash,
        engagement_bucket=engagement,
        current_context=input_data.current_context,
        max_suggestions=input_data.max_suggestions,
    )
    
    return {
        "suggestions": [
            {
                "habit_id": s.habit_id,
                "habit_name": s.habit_name,
                "habit_description": s.habit_description,
                "trigger_context": s.trigger_context,
                "duration_minutes": s.duration_minutes,
                "difficulty_level": s.difficulty_level,
                "personalization_reason": s.personalization_reason,
            }
            for s in suggestions
        ],
        "count": len(suggestions),
    }


@router.post("/patient/trend-explanation")
async def generate_trend_explanation(
    input_data: TrendInput,
    user: User = Depends(require_user),
):
    """
    E.14: Generate safe trend explanation (no raw numbers).
    Available to patients and doctors.
    """
    service = get_genius_patient_service()
    
    try:
        direction = TrendDirection(input_data.trend_direction)
    except ValueError:
        direction = TrendDirection.STABLE
    
    explanation = service.generate_safe_trend_explanation(
        metric_name=input_data.metric_name,
        trend_direction=direction,
        timeframe_days=input_data.timeframe_days,
    )
    
    return {
        "metric_name": explanation.metric_name,
        "trend_direction": explanation.trend_direction.value,
        "explanation": explanation.explanation,
        "timeframe": explanation.timeframe,
        "confidence": explanation.confidence,
        "action_suggestion": explanation.action_suggestion,
    }


@router.get("/health")
async def genius_health():
    """Health check for Genius services"""
    return {
        "status": "healthy",
        "services": {
            "research": "available",
            "clinical": "available",
            "patient": "available",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
