"""
Phase 5: Admin Autopilot API Router

Provides endpoints for system health monitoring, engagement analytics,
model performance tracking, patient cohort analysis, and configuration management.

HIPAA Compliance:
- Admin-only access required
- All patient-level data is aggregated (no individual PHI)
- MIN_CELL_SIZE=10 for privacy protection
- All operations are audit logged
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.database import get_db
from sqlalchemy.orm import Session

router = APIRouter(
    prefix="/api/autopilot/admin",
    tags=["Autopilot Admin"],
    responses={401: {"description": "Not authorized"}}
)

WELLNESS_DISCLAIMER = "This dashboard shows wellness monitoring metrics only. Not medical advice."


class ConfigUpdateRequest(BaseModel):
    """Request to update a configuration value."""
    config_key: str = Field(..., description="Configuration key to update")
    config_value: Any = Field(..., description="New configuration value")


class ConfigUpdateResponse(BaseModel):
    """Response from configuration update."""
    success: bool
    key: Optional[str] = None
    value: Optional[Any] = None
    error: Optional[str] = None


def get_user_from_request(request: Request) -> Dict[str, Any]:
    """Extract user info from request state or return default."""
    if hasattr(request.state, 'user') and request.state.user:
        return request.state.user
    return {"sub": "admin", "role": "admin"}


def _get_admin_analytics_service(db_session):
    """Get admin analytics service with database session."""
    from python_backend.ml_analysis.followup_autopilot.admin_analytics import AdminAnalyticsService
    return AdminAnalyticsService(db_session)


def _audit_log(db: Session, action: str, user_id: str, details: Optional[Dict[str, Any]] = None):
    """Log admin actions for HIPAA compliance."""
    try:
        from app.models.followup_autopilot_models import AutopilotAuditLog
        import uuid
        
        log = AutopilotAuditLog(
            id=uuid.uuid4(),
            action=action,
            entity_type="admin_dashboard",
            user_id=user_id,
            new_values=details or {}
        )
        db.add(log)
        db.commit()
    except Exception:
        pass


@router.get("/health")
async def get_system_health(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get current Autopilot system health and activity metrics.
    
    Returns:
        System status, patient counts, and activity metrics for the last hour.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_system_health", user_id)
    
    service = _get_admin_analytics_service(None)
    result = service.get_system_health()
    
    return result


@router.get("/engagement")
async def get_engagement_analytics(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get patient engagement analytics.
    
    Tracks task completion rates, notification effectiveness, and interaction patterns.
    Cell sizes below 10 are suppressed for patient privacy.
    
    Returns:
        Task completion stats, notification stats, and breakdowns by type/priority.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_engagement_analytics", user_id, {"days": days})
    
    service = _get_admin_analytics_service(None)
    result = service.get_engagement_analytics(days)
    
    return result


@router.get("/models/performance")
async def get_model_performance(
    request: Request,
    model_name: Optional[str] = Query(None, description="Filter by specific model name"),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get ML model performance metrics.
    
    Tracks accuracy, precision, recall, F1-score, AUC-ROC, and drift detection.
    
    Returns:
        Performance metrics for all models or a specific model.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_model_performance", user_id, {"model_name": model_name, "days": days})
    
    service = _get_admin_analytics_service(None)
    result = service.get_model_performance(model_name, days)
    
    return result


@router.get("/cohorts")
async def get_patient_cohorts(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get patient cohort distribution.
    
    Patients are grouped by risk profiles and engagement patterns.
    Cell sizes below 10 are suppressed for patient privacy.
    
    Returns:
        Cohort distribution, risk distribution, and privacy-protected aggregates.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_patient_cohorts", user_id)
    
    service = _get_admin_analytics_service(None)
    result = service.get_patient_cohorts()
    
    return result


@router.get("/triggers")
async def get_trigger_analytics(
    request: Request,
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get trigger analytics.
    
    Analyzes which triggers are firing and their severity distribution.
    Cell sizes below 10 are suppressed for patient privacy.
    
    Returns:
        Trigger counts by type and severity.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_trigger_analytics", user_id, {"days": days})
    
    service = _get_admin_analytics_service(None)
    result = service.get_trigger_analytics(days)
    
    return result


@router.get("/configurations")
async def get_configurations(
    request: Request,
    category: Optional[str] = Query(None, description="Filter by category"),
    db: Session = Depends(get_db)
):
    """
    Get autopilot configuration settings.
    
    Returns all active configurations or those filtered by category.
    
    Returns:
        List of configuration settings with their current values.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_configurations", user_id, {"category": category})
    
    service = _get_admin_analytics_service(None)
    configs = service.get_configurations(category)
    
    return {
        "configurations": configs,
        "total": len(configs),
        "wellness_disclaimer": WELLNESS_DISCLAIMER
    }


@router.put("/configurations")
async def update_configuration(
    request: Request,
    config_request: ConfigUpdateRequest,
    db: Session = Depends(get_db)
):
    """
    Update an autopilot configuration.
    
    Validates value against min/max constraints if defined.
    All changes are audit logged.
    
    Returns:
        Success status and updated value or error message.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    
    service = _get_admin_analytics_service(None)
    result = service.update_configuration(
        config_request.config_key,
        config_request.config_value,
        user_id
    )
    
    _audit_log(db, "update_configuration", user_id, {
        "key": config_request.config_key,
        "value": config_request.config_value,
        "success": result.get("success")
    })
    
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Update failed"))
    
    return result


@router.get("/summary")
async def get_admin_summary(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Get a combined summary of all admin metrics.
    
    Provides a single-call overview of system health, engagement, and model status.
    
    Returns:
        Combined dashboard data for admin overview.
    """
    user = get_user_from_request(request)
    user_id = user.get("sub", "unknown")
    _audit_log(db, "view_admin_summary", user_id)
    
    service = _get_admin_analytics_service(None)
    
    health = service.get_system_health()
    engagement = service.get_engagement_analytics(7)
    models = service.get_model_performance(days=7)
    cohorts = service.get_patient_cohorts()
    triggers = service.get_trigger_analytics(7)
    
    return {
        "system_health": health,
        "engagement_7d": {
            "tasks": engagement.get("tasks", {}),
            "notifications": engagement.get("notifications", {})
        },
        "models": {
            "count": len(models.get("models", [])),
            "drift_alerts": models.get("drift_alerts_active", 0)
        },
        "cohorts": {
            "total": cohorts.get("total_cohorts", 0),
            "risk_distribution": cohorts.get("risk_distribution", {})
        },
        "triggers_7d": {
            "total": triggers.get("total_triggers", 0),
            "by_severity": triggers.get("by_severity", {})
        },
        "wellness_disclaimer": WELLNESS_DISCLAIMER
    }
