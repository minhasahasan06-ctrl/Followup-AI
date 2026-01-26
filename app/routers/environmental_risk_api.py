"""
Environmental Risk Map API
FastAPI endpoints for comprehensive environmental health intelligence.
"""

import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.services.environmental_risk_service import EnvironmentalRiskService, audit_log

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/environment",
    tags=["Environmental Risk Map"]
)


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================

class ProfileCreateRequest(BaseModel):
    """Request to create or update environmental profile."""
    zipCode: str = Field(..., min_length=5, max_length=10)
    conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class ProfileUpdateRequest(BaseModel):
    """Request to update environmental profile settings."""
    zipCode: Optional[str] = None
    conditions: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    alertsEnabled: Optional[bool] = None
    alertThresholds: Optional[dict] = None
    pushNotifications: Optional[bool] = None
    smsNotifications: Optional[bool] = None
    emailDigest: Optional[bool] = None
    digestFrequency: Optional[str] = None
    correlationConsentGiven: Optional[bool] = None


class AlertAcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert."""
    alertId: str


class AlertFeedbackRequest(BaseModel):
    """Request to provide feedback on an alert."""
    alertId: str
    wasHelpful: bool
    feedback: Optional[str] = None


class RefreshDataRequest(BaseModel):
    """Request to refresh environmental data."""
    zipCode: Optional[str] = None


# =============================================================================
# PROFILE ENDPOINTS
# =============================================================================

@router.post("/profile")
async def create_or_update_profile(
    request: ProfileCreateRequest,
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Create or update patient's environmental profile."""
    try:
        service = EnvironmentalRiskService(db)
        profile = await service.get_or_create_profile(
            patient_id=patient_id,
            zip_code=request.zipCode,
            conditions=request.conditions,
            allergies=request.allergies
        )
        
        return {
            "success": True,
            "profile": {
                "id": profile.id,
                "patientId": profile.patient_id,
                "zipCode": profile.zip_code,
                "city": profile.city,
                "state": profile.state,
                "conditions": profile.chronic_conditions,
                "allergies": profile.allergies,
                "alertsEnabled": profile.alerts_enabled,
                "correlationConsent": profile.correlation_consent_given,
            }
        }
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/profile")
async def update_profile(
    request: ProfileUpdateRequest,
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Update patient's environmental profile settings."""
    try:
        service = EnvironmentalRiskService(db)
        
        updates = {}
        if request.zipCode:
            updates["zip_code"] = request.zipCode
        if request.conditions is not None:
            updates["chronic_conditions"] = request.conditions
        if request.allergies is not None:
            updates["allergies"] = request.allergies
        if request.alertsEnabled is not None:
            updates["alerts_enabled"] = request.alertsEnabled
        if request.alertThresholds is not None:
            updates["alert_thresholds"] = request.alertThresholds
        if request.pushNotifications is not None:
            updates["push_notifications"] = request.pushNotifications
        if request.smsNotifications is not None:
            updates["sms_notifications"] = request.smsNotifications
        if request.emailDigest is not None:
            updates["email_digest"] = request.emailDigest
        if request.digestFrequency is not None:
            updates["digest_frequency"] = request.digestFrequency
        if request.correlationConsentGiven is not None:
            updates["correlation_consent_given"] = request.correlationConsentGiven
        
        profile = await service.update_profile(patient_id, updates)
        
        if not profile:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        return {
            "success": True,
            "profile": {
                "id": profile.id,
                "zipCode": profile.zip_code,
                "conditions": profile.chronic_conditions,
                "alertsEnabled": profile.alerts_enabled,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profile")
async def get_profile(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Get patient's environmental profile."""
    try:
        from app.models.environmental_risk import PatientEnvironmentProfile
        
        profile = db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile:
            return {"success": True, "profile": None}
        
        return {
            "success": True,
            "profile": {
                "id": profile.id,
                "patientId": profile.patient_id,
                "zipCode": profile.zip_code,
                "city": profile.city,
                "state": profile.state,
                "conditions": profile.chronic_conditions,
                "allergies": profile.allergies,
                "alertsEnabled": profile.alerts_enabled,
                "alertThresholds": profile.alert_thresholds,
                "pushNotifications": profile.push_notifications,
                "smsNotifications": profile.sms_notifications,
                "emailDigest": profile.email_digest,
                "digestFrequency": profile.digest_frequency,
                "correlationConsent": profile.correlation_consent_given,
                "createdAt": profile.created_at.isoformat() if profile.created_at else None,
            }
        }
    except Exception as e:
        logger.error(f"Error getting profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DATA ENDPOINTS
# =============================================================================

@router.get("/current")
async def get_current_environmental_data(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Get current environmental data and risk assessment for a patient."""
    try:
        service = EnvironmentalRiskService(db)
        data = await service.get_current_data(patient_id)
        
        if "error" in data:
            return {"success": False, "error": data["error"]}
        
        return {"success": True, **data}
    except Exception as e:
        logger.error(f"Error getting current data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refresh")
async def refresh_environmental_data(
    request: RefreshDataRequest,
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Refresh environmental data and recompute risk scores."""
    try:
        service = EnvironmentalRiskService(db)
        
        from app.models.environmental_risk import PatientEnvironmentProfile
        profile = db.query(PatientEnvironmentProfile).filter(
            PatientEnvironmentProfile.patient_id == patient_id,
            PatientEnvironmentProfile.is_active == True
        ).first()
        
        if not profile:
            if not request.zipCode:
                raise HTTPException(status_code=400, detail="ZIP code required for new profile")
            profile = await service.get_or_create_profile(patient_id, request.zipCode)
        
        zip_code = request.zipCode or profile.zip_code
        
        snapshot = await service.ingest_environmental_data(zip_code)
        if not snapshot:
            raise HTTPException(status_code=500, detail="Failed to fetch environmental data")
        
        risk_score = await service.compute_risk_score(patient_id, snapshot)
        
        alerts = []
        if risk_score:
            alerts = await service.check_and_generate_alerts(patient_id, risk_score)
        
        audit_log(
            db,
            patient_id=patient_id,
            action="environment_data_refreshed",
            resource_type="environmental_data",
            resource_id=snapshot.id,
            details={"zip_code": zip_code, "risk_level": risk_score.risk_level if risk_score else None}
        )
        
        data = await service.get_current_data(patient_id)
        
        return {
            "success": True,
            "refreshedAt": datetime.utcnow().isoformat(),
            "alertsGenerated": len(alerts),
            **data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RISK SCORE ENDPOINTS
# =============================================================================

@router.get("/risk")
async def get_risk_score(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Get current risk score for a patient."""
    try:
        from app.models.environmental_risk import PatientEnvironmentRiskScore
        from sqlalchemy import desc
        
        risk_score = db.query(PatientEnvironmentRiskScore).filter(
            PatientEnvironmentRiskScore.patient_id == patient_id
        ).order_by(desc(PatientEnvironmentRiskScore.computed_at)).first()
        
        if not risk_score:
            return {"success": True, "riskScore": None}
        
        return {
            "success": True,
            "riskScore": {
                "id": risk_score.id,
                "compositeScore": float(risk_score.composite_risk_score),
                "riskLevel": risk_score.risk_level,
                "computedAt": risk_score.computed_at.isoformat(),
                "components": {
                    "weather": float(risk_score.weather_risk_score) if risk_score.weather_risk_score else None,
                    "airQuality": float(risk_score.air_quality_risk_score) if risk_score.air_quality_risk_score else None,
                    "allergens": float(risk_score.allergen_risk_score) if risk_score.allergen_risk_score else None,
                    "hazards": float(risk_score.hazard_risk_score) if risk_score.hazard_risk_score else None,
                },
                "trends": {
                    "24hr": float(risk_score.trend_24hr) if risk_score.trend_24hr else None,
                    "48hr": float(risk_score.trend_48hr) if risk_score.trend_48hr else None,
                    "72hr": float(risk_score.trend_72hr) if risk_score.trend_72hr else None,
                },
                "volatility": float(risk_score.volatility_score) if risk_score.volatility_score else None,
                "factorContributions": risk_score.factor_contributions,
                "topRiskFactors": risk_score.top_risk_factors,
            }
        }
    except Exception as e:
        logger.error(f"Error getting risk score: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_risk_history(
    patient_id: str = Query(..., description="Patient ID"),
    days: int = Query(7, ge=1, le=90, description="Number of days of history"),
    db: Session = Depends(get_db)
):
    """Get historical risk scores for a patient."""
    try:
        service = EnvironmentalRiskService(db)
        history = await service.get_history(patient_id, days)
        
        return {
            "success": True,
            "days": days,
            "history": history
        }
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# FORECAST ENDPOINTS
# =============================================================================

@router.get("/forecast")
async def get_forecasts(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Get risk forecasts for a patient."""
    try:
        from app.models.environmental_risk import EnvironmentalForecast
        from sqlalchemy import desc
        
        forecasts = db.query(EnvironmentalForecast).filter(
            EnvironmentalForecast.patient_id == patient_id,
            EnvironmentalForecast.forecast_target_time > datetime.utcnow()
        ).order_by(EnvironmentalForecast.forecast_horizon).all()
        
        return {
            "success": True,
            "forecasts": [
                {
                    "id": f.id,
                    "horizon": f.forecast_horizon,
                    "targetTime": f.forecast_target_time.isoformat(),
                    "predictedScore": float(f.predicted_risk_score),
                    "predictedLevel": f.predicted_risk_level,
                    "confidence": f.confidence_interval,
                    "components": {
                        "weather": float(f.predicted_weather_risk) if f.predicted_weather_risk else None,
                        "airQuality": float(f.predicted_air_quality_risk) if f.predicted_air_quality_risk else None,
                        "allergens": float(f.predicted_allergen_risk) if f.predicted_allergen_risk else None,
                    },
                    "generatedAt": f.generated_at.isoformat(),
                    "modelName": f.model_name,
                }
                for f in forecasts
            ]
        }
    except Exception as e:
        logger.error(f"Error getting forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forecast/generate")
async def generate_forecasts(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Generate new risk forecasts for all horizons."""
    try:
        service = EnvironmentalRiskService(db)
        
        horizons = ["12hr", "24hr", "48hr"]
        forecasts = []
        
        for horizon in horizons:
            forecast = await service.generate_forecast(patient_id, horizon)
            if forecast:
                forecasts.append({
                    "horizon": forecast.forecast_horizon,
                    "predictedScore": float(forecast.predicted_risk_score),
                    "predictedLevel": forecast.predicted_risk_level,
                    "targetTime": forecast.forecast_target_time.isoformat(),
                })
        
        return {
            "success": True,
            "generatedAt": datetime.utcnow().isoformat(),
            "forecasts": forecasts
        }
    except Exception as e:
        logger.error(f"Error generating forecasts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# CORRELATION ENDPOINTS
# =============================================================================

@router.get("/correlations")
async def get_correlations(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Get symptom-environment correlations for a patient."""
    try:
        service = EnvironmentalRiskService(db)
        correlations = await service.get_correlations(patient_id)
        
        return {
            "success": True,
            "correlations": correlations
        }
    except Exception as e:
        logger.error(f"Error getting correlations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correlations/analyze")
async def analyze_correlations(
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Trigger correlation analysis for a patient."""
    try:
        service = EnvironmentalRiskService(db)
        correlations = await service.analyze_symptom_correlations(patient_id)
        
        significant = [c for c in correlations if c.is_statistically_significant]
        
        return {
            "success": True,
            "analyzedAt": datetime.utcnow().isoformat(),
            "totalCorrelations": len(correlations),
            "significantCorrelations": len(significant),
            "correlations": [
                {
                    "symptom": c.symptom_type,
                    "factor": c.environmental_factor,
                    "correlation": float(c.correlation_coefficient),
                    "isSignificant": c.is_statistically_significant,
                    "strength": c.relationship_strength,
                }
                for c in correlations
            ]
        }
    except Exception as e:
        logger.error(f"Error analyzing correlations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ALERT ENDPOINTS
# =============================================================================

@router.get("/alerts")
async def get_alerts(
    patient_id: str = Query(..., description="Patient ID"),
    status: str = Query("active", description="Alert status filter"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """Get alerts for a patient."""
    try:
        from app.models.environmental_risk import EnvironmentalAlert
        from sqlalchemy import desc
        
        query = db.query(EnvironmentalAlert).filter(
            EnvironmentalAlert.patient_id == patient_id
        )
        
        if status != "all":
            query = query.filter(EnvironmentalAlert.status == status)
        
        alerts = query.order_by(desc(EnvironmentalAlert.created_at)).limit(limit).all()
        
        return {
            "success": True,
            "alerts": [
                {
                    "id": a.id,
                    "type": a.alert_type,
                    "triggeredBy": a.triggered_by,
                    "severity": a.severity,
                    "priority": a.priority,
                    "title": a.title,
                    "message": a.message,
                    "recommendations": a.recommendations,
                    "status": a.status,
                    "triggerValue": float(a.trigger_value) if a.trigger_value else None,
                    "thresholdValue": float(a.threshold_value) if a.threshold_value else None,
                    "createdAt": a.created_at.isoformat(),
                    "acknowledgedAt": a.acknowledged_at.isoformat() if a.acknowledged_at else None,
                }
                for a in alerts
            ]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/acknowledge")
async def acknowledge_alert(
    request: AlertAcknowledgeRequest,
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Acknowledge an alert."""
    try:
        service = EnvironmentalRiskService(db)
        alert = await service.acknowledge_alert(request.alertId, patient_id)
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        return {
            "success": True,
            "alert": {
                "id": alert.id,
                "status": alert.status,
                "acknowledgedAt": alert.acknowledged_at.isoformat(),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/feedback")
async def provide_alert_feedback(
    request: AlertFeedbackRequest,
    patient_id: str = Query(..., description="Patient ID"),
    db: Session = Depends(get_db)
):
    """Provide feedback on an alert."""
    try:
        from app.models.environmental_risk import EnvironmentalAlert
        
        alert = db.query(EnvironmentalAlert).filter(
            EnvironmentalAlert.id == request.alertId,
            EnvironmentalAlert.patient_id == patient_id
        ).first()
        
        if not alert:
            raise HTTPException(status_code=404, detail="Alert not found")
        
        alert.was_helpful = request.wasHelpful
        alert.user_feedback = request.feedback
        db.commit()
        
        return {
            "success": True,
            "message": "Feedback recorded. Thank you!"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error providing feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# DOCTOR DASHBOARD ENDPOINT
# =============================================================================

@router.get("/patient/{patient_id}/summary")
async def get_patient_environmental_summary(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get environmental risk summary for a patient.
    Used by doctors in Assistant Lysa dashboard.
    """
    try:
        service = EnvironmentalRiskService(db)
        data = await service.get_current_data(patient_id)
        
        if "error" in data:
            return {
                "success": True,
                "hasProfile": False,
                "summary": None
            }
        
        history = await service.get_history(patient_id, days=7)
        correlations = await service.get_correlations(patient_id)
        
        risk_score = data.get("riskScore")
        current_data = data.get("currentData")
        forecasts = data.get("forecasts", [])
        alerts = data.get("activeAlerts", [])
        
        trend_description = "stable"
        if risk_score and risk_score.get("trends", {}).get("24hr"):
            trend = risk_score["trends"]["24hr"]
            if trend > 0.2:
                trend_description = "worsening"
            elif trend < -0.2:
                trend_description = "improving"
        
        return {
            "success": True,
            "hasProfile": True,
            "summary": {
                "location": {
                    "zipCode": data.get("profile", {}).get("zipCode"),
                    "city": data.get("profile", {}).get("city"),
                    "state": data.get("profile", {}).get("state"),
                },
                "conditions": data.get("profile", {}).get("conditions", []),
                "currentRisk": {
                    "score": risk_score.get("composite") if risk_score else None,
                    "level": risk_score.get("level") if risk_score else None,
                    "trend": trend_description,
                    "topFactors": risk_score.get("topFactors", [])[:3] if risk_score else [],
                },
                "currentConditions": {
                    "aqi": current_data.get("airQuality", {}).get("aqi") if current_data else None,
                    "aqiCategory": current_data.get("airQuality", {}).get("category") if current_data else None,
                    "temperature": current_data.get("weather", {}).get("temperature") if current_data else None,
                    "humidity": current_data.get("weather", {}).get("humidity") if current_data else None,
                    "pollenLevel": current_data.get("allergens", {}).get("pollenCategory") if current_data else None,
                },
                "forecast": {
                    "24hr": next(
                        (f["predictedLevel"] for f in forecasts if f["horizon"] == "24hr"),
                        None
                    ),
                    "48hr": next(
                        (f["predictedLevel"] for f in forecasts if f["horizon"] == "48hr"),
                        None
                    ),
                },
                "activeAlerts": len(alerts),
                "alertsSummary": [
                    {"severity": a["severity"], "title": a["title"]}
                    for a in alerts[:3]
                ],
                "significantCorrelations": [
                    {
                        "symptom": c["symptom"],
                        "factor": c["factor"],
                        "strength": c["strength"],
                    }
                    for c in correlations[:3]
                ],
                "weeklyTrend": {
                    "averageScore": sum(h["compositeScore"] for h in history) / len(history) if history else None,
                    "highestScore": max((h["compositeScore"] for h in history), default=None),
                    "lowestScore": min((h["compositeScore"] for h in history), default=None),
                    "dataPoints": len(history),
                }
            }
        }
    except Exception as e:
        logger.error(f"Error getting patient summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
