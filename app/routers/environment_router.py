"""
Environment Auto-Create Router
Endpoints for auto-creating environmental profiles from GPS location.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional
import logging

from app.database import get_db
from app.services.geocoding_service import reverse_geocode
from app.services.environmental_risk_service import EnvironmentalRiskService
from app.services.ehr_service import get_ehr_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/environment", tags=["environment"])


class AutoCreateRequest(BaseModel):
    """Request for auto-creating environmental profile."""
    consent: bool = Field(..., description="User must consent to location use")
    lat: Optional[float] = Field(None, description="Latitude (optional)")
    lon: Optional[float] = Field(None, description="Longitude (optional)")
    allow_store_latlon: bool = Field(False, description="Allow storing raw coordinates (default false)")


class AutoCreateResponse(BaseModel):
    """Response from auto-create."""
    success: bool
    profile_id: Optional[str] = None
    zip_code: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    conditions: list = []
    message: str = ""


@router.post("/patient/{patient_id}/auto_create", response_model=AutoCreateResponse)
async def auto_create_environmental_profile(
    patient_id: str,
    request: AutoCreateRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Auto-create environmental profile from GPS location and EHR conditions.
    
    Requirements:
    - consent=true is required
    - If lat/lon provided, reverse geocodes to ZIP
    - Otherwise falls back to patient profile ZIP if available
    - Fetches conditions from EHR server-side
    - Creates/updates environmental profile
    
    Privacy:
    - Stores ZIP only by default
    - Never stores raw lat/lon unless allow_store_latlon=true
    - All PHI access is audit logged
    """
    if not request.consent:
        raise HTTPException(
            status_code=400,
            detail="Consent is required to create environmental profile"
        )
    
    zip_code = None
    city = None
    state = None
    
    if request.lat is not None and request.lon is not None:
        geo_result = await reverse_geocode(request.lat, request.lon)
        if geo_result:
            zip_code = geo_result.get("zip")
            city = geo_result.get("city")
            state = geo_result.get("state")
            logger.info(f"Reverse geocoded to ZIP: {zip_code}")
    
    if not zip_code:
        from app.models.user import User
        user = db.query(User).filter(User.id == patient_id).first()
        if user and hasattr(user, 'zip_code') and user.zip_code:
            zip_code = user.zip_code
            logger.info(f"Using patient profile ZIP: {zip_code}")
    
    if not zip_code:
        return AutoCreateResponse(
            success=False,
            message="Could not determine ZIP code. Please provide location or update profile."
        )
    
    ehr_service = get_ehr_service(db)
    problems = await ehr_service.get_problem_list(patient_id, patient_id)
    
    conditions = []
    seen = set()
    for problem in problems:
        name = problem.get("name", "").lower()
        if name and name not in seen:
            seen.add(name)
            conditions.append(name)
    
    allergies = await ehr_service.get_allergies(patient_id, patient_id)
    allergy_list = [a.get("allergen", "") for a in allergies if a.get("allergen")]
    
    env_service = EnvironmentalRiskService(db)
    profile = await env_service.get_or_create_profile(
        patient_id=patient_id,
        zip_code=zip_code,
        conditions=conditions,
        allergies=allergy_list
    )
    
    try:
        await env_service.compute_current_risk(patient_id)
    except Exception as e:
        logger.warning(f"Failed to compute initial risk: {e}")
    
    return AutoCreateResponse(
        success=True,
        profile_id=profile.id,
        zip_code=zip_code,
        city=city,
        state=state,
        conditions=conditions,
        message=f"Environmental profile created with {len(conditions)} conditions"
    )


@router.get("/patient/{patient_id}/profile")
async def get_environmental_profile(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get patient's environmental profile."""
    from app.models.environmental_risk import PatientEnvironmentProfile
    
    profile = db.query(PatientEnvironmentProfile).filter(
        PatientEnvironmentProfile.patient_id == patient_id,
        PatientEnvironmentProfile.is_active == True
    ).first()
    
    if not profile:
        return {"exists": False, "profile": None}
    
    return {
        "exists": True,
        "profile": {
            "id": profile.id,
            "zip_code": profile.zip_code,
            "city": profile.city,
            "state": profile.state,
            "conditions": profile.chronic_conditions,
            "allergies": profile.allergies,
            "alerts_enabled": profile.alerts_enabled
        }
    }
