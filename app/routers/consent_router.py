"""
Voice Consent API Router
========================

API endpoints for HIPAA-compliant consent management
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List

from app.services.voice_consent_service import (
    get_consent_service,
    ConsentType,
    ConsentStatus,
)

router = APIRouter(prefix="/api/consent", tags=["Consent Management"])


class RequestConsentRequest(BaseModel):
    patient_id: str
    consent_type: str


class GrantConsentRequest(BaseModel):
    patient_id: str
    expires_in_days: Optional[int] = 365


class RevokeConsentRequest(BaseModel):
    patient_id: str
    consent_type: str


class CheckConsentRequest(BaseModel):
    patient_id: str
    consent_type: str


@router.get("/types")
async def get_consent_types():
    """Get all available consent types with descriptions"""
    service = get_consent_service()
    
    return {
        "consent_types": [
            {
                "type": ct.value,
                "text": service.get_consent_text(ct),
            }
            for ct in ConsentType
        ]
    }


@router.get("/{patient_id}")
async def get_patient_consents(patient_id: str):
    """Get all consent records for a patient"""
    service = get_consent_service()
    consents = service.get_patient_consents(patient_id)
    
    return {
        "patient_id": patient_id,
        "consents": [
            {
                "consent_id": c.consent_id,
                "consent_type": c.consent_type.value,
                "status": c.status.value,
                "granted_at": c.granted_at.isoformat() if c.granted_at else None,
                "expires_at": c.expires_at.isoformat() if c.expires_at else None,
                "version": c.version,
            }
            for c in consents
        ],
    }


@router.post("/request")
async def request_consent(request: RequestConsentRequest, req: Request):
    """Request consent from a patient"""
    service = get_consent_service()
    
    try:
        consent_type = ConsentType(request.consent_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid consent type: {request.consent_type}")
    
    ip_address = req.client.host if req.client else None
    user_agent = req.headers.get("user-agent")
    
    consent = service.request_consent(
        patient_id=request.patient_id,
        consent_type=consent_type,
        ip_address=ip_address,
        user_agent=user_agent,
    )
    
    return {
        "consent_id": consent.consent_id,
        "consent_type": consent.consent_type.value,
        "consent_text": consent.consent_text,
        "status": consent.status.value,
    }


@router.post("/{consent_id}/grant")
async def grant_consent(consent_id: str, request: GrantConsentRequest, req: Request):
    """Grant a pending consent"""
    service = get_consent_service()
    ip_address = req.client.host if req.client else None
    
    consent = service.grant_consent(
        consent_id=consent_id,
        patient_id=request.patient_id,
        ip_address=ip_address,
        expires_in_days=request.expires_in_days,
    )
    
    if not consent:
        raise HTTPException(status_code=404, detail="Consent not found or access denied")
    
    return {
        "consent_id": consent.consent_id,
        "status": consent.status.value,
        "granted_at": consent.granted_at.isoformat() if consent.granted_at else None,
        "expires_at": consent.expires_at.isoformat() if consent.expires_at else None,
    }


@router.post("/{consent_id}/deny")
async def deny_consent(consent_id: str, request: GrantConsentRequest, req: Request):
    """Deny a pending consent"""
    service = get_consent_service()
    ip_address = req.client.host if req.client else None
    
    consent = service.deny_consent(
        consent_id=consent_id,
        patient_id=request.patient_id,
        ip_address=ip_address,
    )
    
    if not consent:
        raise HTTPException(status_code=404, detail="Consent not found or access denied")
    
    return {
        "consent_id": consent.consent_id,
        "status": consent.status.value,
    }


@router.post("/revoke")
async def revoke_consent(request: RevokeConsentRequest, req: Request):
    """Revoke a previously granted consent"""
    service = get_consent_service()
    
    try:
        consent_type = ConsentType(request.consent_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid consent type: {request.consent_type}")
    
    ip_address = req.client.host if req.client else None
    
    success = service.revoke_consent(
        consent_type=consent_type,
        patient_id=request.patient_id,
        ip_address=ip_address,
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Active consent not found")
    
    return {
        "success": True,
        "consent_type": request.consent_type,
        "message": f"{request.consent_type} consent has been revoked",
    }


@router.get("/{patient_id}/check/{consent_type}")
async def check_consent(patient_id: str, consent_type: str):
    """Check if patient has active consent"""
    service = get_consent_service()
    
    try:
        ct = ConsentType(consent_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid consent type: {consent_type}")
    
    has_consent = service.has_consent(patient_id, ct)
    
    return {
        "patient_id": patient_id,
        "consent_type": consent_type,
        "has_consent": has_consent,
    }


@router.get("/{patient_id}/required/{feature}")
async def get_required_consents(patient_id: str, feature: str):
    """Get missing consents required for a feature"""
    service = get_consent_service()
    
    missing = service.get_required_consents(patient_id, feature)
    
    return {
        "patient_id": patient_id,
        "feature": feature,
        "missing_consents": [ct.value for ct in missing],
        "all_granted": len(missing) == 0,
    }


@router.get("/{patient_id}/audit")
async def get_consent_audit(patient_id: str, limit: int = 100):
    """Get consent audit log for a patient"""
    service = get_consent_service()
    entries = service.get_consent_audit(patient_id, limit)
    
    return {
        "patient_id": patient_id,
        "audit_entries": [
            {
                "audit_id": e.audit_id,
                "consent_id": e.consent_id,
                "action": e.action,
                "old_status": e.old_status.value if e.old_status else None,
                "new_status": e.new_status.value,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in entries
        ],
    }
