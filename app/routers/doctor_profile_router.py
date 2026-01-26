"""
Doctor Profile Router
Handles extended doctor profile management
HIPAA-compliant with proper authentication
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
import logging

from app.database import get_db
from app.models.terms_audit import DoctorProfileExtended
from app.models.user import User
from app.schemas.terms_audit_schemas import DoctorProfileUpdate, DoctorProfileResponse
from app.services.user_audit_service import log_user_audit, AuditEventType
from app.dependencies import get_current_token, TokenPayload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/doctor", tags=["Doctor Profile"])


def get_or_create_doctor_profile(user_id: str, db: Session) -> DoctorProfileExtended:
    """Get existing doctor profile or create new one"""
    profile = db.query(DoctorProfileExtended).filter(
        DoctorProfileExtended.user_id == user_id
    ).first()
    
    if not profile:
        profile = DoctorProfileExtended(user_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    
    return profile


def verify_doctor_role(user_id: str, db: Session) -> User:
    """Verify user is a doctor"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role != "doctor":
        raise HTTPException(status_code=403, detail="Doctor access required")
    return user


@router.get("/profile/extended", response_model=DoctorProfileResponse)
async def get_doctor_profile_extended(
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Get extended doctor profile
    User ID derived from authenticated token
    """
    user_id = token.user_id
    verify_doctor_role(user_id, db)
    
    profile = get_or_create_doctor_profile(user_id, db)
    
    return DoctorProfileResponse(
        user_id=profile.user_id,
        npi=profile.npi,
        hospital_affiliations=profile.hospital_affiliations or [],
        board_certifications=profile.board_certifications or [],
        languages=profile.languages or [],
        accepted_insurances=profile.accepted_insurances or [],
        telemedicine_available=profile.telemedicine_available or False,
        telemedicine_fee=profile.telemedicine_fee,
        consultation_fee=profile.consultation_fee,
        publications=profile.publications or [],
        malpractice_info=profile.malpractice_info or {}
    )


@router.post("/profile/extended")
async def update_doctor_profile_extended(
    payload: DoctorProfileUpdate,
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Update extended doctor profile
    User ID derived from authenticated token
    """
    user_id = token.user_id
    verify_doctor_role(user_id, db)
    
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:500]
    
    profile = get_or_create_doctor_profile(user_id, db)
    
    changes = []
    update_data = payload.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if value is not None:
            old_value = getattr(profile, key)
            if old_value != value:
                setattr(profile, key, value)
                changes.append(key)
    
    db.commit()
    
    if changes:
        log_user_audit(
            user_id=user_id,
            event_type=AuditEventType.PROFILE_UPDATED,
            event_data={"updated_fields": changes, "role": "doctor"},
            ip_address=ip_address,
            user_agent=user_agent,
            db=db
        )
    
    return {"status": "ok", "updated_fields": changes}


@router.get("/profile/extended/{doctor_id}")
async def get_doctor_profile_by_id(
    doctor_id: str,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Get public doctor profile info (for patient viewing)
    Requires authentication but no specific role
    """
    doctor = db.query(User).filter(
        User.id == doctor_id,
        User.role == "doctor"
    ).first()
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    profile = db.query(DoctorProfileExtended).filter(
        DoctorProfileExtended.user_id == doctor_id
    ).first()
    
    return {
        "id": doctor.id,
        "first_name": doctor.first_name,
        "last_name": doctor.last_name,
        "organization": doctor.organization,
        "license_verified": doctor.license_verified,
        "npi": profile.npi if profile else None,
        "hospital_affiliations": profile.hospital_affiliations if profile else [],
        "board_certifications": profile.board_certifications if profile else [],
        "languages": profile.languages if profile else [],
        "accepted_insurances": profile.accepted_insurances if profile else [],
        "telemedicine_available": profile.telemedicine_available if profile else False,
        "telemedicine_fee": profile.telemedicine_fee if profile else None,
        "consultation_fee": profile.consultation_fee if profile else None
    }
