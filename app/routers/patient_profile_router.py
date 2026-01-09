"""
Patient Profile Router
Handles extended patient profile management including doctor assignment
HIPAA-compliant with proper authentication and RBAC
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import Optional
from pydantic import BaseModel
import logging

from app.database import get_db
from app.models.terms_audit import PatientProfileExtended
from app.models.user import User
from app.schemas.terms_audit_schemas import (
    PatientProfileUpdate,
    PatientProfileResponse
)
from app.services.user_audit_service import log_user_audit, AuditEventType
from app.services.audit_logger import AuditLogger
from app.auth.auth0 import get_current_token, TokenPayload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/patient", tags=["Patient Profile"])


class AssignDoctorRequest(BaseModel):
    """Doctor assignment request - patient_id derived from auth context"""
    doctor_id: str


def get_or_create_profile(user_id: str, db: Session) -> PatientProfileExtended:
    """Get existing profile or create new one"""
    profile = db.query(PatientProfileExtended).filter(
        PatientProfileExtended.user_id == user_id
    ).first()
    
    if not profile:
        profile = PatientProfileExtended(user_id=user_id)
        db.add(profile)
        db.commit()
        db.refresh(profile)
    
    return profile


def get_doctor_info(doctor_id: str, db: Session) -> Optional[dict]:
    """Get doctor info for display"""
    doctor = db.query(User).filter(
        User.id == doctor_id,
        User.role == "doctor"
    ).first()
    
    if not doctor:
        return None
    
    return {
        "id": doctor.id,
        "first_name": doctor.first_name,
        "last_name": doctor.last_name,
        "email": doctor.email,
        "organization": doctor.organization,
        "specialty": getattr(doctor, 'specialty', None),
        "license_verified": doctor.license_verified
    }


def verify_patient_role(user_id: str, db: Session) -> User:
    """Verify user is a patient"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.role != "patient":
        raise HTTPException(status_code=403, detail="Patient access required")
    return user


@router.get("/profile/extended", response_model=PatientProfileResponse)
async def get_patient_profile_extended(
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Get extended patient profile with assigned doctor info
    User ID derived from authenticated token
    """
    user_id = token.user_id
    verify_patient_role(user_id, db)
    
    profile = get_or_create_profile(user_id, db)
    
    assigned_doctor = None
    if profile.assigned_doctor_id:
        assigned_doctor = get_doctor_info(profile.assigned_doctor_id, db)
    
    return PatientProfileResponse(
        user_id=profile.user_id,
        assigned_doctor_id=profile.assigned_doctor_id,
        assigned_doctor=assigned_doctor,
        emergency_contacts=profile.emergency_contacts or [],
        medications=profile.medications or [],
        allergies=profile.allergies or [],
        chronic_conditions=profile.chronic_conditions or [],
        recent_labs=profile.recent_labs or [],
        hospitalizations=profile.hospitalizations or [],
        connected_devices=profile.connected_devices or []
    )


@router.post("/profile/extended")
async def update_patient_profile_extended(
    payload: PatientProfileUpdate,
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Update extended patient profile
    User ID derived from authenticated token
    """
    user_id = token.user_id
    verify_patient_role(user_id, db)
    
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:500]
    
    profile = get_or_create_profile(user_id, db)
    
    changes = []
    update_data = payload.model_dump(exclude_unset=True)
    
    for key, value in update_data.items():
        if key == "assigned_doctor_id":
            continue
        
        if value is not None:
            old_value = getattr(profile, key)
            if old_value != value:
                if isinstance(value, list):
                    setattr(profile, key, [
                        item.model_dump() if hasattr(item, 'model_dump') else item 
                        for item in value
                    ])
                else:
                    setattr(profile, key, value)
                changes.append(key)
    
    db.commit()
    
    for change in changes:
        event_type_map = {
            "medications": AuditEventType.MEDICATIONS_UPDATED,
            "allergies": AuditEventType.ALLERGIES_UPDATED,
            "emergency_contacts": AuditEventType.EMERGENCY_CONTACTS_UPDATED,
            "chronic_conditions": AuditEventType.CHRONIC_CONDITIONS_UPDATED,
        }
        event_type = event_type_map.get(change, AuditEventType.PROFILE_UPDATED)
        
        log_user_audit(
            user_id=user_id,
            event_type=event_type,
            event_data={"field": change, "updated": True},
            ip_address=ip_address,
            user_agent=user_agent,
            db=db
        )
    
    return {"status": "ok", "updated_fields": changes}


@router.post("/assign-doctor")
async def assign_doctor(
    payload: AssignDoctorRequest,
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Assign a doctor as patient's primary doctor
    RBAC: Patient ID is derived from auth token - only the authenticated patient
    can assign a doctor to themselves. No spoofing possible.
    """
    user_id = token.user_id
    verify_patient_role(user_id, db)
    
    doctor = db.query(User).filter(
        User.id == payload.doctor_id,
        User.role == "doctor"
    ).first()
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    profile = get_or_create_profile(user_id, db)
    
    old_doctor_id = profile.assigned_doctor_id
    profile.assigned_doctor_id = payload.doctor_id
    db.commit()
    
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent", "")[:500]
    
    log_user_audit(
        user_id=user_id,
        event_type=AuditEventType.DOCTOR_ASSIGNED,
        event_data={
            "doctor_id": payload.doctor_id,
            "doctor_name": f"{doctor.first_name} {doctor.last_name}",
            "previous_doctor_id": old_doctor_id
        },
        ip_address=ip_address,
        user_agent=user_agent,
        db=db
    )
    
    logger.info(f"Doctor {payload.doctor_id} assigned to patient {user_id}")
    
    return {
        "status": "ok",
        "assigned_doctor": get_doctor_info(payload.doctor_id, db)
    }


@router.delete("/unassign-doctor")
async def unassign_doctor(
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Remove assigned doctor from patient profile
    User ID derived from authenticated token
    """
    user_id = token.user_id
    verify_patient_role(user_id, db)
    
    profile = db.query(PatientProfileExtended).filter(
        PatientProfileExtended.user_id == user_id
    ).first()
    
    if not profile or not profile.assigned_doctor_id:
        raise HTTPException(status_code=404, detail="No assigned doctor to remove")
    
    old_doctor_id = profile.assigned_doctor_id
    profile.assigned_doctor_id = None
    db.commit()
    
    log_user_audit(
        user_id=user_id,
        event_type=AuditEventType.DOCTOR_UNASSIGNED,
        event_data={"previous_doctor_id": old_doctor_id},
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent", "")[:500],
        db=db
    )
    
    return {"status": "ok"}


@router.get("/audit-log")
async def get_patient_audit_log(
    page: int = 1,
    page_size: int = 20,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Get audit log for patient (authenticated user only)
    User ID derived from authenticated token
    """
    user_id = token.user_id
    
    from app.services.user_audit_service import get_user_audit_logs
    
    return get_user_audit_logs(user_id, page, page_size, db=db)
