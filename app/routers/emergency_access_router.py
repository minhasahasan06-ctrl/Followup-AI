"""
Emergency Access Router - Break-the-glass emergency access
HIPAA-compliant emergency override with comprehensive audit logging.
"""

import logging
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.dependencies import get_current_user, get_current_doctor
from app.models.user import User
from app.services.access_control import (
    AccessControlService, HIPAAAuditLogger, AccessScope, PHICategory,
    get_access_control
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/emergency", tags=["emergency"])


class EmergencyAccessRequest(BaseModel):
    patient_id: str
    emergency_reason: str
    phi_categories: List[str]
    resource_type: str = "patient_data"


class EmergencyAccessResponse(BaseModel):
    success: bool
    access_granted: bool
    audit_id: str
    expires_in_minutes: int = 60
    message: str


@router.post("/break-glass", response_model=EmergencyAccessResponse)
async def break_glass_access(
    request_data: EmergencyAccessRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_doctor)
):
    """
    Emergency break-the-glass access for doctors.
    Bypasses normal access controls in life-threatening situations.
    
    HIPAA requires:
    1. Logging of who accessed what and why
    2. Notification to patient after emergency
    3. Review by compliance officer
    """
    if not current_user.license_verified:
        raise HTTPException(
            status_code=403,
            detail="Only verified doctors can request emergency access"
        )
    
    if len(request_data.emergency_reason) < 20:
        raise HTTPException(
            status_code=400,
            detail="Emergency reason must be detailed (at least 20 characters)"
        )
    
    patient = db.execute(
        text("SELECT id, email, first_name, last_name FROM users WHERE id = :pid"),
        {"pid": request_data.patient_id}
    ).fetchone()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    valid_categories = [c.value for c in PHICategory]
    for cat in request_data.phi_categories:
        if cat not in valid_categories:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid PHI category: {cat}"
            )
    
    audit_id = HIPAAAuditLogger.log_emergency_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=request_data.patient_id,
        emergency_reason=request_data.emergency_reason,
        phi_categories=request_data.phi_categories,
        resource_type=request_data.resource_type,
        ip_address=request.client.host if request.client else None
    )
    
    db.execute(
        text("""
            INSERT INTO hipaa_audit_logs 
            (id, actor_id, actor_role, patient_id, action, 
             phi_categories, resource_type, access_scope, access_reason,
             consent_verified, ip_address, success, additional_context, created_at)
            VALUES 
            (:id, :actor_id, 'doctor', :patient_id, 'emergency_break_glass',
             :categories, :resource_type, 'emergency', :reason,
             FALSE, :ip, TRUE, :context, NOW())
        """),
        {
            "id": audit_id + "_btg",
            "actor_id": current_user.id,
            "patient_id": request_data.patient_id,
            "categories": str(request_data.phi_categories),
            "resource_type": request_data.resource_type,
            "reason": request_data.emergency_reason,
            "ip": request.client.host if request.client else None,
            "context": f'{{"break_the_glass": true, "verified_doctor": true}}'
        }
    )
    db.commit()
    
    logger.warning(
        f"[EMERGENCY ACCESS] Doctor {current_user.id} accessed patient {request_data.patient_id} - "
        f"Reason: {request_data.emergency_reason}"
    )
    
    return EmergencyAccessResponse(
        success=True,
        access_granted=True,
        audit_id=audit_id,
        expires_in_minutes=60,
        message="Emergency access granted. This access is logged and will be reviewed."
    )


@router.get("/access-log/{patient_id}")
async def get_emergency_access_log(
    patient_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get emergency access log for a patient.
    Patients can see who accessed their data in emergencies.
    Doctors can see their own emergency accesses.
    """
    if current_user.role == "patient" and current_user.id != patient_id:
        raise HTTPException(status_code=403, detail="Can only view own access logs")
    
    if current_user.role == "patient":
        rows = db.execute(
            text("""
                SELECT id, actor_id, action, access_reason, created_at
                FROM hipaa_audit_logs
                WHERE patient_id = :pid 
                AND access_scope = 'emergency'
                ORDER BY created_at DESC
                LIMIT 100
            """),
            {"pid": patient_id}
        ).fetchall()
    else:
        rows = db.execute(
            text("""
                SELECT id, patient_id, action, access_reason, created_at
                FROM hipaa_audit_logs
                WHERE actor_id = :did 
                AND access_scope = 'emergency'
                ORDER BY created_at DESC
                LIMIT 100
            """),
            {"did": current_user.id}
        ).fetchall()
    
    return {
        "accesses": [
            {
                "id": row[0],
                "actor_or_patient_id": row[1],
                "action": row[2],
                "reason": row[3],
                "timestamp": row[4].isoformat() if row[4] else None
            }
            for row in rows
        ]
    }
