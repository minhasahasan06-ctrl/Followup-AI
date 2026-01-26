"""
Doctor Billing Router - License verification and billing management
Admin endpoints for verifying doctor licenses.
"""

import logging
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.dependencies import get_current_user, require_admin
from app.models.user import User
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/doctor", tags=["doctor-billing"])


class VerifyLicenseRequest(BaseModel):
    doctor_id: str
    license_number: str
    license_country: str
    verified: bool = True
    notes: Optional[str] = None


class SendVerificationRequest(BaseModel):
    doctor_id: str
    method: str = "email"


@router.post("/verify-license")
async def verify_license(
    request_data: VerifyLicenseRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin_user = Depends(require_admin)
):
    """
    Verify a doctor's medical license (Admin only).
    Sets license_verified flag to allow clinical tools access.
    """
    doctor = db.execute(
        text("SELECT id, email, role, license_verified FROM users WHERE id = :did"),
        {"did": request_data.doctor_id}
    ).fetchone()
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    if doctor[2] != "doctor":
        raise HTTPException(status_code=400, detail="User is not a doctor")
    
    db.execute(
        text("""
            UPDATE users 
            SET medical_license_number = :license,
                license_country = :country,
                license_verified = :verified,
                admin_verified = :verified,
                admin_verified_at = CASE WHEN :verified THEN NOW() ELSE NULL END,
                admin_verified_by = CASE WHEN :verified THEN :admin_id ELSE NULL END,
                updated_at = NOW()
            WHERE id = :did
        """),
        {
            "did": request_data.doctor_id,
            "license": request_data.license_number,
            "country": request_data.license_country,
            "verified": request_data.verified,
            "admin_id": admin_user.client_id
        }
    )
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=admin_user.client_id,
        actor_role="admin",
        patient_id="system",
        action="doctor_license_verified" if request_data.verified else "doctor_license_rejected",
        phi_categories=["doctor_credentials"],
        resource_type="user",
        resource_id=request_data.doctor_id,
        ip_address=request.client.host if request.client else None,
        success=True,
        additional_context={
            "license_number": request_data.license_number[-4:],
            "country": request_data.license_country,
            "notes": request_data.notes
        }
    )
    
    return {
        "success": True,
        "doctor_id": request_data.doctor_id,
        "license_verified": request_data.verified,
        "message": "License verification updated"
    }


@router.post("/send-verification")
async def send_verification(
    request_data: SendVerificationRequest,
    request: Request,
    db: Session = Depends(get_db),
    admin_user = Depends(require_admin)
):
    """
    Send email/SMS verification to a doctor using Stytch.
    Used for initial identity verification before license approval.
    """
    doctor = db.execute(
        text("SELECT id, email, phone_number, role FROM users WHERE id = :did"),
        {"did": request_data.doctor_id}
    ).fetchone()
    
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    if doctor[3] != "doctor":
        raise HTTPException(status_code=400, detail="User is not a doctor")
    
    stytch_project_id = os.getenv("STYTCH_PROJECT_ID")
    stytch_secret = os.getenv("STYTCH_SECRET")
    
    if not stytch_project_id or not stytch_secret:
        raise HTTPException(status_code=503, detail="Stytch not configured")
    
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            if request_data.method == "email":
                response = await client.post(
                    "https://api.stytch.com/v1/magic_links/email/send",
                    auth=(stytch_project_id, stytch_secret),
                    json={
                        "email": doctor[1],
                        "login_magic_link_url": f"{os.getenv('FRONTEND_BASE_URL', 'http://localhost:5000')}/verify-email",
                        "signup_magic_link_url": f"{os.getenv('FRONTEND_BASE_URL', 'http://localhost:5000')}/verify-email"
                    }
                )
            elif request_data.method == "sms":
                if not doctor[2]:
                    raise HTTPException(status_code=400, detail="Doctor has no phone number")
                
                response = await client.post(
                    "https://api.stytch.com/v1/otps/sms/send",
                    auth=(stytch_project_id, stytch_secret),
                    json={
                        "phone_number": doctor[2]
                    }
                )
            else:
                raise HTTPException(status_code=400, detail="Invalid verification method")
            
            if response.status_code not in [200, 201]:
                logger.error(f"Stytch verification failed: {response.text}")
                raise HTTPException(status_code=502, detail="Verification service error")
        
        return {
            "success": True,
            "doctor_id": request_data.doctor_id,
            "method": request_data.method,
            "message": f"Verification {request_data.method} sent"
        }
        
    except httpx.RequestError as e:
        logger.error(f"Stytch request failed: {e}")
        raise HTTPException(status_code=503, detail="Verification service unavailable")


@router.get("/unverified")
async def list_unverified_doctors(
    db: Session = Depends(get_db),
    admin_user = Depends(require_admin)
):
    """List doctors pending license verification (Admin only)."""
    rows = db.execute(
        text("""
            SELECT id, email, first_name, last_name, 
                   medical_license_number, license_country,
                   google_drive_application_url, created_at
            FROM users 
            WHERE role = 'doctor' AND license_verified = FALSE
            ORDER BY created_at DESC
            LIMIT 100
        """)
    ).fetchall()
    
    return {
        "doctors": [
            {
                "id": row[0],
                "email": row[1],
                "first_name": row[2],
                "last_name": row[3],
                "license_number": row[4],
                "license_country": row[5],
                "application_url": row[6],
                "created_at": row[7].isoformat() if row[7] else None
            }
            for row in rows
        ]
    }
