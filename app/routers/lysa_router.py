"""
Lysa Clinical Documentation Router
Endpoints for AI-generated clinical documentation drafts.
Doctor-only access - all drafts require clinician review before use.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.database import get_db
from app.services.lysa_documentation_service import get_lysa_documentation_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/lysa/patient", tags=["lysa"])


class DifferentialRequest(BaseModel):
    """Request for generating differential diagnosis."""
    question: Optional[str] = Field(None, description="Clinical question or context")


class ReviseRequest(BaseModel):
    """Request for revising a draft."""
    instruction: str = Field(..., description="Revision instruction from clinician")


class ApproveRequest(BaseModel):
    """Request for approving a draft."""
    insert_to_chart: bool = Field(False, description="Insert to EHR after approval")
    confirmation: bool = Field(False, description="Explicit confirmation required")


@router.post("/{patient_id}/differential")
async def generate_differential(
    patient_id: str,
    request: DifferentialRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Generate a structured differential diagnosis draft.
    
    This endpoint:
    1. Fetches full EHR record server-side (notes, problem list, meds, allergies, labs, imaging, vitals)
    2. Fetches recent followup answers and trends
    3. Builds sanitized summary with provenance tracking
    4. Generates structured differential using AI
    5. Saves draft to database with audit logging
    
    Returns a draft for clinician review - NOT autonomous medical advice.
    """
    doctor_id = http_request.headers.get("X-Doctor-Id")
    
    if not doctor_id:
        from app.models.patient_doctor_connection import PatientDoctorConnection
        connection = db.query(PatientDoctorConnection).filter(
            PatientDoctorConnection.patient_id == patient_id,
            PatientDoctorConnection.status == "active"
        ).first()
        if connection:
            doctor_id = connection.doctor_id
        else:
            from app.models.user import User
            fallback_doctor = db.query(User).filter(User.role == "doctor").first()
            if fallback_doctor:
                doctor_id = fallback_doctor.id
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No doctor assigned to patient. Please provide X-Doctor-Id header."
                )
    
    lysa_service = get_lysa_documentation_service(db)
    
    try:
        result = await lysa_service.generate_differential(
            patient_id=patient_id,
            doctor_id=doctor_id,
            question=request.question,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("User-Agent")
        )
        return result
    except Exception as e:
        logger.error(f"Error generating differential: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_id}/drafts")
async def get_drafts(
    patient_id: str,
    http_request: Request,
    draft_type: Optional[str] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get all drafts for a patient."""
    doctor_id = http_request.headers.get("X-Doctor-Id")
    
    if not doctor_id:
        from app.models.patient_doctor_connection import PatientDoctorConnection
        connection = db.query(PatientDoctorConnection).filter(
            PatientDoctorConnection.patient_id == patient_id,
            PatientDoctorConnection.status == "active"
        ).first()
        if connection:
            doctor_id = connection.doctor_id
        else:
            doctor_id = None
    
    lysa_service = get_lysa_documentation_service(db)
    
    try:
        drafts = await lysa_service.get_drafts(
            patient_id=patient_id,
            doctor_id=doctor_id,
            draft_type=draft_type,
            status=status
        )
        return {"drafts": drafts}
    except Exception as e:
        logger.error(f"Error fetching drafts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{patient_id}/drafts/{draft_id}/revise")
async def revise_draft(
    patient_id: str,
    draft_id: str,
    request: ReviseRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Revise a draft based on clinician instruction.
    
    The original draft is preserved in revision history.
    """
    doctor_id = http_request.headers.get("X-Doctor-Id", "unknown")
    
    lysa_service = get_lysa_documentation_service(db)
    
    try:
        result = await lysa_service.revise_draft(
            draft_id=draft_id,
            doctor_id=doctor_id,
            instruction=request.instruction,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("User-Agent")
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error revising draft: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{patient_id}/drafts/{draft_id}/approve")
async def approve_draft(
    patient_id: str,
    draft_id: str,
    request: ApproveRequest,
    http_request: Request,
    db: Session = Depends(get_db)
):
    """
    Approve a draft and optionally insert to chart.
    
    IMPORTANT: Requires explicit confirmation (confirmation=true).
    The draft becomes part of the official medical record only after approval.
    """
    if not request.confirmation:
        raise HTTPException(
            status_code=400,
            detail="Explicit confirmation required. Set confirmation=true to approve."
        )
    
    doctor_id = http_request.headers.get("X-Doctor-Id", "unknown")
    
    lysa_service = get_lysa_documentation_service(db)
    
    try:
        result = await lysa_service.approve_draft(
            draft_id=draft_id,
            doctor_id=doctor_id,
            insert_to_chart=request.insert_to_chart,
            ip_address=http_request.client.host if http_request.client else None,
            user_agent=http_request.headers.get("User-Agent")
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error approving draft: {e}")
        raise HTTPException(status_code=500, detail=str(e))
