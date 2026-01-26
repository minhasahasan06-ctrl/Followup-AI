"""
Doctor Lysa Router - AI-powered clinical documentation with PHI protection
Secure endpoints for differential diagnosis and draft management.
"""

import logging
import json
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
    get_access_control, require_patient_access
)
from app.services.llm_safe_client import get_llm_safe_client
from app.services.phi_detection_service import get_phi_detection_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/doctor/lysa", tags=["lysa"])


def require_verified_doctor(current_user: User = Depends(get_current_doctor)) -> User:
    """Require a verified doctor for clinical tools."""
    if not current_user.license_verified:
        raise HTTPException(
            status_code=403,
            detail="Doctor license must be verified to access clinical tools"
        )
    return current_user


class DifferentialRequest(BaseModel):
    symptoms: List[str]
    patient_history: Optional[str] = None
    medications: Optional[List[str]] = None
    lab_results: Optional[str] = None


class DraftApprovalRequest(BaseModel):
    notes: Optional[str] = None


@router.post("/{patient_id}/differential")
async def generate_differential(
    patient_id: str,
    request_data: DifferentialRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_verified_doctor)
):
    """
    Generate differential diagnosis for a patient.
    Requires verified doctor with patient access.
    """
    acs = get_access_control()
    decision = acs.verify_doctor_patient_access(
        db=db,
        doctor_id=current_user.id,
        patient_id=patient_id,
        required_scope=AccessScope.FULL,
        phi_categories=[PHICategory.SYMPTOMS.value, PHICategory.LAB_RESULTS.value]
    )
    
    if not decision.allowed:
        HIPAAAuditLogger.log_phi_access(
            actor_id=current_user.id,
            actor_role="doctor",
            patient_id=patient_id,
            action="differential_diagnosis_denied",
            phi_categories=[PHICategory.SYMPTOMS.value],
            resource_type="lysa_differential",
            success=False,
            error_message=decision.reason,
            ip_address=request.client.host if request.client else None
        )
        raise HTTPException(status_code=403, detail=f"Access denied: {decision.reason}")
    
    patient = db.execute(
        text("SELECT first_name, last_name FROM users WHERE id = :pid"),
        {"pid": patient_id}
    ).fetchone()
    
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    llm_client = get_llm_safe_client()
    
    prompt = f"""You are a clinical decision support system for differential diagnosis.
Based on the following patient information, provide a differential diagnosis with:
1. Top 5 most likely diagnoses ranked by probability
2. Key supporting findings for each
3. Recommended tests to confirm/rule out
4. Red flags to watch for

Patient Symptoms: {', '.join(request_data.symptoms)}
Patient History: {request_data.patient_history or 'Not provided'}
Current Medications: {', '.join(request_data.medications) if request_data.medications else 'None reported'}
Lab Results: {request_data.lab_results or 'No lab results provided'}

Provide response as JSON with structure:
{{
    "diagnoses": [
        {{
            "name": "Diagnosis name",
            "probability": "High/Medium/Low",
            "icd10_code": "Code if known",
            "supporting_findings": ["finding1", "finding2"],
            "tests_recommended": ["test1", "test2"],
            "red_flags": ["flag1"]
        }}
    ],
    "summary": "Brief clinical summary",
    "urgency": "routine/urgent/emergent"
}}
"""
    
    result = llm_client.safe_completion(
        messages=[
            {"role": "system", "content": "You are a clinical decision support AI. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=patient_id,
        model="gpt-4o",
        temperature=0.3,
        allow_phi=True,
        redact_phi=False,
        access_reason="differential_diagnosis"
    )
    
    if not result.success:
        raise HTTPException(status_code=500, detail=f"AI generation failed: {result.error}")
    
    try:
        content_json = json.loads(result.content)
    except json.JSONDecodeError:
        content_json = {"raw_response": result.content}
    
    draft_result = db.execute(
        text("""
            INSERT INTO lysa_drafts 
            (patient_id, doctor_id, draft_type, status, question, content_json, raw_output, provenance)
            VALUES (:pid, :did, 'differential', 'draft', :question, :content, :raw, :provenance)
            RETURNING id
        """),
        {
            "pid": patient_id,
            "did": current_user.id,
            "question": f"Differential for: {', '.join(request_data.symptoms)}",
            "content": json.dumps(content_json),
            "raw": result.content,
            "provenance": json.dumps({
                "model": result.model,
                "phi_detected": result.phi_detected,
                "audit_id": result.audit_id,
                "generated_at": __import__("datetime").datetime.utcnow().isoformat()
            })
        }
    )
    draft_id = draft_result.fetchone()[0]
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=patient_id,
        action="differential_diagnosis_generated",
        phi_categories=[PHICategory.SYMPTOMS.value, PHICategory.CLINICAL_NOTES.value],
        resource_type="lysa_draft",
        resource_id=draft_id,
        access_scope=decision.access_scope.value,
        assignment_id=decision.assignment_id,
        ip_address=request.client.host if request.client else None,
        success=True
    )
    
    return {
        "success": True,
        "draft_id": draft_id,
        "content": content_json,
        "status": "draft",
        "audit_id": result.audit_id
    }


@router.post("/{patient_id}/drafts/{draft_id}/approve")
async def approve_draft(
    patient_id: str,
    draft_id: str,
    request_data: DraftApprovalRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_verified_doctor)
):
    """
    Approve a Lysa draft and commit to clinical notes.
    """
    acs = get_access_control()
    decision = acs.verify_doctor_patient_access(
        db=db,
        doctor_id=current_user.id,
        patient_id=patient_id,
        required_scope=AccessScope.FULL,
        phi_categories=[PHICategory.CLINICAL_NOTES.value]
    )
    
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Access denied: {decision.reason}")
    
    draft = db.execute(
        text("""
            SELECT id, patient_id, doctor_id, status, content_json 
            FROM lysa_drafts 
            WHERE id = :did AND patient_id = :pid
        """),
        {"did": draft_id, "pid": patient_id}
    ).fetchone()
    
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    
    if draft[2] != current_user.id:
        raise HTTPException(status_code=403, detail="Only the creating doctor can approve")
    
    if draft[3] == "approved":
        raise HTTPException(status_code=400, detail="Draft already approved")
    
    db.execute(
        text("""
            UPDATE lysa_drafts 
            SET status = 'approved', 
                approved_at = NOW(), 
                approved_by = :aid,
                revision_history = revision_history || :note
            WHERE id = :did
        """),
        {
            "did": draft_id,
            "aid": current_user.id,
            "note": json.dumps([{
                "action": "approved",
                "by": current_user.id,
                "at": __import__("datetime").datetime.utcnow().isoformat(),
                "notes": request_data.notes
            }])
        }
    )
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="doctor",
        patient_id=patient_id,
        action="draft_approved",
        phi_categories=[PHICategory.CLINICAL_NOTES.value],
        resource_type="lysa_draft",
        resource_id=draft_id,
        ip_address=request.client.host if request.client else None,
        success=True,
        additional_context={"notes": request_data.notes}
    )
    
    return {
        "success": True,
        "draft_id": draft_id,
        "status": "approved"
    }


@router.get("/{patient_id}/drafts")
async def list_drafts(
    patient_id: str,
    status: Optional[str] = None,
    request: Request = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_verified_doctor)
):
    """List drafts for a patient."""
    acs = get_access_control()
    decision = acs.verify_doctor_patient_access(
        db=db,
        doctor_id=current_user.id,
        patient_id=patient_id,
        required_scope=AccessScope.LIMITED
    )
    
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=f"Access denied: {decision.reason}")
    
    query = """
        SELECT id, draft_type, status, question, created_at, approved_at
        FROM lysa_drafts
        WHERE patient_id = :pid AND doctor_id = :did
    """
    params = {"pid": patient_id, "did": current_user.id}
    
    if status:
        query += " AND status = :status"
        params["status"] = status
    
    query += " ORDER BY created_at DESC LIMIT 50"
    
    rows = db.execute(text(query), params).fetchall()
    
    return {
        "drafts": [
            {
                "id": row[0],
                "draft_type": row[1],
                "status": row[2],
                "question": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "approved_at": row[5].isoformat() if row[5] else None
            }
            for row in rows
        ]
    }
