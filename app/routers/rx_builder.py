"""
Rx Builder API - AI-Assisted Prescription System with SOAP Notes and ICD-10 Suggestions
HIPAA-compliant with comprehensive audit logging
"""

import logging
import uuid
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field

from app.database import get_db
from app.auth import get_current_user
from app.models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/rx-builder", tags=["rx-builder"])


class SOAPNoteCreate(BaseModel):
    patient_id: str
    encounter_date: Optional[datetime] = None
    chief_complaint: Optional[str] = Field(None, max_length=500)
    subjective: Optional[str] = None
    history_present_illness: Optional[str] = None
    review_of_systems: Optional[Dict[str, Any]] = None
    objective: Optional[str] = None
    vital_signs: Optional[Dict[str, Any]] = None
    physical_exam: Optional[str] = None
    lab_results: Optional[List[Dict[str, Any]]] = None
    assessment: Optional[str] = None
    primary_diagnosis: Optional[str] = Field(None, max_length=500)
    primary_icd10: Optional[str] = Field(None, max_length=20)
    secondary_diagnoses: Optional[List[Dict[str, str]]] = None
    differential_diagnoses: Optional[List[str]] = None
    plan: Optional[str] = None
    medications_prescribed: Optional[List[Dict[str, Any]]] = None
    procedures_ordered: Optional[List[str]] = None
    referrals: Optional[List[str]] = None
    patient_education: Optional[str] = None
    follow_up_instructions: Optional[str] = None
    follow_up_date: Optional[datetime] = None
    linked_appointment_id: Optional[str] = None


class SOAPNoteUpdate(BaseModel):
    subjective: Optional[str] = None
    chief_complaint: Optional[str] = None
    objective: Optional[str] = None
    assessment: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    primary_icd10: Optional[str] = None
    secondary_diagnoses: Optional[List[Dict[str, str]]] = None
    plan: Optional[str] = None
    medications_prescribed: Optional[List[Dict[str, Any]]] = None
    follow_up_instructions: Optional[str] = None
    follow_up_date: Optional[datetime] = None


class ICD10SuggestionRequest(BaseModel):
    symptoms: List[str]
    chief_complaint: Optional[str] = None
    history: Optional[str] = None
    exam_findings: Optional[str] = None


class PrescriptionSuggestionRequest(BaseModel):
    diagnosis: str
    icd10_code: Optional[str] = None
    patient_id: str
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None


def verify_doctor_patient_access(doctor_id: str, patient_id: str, db: Session) -> bool:
    """Verify doctor has an active connection with patient"""
    result = db.execute(
        text("""
            SELECT id FROM patient_doctor_connections
            WHERE doctor_id = :doctor_id 
            AND patient_id = :patient_id 
            AND status = 'active'
        """),
        {"doctor_id": doctor_id, "patient_id": patient_id}
    )
    return result.fetchone() is not None


def audit_log(db: Session, user_id: str, action: str, resource_type: str, 
              resource_id: str, details: Optional[Dict[str, Any]] = None):
    """HIPAA-compliant audit logging"""
    try:
        db.execute(
            text("""
                INSERT INTO hipaa_audit_logs (id, user_id, action, resource_type, resource_id, details, created_at)
                VALUES (:id, :user_id, :action, :resource_type, :resource_id, :details, NOW())
            """),
            {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": json.dumps(details if details else {})
            }
        )
        db.commit()
    except Exception as e:
        logger.warning(f"Audit log failed: {e}")


@router.post("/soap-notes", response_model=Dict[str, Any])
def create_soap_note(
    note: SOAPNoteCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new SOAP note for a patient encounter"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can create SOAP notes")
    
    if not verify_doctor_patient_access(current_user.id, note.patient_id, db):
        raise HTTPException(status_code=403, detail="No active connection with this patient")
    
    note_id = str(uuid.uuid4())
    
    try:
        db.execute(
            text("""
                INSERT INTO soap_notes (
                    id, patient_id, doctor_id, encounter_date,
                    chief_complaint, subjective, history_present_illness, review_of_systems,
                    objective, vital_signs, physical_exam, lab_results,
                    assessment, primary_diagnosis, primary_icd10, secondary_diagnoses, differential_diagnoses,
                    plan, medications_prescribed, procedures_ordered, referrals,
                    patient_education, follow_up_instructions, follow_up_date,
                    linked_appointment_id, status, created_at, updated_at
                ) VALUES (
                    :id, :patient_id, :doctor_id, :encounter_date,
                    :chief_complaint, :subjective, :history_present_illness, :review_of_systems,
                    :objective, :vital_signs, :physical_exam, :lab_results,
                    :assessment, :primary_diagnosis, :primary_icd10, :secondary_diagnoses, :differential_diagnoses,
                    :plan, :medications_prescribed, :procedures_ordered, :referrals,
                    :patient_education, :follow_up_instructions, :follow_up_date,
                    :linked_appointment_id, 'draft', NOW(), NOW()
                )
            """),
            {
                "id": note_id,
                "patient_id": note.patient_id,
                "doctor_id": current_user.id,
                "encounter_date": note.encounter_date or datetime.utcnow(),
                "chief_complaint": note.chief_complaint,
                "subjective": note.subjective,
                "history_present_illness": note.history_present_illness,
                "review_of_systems": json.dumps(note.review_of_systems) if note.review_of_systems else None,
                "objective": note.objective,
                "vital_signs": json.dumps(note.vital_signs) if note.vital_signs else None,
                "physical_exam": note.physical_exam,
                "lab_results": json.dumps(note.lab_results) if note.lab_results else None,
                "assessment": note.assessment,
                "primary_diagnosis": note.primary_diagnosis,
                "primary_icd10": note.primary_icd10,
                "secondary_diagnoses": json.dumps(note.secondary_diagnoses) if note.secondary_diagnoses else None,
                "differential_diagnoses": json.dumps(note.differential_diagnoses) if note.differential_diagnoses else None,
                "plan": note.plan,
                "medications_prescribed": json.dumps(note.medications_prescribed) if note.medications_prescribed else None,
                "procedures_ordered": json.dumps(note.procedures_ordered) if note.procedures_ordered else None,
                "referrals": json.dumps(note.referrals) if note.referrals else None,
                "patient_education": note.patient_education,
                "follow_up_instructions": note.follow_up_instructions,
                "follow_up_date": note.follow_up_date,
                "linked_appointment_id": note.linked_appointment_id,
            }
        )
        db.commit()
        
        audit_log(db, current_user.id, "create", "soap_note", note_id, 
                  {"patient_id": note.patient_id, "chief_complaint": note.chief_complaint})
        
        return {
            "id": note_id,
            "status": "draft",
            "message": "SOAP note created successfully"
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to create SOAP note: {e}")
        raise HTTPException(status_code=500, detail="Failed to create SOAP note")


@router.get("/soap-notes/patient/{patient_id}")
def get_patient_soap_notes(
    patient_id: str,
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all SOAP notes for a patient"""
    if current_user.role == "patient" and current_user.id != patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if current_user.role == "doctor":
        if not verify_doctor_patient_access(current_user.id, patient_id, db):
            raise HTTPException(status_code=403, detail="No active connection with this patient")
    
    result = db.execute(
        text("""
            SELECT id, patient_id, doctor_id, encounter_date, chief_complaint,
                   primary_diagnosis, primary_icd10, status, signed_at, created_at
            FROM soap_notes
            WHERE patient_id = :patient_id
            ORDER BY encounter_date DESC
            LIMIT :limit
        """),
        {"patient_id": patient_id, "limit": limit}
    )
    
    notes = []
    for row in result.fetchall():
        notes.append({
            "id": row[0],
            "patient_id": row[1],
            "doctor_id": row[2],
            "encounter_date": row[3].isoformat() if row[3] else None,
            "chief_complaint": row[4],
            "primary_diagnosis": row[5],
            "primary_icd10": row[6],
            "status": row[7],
            "signed_at": row[8].isoformat() if row[8] else None,
            "created_at": row[9].isoformat() if row[9] else None,
        })
    
    audit_log(db, current_user.id, "view_list", "soap_notes", patient_id)
    return {"notes": notes, "count": len(notes)}


@router.get("/soap-notes/{note_id}")
def get_soap_note(
    note_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific SOAP note by ID"""
    result = db.execute(
        text("""
            SELECT id, patient_id, doctor_id, encounter_date,
                   chief_complaint, subjective, history_present_illness, review_of_systems,
                   objective, vital_signs, physical_exam, lab_results,
                   assessment, primary_diagnosis, primary_icd10, secondary_diagnoses, differential_diagnoses,
                   plan, medications_prescribed, procedures_ordered, referrals,
                   patient_education, follow_up_instructions, follow_up_date,
                   linked_appointment_id, linked_prescription_ids, status, signed_at, signed_by,
                   ai_suggestions_used, ai_icd10_suggestions, created_at, updated_at
            FROM soap_notes
            WHERE id = :note_id
        """),
        {"note_id": note_id}
    )
    
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="SOAP note not found")
    
    patient_id = row[1]
    if current_user.role == "patient" and current_user.id != patient_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if current_user.role == "doctor":
        if current_user.id != row[2] and not verify_doctor_patient_access(current_user.id, patient_id, db):
            raise HTTPException(status_code=403, detail="Access denied")
    
    def safe_json_load(val):
        if val is None:
            return None
        if isinstance(val, (dict, list)):
            return val
        try:
            return json.loads(val)
        except:
            return val
    
    note = {
        "id": row[0],
        "patient_id": row[1],
        "doctor_id": row[2],
        "encounter_date": row[3].isoformat() if row[3] else None,
        "chief_complaint": row[4],
        "subjective": row[5],
        "history_present_illness": row[6],
        "review_of_systems": safe_json_load(row[7]),
        "objective": row[8],
        "vital_signs": safe_json_load(row[9]),
        "physical_exam": row[10],
        "lab_results": safe_json_load(row[11]),
        "assessment": row[12],
        "primary_diagnosis": row[13],
        "primary_icd10": row[14],
        "secondary_diagnoses": safe_json_load(row[15]),
        "differential_diagnoses": safe_json_load(row[16]),
        "plan": row[17],
        "medications_prescribed": safe_json_load(row[18]),
        "procedures_ordered": safe_json_load(row[19]),
        "referrals": safe_json_load(row[20]),
        "patient_education": row[21],
        "follow_up_instructions": row[22],
        "follow_up_date": row[23].isoformat() if row[23] else None,
        "linked_appointment_id": row[24],
        "linked_prescription_ids": safe_json_load(row[25]),
        "status": row[26],
        "signed_at": row[27].isoformat() if row[27] else None,
        "signed_by": row[28],
        "ai_suggestions_used": row[29],
        "ai_icd10_suggestions": safe_json_load(row[30]),
        "created_at": row[31].isoformat() if row[31] else None,
        "updated_at": row[32].isoformat() if row[32] else None,
    }
    
    audit_log(db, current_user.id, "view", "soap_note", note_id)
    return note


@router.put("/soap-notes/{note_id}")
def update_soap_note(
    note_id: str,
    update: SOAPNoteUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a SOAP note (only draft notes can be updated)"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can update SOAP notes")
    
    result = db.execute(
        text("SELECT doctor_id, status FROM soap_notes WHERE id = :id"),
        {"id": note_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="SOAP note not found")
    
    if row[0] != current_user.id:
        raise HTTPException(status_code=403, detail="Only the author can update this note")
    
    if row[1] == "signed":
        raise HTTPException(status_code=400, detail="Signed notes cannot be modified")
    
    update_fields = []
    params = {"id": note_id}
    
    for field, value in update.model_dump(exclude_unset=True).items():
        if value is not None:
            if isinstance(value, (dict, list)):
                params[field] = json.dumps(value)
            else:
                params[field] = value
            update_fields.append(f"{field} = :{field}")
    
    if not update_fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    update_fields.append("updated_at = NOW()")
    
    db.execute(
        text(f"UPDATE soap_notes SET {', '.join(update_fields)} WHERE id = :id"),
        params
    )
    db.commit()
    
    audit_log(db, current_user.id, "update", "soap_note", note_id, 
              {"updated_fields": list(update.model_dump(exclude_unset=True).keys())})
    
    return {"message": "SOAP note updated successfully"}


@router.post("/soap-notes/{note_id}/sign")
def sign_soap_note(
    note_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Sign and finalize a SOAP note"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can sign SOAP notes")
    
    result = db.execute(
        text("SELECT doctor_id, status FROM soap_notes WHERE id = :id"),
        {"id": note_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="SOAP note not found")
    
    if row[0] != current_user.id:
        raise HTTPException(status_code=403, detail="Only the author can sign this note")
    
    if row[1] == "signed":
        raise HTTPException(status_code=400, detail="Note is already signed")
    
    db.execute(
        text("""
            UPDATE soap_notes 
            SET status = 'signed', signed_at = NOW(), signed_by = :doctor_id, updated_at = NOW()
            WHERE id = :id
        """),
        {"id": note_id, "doctor_id": current_user.id}
    )
    db.commit()
    
    audit_log(db, current_user.id, "sign", "soap_note", note_id)
    return {"message": "SOAP note signed successfully", "status": "signed"}


@router.post("/icd10/suggest")
async def suggest_icd10_codes(
    request: ICD10SuggestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI-powered ICD-10 code suggestions based on clinical presentation"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can request ICD-10 suggestions")
    
    try:
        import openai
        client = openai.OpenAI()
        
        prompt = f"""You are a medical coding expert. Based on the following clinical presentation, suggest the most appropriate ICD-10-CM diagnosis codes.

Chief Complaint: {request.chief_complaint or 'Not specified'}

Symptoms: {', '.join(request.symptoms)}

History: {request.history or 'Not provided'}

Exam Findings: {request.exam_findings or 'Not provided'}

Provide 3-5 ICD-10-CM codes with their descriptions, ordered by likelihood. Format as JSON array:
[
  {{"code": "X00.0", "description": "Diagnosis description", "confidence": 0.95, "specificity": "billable"}},
  ...
]

Important: Only suggest valid, billable ICD-10-CM codes. Include the full code with all applicable digits."""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical coding AI that provides accurate ICD-10-CM code suggestions. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        suggestions = json.loads(content)
        
        if isinstance(suggestions, dict) and "codes" in suggestions:
            suggestions = suggestions["codes"]
        elif isinstance(suggestions, dict) and not isinstance(suggestions, list):
            suggestions = [suggestions]
        
        audit_log(db, current_user.id, "ai_suggest", "icd10_codes", "system",
                  {"symptoms_count": len(request.symptoms), "suggestions_count": len(suggestions)})
        
        return {
            "suggestions": suggestions,
            "disclaimer": "AI-generated suggestions for clinical review only. Verify accuracy before use."
        }
    except Exception as e:
        logger.error(f"ICD-10 suggestion error: {e}")
        return {
            "suggestions": get_common_icd10_fallback(request.symptoms),
            "disclaimer": "Fallback suggestions provided. AI service unavailable."
        }


@router.post("/prescription/suggest")
async def suggest_prescription(
    request: PrescriptionSuggestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """AI-powered prescription suggestions based on diagnosis"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can request prescription suggestions")
    
    if not verify_doctor_patient_access(current_user.id, request.patient_id, db):
        raise HTTPException(status_code=403, detail="No active connection with this patient")
    
    try:
        import openai
        client = openai.OpenAI()
        
        prompt = f"""You are a clinical pharmacology expert. Based on the following diagnosis, suggest appropriate medication options.

Diagnosis: {request.diagnosis}
ICD-10 Code: {request.icd10_code or 'Not specified'}

Current Medications: {', '.join(request.current_medications) if request.current_medications else 'None reported'}
Known Allergies: {', '.join(request.allergies) if request.allergies else 'None reported'}
Contraindications: {', '.join(request.contraindications) if request.contraindications else 'None specified'}

IMPORTANT: This is for an immunocompromised patient. Consider drug interactions and immunosuppressive effects.

Provide 2-4 medication options with dosing. Format as JSON:
{{
  "medications": [
    {{
      "name": "Drug Name",
      "generic_name": "generic name",
      "drug_class": "class",
      "suggested_dosage": "dosage",
      "frequency": "frequency",
      "duration_days": number or null for continuous,
      "route": "oral/IV/etc",
      "rationale": "why this medication",
      "warnings": ["list of warnings for immunocompromised"],
      "monitoring": ["what to monitor"]
    }}
  ],
  "general_recommendations": "string",
  "interaction_warnings": ["any interaction concerns"]
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a clinical pharmacology AI specializing in immunocompromised patients. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        suggestions = json.loads(content)
        
        audit_log(db, current_user.id, "ai_suggest", "prescription", request.patient_id,
                  {"diagnosis": request.diagnosis})
        
        return {
            **suggestions,
            "disclaimer": "AI-generated suggestions for clinical review only. Doctor must verify appropriateness."
        }
    except Exception as e:
        logger.error(f"Prescription suggestion error: {e}")
        return {
            "medications": [],
            "general_recommendations": "AI service unavailable. Please consult formulary.",
            "disclaimer": "AI service unavailable. Please use clinical judgment."
        }


def get_common_icd10_fallback(symptoms: List[str]) -> List[Dict[str, Any]]:
    """Fallback ICD-10 suggestions based on common symptom patterns"""
    symptom_lower = [s.lower() for s in symptoms]
    suggestions = []
    
    symptom_map = {
        "fever": {"code": "R50.9", "description": "Fever, unspecified"},
        "cough": {"code": "R05.9", "description": "Cough, unspecified"},
        "headache": {"code": "R51.9", "description": "Headache, unspecified"},
        "fatigue": {"code": "R53.83", "description": "Other fatigue"},
        "pain": {"code": "R52", "description": "Pain, unspecified"},
        "nausea": {"code": "R11.0", "description": "Nausea"},
        "vomiting": {"code": "R11.10", "description": "Vomiting, unspecified"},
        "diarrhea": {"code": "R19.7", "description": "Diarrhea, unspecified"},
        "shortness of breath": {"code": "R06.02", "description": "Shortness of breath"},
        "chest pain": {"code": "R07.9", "description": "Chest pain, unspecified"},
    }
    
    for symptom in symptom_lower:
        for key, value in symptom_map.items():
            if key in symptom:
                suggestions.append({**value, "confidence": 0.7, "specificity": "billable"})
                break
    
    return suggestions[:5] if suggestions else [
        {"code": "R69", "description": "Illness, unspecified", "confidence": 0.5, "specificity": "billable"}
    ]


@router.post("/soap-notes/{note_id}/link-prescription")
def link_prescription_to_soap(
    note_id: str,
    prescription_id: str = Query(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Link a prescription to a SOAP note"""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can link prescriptions")
    
    result = db.execute(
        text("SELECT doctor_id, linked_prescription_ids FROM soap_notes WHERE id = :id"),
        {"id": note_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="SOAP note not found")
    
    if row[0] != current_user.id:
        raise HTTPException(status_code=403, detail="Only the author can modify this note")
    
    current_prescriptions = json.loads(row[1]) if row[1] else []
    if prescription_id not in current_prescriptions:
        current_prescriptions.append(prescription_id)
    
    db.execute(
        text("""
            UPDATE soap_notes 
            SET linked_prescription_ids = :prescriptions, updated_at = NOW()
            WHERE id = :id
        """),
        {"id": note_id, "prescriptions": json.dumps(current_prescriptions)}
    )
    db.commit()
    
    audit_log(db, current_user.id, "link_prescription", "soap_note", note_id,
              {"prescription_id": prescription_id})
    
    return {"message": "Prescription linked successfully"}
