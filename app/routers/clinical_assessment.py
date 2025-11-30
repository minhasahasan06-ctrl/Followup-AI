"""
Clinical Assessment Aggregation Router
HIPAA-compliant endpoint for aggregating patient data for AI clinical assessment
Uses doctor_patient_assignments table for authorization (primary)
Optionally checks PatientSharingLink for additional data sharing preferences
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, or_, text
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import uuid
import logging
import os

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.patient_sharing import PatientSharingLink, SharingAccessLog, SharingStatus
from app.models.medication_side_effects import MedicationTimeline
from app.schemas.clinical_assessment_schemas import (
    ConsentedPatient,
    MedicalFile,
    HealthAlert,
    MLInferenceResult,
    CurrentMedication,
    FollowupSummary,
    PatientDataAggregation,
    ClinicalAssessmentRequest,
    ClinicalAssessmentWithContextRequest,
    ClinicalAssessmentResult,
    DiagnosisSuggestion,
    AuditLogEntry
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/clinical-assessment", tags=["Clinical Assessment"])

HIPAA_DISCLAIMER = "This is clinical decision support only. Not a diagnosis or substitute for professional medical judgment."


def log_data_access(
    db: Session,
    doctor_id: str,
    patient_id: str,
    action: str,
    resource_type: str,
    sharing_link_id: Optional[str] = None,
    assignment_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    details: Optional[dict] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
) -> str:
    """
    Log data access for HIPAA compliance audit trail.
    Supports both PatientSharingLink and doctor_patient_assignments authorization sources.
    """
    audit_id = str(uuid.uuid4())
    
    log_details = details.copy() if details else {}
    if assignment_id and not sharing_link_id:
        log_details["authorization_source"] = "doctor_patient_assignment"
        log_details["assignment_id"] = assignment_id
    elif sharing_link_id:
        log_details["authorization_source"] = "patient_sharing_link"
    
    if sharing_link_id:
        try:
            log_entry = SharingAccessLog(
                id=audit_id,
                sharing_link_id=sharing_link_id,
                doctor_id=doctor_id,
                patient_id=patient_id,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=log_details,
                ip_address=ip_address,
                user_agent=user_agent
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to log to SharingAccessLog: {e}")
            db.rollback()
    else:
        try:
            audit_query = text("""
                INSERT INTO hipaa_audit_logs (id, doctor_id, patient_id, action, 
                    resource_type, resource_id, details, ip_address, user_agent, created_at)
                VALUES (:id, :doctor_id, :patient_id, :action, 
                    :resource_type, :resource_id, :details::jsonb, :ip_address, :user_agent, NOW())
            """)
            import json
            db.execute(audit_query, {
                "id": audit_id,
                "doctor_id": doctor_id,
                "patient_id": patient_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "details": json.dumps(log_details),
                "ip_address": ip_address,
                "user_agent": user_agent
            })
            db.commit()
        except Exception as e:
            logger.warning(f"Failed to log to hipaa_audit_logs: {e}, falling back to logger")
            db.rollback()
            logger.info(f"HIPAA_AUDIT: {action} by doctor {doctor_id} on patient {patient_id} - {log_details}")
    
    return audit_id


def verify_doctor_patient_assignment(
    db: Session,
    doctor_id: str,
    patient_id: str
) -> Tuple[bool, Optional[dict]]:
    """
    Verify doctor has active assignment to patient via doctor_patient_assignments table.
    This is the PRIMARY authorization check per HIPAA compliance requirements.
    Returns: (is_authorized, assignment_info)
    """
    try:
        assignment_query = text("""
            SELECT id, status, patient_consented, access_scope, 
                   is_primary_care_provider, assignment_source
            FROM doctor_patient_assignments 
            WHERE doctor_id = :doctor_id 
            AND patient_id = :patient_id 
            AND status = 'active'
            LIMIT 1
        """)
        
        result = db.execute(assignment_query, {
            "doctor_id": doctor_id,
            "patient_id": patient_id
        })
        row = result.first()
        
        if row:
            return True, {
                "assignment_id": row[0],
                "status": row[1],
                "patient_consented": row[2],
                "access_scope": row[3] or "full",
                "is_primary_care_provider": row[4],
                "assignment_source": row[5]
            }
        return False, None
    except Exception as e:
        logger.warning(f"Error checking doctor_patient_assignments: {e}")
        return False, None


def get_sharing_preferences(
    db: Session,
    doctor_id: str,
    patient_id: str
) -> Optional[PatientSharingLink]:
    """Get optional sharing preferences from PatientSharingLink if available."""
    sharing_link = db.query(PatientSharingLink).filter(
        and_(
            PatientSharingLink.doctor_id == doctor_id,
            PatientSharingLink.patient_id == patient_id,
            PatientSharingLink.status == SharingStatus.ACTIVE.value
        )
    ).first()
    
    if sharing_link:
        if sharing_link.expires_at and datetime.utcnow() > sharing_link.expires_at:
            sharing_link.status = SharingStatus.EXPIRED.value
            db.commit()
            return None
    
    return sharing_link


def verify_patient_consent(
    db: Session,
    doctor_id: str,
    patient_id: str
) -> Tuple[bool, Optional[dict], Optional[PatientSharingLink]]:
    """
    Verify doctor has authorization to access patient data.
    Uses doctor_patient_assignments as primary check (per HIPAA requirements).
    Also retrieves optional sharing preferences if available.
    Returns: (is_authorized, assignment_info, sharing_link)
    """
    is_authorized, assignment_info = verify_doctor_patient_assignment(db, doctor_id, patient_id)
    sharing_link = get_sharing_preferences(db, doctor_id, patient_id)
    
    if not is_authorized:
        if sharing_link:
            is_authorized = True
            assignment_info = {
                "assignment_id": sharing_link.id,
                "status": sharing_link.status,
                "patient_consented": True,
                "access_scope": sharing_link.access_level,
                "is_primary_care_provider": False,
                "assignment_source": "sharing_link"
            }
    
    return is_authorized, assignment_info, sharing_link


@router.get("/patients", response_model=List[ConsentedPatient])
async def get_consented_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Get list of patients with active consent for clinical assessment.
    Uses doctor_patient_assignments as primary source (HIPAA requirement).
    Supplements with PatientSharingLink for detailed sharing preferences.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient data")
    
    doctor_id = current_user.id
    consented_patients = []
    seen_patient_ids = set()
    
    try:
        assignments_query = text("""
            SELECT dpa.id, dpa.patient_id, dpa.status, dpa.patient_consented,
                   dpa.consented_at, dpa.access_scope, dpa.is_primary_care_provider,
                   u.email, u.first_name, u.last_name
            FROM doctor_patient_assignments dpa
            JOIN users u ON u.id = dpa.patient_id
            WHERE dpa.doctor_id = :doctor_id 
            AND dpa.status = 'active'
            ORDER BY dpa.is_primary_care_provider DESC, dpa.created_at DESC
        """)
        
        result = db.execute(assignments_query, {"doctor_id": doctor_id})
        
        for row in result:
            patient_id = row[1]
            if patient_id in seen_patient_ids:
                continue
            seen_patient_ids.add(patient_id)
            
            patient_name = f"{row[8] or ''} {row[9] or ''}".strip() or row[7]
            
            sharing_link = get_sharing_preferences(db, doctor_id, patient_id)
            
            if sharing_link:
                consented_patients.append(ConsentedPatient(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    patient_email=row[7],
                    sharing_link_id=sharing_link.id,
                    consent_status=row[2],
                    access_level=row[5] or "full",
                    consent_given_at=row[4] or sharing_link.consent_given_at,
                    share_vitals=sharing_link.share_vitals,
                    share_symptoms=sharing_link.share_symptoms,
                    share_medications=sharing_link.share_medications,
                    share_activities=sharing_link.share_activities,
                    share_mental_health=sharing_link.share_mental_health,
                    share_video_exams=sharing_link.share_video_exams,
                    share_audio_exams=sharing_link.share_audio_exams
                ))
            else:
                consented_patients.append(ConsentedPatient(
                    patient_id=patient_id,
                    patient_name=patient_name,
                    patient_email=row[7],
                    sharing_link_id=row[0],
                    consent_status=row[2],
                    access_level=row[5] or "full",
                    consent_given_at=row[4],
                    share_vitals=True,
                    share_symptoms=True,
                    share_medications=True,
                    share_activities=True,
                    share_mental_health=True,
                    share_video_exams=True,
                    share_audio_exams=True
                ))
    except Exception as e:
        logger.warning(f"Error querying doctor_patient_assignments: {e}")
    
    active_links = db.query(PatientSharingLink).filter(
        and_(
            PatientSharingLink.doctor_id == doctor_id,
            PatientSharingLink.status == SharingStatus.ACTIVE.value
        )
    ).all()
    
    for link in active_links:
        if link.patient_id in seen_patient_ids:
            continue
            
        if link.expires_at and datetime.utcnow() > link.expires_at:
            link.status = SharingStatus.EXPIRED.value
            db.commit()
            continue
        
        seen_patient_ids.add(link.patient_id)
        patient = db.query(User).filter(User.id == link.patient_id).first()
        if patient:
            consented_patients.append(ConsentedPatient(
                patient_id=patient.id,
                patient_name=f"{patient.first_name or ''} {patient.last_name or ''}".strip() or patient.email,
                patient_email=patient.email,
                sharing_link_id=link.id,
                consent_status=link.status,
                access_level=link.access_level,
                consent_given_at=link.consent_given_at,
                share_vitals=link.share_vitals,
                share_symptoms=link.share_symptoms,
                share_medications=link.share_medications,
                share_activities=link.share_activities,
                share_mental_health=link.share_mental_health,
                share_video_exams=link.share_video_exams,
                share_audio_exams=link.share_audio_exams
            ))
    
    return consented_patients


@router.get("/patient/{patient_id}/data", response_model=PatientDataAggregation)
async def get_patient_data_aggregation(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Aggregate all available patient data for clinical assessment.
    Verifies consent via doctor_patient_assignments and logs access for HIPAA compliance.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can access patient data")
    
    doctor_id = current_user.id
    
    is_authorized, assignment_info, sharing_link = verify_patient_consent(db, doctor_id, patient_id)
    if not is_authorized:
        raise HTTPException(
            status_code=403,
            detail="No active assignment or consent from patient for data access"
        )
    
    ip_address = request.client.host if request and request.client else None
    user_agent = request.headers.get("user-agent") if request else None
    
    assignment_id = assignment_info.get("assignment_id", "") if assignment_info else ""
    access_scope = assignment_info.get("access_scope", "full") if assignment_info else "full"
    
    audit_id = log_data_access(
        db=db,
        doctor_id=doctor_id,
        patient_id=patient_id,
        action="aggregate_patient_data",
        resource_type="clinical_assessment",
        sharing_link_id=sharing_link.id if sharing_link else None,
        assignment_id=assignment_id if not sharing_link else None,
        details={"access_level": access_scope, "source": assignment_info.get("assignment_source") if assignment_info else "unknown"},
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    patient = db.query(User).filter(User.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    patient_name = f"{patient.first_name or ''} {patient.last_name or ''}".strip() or patient.email
    
    medical_files = []
    health_alerts = []
    ml_inference_results = []
    current_medications = []
    last_followup = None
    
    is_full_or_limited = access_scope in ("full", "limited")
    is_emergency = access_scope == "emergency_only"
    
    if sharing_link:
        share_medications = sharing_link.share_medications
        share_vitals = sharing_link.share_vitals
        share_symptoms = sharing_link.share_symptoms
    else:
        share_medications = is_full_or_limited or is_emergency
        share_vitals = is_full_or_limited or is_emergency
        share_symptoms = is_full_or_limited
    
    if share_medications:
        meds = db.query(MedicationTimeline).filter(
            and_(
                MedicationTimeline.patient_id == patient_id,
                MedicationTimeline.is_active == True
            )
        ).order_by(desc(MedicationTimeline.started_at)).all()
        
        for med in meds:
            current_medications.append(CurrentMedication(
                id=med.id,
                medication_name=med.medication_name,
                generic_name=med.generic_name,
                drug_class=med.drug_class,
                dosage=med.dosage,
                frequency=med.frequency,
                route=med.route,
                started_at=med.started_at,
                prescribed_by=med.prescribed_by,
                prescription_reason=med.prescription_reason,
                is_active=med.is_active
            ))
        
        log_data_access(
            db=db,
            doctor_id=doctor_id,
            patient_id=patient_id,
            action="read_medications",
            resource_type="medications",
            sharing_link_id=sharing_link.id if sharing_link else None,
            assignment_id=assignment_id if not sharing_link else None,
            details={"count": len(current_medications)},
            ip_address=ip_address
        )
    
    if share_vitals or share_symptoms:
        try:
            from sqlalchemy import text
            
            alerts_query = text("""
                SELECT id, alert_type, alert_category, severity, priority, title, message, 
                       status, created_at, contributing_metrics
                FROM ai_health_alerts 
                WHERE patient_id = :patient_id 
                AND status = 'active'
                AND created_at > :since_date
                ORDER BY priority DESC, created_at DESC
                LIMIT 20
            """)
            
            since_date = datetime.utcnow() - timedelta(days=30)
            result = db.execute(alerts_query, {"patient_id": patient_id, "since_date": since_date})
            
            for row in result:
                health_alerts.append(HealthAlert(
                    id=str(row[0]),
                    alert_type=row[1] or "unknown",
                    alert_category=row[2] or "general",
                    severity=row[3] or "low",
                    priority=row[4] or 0,
                    title=row[5] or "Health Alert",
                    message=row[6] or "",
                    status=row[7] or "active",
                    created_at=row[8] or datetime.utcnow(),
                    contributing_metrics=row[9] if row[9] else None
                ))
            
            log_data_access(
                db=db,
                doctor_id=doctor_id,
                patient_id=patient_id,
                action="read_health_alerts",
                resource_type="health_alerts",
                sharing_link_id=sharing_link.id if sharing_link else None,
                assignment_id=assignment_id if not sharing_link else None,
                details={"count": len(health_alerts)},
                ip_address=ip_address
            )
        except Exception as e:
            logger.warning(f"Failed to load health alerts: {e}")
    
    try:
        from sqlalchemy import text
        
        ml_query = text("""
            SELECT model_name, prediction_type, risk_score, risk_level, confidence, 
                   details, computed_at
            FROM ml_predictions 
            WHERE patient_id = :patient_id 
            AND computed_at > :since_date
            ORDER BY computed_at DESC
            LIMIT 10
        """)
        
        since_date = datetime.utcnow() - timedelta(days=7)
        result = db.execute(ml_query, {"patient_id": patient_id, "since_date": since_date})
        
        for row in result:
            ml_inference_results.append(MLInferenceResult(
                model_name=row[0] or "unknown",
                prediction_type=row[1] or "risk_assessment",
                risk_score=float(row[2]) if row[2] else None,
                risk_level=row[3],
                confidence=float(row[4]) if row[4] else None,
                details=row[5] if row[5] else None,
                computed_at=row[6] or datetime.utcnow()
            ))
        
        log_data_access(
            db=db,
            doctor_id=doctor_id,
            patient_id=patient_id,
            action="read_ml_inference",
            resource_type="ml_predictions",
            sharing_link_id=sharing_link.id if sharing_link else None,
            assignment_id=assignment_id if not sharing_link else None,
            details={"count": len(ml_inference_results)},
            ip_address=ip_address
        )
    except Exception as e:
        logger.warning(f"Failed to load ML inference results: {e}")
    
    try:
        from sqlalchemy import text
        
        followup_query = text("""
            SELECT recorded_at, vital_signs, symptom_summary, pain_level,
                   mental_health_scores, video_exam_summary, audio_exam_summary,
                   overall_status, risk_indicators
            FROM daily_followups 
            WHERE patient_id = :patient_id 
            ORDER BY recorded_at DESC
            LIMIT 1
        """)
        
        result = db.execute(followup_query, {"patient_id": patient_id})
        row = result.first()
        
        if row:
            last_followup = FollowupSummary(
                summary_date=row[0] or datetime.utcnow(),
                vital_signs=row[1] if row[1] else None,
                symptom_summary=row[2],
                pain_level=float(row[3]) if row[3] else None,
                mental_health_score=row[4] if row[4] else None,
                video_exam_findings=row[5] if row[5] else None,
                audio_exam_findings=row[6] if row[6] else None,
                overall_status=row[7],
                risk_indicators=row[8] if row[8] else None
            )
            
            log_data_access(
                db=db,
                doctor_id=doctor_id,
                patient_id=patient_id,
                action="read_followup_summary",
                resource_type="daily_followup",
                sharing_link_id=sharing_link.id if sharing_link else None,
                assignment_id=assignment_id if not sharing_link else None,
                ip_address=ip_address
            )
    except Exception as e:
        logger.warning(f"Failed to load followup summary: {e}")
    
    if sharing_link:
        consent_info = ConsentedPatient(
            patient_id=patient.id,
            patient_name=patient_name,
            patient_email=patient.email,
            sharing_link_id=sharing_link.id,
            consent_status=sharing_link.status,
            access_level=sharing_link.access_level,
            consent_given_at=sharing_link.consent_given_at,
            share_vitals=sharing_link.share_vitals,
            share_symptoms=sharing_link.share_symptoms,
            share_medications=sharing_link.share_medications,
            share_activities=sharing_link.share_activities,
            share_mental_health=sharing_link.share_mental_health,
            share_video_exams=sharing_link.share_video_exams,
            share_audio_exams=sharing_link.share_audio_exams
        )
    else:
        is_full_access = access_scope == "full"
        is_limited_access = access_scope == "limited"
        is_emergency_only = access_scope == "emergency_only"
        
        consent_info = ConsentedPatient(
            patient_id=patient.id,
            patient_name=patient_name,
            patient_email=patient.email,
            sharing_link_id=assignment_id,
            consent_status=assignment_info.get("status", "active") if assignment_info else "active",
            access_level=access_scope,
            consent_given_at=None,
            share_vitals=is_full_access or is_limited_access or is_emergency_only,
            share_symptoms=is_full_access or is_limited_access,
            share_medications=is_full_access or is_limited_access or is_emergency_only,
            share_activities=is_full_access,
            share_mental_health=is_full_access,
            share_video_exams=is_full_access,
            share_audio_exams=is_full_access
        )
    
    return PatientDataAggregation(
        patient_id=patient_id,
        patient_name=patient_name,
        patient_age=getattr(patient, 'age', None),
        patient_sex=getattr(patient, 'biological_sex', None),
        consent_info=consent_info,
        medical_files=medical_files,
        health_alerts=health_alerts,
        ml_inference_results=ml_inference_results,
        current_medications=current_medications,
        last_followup=last_followup,
        aggregated_at=datetime.utcnow(),
        audit_id=audit_id
    )


@router.post("/analyze", response_model=ClinicalAssessmentResult)
async def analyze_with_patient_context(
    request_data: ClinicalAssessmentWithContextRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: Request = None
):
    """
    Perform comprehensive AI clinical assessment with full patient context.
    Uses GPT-4 to analyze symptoms along with patient's medical history,
    current medications, health alerts, and ML inference results.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can perform clinical assessment")
    
    doctor_id = current_user.id
    patient_id = request_data.patient_id
    
    is_authorized, assignment_info, sharing_link = verify_patient_consent(db, doctor_id, patient_id)
    if not is_authorized:
        raise HTTPException(
            status_code=403,
            detail="No active assignment or consent from patient for clinical assessment"
        )
    
    ip_address = request.client.host if request and request.client else None
    assignment_id = assignment_info.get("assignment_id", "") if assignment_info else ""
    
    audit_id = log_data_access(
        db=db,
        doctor_id=doctor_id,
        patient_id=patient_id,
        action="ai_clinical_assessment",
        resource_type="diagnosis_support",
        sharing_link_id=sharing_link.id if sharing_link else None,
        assignment_id=assignment_id if not sharing_link else None,
        details={
            "symptom_count": len(request_data.symptoms),
            "has_patient_data": request_data.patient_data is not None,
            "authorization_source": assignment_info.get("assignment_source") if assignment_info else "unknown"
        },
        ip_address=ip_address
    )
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        context_parts = []
        
        if request_data.patient_data:
            pd = request_data.patient_data
            
            context_parts.append(f"Patient: {pd.patient_name}")
            if pd.patient_age:
                context_parts.append(f"Age: {pd.patient_age}")
            if pd.patient_sex:
                context_parts.append(f"Sex: {pd.patient_sex}")
            
            if pd.current_medications:
                med_list = [f"- {m.medication_name} ({m.dosage}, {m.frequency})" for m in pd.current_medications]
                context_parts.append(f"\nCurrent Medications:\n" + "\n".join(med_list))
            
            if pd.health_alerts:
                alert_list = [f"- [{a.severity.upper()}] {a.title}: {a.message}" for a in pd.health_alerts[:5]]
                context_parts.append(f"\nRecent Health Alerts:\n" + "\n".join(alert_list))
            
            if pd.ml_inference_results:
                ml_list = []
                for m in pd.ml_inference_results[:3]:
                    risk_info = f"Risk: {m.risk_level}" if m.risk_level else f"Score: {m.risk_score:.2f}" if m.risk_score else ""
                    ml_list.append(f"- {m.model_name} ({m.prediction_type}): {risk_info}")
                context_parts.append(f"\nML Risk Predictions:\n" + "\n".join(ml_list))
            
            if pd.last_followup:
                lf = pd.last_followup
                followup_info = [f"Date: {lf.summary_date.strftime('%Y-%m-%d')}"]
                if lf.pain_level is not None:
                    followup_info.append(f"Pain Level: {lf.pain_level}/10")
                if lf.overall_status:
                    followup_info.append(f"Status: {lf.overall_status}")
                if lf.symptom_summary:
                    followup_info.append(f"Symptoms: {lf.symptom_summary}")
                context_parts.append(f"\nLast Follow-up Summary:\n" + "\n".join(followup_info))
        
        symptoms_text = ""
        for s in request_data.symptoms:
            symptoms_text += f"- {s.get('name', 'Unknown')}"
            if s.get('duration'):
                symptoms_text += f" (Duration: {s['duration']})"
            if s.get('severity'):
                symptoms_text += f" [Severity: {s['severity']}]"
            symptoms_text += "\n"
        
        if request_data.additional_notes:
            context_parts.append(f"\nAdditional Notes: {request_data.additional_notes}")
        
        patient_context = "\n".join(context_parts)
        
        system_prompt = """You are an AI clinical decision support assistant for healthcare providers.
Your role is to help analyze patient symptoms and provide differential diagnosis suggestions.

IMPORTANT DISCLAIMERS:
- This is decision SUPPORT only, not a diagnosis
- All suggestions require physician verification
- Never replace professional medical judgment
- Always recommend appropriate testing/consultation

When analyzing, consider:
1. Symptom patterns and combinations
2. Patient's current medications (drug interactions, side effects)
3. Recent health alerts and risk indicators
4. ML-based deterioration predictions if available
5. Last follow-up findings

Provide structured output with:
- Primary suspected condition with probability estimate
- Differential diagnoses ranked by likelihood
- Red flags requiring immediate attention
- Recommended tests or referrals
- Medication considerations (interactions, contraindications)
- Clinical insights based on the patient's health data"""

        user_prompt = f"""Analyze the following clinical presentation:

{patient_context}

PRESENTING SYMPTOMS:
{symptoms_text}

Please provide:
1. PRIMARY DIAGNOSIS: Most likely condition with probability estimate
2. DIFFERENTIAL DIAGNOSES: 2-4 alternative conditions to consider
3. RED FLAGS: Any warning signs requiring immediate attention
4. RECOMMENDED ACTIONS: Tests, referrals, or immediate steps
5. MEDICATION CONSIDERATIONS: Any interactions or concerns with current medications
6. CLINICAL INSIGHTS: Key observations from the patient's health data

Format your response as structured JSON with the following schema:
{{
    "primary_diagnosis": {{
        "condition": "string",
        "probability": 0.0-1.0,
        "matching_symptoms": ["list"],
        "missing_symptoms": ["list"],
        "urgency": "low|moderate|high|emergency",
        "description": "string",
        "recommended_tests": ["list"],
        "differential_diagnosis": ["list"]
    }},
    "differential_diagnoses": [same structure as primary],
    "red_flags": ["list of concerning findings"],
    "recommended_actions": ["list of next steps"],
    "medication_considerations": ["list of drug-related concerns"],
    "clinical_insights": ["list of observations from health data"],
    "patient_context_summary": "brief summary of patient's current health status"
}}"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        
        import json
        result_json = json.loads(response.choices[0].message.content)
        
        primary = result_json.get("primary_diagnosis", {})
        primary_diagnosis = DiagnosisSuggestion(
            condition=primary.get("condition", "Unable to determine"),
            probability=primary.get("probability", 0.5),
            matching_symptoms=primary.get("matching_symptoms", []),
            missing_symptoms=primary.get("missing_symptoms", []),
            urgency=primary.get("urgency", "moderate"),
            description=primary.get("description", ""),
            recommended_tests=primary.get("recommended_tests", []),
            differential_diagnosis=primary.get("differential_diagnosis", [])
        ) if primary else None
        
        differential_diagnoses = []
        for diff in result_json.get("differential_diagnoses", []):
            differential_diagnoses.append(DiagnosisSuggestion(
                condition=diff.get("condition", "Unknown"),
                probability=diff.get("probability", 0.3),
                matching_symptoms=diff.get("matching_symptoms", []),
                missing_symptoms=diff.get("missing_symptoms", []),
                urgency=diff.get("urgency", "moderate"),
                description=diff.get("description", ""),
                recommended_tests=diff.get("recommended_tests", []),
                differential_diagnosis=diff.get("differential_diagnosis", [])
            ))
        
        return ClinicalAssessmentResult(
            primary_diagnosis=primary_diagnosis,
            differential_diagnoses=differential_diagnoses,
            clinical_insights=result_json.get("clinical_insights", []),
            recommended_actions=result_json.get("recommended_actions", []),
            red_flags=result_json.get("red_flags", []),
            patient_context_summary=result_json.get("patient_context_summary"),
            medication_considerations=result_json.get("medication_considerations", []),
            references=[],
            disclaimer=HIPAA_DISCLAIMER,
            analyzed_at=datetime.utcnow(),
            audit_id=audit_id
        )
        
    except Exception as e:
        logger.error(f"AI clinical assessment failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Clinical assessment analysis failed: {str(e)}"
        )


@router.get("/audit-log/{patient_id}", response_model=List[AuditLogEntry])
async def get_access_audit_log(
    patient_id: str,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get HIPAA audit log of data access for a patient.
    Doctors can only see their own access logs.
    """
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can view audit logs")
    
    doctor_id = current_user.id
    since_date = datetime.utcnow() - timedelta(days=days)
    
    logs = db.query(SharingAccessLog).filter(
        and_(
            SharingAccessLog.doctor_id == doctor_id,
            SharingAccessLog.patient_id == patient_id,
            SharingAccessLog.created_at >= since_date
        )
    ).order_by(desc(SharingAccessLog.created_at)).limit(100).all()
    
    return [
        AuditLogEntry(
            id=log.id,
            action=log.action,
            resource_type=log.resource_type,
            resource_id=log.resource_id,
            doctor_id=log.doctor_id,
            patient_id=log.patient_id,
            details=log.details,
            ip_address=log.ip_address,
            created_at=log.created_at
        )
        for log in logs
    ]
