"""
Clinical Assessment Aggregation Schemas
Pydantic models for comprehensive patient data aggregation with consent verification
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ConsentStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ConsentedPatient(BaseModel):
    """Patient with active consent for data sharing"""
    patient_id: str
    patient_name: str
    patient_email: Optional[str] = None
    sharing_link_id: str
    consent_status: str
    access_level: str
    consent_given_at: Optional[datetime] = None
    share_vitals: bool = True
    share_symptoms: bool = True
    share_medications: bool = True
    share_activities: bool = True
    share_mental_health: bool = False
    share_video_exams: bool = False
    share_audio_exams: bool = False


class MedicalFile(BaseModel):
    """Medical file/document record"""
    id: str
    file_name: str
    file_type: str
    file_category: Optional[str] = None
    description: Optional[str] = None
    uploaded_at: datetime
    file_size: Optional[int] = None
    is_shared: bool = True


class HealthAlert(BaseModel):
    """Health alert from deterioration detection"""
    id: str
    alert_type: str
    alert_category: str
    severity: str
    priority: int
    title: str
    message: str
    status: str
    created_at: datetime
    contributing_metrics: Optional[List[Dict[str, Any]]] = None


class MLInferenceResult(BaseModel):
    """ML inference/prediction result"""
    model_name: str
    prediction_type: str
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    confidence: Optional[float] = None
    details: Optional[Dict[str, Any]] = None
    computed_at: datetime


class CurrentMedication(BaseModel):
    """Current active medication"""
    id: int
    medication_name: str
    generic_name: Optional[str] = None
    drug_class: Optional[str] = None
    dosage: str
    frequency: str
    route: Optional[str] = None
    started_at: datetime
    prescribed_by: Optional[str] = None
    prescription_reason: Optional[str] = None
    is_active: bool = True


class FollowupSummary(BaseModel):
    """Last follow-up summary data"""
    summary_date: datetime
    vital_signs: Optional[Dict[str, Any]] = None
    symptom_summary: Optional[str] = None
    pain_level: Optional[float] = None
    mental_health_score: Optional[Dict[str, Any]] = None
    video_exam_findings: Optional[Dict[str, Any]] = None
    audio_exam_findings: Optional[Dict[str, Any]] = None
    overall_status: Optional[str] = None
    risk_indicators: Optional[List[str]] = None


class PatientDataAggregation(BaseModel):
    """Complete patient data aggregation for clinical assessment"""
    patient_id: str
    patient_name: str
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    consent_info: ConsentedPatient
    medical_files: List[MedicalFile] = []
    health_alerts: List[HealthAlert] = []
    ml_inference_results: List[MLInferenceResult] = []
    current_medications: List[CurrentMedication] = []
    last_followup: Optional[FollowupSummary] = None
    aggregated_at: datetime = Field(default_factory=datetime.utcnow)
    audit_id: Optional[str] = None


class ClinicalAssessmentRequest(BaseModel):
    """Request for comprehensive clinical assessment"""
    patient_id: str
    symptoms: List[Dict[str, Any]] = []
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    medical_history: Optional[str] = None
    current_medications: Optional[str] = None
    additional_notes: Optional[str] = None
    include_patient_data: bool = True


class ClinicalAssessmentWithContextRequest(BaseModel):
    """Request for AI analysis with full patient context"""
    patient_id: str
    symptoms: List[Dict[str, Any]] = []
    additional_notes: Optional[str] = None
    patient_data: Optional[PatientDataAggregation] = None


class DiagnosisSuggestion(BaseModel):
    """AI diagnosis suggestion"""
    condition: str
    probability: float
    matching_symptoms: List[str] = []
    missing_symptoms: List[str] = []
    urgency: str = "moderate"
    description: str
    recommended_tests: List[str] = []
    differential_diagnosis: List[str] = []


class ClinicalAssessmentResult(BaseModel):
    """Complete AI clinical assessment result"""
    primary_diagnosis: Optional[DiagnosisSuggestion] = None
    differential_diagnoses: List[DiagnosisSuggestion] = []
    clinical_insights: List[str] = []
    recommended_actions: List[str] = []
    red_flags: List[str] = []
    patient_context_summary: Optional[str] = None
    medication_considerations: List[str] = []
    references: List[str] = []
    disclaimer: str = "This is clinical decision support only. Not a diagnosis or substitute for professional medical judgment."
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    audit_id: Optional[str] = None


class AuditLogEntry(BaseModel):
    """HIPAA audit log entry for data access"""
    id: str
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    doctor_id: str
    patient_id: str
    details: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    created_at: datetime
