"""
Pydantic schemas for Terms, Audit, and Profile management
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class TermsAcceptPayload(BaseModel):
    """Payload for recording terms acceptance"""
    user_id: str
    terms_version: str = "v2025-01"
    accepted_at: Optional[str] = None
    research_consent: bool = False


class TermsAcceptResponse(BaseModel):
    """Response after recording terms acceptance"""
    status: str = "ok"
    recorded_at: datetime


class EmergencyContact(BaseModel):
    """Emergency contact structure"""
    name: str
    relationship: str
    phone: str
    preferred: bool = False


class Medication(BaseModel):
    """Medication structure"""
    name: str
    dose: str
    frequency: str
    start_date: Optional[str] = None
    instructions: Optional[str] = None


class PatientProfileUpdate(BaseModel):
    """Patient profile update payload"""
    assigned_doctor_id: Optional[str] = None
    emergency_contacts: Optional[List[EmergencyContact]] = None
    medications: Optional[List[Medication]] = None
    allergies: Optional[List[str]] = None
    chronic_conditions: Optional[List[str]] = None
    recent_labs: Optional[List[Dict[str, Any]]] = None
    hospitalizations: Optional[List[Dict[str, Any]]] = None
    connected_devices: Optional[List[str]] = None


class DoctorProfileUpdate(BaseModel):
    """Doctor profile update payload"""
    npi: Optional[str] = None
    hospital_affiliations: Optional[List[str]] = None
    board_certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    accepted_insurances: Optional[List[str]] = None
    telemedicine_available: Optional[bool] = None
    telemedicine_fee: Optional[int] = None
    consultation_fee: Optional[int] = None
    publications: Optional[List[Dict[str, Any]]] = None
    malpractice_info: Optional[Dict[str, Any]] = None


class AssignDoctorPayload(BaseModel):
    """Payload for assigning a doctor to a patient"""
    patient_id: str
    doctor_id: str


class AuditLogEntry(BaseModel):
    """Single audit log entry"""
    id: int
    event_type: str
    event_data: Dict[str, Any]
    actor_id: Optional[str] = None
    created_at: datetime


class AuditLogResponse(BaseModel):
    """Response for audit log query"""
    logs: List[AuditLogEntry]
    total: int
    page: int
    page_size: int


class PatientProfileResponse(BaseModel):
    """Patient profile response with assigned doctor info"""
    user_id: str
    assigned_doctor_id: Optional[str] = None
    assigned_doctor: Optional[Dict[str, Any]] = None
    emergency_contacts: List[EmergencyContact] = []
    medications: List[Medication] = []
    allergies: List[str] = []
    chronic_conditions: List[str] = []
    recent_labs: List[Dict[str, Any]] = []
    hospitalizations: List[Dict[str, Any]] = []
    connected_devices: List[str] = []


class DoctorProfileResponse(BaseModel):
    """Doctor profile response"""
    user_id: str
    npi: Optional[str] = None
    hospital_affiliations: List[str] = []
    board_certifications: List[str] = []
    languages: List[str] = []
    accepted_insurances: List[str] = []
    telemedicine_available: bool = False
    telemedicine_fee: Optional[int] = None
    consultation_fee: Optional[int] = None
    publications: List[Dict[str, Any]] = []
    malpractice_info: Dict[str, Any] = {}
