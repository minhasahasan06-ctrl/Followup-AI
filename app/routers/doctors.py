"""
Doctor search and connection API endpoints.
Allows patients to search for doctors and manage connections.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db
from app.dependencies import get_current_user, require_role
from app.models.user import User
from app.services.doctor_search_service import DoctorSearchService
from app.services.access_control import HIPAAAuditLogger, PHICategory


router = APIRouter(prefix="/api/doctors", tags=["doctors"])


class DoctorSearchResponse(BaseModel):
    id: str
    first_name: Optional[str]
    last_name: Optional[str]
    email: str
    specialty: Optional[str]
    years_of_experience: Optional[int]
    hospital_name: Optional[str]
    location_city: Optional[str]
    location_state: Optional[str]
    location_country: Optional[str]
    linkedin_url: Optional[str]
    availability_status: Optional[str]
    bio: Optional[str]


class DoctorProfileResponse(BaseModel):
    id: str
    first_name: Optional[str]
    last_name: Optional[str]
    email: str
    specialty: Optional[str]
    years_of_experience: Optional[int]
    medical_license_number: Optional[str]
    hospital_name: Optional[str]
    location_city: Optional[str]
    location_state: Optional[str]
    location_country: Optional[str]
    linkedin_url: Optional[str]
    availability_status: Optional[str]
    bio: Optional[str]
    created_at: Optional[str]


class ConnectDoctorRequest(BaseModel):
    doctor_id: str
    connection_type: str = "primary_care"
    notes: Optional[str] = None


class ConnectionResponse(BaseModel):
    id: int
    patient_id: str
    doctor_id: str
    status: str
    connection_type: str
    connected_at: Optional[str]


@router.get("/search", response_model=List[DoctorSearchResponse])
async def search_doctors(
    request: Request,
    query: Optional[str] = Query(None, description="Search by name, email, or LinkedIn"),
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    location_city: Optional[str] = Query(None, description="Filter by city"),
    location_state: Optional[str] = Query(None, description="Filter by state"),
    hospital_name: Optional[str] = Query(None, description="Filter by hospital"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Search for doctors with various filters.
    Patients can search by name, email, LinkedIn profile, specialty, location, or hospital.
    """
    doctors = DoctorSearchService.search_doctors(
        db=db,
        query=query,
        specialty=specialty,
        location_city=location_city,
        location_state=location_state,
        hospital_name=hospital_name,
        limit=limit,
        offset=offset
    )
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role=str(current_user.role),
        patient_id=None,
        action="search_doctors",
        phi_categories=[PHICategory.PROVIDER_INFO.value],
        resource_type="doctor_search",
        success=True,
        ip_address=request.client.host if request.client else None
    )
    
    return doctors


@router.get("/{doctor_id}", response_model=DoctorProfileResponse)
async def get_doctor_profile(
    doctor_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detailed profile of a specific doctor"""
    profile = DoctorSearchService.get_doctor_profile(db=db, doctor_id=doctor_id)
    
    if not profile:
        raise HTTPException(status_code=404, detail="Doctor not found")
    
    return profile


@router.get("/my-doctors/list")
async def get_my_doctors(
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get list of doctors connected to the current patient.
    Only accessible by patients.
    """
    connections = DoctorSearchService.get_patient_connected_doctors(
        db=db,
        patient_id=current_user.id
    )
    
    return {"connections": connections}


@router.post("/connect", response_model=ConnectionResponse)
async def connect_to_doctor(
    connect_request: ConnectDoctorRequest,
    request: Request,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Connect patient to a doctor.
    Only accessible by patients.
    """
    try:
        connection = DoctorSearchService.connect_patient_to_doctor(
            db=db,
            patient_id=current_user.id,
            doctor_id=connect_request.doctor_id,
            connection_type=connect_request.connection_type,
            notes=connect_request.notes
        )
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=current_user.id,
            actor_role="patient",
            patient_id=current_user.id,
            action="connect_to_doctor",
            phi_categories=[PHICategory.PROVIDER_INFO.value],
            resource_type="doctor_connection",
            resource_id=str(connection.id) if hasattr(connection, 'id') else None,
            success=True,
            ip_address=request.client.host if request.client else None
        )
        
        return connection
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/disconnect/{doctor_id}")
async def disconnect_from_doctor(
    doctor_id: str,
    request: Request,
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Disconnect patient from a doctor.
    Only accessible by patients.
    """
    success = DoctorSearchService.disconnect_patient_from_doctor(
        db=db,
        patient_id=current_user.id,
        doctor_id=doctor_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=current_user.id,
        actor_role="patient",
        patient_id=current_user.id,
        action="disconnect_from_doctor",
        phi_categories=[PHICategory.PROVIDER_INFO.value],
        resource_type="doctor_connection",
        success=True,
        ip_address=request.client.host if request.client else None
    )
    
    return {"message": "Successfully disconnected from doctor"}


@router.get("/suggest/by-specialty")
async def suggest_doctors_by_specialty(
    specialty: str = Query(..., description="Required specialty"),
    current_user: User = Depends(require_role("patient")),
    db: Session = Depends(get_db)
):
    """
    Get doctor suggestions by specialty, considering patient's location.
    Used by Agent Clona to suggest specialists based on symptoms.
    """
    suggestions = DoctorSearchService.suggest_doctors_by_specialty(
        db=db,
        patient_location_city=current_user.location_city,
        patient_location_state=current_user.location_state,
        specialty=specialty,
        limit=5
    )
    
    return {"suggested_doctors": suggestions}
