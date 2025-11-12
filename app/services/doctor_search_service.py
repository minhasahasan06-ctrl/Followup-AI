"""
Doctor search and connection service.
Handles doctor discovery, filtering, and patient-doctor relationships.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from datetime import datetime

from app.models.user import User
from app.models.hospital import Hospital
from app.models.specialty import Specialty, DoctorSpecialty
from app.models.patient_doctor_connection import PatientDoctorConnection, PatientConsultation


class DoctorSearchService:
    """Service for searching and managing doctor connections"""

    @staticmethod
    def search_doctors(
        db: Session,
        query: Optional[str] = None,
        specialty: Optional[str] = None,
        location_city: Optional[str] = None,
        location_state: Optional[str] = None,
        hospital_name: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Search for doctors with various filters.
        
        Args:
            db: Database session
            query: Search query (name, email, LinkedIn)
            specialty: Filter by specialty
            location_city: Filter by city
            location_state: Filter by state
            hospital_name: Filter by hospital
            limit: Maximum results to return
            offset: Pagination offset
            
        Returns:
            List of doctor profiles matching search criteria
        """
        filters = [User.role == "doctor"]
        
        if query:
            search_term = f"%{query}%"
            filters.append(
                or_(
                    func.concat(User.first_name, ' ', User.last_name).ilike(search_term),
                    User.email.ilike(search_term),
                    User.linkedin_url.ilike(search_term)
                )
            )
        
        if specialty:
            filters.append(User.specialty.ilike(f"%{specialty}%"))
        
        if location_city:
            filters.append(User.location_city.ilike(f"%{location_city}%"))
        
        if location_state:
            filters.append(User.location_state.ilike(f"%{location_state}%"))
        
        if hospital_name:
            filters.append(User.hospital_name.ilike(f"%{hospital_name}%"))
        
        doctors = (
            db.query(User)
            .filter(and_(*filters))
            .offset(offset)
            .limit(limit)
            .all()
        )
        
        return [
            {
                "id": doctor.id,
                "first_name": doctor.first_name,
                "last_name": doctor.last_name,
                "email": doctor.email,
                "specialty": doctor.specialty,
                "years_of_experience": doctor.years_of_experience,
                "hospital_name": doctor.hospital_name,
                "location_city": doctor.location_city,
                "location_state": doctor.location_state,
                "location_country": doctor.location_country,
                "linkedin_url": doctor.linkedin_url,
                "availability_status": doctor.availability_status,
                "bio": doctor.bio,
            }
            for doctor in doctors
        ]
    
    @staticmethod
    def get_doctor_profile(db: Session, doctor_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed doctor profile by ID"""
        doctor = db.query(User).filter(
            and_(User.id == doctor_id, User.role == "doctor")
        ).first()
        
        if not doctor:
            return None
        
        created_at_value = doctor.created_at
        return {
            "id": doctor.id,
            "first_name": doctor.first_name,
            "last_name": doctor.last_name,
            "email": doctor.email,
            "specialty": doctor.specialty,
            "years_of_experience": doctor.years_of_experience,
            "medical_license_number": doctor.medical_license_number,
            "hospital_name": doctor.hospital_name,
            "location_city": doctor.location_city,
            "location_state": doctor.location_state,
            "location_country": doctor.location_country,
            "linkedin_url": doctor.linkedin_url,
            "availability_status": doctor.availability_status,
            "bio": doctor.bio,
            "created_at": created_at_value.isoformat() if created_at_value is not None else None,
        }
    
    @staticmethod
    def get_patient_connected_doctors(
        db: Session,
        patient_id: str
    ) -> List[Dict[str, Any]]:
        """Get list of doctors connected to a patient"""
        connections = (
            db.query(PatientDoctorConnection, User)
            .join(User, PatientDoctorConnection.doctor_id == User.id)
            .filter(
                and_(
                    PatientDoctorConnection.patient_id == patient_id,
                    PatientDoctorConnection.status == "connected"
                )
            )
            .all()
        )
        
        result = []
        for conn, doctor in connections:
            connected_at_value = conn.connected_at
            result.append({
                "connection_id": conn.id,
                "connection_type": conn.connection_type,
                "connected_at": connected_at_value.isoformat() if connected_at_value is not None else None,
                "doctor": {
                    "id": doctor.id,
                    "first_name": doctor.first_name,
                    "last_name": doctor.last_name,
                    "email": doctor.email,
                    "specialty": doctor.specialty,
                    "hospital_name": doctor.hospital_name,
                    "location_city": doctor.location_city,
                    "linkedin_url": doctor.linkedin_url,
                    "availability_status": doctor.availability_status,
                }
            })
        return result
    
    @staticmethod
    def connect_patient_to_doctor(
        db: Session,
        patient_id: str,
        doctor_id: str,
        connection_type: str = "primary_care",
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Connect a patient to a doctor.
        Creates a pending connection request.
        """
        existing = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == doctor_id,
                PatientDoctorConnection.status.in_(["pending", "connected"])
            )
        ).first()
        
        if existing:
            raise ValueError("Connection already exists")
        
        connection = PatientDoctorConnection(
            patient_id=patient_id,
            doctor_id=doctor_id,
            status="connected",
            connection_type=connection_type,
            notes=notes,
            connected_at=datetime.utcnow()
        )
        
        db.add(connection)
        db.commit()
        db.refresh(connection)
        
        return {
            "id": connection.id,
            "patient_id": connection.patient_id,
            "doctor_id": connection.doctor_id,
            "status": connection.status,
            "connection_type": connection.connection_type,
            "connected_at": connection.connected_at.isoformat() if connection.connected_at else None,
        }
    
    @staticmethod
    def disconnect_patient_from_doctor(
        db: Session,
        patient_id: str,
        doctor_id: str
    ) -> bool:
        """Disconnect a patient from a doctor"""
        connection = db.query(PatientDoctorConnection).filter(
            and_(
                PatientDoctorConnection.patient_id == patient_id,
                PatientDoctorConnection.doctor_id == doctor_id,
                PatientDoctorConnection.status == "connected"
            )
        ).first()
        
        if not connection:
            return False
        
        connection.status = "disconnected"
        connection.disconnected_at = datetime.utcnow()
        
        db.commit()
        return True
    
    @staticmethod
    def suggest_doctors_by_specialty(
        db: Session,
        patient_location_city: Optional[str],
        patient_location_state: Optional[str],
        specialty: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Suggest doctors based on specialty and patient location.
        Used by Agent Clona to recommend specialists.
        """
        filters = [
            User.role == "doctor",
            User.specialty.ilike(f"%{specialty}%"),
            User.availability_status == "available"
        ]
        
        if patient_location_city:
            filters.append(User.location_city.ilike(f"%{patient_location_city}%"))
        
        if patient_location_state:
            filters.append(User.location_state.ilike(f"%{patient_location_state}%"))
        
        doctors = (
            db.query(User)
            .filter(and_(*filters))
            .limit(limit)
            .all()
        )
        
        return [
            {
                "id": doctor.id,
                "first_name": doctor.first_name,
                "last_name": doctor.last_name,
                "specialty": doctor.specialty,
                "hospital_name": doctor.hospital_name,
                "location_city": doctor.location_city,
                "location_state": doctor.location_state,
                "linkedin_url": doctor.linkedin_url,
                "years_of_experience": doctor.years_of_experience,
            }
            for doctor in doctors
        ]
