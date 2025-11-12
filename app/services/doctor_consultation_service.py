from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app.models.consultation import DoctorConsultation, ConsultationRecordAccess
from app.models.user import User
from datetime import datetime


class DoctorConsultationService:
    def __init__(self, db: Session):
        self.db = db
    
    def request_consultation(
        self,
        requesting_doctor_id: str,
        consulted_doctor_id: str,
        patient_id: str,
        reason: str
    ) -> Dict:
        consultation = DoctorConsultation(
            requesting_doctor_id=requesting_doctor_id,
            consulted_doctor_id=consulted_doctor_id,
            patient_id=patient_id,
            consultation_reason=reason,
            status="pending"
        )
        
        self.db.add(consultation)
        self.db.commit()
        self.db.refresh(consultation)
        
        record_access = ConsultationRecordAccess(
            consultation_id=consultation.id,
            doctor_id=consulted_doctor_id,
            patient_id=patient_id
        )
        self.db.add(record_access)
        self.db.commit()
        
        return {
            "success": True,
            "consultation_id": consultation.id,
            "status": "pending"
        }
    
    def get_consultations(self, doctor_id: str) -> List[Dict]:
        consultations = self.db.query(DoctorConsultation).filter(
            (DoctorConsultation.requesting_doctor_id == doctor_id) |
            (DoctorConsultation.consulted_doctor_id == doctor_id)
        ).order_by(DoctorConsultation.created_at.desc()).all()
        
        return [
            {
                "id": c.id,
                "requesting_doctor_id": c.requesting_doctor_id,
                "consulted_doctor_id": c.consulted_doctor_id,
                "patient_id": c.patient_id,
                "reason": c.consultation_reason,
                "status": c.status,
                "response": c.response,
                "created_at": c.created_at.isoformat()
            }
            for c in consultations
        ]
    
    def approve_consultation(self, consultation_id: int, doctor_id: str) -> Dict:
        consultation = self.db.query(DoctorConsultation).filter(
            DoctorConsultation.id == consultation_id,
            DoctorConsultation.consulted_doctor_id == doctor_id
        ).first()
        
        if not consultation:
            return {"error": "Consultation not found"}
        
        consultation.status = "approved"
        self.db.commit()
        
        return {"success": True, "status": "approved"}
    
    def decline_consultation(self, consultation_id: int, doctor_id: str, reason: str) -> Dict:
        consultation = self.db.query(DoctorConsultation).filter(
            DoctorConsultation.id == consultation_id,
            DoctorConsultation.consulted_doctor_id == doctor_id
        ).first()
        
        if not consultation:
            return {"error": "Consultation not found"}
        
        consultation.status = "declined"
        consultation.response = reason
        consultation.response_at = datetime.utcnow()
        self.db.commit()
        
        return {"success": True, "status": "declined"}
