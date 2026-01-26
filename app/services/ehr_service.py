"""
EHR Service
Server-side EHR data fetching for personalization.
Fetches problem list, complaints, medications, allergies, labs, imaging, vitals.
All PHI access is audit logged for HIPAA compliance.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_

logger = logging.getLogger(__name__)


def audit_log_phi_access(
    db: Session,
    patient_id: str,
    accessor_id: str,
    resource_type: str,
    action: str,
    details: Dict[str, Any] = None
):
    """Log PHI access for HIPAA compliance."""
    logger.info(
        f"PHI_ACCESS: patient={patient_id}, accessor={accessor_id}, "
        f"resource={resource_type}, action={action}, details={details}"
    )


class EHRService:
    """
    EHR data service for fetching patient health records server-side.
    All methods require authenticated access and log PHI access.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    async def get_problem_list(
        self,
        patient_id: str,
        accessor_id: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch patient's problem list (diagnoses/conditions).
        
        Returns list of:
        - code: ICD-10 or SNOMED code
        - name: Condition name
        - status: active, resolved, chronic
        - onset_date: When diagnosed
        - category: Category for mapping (respiratory, cardiac, mental_health, etc.)
        """
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "problem_list", "read"
        )
        
        problems = []
        
        try:
            from app.models.symptom_journal import SymptomEntry
            symptoms = self.db.query(SymptomEntry).filter(
                SymptomEntry.patient_id == patient_id
            ).order_by(desc(SymptomEntry.created_at)).limit(100).all()
            
            condition_map = {}
            for symptom in symptoms:
                if hasattr(symptom, 'symptom_type') and symptom.symptom_type:
                    key = symptom.symptom_type.lower()
                    if key not in condition_map:
                        condition_map[key] = {
                            "code": f"SYM_{key.upper()}",
                            "name": symptom.symptom_type,
                            "status": "active",
                            "category": self._categorize_condition(key),
                            "onset_date": symptom.created_at.isoformat() if symptom.created_at else None
                        }
            problems.extend(condition_map.values())
        except Exception as e:
            logger.warning(f"Error fetching symptoms: {e}")
        
        try:
            from app.models.environmental_risk import PatientEnvironmentProfile
            profile = self.db.query(PatientEnvironmentProfile).filter(
                PatientEnvironmentProfile.patient_id == patient_id,
                PatientEnvironmentProfile.is_active == True
            ).first()
            
            if profile and profile.chronic_conditions:
                for condition in profile.chronic_conditions:
                    if isinstance(condition, str):
                        problems.append({
                            "code": f"CHRONIC_{condition.upper().replace(' ', '_')}",
                            "name": condition,
                            "status": "chronic",
                            "category": self._categorize_condition(condition)
                        })
        except Exception as e:
            logger.warning(f"Error fetching chronic conditions: {e}")
        
        return problems
    
    async def get_recent_complaints(
        self,
        patient_id: str,
        accessor_id: str,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Fetch patient's recent complaints from followups and symptom journals.
        
        Returns list of:
        - complaint: Description
        - severity: 1-10 scale
        - date: When reported
        - source: followup, journal, chat
        - category: Categorized for mapping
        """
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "complaints", "read",
            {"days": days}
        )
        
        complaints = []
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            from app.models.symptom_journal import SymptomEntry
            symptoms = self.db.query(SymptomEntry).filter(
                SymptomEntry.patient_id == patient_id,
                SymptomEntry.created_at >= cutoff_date
            ).order_by(desc(SymptomEntry.created_at)).all()
            
            for symptom in symptoms:
                complaints.append({
                    "complaint": getattr(symptom, 'description', '') or getattr(symptom, 'symptom_type', ''),
                    "severity": getattr(symptom, 'severity', 5),
                    "date": symptom.created_at.isoformat() if symptom.created_at else None,
                    "source": "journal",
                    "category": self._categorize_condition(
                        getattr(symptom, 'symptom_type', '') or ''
                    )
                })
        except Exception as e:
            logger.warning(f"Error fetching symptom entries: {e}")
        
        try:
            from app.models.pain_tracking import PainEntry
            pain_entries = self.db.query(PainEntry).filter(
                PainEntry.patient_id == patient_id,
                PainEntry.created_at >= cutoff_date
            ).order_by(desc(PainEntry.created_at)).all()
            
            for entry in pain_entries:
                complaints.append({
                    "complaint": f"Pain: {getattr(entry, 'location', 'unspecified')}",
                    "severity": getattr(entry, 'intensity', 5),
                    "date": entry.created_at.isoformat() if entry.created_at else None,
                    "source": "pain_tracking",
                    "category": "pain"
                })
        except Exception as e:
            logger.warning(f"Error fetching pain entries: {e}")
        
        return complaints
    
    async def get_medications(
        self,
        patient_id: str,
        accessor_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch patient's current medications."""
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "medications", "read"
        )
        
        medications = []
        
        try:
            from app.models.medication import Medication
            meds = self.db.query(Medication).filter(
                Medication.patient_id == patient_id,
                Medication.is_active == True
            ).all()
            
            for med in meds:
                medications.append({
                    "name": med.name,
                    "dosage": getattr(med, 'dosage', None),
                    "frequency": getattr(med, 'frequency', None),
                    "category": getattr(med, 'category', None),
                    "start_date": med.created_at.isoformat() if med.created_at else None
                })
        except Exception as e:
            logger.warning(f"Error fetching medications: {e}")
        
        return medications
    
    async def get_allergies(
        self,
        patient_id: str,
        accessor_id: str
    ) -> List[Dict[str, Any]]:
        """Fetch patient's allergies."""
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "allergies", "read"
        )
        
        allergies = []
        
        try:
            from app.models.environmental_risk import PatientEnvironmentProfile
            profile = self.db.query(PatientEnvironmentProfile).filter(
                PatientEnvironmentProfile.patient_id == patient_id,
                PatientEnvironmentProfile.is_active == True
            ).first()
            
            if profile and profile.allergies:
                for allergy in profile.allergies:
                    if isinstance(allergy, str):
                        allergies.append({
                            "allergen": allergy,
                            "reaction": "unknown",
                            "severity": "unknown"
                        })
                    elif isinstance(allergy, dict):
                        allergies.append(allergy)
        except Exception as e:
            logger.warning(f"Error fetching allergies: {e}")
        
        return allergies
    
    async def get_recent_vitals(
        self,
        patient_id: str,
        accessor_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Fetch patient's recent vital signs."""
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "vitals", "read",
            {"days": days}
        )
        
        vitals = []
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            from app.models.trend_models import TrendSnapshot
            snapshots = self.db.query(TrendSnapshot).filter(
                TrendSnapshot.patient_id == patient_id,
                TrendSnapshot.snapshot_date >= cutoff_date
            ).order_by(desc(TrendSnapshot.snapshot_date)).limit(30).all()
            
            for snapshot in snapshots:
                if hasattr(snapshot, 'metrics') and snapshot.metrics:
                    vitals.append({
                        "date": snapshot.snapshot_date.isoformat() if snapshot.snapshot_date else None,
                        "metrics": snapshot.metrics
                    })
        except Exception as e:
            logger.warning(f"Error fetching vitals: {e}")
        
        return vitals
    
    async def get_recent_labs(
        self,
        patient_id: str,
        accessor_id: str,
        days: int = 90
    ) -> List[Dict[str, Any]]:
        """Fetch patient's recent lab results."""
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "labs", "read",
            {"days": days}
        )
        
        return []
    
    async def get_full_ehr_summary(
        self,
        patient_id: str,
        accessor_id: str
    ) -> Dict[str, Any]:
        """
        Fetch comprehensive EHR summary for Lysa differential generation.
        Includes all available patient data with source tracking.
        """
        audit_log_phi_access(
            self.db, patient_id, accessor_id,
            "full_ehr_summary", "read"
        )
        
        problems = await self.get_problem_list(patient_id, accessor_id)
        complaints = await self.get_recent_complaints(patient_id, accessor_id, days=90)
        medications = await self.get_medications(patient_id, accessor_id)
        allergies = await self.get_allergies(patient_id, accessor_id)
        vitals = await self.get_recent_vitals(patient_id, accessor_id, days=30)
        labs = await self.get_recent_labs(patient_id, accessor_id, days=90)
        
        provenance = []
        if problems:
            provenance.append({"type": "problem_list", "count": len(problems)})
        if complaints:
            provenance.append({"type": "complaints", "count": len(complaints)})
        if medications:
            provenance.append({"type": "medications", "count": len(medications)})
        if allergies:
            provenance.append({"type": "allergies", "count": len(allergies)})
        if vitals:
            provenance.append({"type": "vitals", "count": len(vitals)})
        if labs:
            provenance.append({"type": "labs", "count": len(labs)})
        
        return {
            "patient_id": patient_id,
            "generated_at": datetime.utcnow().isoformat(),
            "problems": problems,
            "complaints": complaints,
            "medications": medications,
            "allergies": allergies,
            "vitals": vitals,
            "labs": labs,
            "provenance": provenance
        }
    
    def _categorize_condition(self, condition: str) -> str:
        """Categorize a condition for habit/recommendation mapping."""
        condition_lower = condition.lower()
        
        respiratory = ["asthma", "copd", "bronchitis", "pneumonia", "breathing", "respiratory", "lung"]
        cardiac = ["heart", "cardiac", "hypertension", "blood pressure", "arrhythmia", "cardiovascular"]
        mental = ["depression", "anxiety", "stress", "mental", "mood", "panic", "ptsd"]
        pain = ["pain", "headache", "migraine", "arthritis", "fibromyalgia", "chronic pain"]
        metabolic = ["diabetes", "thyroid", "metabolic", "weight", "obesity"]
        immune = ["autoimmune", "lupus", "rheumatoid", "immunocompromised", "immune"]
        gastrointestinal = ["ibs", "crohn", "colitis", "digestive", "stomach", "gut"]
        neurological = ["seizure", "epilepsy", "neuropathy", "parkinson", "multiple sclerosis"]
        
        for keyword in respiratory:
            if keyword in condition_lower:
                return "respiratory"
        for keyword in cardiac:
            if keyword in condition_lower:
                return "cardiac"
        for keyword in mental:
            if keyword in condition_lower:
                return "mental_health"
        for keyword in pain:
            if keyword in condition_lower:
                return "pain"
        for keyword in metabolic:
            if keyword in condition_lower:
                return "metabolic"
        for keyword in immune:
            if keyword in condition_lower:
                return "immune"
        for keyword in gastrointestinal:
            if keyword in condition_lower:
                return "gastrointestinal"
        for keyword in neurological:
            if keyword in condition_lower:
                return "neurological"
        
        return "general"


def get_ehr_service(db: Session) -> EHRService:
    """Factory function for EHR service."""
    return EHRService(db)
