"""
Consent Verification Service
Verifies doctor-patient relationships before allowing cross-party communication.
All message routing between doctors/patients/agents must pass through this service.
"""

import logging
import json
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=5)


class CommunicationParty(str, Enum):
    """Types of parties in communication"""
    PATIENT = "patient"
    DOCTOR = "doctor"
    AGENT_CLONA = "clona"      # Patient's AI companion
    AGENT_LYSA = "lysa"        # Doctor's AI assistant
    SYSTEM = "system"


class ConsentStatus(str, Enum):
    """Status of consent/connection"""
    ACTIVE = "active"
    PENDING = "pending"
    REVOKED = "revoked"
    EXPIRED = "expired"


def _get_db_session():
    """Get a fresh database session"""
    try:
        from app.database import SessionLocal
        return SessionLocal()
    except Exception as e:
        logger.error(f"Failed to create database session: {e}")
        return None


class ConsentService:
    """
    Verifies doctor-patient relationships for message routing.
    
    Communication Rules:
    1. Patient ↔ Clona: ALWAYS allowed (patient's own agent)
    2. Doctor ↔ Lysa: ALWAYS allowed (doctor's own assistant)
    3. Patient ↔ Doctor: Requires active doctor_patient_assignment
    4. Patient ↔ Lysa: Requires active assignment (patient talks to doctor's assistant)
    5. Doctor ↔ Clona: Requires active assignment (doctor talks to patient's agent)
    6. Clona ↔ Lysa: Requires active assignment between their respective users
    """
    
    # Cache for connection lookups (short TTL in production)
    _connection_cache: Dict[str, Tuple[bool, datetime]] = {}
    _cache_ttl_seconds: int = 60  # 1 minute cache
    
    def __init__(self):
        self._initialized = False
    
    def _get_party_type(self, actor_type: str, actor_id: str) -> CommunicationParty:
        """Determine the party type from actor info"""
        if actor_type == "agent":
            if actor_id == "clona":
                return CommunicationParty.AGENT_CLONA
            elif actor_id == "lysa":
                return CommunicationParty.AGENT_LYSA
        elif actor_type == "user":
            # Determine if user is patient or doctor by looking up their role
            role = self._get_user_role(actor_id)
            if role == "doctor":
                return CommunicationParty.DOCTOR
            else:
                return CommunicationParty.PATIENT
        elif actor_type == "system":
            return CommunicationParty.SYSTEM
        
        return CommunicationParty.PATIENT  # Default to patient for safety
    
    def _get_user_role(self, user_id: str) -> str:
        """Get user's role from database"""
        db = _get_db_session()
        if not db:
            return "patient"
        
        try:
            result = db.execute(
                text("SELECT role FROM users WHERE id = :user_id"),
                {"user_id": user_id}
            )
            row = result.fetchone()
            return row[0] if row else "patient"
        except Exception as e:
            logger.warning(f"Failed to get user role: {e}")
            return "patient"
        finally:
            db.close()
    
    def _get_user_id_for_agent(self, agent_id: str, context_user_id: Optional[str] = None) -> Optional[str]:
        """
        Get the user ID associated with an agent in a conversation context.
        - Clona is associated with the patient user
        - Lysa is associated with the doctor user
        """
        if context_user_id:
            role = self._get_user_role(context_user_id)
            if agent_id == "clona" and role == "patient":
                return context_user_id
            elif agent_id == "lysa" and role == "doctor":
                return context_user_id
        return None
    
    def get_cache_key(self, doctor_id: str, patient_id: str) -> str:
        """Generate cache key for connection lookup"""
        return f"consent:{doctor_id}:{patient_id}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid"""
        if cache_key not in self._connection_cache:
            return False
        _, cached_at = self._connection_cache[cache_key]
        return (datetime.utcnow() - cached_at).total_seconds() < self._cache_ttl_seconds
    
    def verify_connection(
        self,
        doctor_id: str,
        patient_id: str,
        require_consent: bool = True
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify active doctor-patient connection exists.
        
        Returns:
            Tuple of (is_connected, connection_details)
        """
        cache_key = self.get_cache_key(doctor_id, patient_id)
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            is_connected, _ = self._connection_cache[cache_key]
            if is_connected:
                return True, {"source": "cache"}
        
        db = _get_db_session()
        if not db:
            logger.error("No database session for consent verification")
            return False, {"error": "database_unavailable"}
        
        try:
            # Check doctor_patient_assignments table
            result = db.execute(
                text("""
                    SELECT id, status, patient_consented, access_scope,
                           is_primary_care_provider, assignment_source, created_at
                    FROM doctor_patient_assignments
                    WHERE doctor_id = :doctor_id
                    AND patient_id = :patient_id
                    AND status = 'active'
                    LIMIT 1
                """),
                {"doctor_id": doctor_id, "patient_id": patient_id}
            )
            row = result.fetchone()
            
            if row:
                connection_info = {
                    "assignment_id": str(row[0]),
                    "status": row[1],
                    "patient_consented": row[2],
                    "access_scope": row[3] or "full",
                    "is_primary_care_provider": row[4],
                    "assignment_source": row[5],
                    "connected_since": row[6].isoformat() if row[6] else None
                }
                
                # Check if consent is required and given
                if require_consent and not row[2]:
                    logger.warning(f"Connection exists but patient consent not given: {doctor_id} -> {patient_id}")
                    return False, {"error": "consent_not_given", "connection": connection_info}
                
                # Cache successful result
                self._connection_cache[cache_key] = (True, datetime.utcnow())
                
                return True, connection_info
            
            # Also check PatientSharingLink as fallback
            sharing_result = db.execute(
                text("""
                    SELECT id, status, access_level, expires_at, created_at
                    FROM patient_sharing_links
                    WHERE doctor_id = :doctor_id
                    AND patient_id = :patient_id
                    AND status = 'active'
                    AND (expires_at IS NULL OR expires_at > NOW())
                    LIMIT 1
                """),
                {"doctor_id": doctor_id, "patient_id": patient_id}
            )
            sharing_row = sharing_result.fetchone()
            
            if sharing_row:
                connection_info = {
                    "sharing_link_id": str(sharing_row[0]),
                    "status": sharing_row[1],
                    "access_level": sharing_row[2],
                    "expires_at": sharing_row[3].isoformat() if sharing_row[3] else None,
                    "assignment_source": "sharing_link",
                    "connected_since": sharing_row[4].isoformat() if sharing_row[4] else None
                }
                
                # Cache successful result
                self._connection_cache[cache_key] = (True, datetime.utcnow())
                
                return True, connection_info
            
            # No connection found
            self._connection_cache[cache_key] = (False, datetime.utcnow())
            return False, {"error": "no_connection"}
            
        except Exception as e:
            logger.error(f"Error verifying connection: {e}")
            return False, {"error": str(e)}
        finally:
            db.close()
    
    def can_communicate(
        self,
        from_type: str,
        from_id: str,
        to_type: str,
        to_id: str,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Check if communication is allowed between two parties.
        
        Args:
            from_type: Sender type (user, agent, system)
            from_id: Sender ID
            to_type: Recipient type
            to_id: Recipient ID
            conversation_context: Optional context with patient_id, doctor_id
            
        Returns:
            Tuple of (allowed, reason, connection_info)
        """
        from_party = self._get_party_type(from_type, from_id)
        to_party = self._get_party_type(to_type, to_id)
        
        logger.debug(f"Checking communication: {from_party}({from_id}) -> {to_party}({to_id})")
        
        # Rule 1: Patient ↔ Clona - Always allowed (own agent)
        if (from_party == CommunicationParty.PATIENT and to_party == CommunicationParty.AGENT_CLONA) or \
           (from_party == CommunicationParty.AGENT_CLONA and to_party == CommunicationParty.PATIENT):
            # Verify it's the patient's own Clona instance
            # In context, patient_id should match the user
            if conversation_context:
                ctx_patient = conversation_context.get("patient_id")
                if from_party == CommunicationParty.PATIENT and ctx_patient and ctx_patient != from_id:
                    return False, "Cannot communicate with another patient's agent", None
                if to_party == CommunicationParty.PATIENT and ctx_patient and ctx_patient != to_id:
                    return False, "Agent cannot communicate with unrelated patient", None
            return True, "own_agent", {"relationship": "patient_clona"}
        
        # Rule 2: Doctor ↔ Lysa - Always allowed (own assistant)
        if (from_party == CommunicationParty.DOCTOR and to_party == CommunicationParty.AGENT_LYSA) or \
           (from_party == CommunicationParty.AGENT_LYSA and to_party == CommunicationParty.DOCTOR):
            # Verify it's the doctor's own Lysa instance
            if conversation_context:
                ctx_doctor = conversation_context.get("doctor_id")
                if from_party == CommunicationParty.DOCTOR and ctx_doctor and ctx_doctor != from_id:
                    return False, "Cannot communicate with another doctor's assistant", None
                if to_party == CommunicationParty.DOCTOR and ctx_doctor and ctx_doctor != to_id:
                    return False, "Assistant cannot communicate with unrelated doctor", None
            return True, "own_assistant", {"relationship": "doctor_lysa"}
        
        # Rule 3: System messages - Always allowed
        if from_party == CommunicationParty.SYSTEM or to_party == CommunicationParty.SYSTEM:
            return True, "system_message", {"relationship": "system"}
        
        # For all other communication, we need to verify doctor-patient connection
        # Extract doctor_id and patient_id based on party types
        doctor_id = None
        patient_id = None
        
        # Patient ↔ Doctor
        if from_party == CommunicationParty.PATIENT and to_party == CommunicationParty.DOCTOR:
            patient_id = from_id
            doctor_id = to_id
        elif from_party == CommunicationParty.DOCTOR and to_party == CommunicationParty.PATIENT:
            doctor_id = from_id
            patient_id = to_id
        
        # Patient ↔ Lysa (patient talks to doctor's assistant)
        elif from_party == CommunicationParty.PATIENT and to_party == CommunicationParty.AGENT_LYSA:
            patient_id = from_id
            # Get doctor from context
            doctor_id = conversation_context.get("doctor_id") if conversation_context else None
        elif from_party == CommunicationParty.AGENT_LYSA and to_party == CommunicationParty.PATIENT:
            patient_id = to_id
            doctor_id = conversation_context.get("doctor_id") if conversation_context else None
        
        # Doctor ↔ Clona (doctor talks to patient's agent)
        elif from_party == CommunicationParty.DOCTOR and to_party == CommunicationParty.AGENT_CLONA:
            doctor_id = from_id
            patient_id = conversation_context.get("patient_id") if conversation_context else None
        elif from_party == CommunicationParty.AGENT_CLONA and to_party == CommunicationParty.DOCTOR:
            doctor_id = to_id
            patient_id = conversation_context.get("patient_id") if conversation_context else None
        
        # Clona ↔ Lysa (inter-agent communication)
        elif (from_party == CommunicationParty.AGENT_CLONA and to_party == CommunicationParty.AGENT_LYSA) or \
             (from_party == CommunicationParty.AGENT_LYSA and to_party == CommunicationParty.AGENT_CLONA):
            # Both IDs must come from context
            if conversation_context:
                patient_id = conversation_context.get("patient_id")
                doctor_id = conversation_context.get("doctor_id")
            else:
                return False, "Inter-agent communication requires conversation context", None
        
        # Validate we have both IDs
        if not doctor_id or not patient_id:
            logger.warning(f"Missing IDs for consent check: doctor={doctor_id}, patient={patient_id}")
            return False, "Cannot determine doctor-patient relationship", None
        
        # Verify the connection
        is_connected, connection_info = self.verify_connection(doctor_id, patient_id)
        
        if not is_connected:
            error = connection_info.get("error", "no_connection") if connection_info else "no_connection"
            logger.warning(f"Communication blocked: no connection between doctor {doctor_id} and patient {patient_id}")
            return False, f"No active connection between doctor and patient: {error}", connection_info
        
        return True, "connected", connection_info
    
    def get_connected_doctors(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all doctors connected to a patient"""
        db = _get_db_session()
        if not db:
            return []
        
        try:
            result = db.execute(
                text("""
                    SELECT DISTINCT dpa.doctor_id, u.email, u.first_name, u.last_name,
                           dpa.is_primary_care_provider, dpa.created_at
                    FROM doctor_patient_assignments dpa
                    JOIN users u ON u.id = dpa.doctor_id
                    WHERE dpa.patient_id = :patient_id
                    AND dpa.status = 'active'
                    AND dpa.patient_consented = true
                    ORDER BY dpa.is_primary_care_provider DESC, dpa.created_at ASC
                """),
                {"patient_id": patient_id}
            )
            rows = result.fetchall()
            return [
                {
                    "doctor_id": row[0],
                    "email": row[1],
                    "first_name": row[2],
                    "last_name": row[3],
                    "is_primary": row[4],
                    "connected_since": row[5].isoformat() if row[5] else None
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting connected doctors: {e}")
            return []
        finally:
            db.close()
    
    def get_connected_patients(self, doctor_id: str) -> List[Dict[str, Any]]:
        """Get all patients connected to a doctor"""
        db = _get_db_session()
        if not db:
            return []
        
        try:
            result = db.execute(
                text("""
                    SELECT DISTINCT dpa.patient_id, u.email, u.first_name, u.last_name,
                           dpa.is_primary_care_provider, dpa.access_scope, dpa.created_at
                    FROM doctor_patient_assignments dpa
                    JOIN users u ON u.id = dpa.patient_id
                    WHERE dpa.doctor_id = :doctor_id
                    AND dpa.status = 'active'
                    AND dpa.patient_consented = true
                    ORDER BY dpa.created_at DESC
                """),
                {"doctor_id": doctor_id}
            )
            rows = result.fetchall()
            return [
                {
                    "patient_id": row[0],
                    "email": row[1],
                    "first_name": row[2],
                    "last_name": row[3],
                    "is_primary_doctor": row[4],
                    "access_scope": row[5] or "full",
                    "connected_since": row[6].isoformat() if row[6] else None
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Error getting connected patients: {e}")
            return []
        finally:
            db.close()
    
    def invalidate_cache(self, doctor_id: str, patient_id: str):
        """Invalidate cache for a specific connection"""
        cache_key = self.get_cache_key(doctor_id, patient_id)
        if cache_key in self._connection_cache:
            del self._connection_cache[cache_key]
    
    def clear_cache(self):
        """Clear all cached connections"""
        self._connection_cache.clear()


# Singleton instance
_consent_service: Optional[ConsentService] = None


def get_consent_service() -> ConsentService:
    """Get or create the consent service singleton"""
    global _consent_service
    if _consent_service is None:
        _consent_service = ConsentService()
    return _consent_service
