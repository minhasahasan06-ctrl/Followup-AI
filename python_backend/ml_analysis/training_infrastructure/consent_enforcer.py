"""
Consent Enforcer
=================
Production-grade consent enforcement for ML data pipelines with:
- Pipeline-level consent verification
- Data type filtering based on consent
- K-anonymity enforcement
- Comprehensive audit logging

HIPAA-compliant with full consent tracking.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class ConsentLevel(str, Enum):
    """Patient consent levels"""
    NONE = "none"
    BASIC = "basic"  # Aggregate statistics only
    STANDARD = "standard"  # ML training with anonymization
    FULL = "full"  # Full data access for research


class DataCategory(str, Enum):
    """Data categories requiring consent"""
    DEMOGRAPHICS = "demographics"
    VITALS = "vitals"
    MEDICATIONS = "medications"
    DIAGNOSES = "diagnoses"
    LAB_RESULTS = "lab_results"
    DEVICE_DATA = "device_data"
    BEHAVIORAL = "behavioral"
    MENTAL_HEALTH = "mental_health"
    GENETIC = "genetic"
    ENVIRONMENTAL = "environmental"


@dataclass
class ConsentPolicy:
    """Policy defining consent requirements for data access"""
    policy_id: str
    name: str
    description: str
    required_consent_level: ConsentLevel
    allowed_categories: List[DataCategory]
    min_aggregation_size: int = 10  # k-anonymity minimum
    requires_irb_approval: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConsentCheckResult:
    """Result of a consent check operation"""
    allowed: bool
    patient_ids_with_consent: Set[str]
    patient_ids_denied: Set[str]
    categories_allowed: List[str]
    categories_denied: List[str]
    k_anonymity_met: bool
    audit_log_id: str
    message: str


class ConsentEnforcer:
    """
    Enforces consent requirements in all ML data pipelines.
    
    Features:
    - Verify patient consent before data extraction
    - Filter data categories based on consent level
    - Enforce k-anonymity requirements
    - Audit log all consent checks
    - Block access to non-consented data
    """
    
    MIN_K_ANONYMITY = 10  # Minimum group size for privacy
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self._policies_cache: Dict[str, ConsentPolicy] = {}
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def check_patient_consent(
        self,
        patient_ids: List[str],
        data_categories: List[DataCategory],
        purpose: str,
        requester_id: str,
        min_consent_level: ConsentLevel = ConsentLevel.STANDARD
    ) -> ConsentCheckResult:
        """
        Check if patients have consented to data access for given categories.
        
        Args:
            patient_ids: List of patient IDs to check
            data_categories: Data categories being accessed
            purpose: Purpose of data access (e.g., "ml_training", "research")
            requester_id: ID of the user/system requesting access
            min_consent_level: Minimum required consent level
            
        Returns:
            ConsentCheckResult with allowed patients and categories
        """
        consented_patients: Set[str] = set()
        denied_patients: Set[str] = set()
        allowed_categories: List[str] = []
        denied_categories: List[str] = []
        
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get consent status for each patient
            for patient_id in patient_ids:
                consent_status = self._get_patient_consent(cur, patient_id)
                
                if consent_status and self._meets_consent_level(consent_status, min_consent_level):
                    consented_patients.add(patient_id)
                else:
                    denied_patients.add(patient_id)
            
            # Check category-level consent
            for category in data_categories:
                if self._is_category_allowed(category, min_consent_level):
                    allowed_categories.append(category.value if isinstance(category, DataCategory) else category)
                else:
                    denied_categories.append(category.value if isinstance(category, DataCategory) else category)
            
            # Check k-anonymity
            k_anonymity_met = len(consented_patients) >= self.MIN_K_ANONYMITY
            
            # Log the consent check
            audit_log_id = self._log_consent_check(
                cur=cur,
                patient_ids=list(patient_ids),
                consented_count=len(consented_patients),
                denied_count=len(denied_patients),
                categories=data_categories,
                purpose=purpose,
                requester_id=requester_id,
                result="allowed" if (consented_patients and k_anonymity_met) else "denied"
            )
            
            conn.commit()
            cur.close()
            conn.close()
            
            allowed = bool(consented_patients) and k_anonymity_met and bool(allowed_categories)
            
            return ConsentCheckResult(
                allowed=allowed,
                patient_ids_with_consent=consented_patients,
                patient_ids_denied=denied_patients,
                categories_allowed=allowed_categories,
                categories_denied=denied_categories,
                k_anonymity_met=k_anonymity_met,
                audit_log_id=audit_log_id,
                message=self._get_result_message(
                    consented_patients, denied_patients, k_anonymity_met, allowed_categories
                )
            )
            
        except Exception as e:
            logger.error(f"Error checking consent: {e}")
            return ConsentCheckResult(
                allowed=False,
                patient_ids_with_consent=set(),
                patient_ids_denied=set(patient_ids),
                categories_allowed=[],
                categories_denied=[c.value if isinstance(c, DataCategory) else c for c in data_categories],
                k_anonymity_met=False,
                audit_log_id="error",
                message=f"Consent check failed: {e}"
            )
    
    def _get_patient_consent(self, cur, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get consent status for a patient"""
        try:
            cur.execute("""
                SELECT 
                    ml_training_consent,
                    research_consent,
                    device_data_consent,
                    genetic_consent,
                    consent_level,
                    consent_date,
                    revoked_at
                FROM patient_consent
                WHERE patient_id = %s
                AND (revoked_at IS NULL OR revoked_at > NOW())
            """, (patient_id,))
            
            row = cur.fetchone()
            return dict(row) if row else None
            
        except Exception as e:
            logger.error(f"Error getting patient consent: {e}")
            return None
    
    def _meets_consent_level(self, consent_status: Dict[str, Any], required_level: ConsentLevel) -> bool:
        """Check if consent status meets required level"""
        if not consent_status.get('ml_training_consent'):
            return False
        
        patient_level = consent_status.get('consent_level', 'basic')
        
        level_hierarchy = {
            ConsentLevel.NONE: 0,
            ConsentLevel.BASIC: 1,
            ConsentLevel.STANDARD: 2,
            ConsentLevel.FULL: 3
        }
        
        required_value = level_hierarchy.get(required_level, 2)
        patient_value = level_hierarchy.get(ConsentLevel(patient_level) if patient_level in [e.value for e in ConsentLevel] else ConsentLevel.BASIC, 1)
        
        return patient_value >= required_value
    
    def _is_category_allowed(self, category: DataCategory, consent_level: ConsentLevel) -> bool:
        """Check if data category is allowed at consent level"""
        # Genetic data requires FULL consent
        if category == DataCategory.GENETIC:
            return consent_level == ConsentLevel.FULL
        
        # Mental health data requires at least STANDARD
        if category == DataCategory.MENTAL_HEALTH:
            return consent_level in [ConsentLevel.STANDARD, ConsentLevel.FULL]
        
        # Most categories allowed at STANDARD or higher
        return consent_level in [ConsentLevel.STANDARD, ConsentLevel.FULL]
    
    def _get_result_message(
        self,
        consented: Set[str],
        denied: Set[str],
        k_met: bool,
        categories: List[str]
    ) -> str:
        """Generate human-readable result message"""
        parts = []
        
        if consented:
            parts.append(f"{len(consented)} patients with valid consent")
        if denied:
            parts.append(f"{len(denied)} patients without consent")
        if not k_met:
            parts.append(f"k-anonymity requirement not met (minimum {self.MIN_K_ANONYMITY})")
        if categories:
            parts.append(f"allowed categories: {', '.join(categories)}")
        
        return "; ".join(parts) if parts else "Consent check complete"
    
    def _log_consent_check(
        self,
        cur,
        patient_ids: List[str],
        consented_count: int,
        denied_count: int,
        categories: List[DataCategory],
        purpose: str,
        requester_id: str,
        result: str
    ) -> str:
        """Log consent check for HIPAA audit"""
        import uuid
        audit_log_id = str(uuid.uuid4())
        
        try:
            cur.execute("""
                INSERT INTO ml_consent_audit_log (
                    audit_id,
                    check_type,
                    patient_count,
                    consented_count,
                    denied_count,
                    data_categories,
                    purpose,
                    requester_id,
                    result,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                audit_log_id,
                'consent_check',
                len(patient_ids),
                consented_count,
                denied_count,
                json.dumps([c.value if isinstance(c, DataCategory) else c for c in categories]),
                purpose,
                requester_id,
                result
            ))
            
        except Exception as e:
            # Table might not exist, create it
            try:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS ml_consent_audit_log (
                        id SERIAL PRIMARY KEY,
                        audit_id VARCHAR(50) UNIQUE NOT NULL,
                        check_type VARCHAR(50) NOT NULL,
                        patient_count INTEGER,
                        consented_count INTEGER,
                        denied_count INTEGER,
                        data_categories JSONB,
                        purpose VARCHAR(200),
                        requester_id VARCHAR(100),
                        result VARCHAR(50),
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    CREATE INDEX IF NOT EXISTS idx_consent_audit_created 
                        ON ml_consent_audit_log(created_at DESC);
                """)
                
                cur.execute("""
                    INSERT INTO ml_consent_audit_log (
                        audit_id, check_type, patient_count, consented_count,
                        denied_count, data_categories, purpose, requester_id, result
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    audit_log_id, 'consent_check', len(patient_ids),
                    consented_count, denied_count,
                    json.dumps([c.value if isinstance(c, DataCategory) else c for c in categories]),
                    purpose, requester_id, result
                ))
            except Exception as e2:
                logger.error(f"Error creating audit log: {e2}")
        
        return audit_log_id
    
    def enforce_in_pipeline(
        self,
        patient_ids: List[str],
        data_categories: List[DataCategory],
        purpose: str,
        requester_id: str
    ) -> List[str]:
        """
        Enforce consent in a data pipeline, returning only consented patient IDs.
        
        This is the main entry point for pipelines to use.
        Raises exception if k-anonymity is not met.
        
        Args:
            patient_ids: List of patient IDs to filter
            data_categories: Data categories being accessed
            purpose: Purpose of access
            requester_id: Requester ID for audit
            
        Returns:
            List of patient IDs that have valid consent
            
        Raises:
            ValueError: If k-anonymity requirement not met
        """
        result = self.check_patient_consent(
            patient_ids=patient_ids,
            data_categories=data_categories,
            purpose=purpose,
            requester_id=requester_id
        )
        
        if not result.k_anonymity_met:
            raise ValueError(
                f"K-anonymity requirement not met. Only {len(result.patient_ids_with_consent)} "
                f"patients have consent, minimum required is {self.MIN_K_ANONYMITY}"
            )
        
        if not result.allowed:
            raise ValueError(f"Consent check failed: {result.message}")
        
        return list(result.patient_ids_with_consent)
    
    def get_consented_categories(
        self,
        patient_id: str,
        requester_id: str
    ) -> List[DataCategory]:
        """Get list of data categories a patient has consented to"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            consent = self._get_patient_consent(cur, patient_id)
            
            cur.close()
            conn.close()
            
            if not consent:
                return []
            
            allowed = []
            consent_level = ConsentLevel(consent.get('consent_level', 'basic'))
            
            for category in DataCategory:
                if self._is_category_allowed(category, consent_level):
                    allowed.append(category)
            
            # Check specific consent flags
            if not consent.get('device_data_consent'):
                allowed = [c for c in allowed if c != DataCategory.DEVICE_DATA]
            if not consent.get('genetic_consent'):
                allowed = [c for c in allowed if c != DataCategory.GENETIC]
            
            return allowed
            
        except Exception as e:
            logger.error(f"Error getting consented categories: {e}")
            return []
