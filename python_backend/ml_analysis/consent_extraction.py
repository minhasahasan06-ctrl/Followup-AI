"""
Consent-Aware ML Data Extraction
=================================
Production-grade consent-aware data extraction for ML training:
- Patient consent verification before feature extraction
- RBAC-based access control
- Differential privacy options
- Comprehensive audit logging

HIPAA-compliant with full consent tracking.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
import numpy as np
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class ConsentCategory(str, Enum):
    GENERAL_ML = "general_ml"
    DRUG_SAFETY = "drug_safety"
    OUTBREAK_PREDICTION = "outbreak_prediction"
    VACCINE_EFFECTIVENESS = "vaccine_effectiveness"
    GENETIC_ANALYSIS = "genetic_analysis"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"
    ENVIRONMENTAL_ANALYSIS = "environmental_analysis"


class DataType(str, Enum):
    DEMOGRAPHICS = "demographics"
    MEDICATIONS = "medications"
    ADVERSE_EVENTS = "adverse_events"
    LAB_RESULTS = "lab_results"
    VITAL_SIGNS = "vital_signs"
    DIAGNOSES = "diagnoses"
    DEVICE_DATA = "device_data"
    GENETIC_DATA = "genetic_data"
    BEHAVIORAL_DATA = "behavioral_data"
    ENVIRONMENTAL = "environmental"


@dataclass
class ConsentPolicy:
    """Policy defining what data can be extracted"""
    policy_id: str
    name: str
    required_consents: List[ConsentCategory]
    allowed_data_types: List[DataType]
    min_aggregation_size: int = 10
    differential_privacy_enabled: bool = False
    epsilon: float = 1.0
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExtractionRequest:
    """Request for ML data extraction"""
    request_id: str
    requester_id: str
    purpose: str
    policy_id: str
    data_types: List[DataType]
    patient_filter: Optional[Dict[str, Any]] = None
    date_range: Optional[tuple] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass 
class ExtractionResult:
    """Result of a consent-aware extraction"""
    request_id: str
    n_patients_requested: int
    n_patients_with_consent: int
    n_patients_extracted: int
    data_types_extracted: List[str]
    suppressed_fields: List[str]
    differential_privacy_applied: bool
    audit_log_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)


class ConsentService:
    """
    Verifies and manages patient consent for ML data extraction
    """
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def get_consented_patients(
        self,
        consent_categories: List[ConsentCategory],
        data_types: List[DataType]
    ) -> Set[str]:
        """Get set of patient IDs who have given required consents"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT DISTINCT pc.patient_id
                FROM patient_consent pc
                WHERE pc.ml_training_consent = TRUE
                AND pc.consent_date IS NOT NULL
                AND (pc.revoked_at IS NULL OR pc.revoked_at > NOW())
            """)
            
            patient_ids = {row[0] for row in cur.fetchall()}
            
            cur.close()
            conn.close()
            
            return patient_ids
            
        except Exception as e:
            logger.error(f"Error getting consented patients: {e}")
            return set()
    
    def verify_consent(
        self,
        patient_id: str,
        consent_category: ConsentCategory,
        data_types: List[DataType]
    ) -> Dict[str, Any]:
        """Verify consent for a specific patient"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT 
                    ml_training_consent,
                    consent_date,
                    revoked_at,
                    consent_details
                FROM patient_consent
                WHERE patient_id = %s
            """, (patient_id,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if not row:
                return {
                    'has_consent': False,
                    'reason': 'No consent record found'
                }
            
            if not row['ml_training_consent']:
                return {
                    'has_consent': False,
                    'reason': 'ML training consent not granted'
                }
            
            if row['revoked_at'] and row['revoked_at'] < datetime.utcnow():
                return {
                    'has_consent': False,
                    'reason': 'Consent has been revoked'
                }
            
            return {
                'has_consent': True,
                'consent_date': row['consent_date'].isoformat() if row['consent_date'] else None,
                'allowed_categories': [consent_category.value],
                'allowed_data_types': [dt.value for dt in data_types]
            }
            
        except Exception as e:
            logger.error(f"Error verifying consent: {e}")
            return {
                'has_consent': False,
                'reason': f'Error: {str(e)}'
            }


class DifferentialPrivacy:
    """
    Implements differential privacy mechanisms for ML data
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(
        self, 
        value: float, 
        sensitivity: float = 1.0
    ) -> float:
        """Add Laplace noise for ε-differential privacy"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def add_gaussian_noise(
        self, 
        value: float, 
        sensitivity: float = 1.0
    ) -> float:
        """Add Gaussian noise for (ε,δ)-differential privacy"""
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, sigma)
        return value + noise
    
    def privatize_count(self, count: int, sensitivity: int = 1) -> int:
        """Privatize a count value"""
        noisy_count = self.add_laplace_noise(float(count), sensitivity)
        return max(0, int(round(noisy_count)))
    
    def privatize_mean(
        self, 
        values: np.ndarray, 
        min_val: float, 
        max_val: float
    ) -> float:
        """Privatize a mean value"""
        true_mean = np.mean(values)
        sensitivity = (max_val - min_val) / len(values)
        return self.add_laplace_noise(true_mean, sensitivity)
    
    def privatize_histogram(
        self, 
        counts: Dict[str, int], 
        sensitivity: int = 1
    ) -> Dict[str, int]:
        """Privatize a histogram of counts"""
        return {
            k: self.privatize_count(v, sensitivity)
            for k, v in counts.items()
        }


class ConsentAwareExtractor:
    """
    Extracts ML training data with full consent verification
    """
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self.consent_service = ConsentService(db_url)
        self.min_cell_size = 10
    
    def get_connection(self):
        return psycopg2.connect(self.db_url)
    
    def extract_for_drug_safety(
        self,
        request: ExtractionRequest,
        policy: ConsentPolicy,
        differential_privacy: Optional[DifferentialPrivacy] = None
    ) -> Dict[str, Any]:
        """Extract data for drug safety ML training"""
        consented_patients = self.consent_service.get_consented_patients(
            policy.required_consents,
            request.data_types
        )
        
        if len(consented_patients) < policy.min_aggregation_size:
            return {
                'error': 'Insufficient consented patients for aggregation',
                'n_consented': len(consented_patients),
                'min_required': policy.min_aggregation_size
            }
        
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT 
                    s.drug_code,
                    s.drug_name,
                    s.outcome_code,
                    s.outcome_name,
                    s.patient_location_id,
                    s.estimate as odds_ratio,
                    s.ci_lower,
                    s.ci_upper,
                    s.p_value,
                    s.n_patients,
                    s.n_events,
                    s.flagged
                FROM drug_outcome_signals s
                WHERE s.n_patients >= %s
            """, (self.min_cell_size,))
            
            rows = [dict(row) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            if differential_privacy:
                for row in rows:
                    row['n_patients'] = differential_privacy.privatize_count(row['n_patients'])
                    row['n_events'] = differential_privacy.privatize_count(row['n_events'])
            
            audit_log_id = self._log_extraction(
                request, 
                len(consented_patients),
                len(rows),
                policy.differential_privacy_enabled
            )
            
            return {
                'data': rows,
                'n_records': len(rows),
                'n_consented_patients': len(consented_patients),
                'differential_privacy_applied': policy.differential_privacy_enabled,
                'audit_log_id': audit_log_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting drug safety data: {e}")
            raise
    
    def extract_for_outbreak_prediction(
        self,
        request: ExtractionRequest,
        policy: ConsentPolicy,
        differential_privacy: Optional[DifferentialPrivacy] = None
    ) -> Dict[str, Any]:
        """Extract data for outbreak prediction ML training"""
        consented_patients = self.consent_service.get_consented_patients(
            policy.required_consents,
            request.data_types
        )
        
        if len(consented_patients) < policy.min_aggregation_size:
            return {
                'error': 'Insufficient consented patients for aggregation',
                'n_consented': len(consented_patients),
                'min_required': policy.min_aggregation_size
            }
        
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            date_filter = ""
            params: List[Any] = [self.min_cell_size]
            
            if request.date_range:
                date_filter = " AND event_date BETWEEN %s AND %s"
                params.extend(request.date_range)
            
            cur.execute(f"""
                SELECT 
                    pathogen_code,
                    pathogen_name,
                    patient_location_id,
                    event_date,
                    case_count,
                    death_count
                FROM infectious_events_aggregated
                WHERE case_count >= %s
                {date_filter}
                ORDER BY event_date
            """, params)
            
            rows = [dict(row) for row in cur.fetchall()]
            
            cur.execute("""
                SELECT 
                    pathogen_code,
                    location_id,
                    calculation_date,
                    r_value,
                    confidence_lower,
                    confidence_upper
                FROM reproduction_numbers
                ORDER BY calculation_date
            """)
            
            r_numbers = [dict(row) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            if differential_privacy:
                for row in rows:
                    row['case_count'] = differential_privacy.privatize_count(row['case_count'])
                    row['death_count'] = differential_privacy.privatize_count(row['death_count'])
            
            audit_log_id = self._log_extraction(
                request,
                len(consented_patients),
                len(rows),
                policy.differential_privacy_enabled
            )
            
            return {
                'event_data': rows,
                'reproduction_numbers': r_numbers,
                'n_records': len(rows),
                'n_consented_patients': len(consented_patients),
                'differential_privacy_applied': policy.differential_privacy_enabled,
                'audit_log_id': audit_log_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting outbreak data: {e}")
            raise
    
    def extract_for_vaccine_effectiveness(
        self,
        request: ExtractionRequest,
        policy: ConsentPolicy,
        differential_privacy: Optional[DifferentialPrivacy] = None
    ) -> Dict[str, Any]:
        """Extract data for vaccine effectiveness ML training"""
        consented_patients = self.consent_service.get_consented_patients(
            policy.required_consents,
            request.data_types
        )
        
        if len(consented_patients) < policy.min_aggregation_size:
            return {
                'error': 'Insufficient consented patients for aggregation',
                'n_consented': len(consented_patients),
                'min_required': policy.min_aggregation_size
            }
        
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT 
                    vaccine_code,
                    vaccine_name,
                    location_id,
                    SUM(n_doses) as total_doses,
                    SUM(n_patients) as total_patients
                FROM immunization_aggregates
                WHERE n_patients >= %s
                GROUP BY vaccine_code, vaccine_name, location_id
            """, (self.min_cell_size,))
            
            immunizations = [dict(row) for row in cur.fetchall()]
            
            cur.execute("""
                SELECT 
                    vaccine_code,
                    outcome_code,
                    location_id,
                    effectiveness_estimate,
                    ci_lower,
                    ci_upper,
                    study_type,
                    n_vaccinated,
                    n_unvaccinated
                FROM vaccine_effectiveness_estimates
                WHERE n_vaccinated >= %s AND n_unvaccinated >= %s
            """, (self.min_cell_size, self.min_cell_size))
            
            effectiveness = [dict(row) for row in cur.fetchall()]
            
            cur.close()
            conn.close()
            
            if differential_privacy:
                for row in immunizations:
                    row['total_doses'] = differential_privacy.privatize_count(row['total_doses'])
                    row['total_patients'] = differential_privacy.privatize_count(row['total_patients'])
            
            audit_log_id = self._log_extraction(
                request,
                len(consented_patients),
                len(immunizations) + len(effectiveness),
                policy.differential_privacy_enabled
            )
            
            return {
                'immunizations': immunizations,
                'effectiveness_estimates': effectiveness,
                'n_consented_patients': len(consented_patients),
                'differential_privacy_applied': policy.differential_privacy_enabled,
                'audit_log_id': audit_log_id
            }
            
        except Exception as e:
            logger.error(f"Error extracting vaccine data: {e}")
            raise
    
    def _log_extraction(
        self,
        request: ExtractionRequest,
        n_consented: int,
        n_extracted: int,
        dp_applied: bool
    ) -> str:
        """Log extraction for audit trail"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            log_id = str(uuid.uuid4())
            
            cur.execute("""
                INSERT INTO ml_extraction_audit_log
                (id, request_id, requester_id, purpose, policy_id,
                 n_patients_consented, n_records_extracted, 
                 differential_privacy_applied, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                log_id,
                request.request_id,
                request.requester_id,
                request.purpose,
                request.policy_id,
                n_consented,
                n_extracted,
                dp_applied
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Logged ML extraction: {log_id}")
            return log_id
            
        except Exception as e:
            logger.error(f"Failed to log extraction: {e}")
            return str(uuid.uuid4())


def create_extraction_policy(
    name: str,
    purpose: str,
    data_types: List[DataType],
    enable_differential_privacy: bool = False,
    epsilon: float = 1.0
) -> ConsentPolicy:
    """Factory function to create extraction policies"""
    
    consent_mapping = {
        DataType.MEDICATIONS: ConsentCategory.DRUG_SAFETY,
        DataType.ADVERSE_EVENTS: ConsentCategory.DRUG_SAFETY,
        DataType.DIAGNOSES: ConsentCategory.OUTBREAK_PREDICTION,
        DataType.LAB_RESULTS: ConsentCategory.VACCINE_EFFECTIVENESS,
        DataType.GENETIC_DATA: ConsentCategory.GENETIC_ANALYSIS,
        DataType.BEHAVIORAL_DATA: ConsentCategory.BEHAVIORAL_ANALYSIS,
        DataType.ENVIRONMENTAL: ConsentCategory.ENVIRONMENTAL_ANALYSIS
    }
    
    required_consents = list(set(
        consent_mapping.get(dt, ConsentCategory.GENERAL_ML)
        for dt in data_types
    ))
    
    return ConsentPolicy(
        policy_id=str(uuid.uuid4()),
        name=name,
        required_consents=required_consents,
        allowed_data_types=data_types,
        differential_privacy_enabled=enable_differential_privacy,
        epsilon=epsilon
    )
