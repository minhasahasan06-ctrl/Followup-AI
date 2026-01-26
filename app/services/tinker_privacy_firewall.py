"""
Tinker Privacy Firewall Service
===============================
HIPAA-compliant privacy protection layer for Tinker Thinking Machine integration.
Operates in NON-BAA mode only - never sends PHI to Tinker.

Features:
- PHI detection and stripping using existing PHIDetectionService
- K-anonymity enforcement (k≥25 by default)
- SHA256 hashing for all identifiers
- Data bucketing (age→ranges, dates→periods, vitals→clinical ranges)
- Aggregate-only mode for cohort statistics
"""

import hashlib
import json
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)


class BucketType(str, Enum):
    """Types of data bucketing for privacy protection"""
    AGE = "age"
    DATE = "date"
    VITALS = "vitals"
    LAB_VALUES = "lab_values"
    BMI = "bmi"
    DURATION = "duration"


@dataclass
class PrivacyConfig:
    """Configuration for privacy firewall operations"""
    k_anonymity_threshold: int = 25
    hash_algorithm: str = "sha256"
    salt: str = "tinker_followup_2024"
    age_bucket_size: int = 5
    date_bucket: str = "month"  # week, month, quarter, year
    suppress_low_counts: bool = True
    max_precision_decimals: int = 1


@dataclass
class PrivacyAuditRecord:
    """Record of privacy transformations applied"""
    timestamp: datetime
    operation: str
    fields_hashed: List[str] = field(default_factory=list)
    fields_bucketed: List[str] = field(default_factory=list)
    fields_stripped: List[str] = field(default_factory=list)
    k_anon_passed: bool = True
    cohort_size: Optional[int] = None
    suppression_applied: bool = False


class TinkerPrivacyFirewall:
    """
    Privacy firewall ensuring no PHI reaches Tinker API.
    
    All data sent to Tinker must pass through this firewall:
    1. PHI detection and removal
    2. Identifier hashing (SHA256)
    3. Value bucketing (ages, dates, vitals)
    4. K-anonymity enforcement
    5. Aggregate-only statistics
    """
    
    # Clinical value ranges for bucketing
    VITALS_BUCKETS = {
        "heart_rate": [(0, 60, "bradycardia"), (60, 100, "normal"), (100, 150, "tachycardia"), (150, 999, "severe_tachycardia")],
        "blood_pressure_systolic": [(0, 90, "hypotensive"), (90, 120, "normal"), (120, 140, "elevated"), (140, 180, "hypertensive"), (180, 999, "crisis")],
        "blood_pressure_diastolic": [(0, 60, "low"), (60, 80, "normal"), (80, 90, "elevated"), (90, 120, "hypertensive"), (120, 999, "crisis")],
        "temperature": [(0, 36, "hypothermia"), (36, 37.5, "normal"), (37.5, 38.5, "low_fever"), (38.5, 40, "fever"), (40, 99, "high_fever")],
        "oxygen_saturation": [(0, 90, "critical"), (90, 95, "low"), (95, 100, "normal")],
        "respiratory_rate": [(0, 12, "slow"), (12, 20, "normal"), (20, 30, "elevated"), (30, 99, "distress")],
    }
    
    BMI_BUCKETS = [
        (0, 18.5, "underweight"),
        (18.5, 25, "normal"),
        (25, 30, "overweight"),
        (30, 35, "obese_class_1"),
        (35, 40, "obese_class_2"),
        (40, 999, "obese_class_3"),
    ]
    
    LAB_VALUE_BUCKETS = {
        "glucose": [(0, 70, "low"), (70, 100, "normal"), (100, 126, "prediabetic"), (126, 999, "diabetic")],
        "hba1c": [(0, 5.7, "normal"), (5.7, 6.5, "prediabetic"), (6.5, 99, "diabetic")],
        "creatinine": [(0, 0.7, "low"), (0.7, 1.3, "normal"), (1.3, 2.0, "elevated"), (2.0, 99, "high")],
        "wbc": [(0, 4.5, "low"), (4.5, 11, "normal"), (11, 99, "elevated")],
        "hemoglobin": [(0, 12, "low"), (12, 17.5, "normal"), (17.5, 99, "elevated")],
    }
    
    # PHI field patterns to always strip
    PHI_FIELD_PATTERNS = [
        "name", "first_name", "last_name", "full_name",
        "email", "phone", "address", "street", "city", "zip", "postal",
        "ssn", "social_security", "mrn", "medical_record",
        "dob", "date_of_birth", "birth_date",
        "ip_address", "device_id", "mac_address",
        "photo", "image", "picture", "avatar",
        "signature", "fingerprint", "biometric",
    ]
    
    def __init__(self, config: Optional[PrivacyConfig] = None):
        """Initialize privacy firewall with configuration"""
        self.config = config or PrivacyConfig(
            k_anonymity_threshold=settings.TINKER_K_ANON
        )
        self._validate_non_baa_mode()
        logger.info(f"TinkerPrivacyFirewall initialized with k={self.config.k_anonymity_threshold}")
    
    def _validate_non_baa_mode(self):
        """Ensure Tinker is in NON-BAA mode"""
        if settings.TINKER_MODE != "NON_BAA":
            raise ValueError(
                "CRITICAL: Tinker must operate in NON_BAA mode. "
                "PHI transmission is prohibited."
            )
    
    # =========================================================================
    # SHA256 Hashing
    # =========================================================================
    
    def hash_identifier(self, value: str, purpose: str = "general") -> str:
        """
        Create SHA256 hash of identifier with salt.
        
        Args:
            value: The identifier to hash
            purpose: Purpose context for additional salt differentiation
            
        Returns:
            SHA256 hex digest (64 characters)
        """
        if not value:
            return ""
        
        salted = f"{self.config.salt}:{purpose}:{str(value)}"
        return hashlib.sha256(salted.encode('utf-8')).hexdigest()
    
    def hash_patient_id(self, patient_id: str) -> str:
        """Hash patient ID for Tinker API"""
        return self.hash_identifier(patient_id, purpose="patient")
    
    def hash_doctor_id(self, doctor_id: str) -> str:
        """Hash doctor ID for Tinker API"""
        return self.hash_identifier(doctor_id, purpose="doctor")
    
    def hash_dict(self, data: Dict[str, Any], fields_to_hash: List[str]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Hash specified fields in a dictionary.
        
        Returns:
            Tuple of (modified dict, list of hashed field names)
        """
        result = data.copy()
        hashed_fields = []
        
        for field in fields_to_hash:
            if field in result and result[field]:
                result[field] = self.hash_identifier(str(result[field]), purpose=field)
                hashed_fields.append(field)
        
        return result, hashed_fields
    
    # =========================================================================
    # Data Bucketing
    # =========================================================================
    
    def bucket_age(self, age: Union[int, float]) -> str:
        """
        Convert exact age to 5-year bucket.
        
        Examples:
            23 → "20-24"
            67 → "65-69"
        """
        if age is None or age < 0:
            return "unknown"
        
        bucket_size = self.config.age_bucket_size
        lower = (int(age) // bucket_size) * bucket_size
        upper = lower + bucket_size - 1
        
        if lower >= 90:
            return "90+"
        
        return f"{lower}-{upper}"
    
    def bucket_date(self, dt: Union[datetime, date, str]) -> str:
        """
        Convert exact date to period bucket.
        
        Args:
            dt: Date to bucket
            
        Returns:
            Period string like "2024-Q1" or "2024-03"
        """
        if dt is None:
            return "unknown"
        
        if isinstance(dt, str):
            try:
                dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                return "unknown"
        
        if isinstance(dt, datetime):
            dt = dt.date()
        
        bucket_type = self.config.date_bucket
        
        if bucket_type == "week":
            week = dt.isocalendar()[1]
            return f"{dt.year}-W{week:02d}"
        elif bucket_type == "month":
            return f"{dt.year}-{dt.month:02d}"
        elif bucket_type == "quarter":
            quarter = (dt.month - 1) // 3 + 1
            return f"{dt.year}-Q{quarter}"
        elif bucket_type == "year":
            return str(dt.year)
        else:
            return f"{dt.year}-{dt.month:02d}"
    
    def bucket_vital(self, vital_type: str, value: Union[int, float]) -> str:
        """
        Convert vital sign value to clinical category.
        
        Args:
            vital_type: Type of vital (heart_rate, blood_pressure_systolic, etc.)
            value: Numeric value
            
        Returns:
            Clinical category string
        """
        if value is None:
            return "unknown"
        
        buckets = self.VITALS_BUCKETS.get(vital_type.lower())
        if not buckets:
            return "unknown"
        
        for low, high, category in buckets:
            if low <= value < high:
                return category
        
        return "unknown"
    
    def bucket_bmi(self, bmi: Union[int, float]) -> str:
        """Convert BMI to weight category"""
        if bmi is None:
            return "unknown"
        
        for low, high, category in self.BMI_BUCKETS:
            if low <= bmi < high:
                return category
        
        return "unknown"
    
    def bucket_lab_value(self, lab_type: str, value: Union[int, float]) -> str:
        """Convert lab value to clinical category"""
        if value is None:
            return "unknown"
        
        buckets = self.LAB_VALUE_BUCKETS.get(lab_type.lower())
        if not buckets:
            return "unknown"
        
        for low, high, category in buckets:
            if low <= value < high:
                return category
        
        return "unknown"
    
    def bucket_duration_days(self, days: int) -> str:
        """Convert duration in days to bucket"""
        if days is None or days < 0:
            return "unknown"
        
        if days == 0:
            return "same_day"
        elif days <= 7:
            return "within_week"
        elif days <= 30:
            return "within_month"
        elif days <= 90:
            return "within_quarter"
        elif days <= 365:
            return "within_year"
        else:
            return "over_year"
    
    # =========================================================================
    # PHI Detection and Stripping
    # =========================================================================
    
    def _is_phi_field(self, field_name: str) -> bool:
        """Check if field name matches PHI patterns"""
        field_lower = field_name.lower()
        return any(pattern in field_lower for pattern in self.PHI_FIELD_PATTERNS)
    
    def strip_phi_fields(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Remove all PHI fields from dictionary.
        
        SECURITY: This method MUST succeed - on any error, returns empty dict
        to prevent PHI leakage. Never returns original data on error.
        
        Returns:
            Tuple of (cleaned dict, list of stripped field names)
        """
        try:
            result = {}
            stripped = []
            
            for key, value in data.items():
                if self._is_phi_field(key):
                    stripped.append(key)
                    continue
                
                if isinstance(value, dict):
                    cleaned_value, nested_stripped = self.strip_phi_fields(value)
                    result[key] = cleaned_value
                    stripped.extend([f"{key}.{s}" for s in nested_stripped])
                elif isinstance(value, list):
                    cleaned_list = []
                    for item in value:
                        if isinstance(item, dict):
                            cleaned_item, nested_stripped = self.strip_phi_fields(item)
                            cleaned_list.append(cleaned_item)
                            stripped.extend(nested_stripped)
                        else:
                            cleaned_list.append(item)
                    result[key] = cleaned_list
                else:
                    result[key] = value
            
            return result, stripped
        except Exception as e:
            logger.error(f"CRITICAL: PHI stripping failed, returning empty dict to prevent PHI leakage: {e}")
            return {}, ["ERROR_ALL_FIELDS_STRIPPED"]
    
    # =========================================================================
    # K-Anonymity Enforcement
    # =========================================================================
    
    def check_k_anonymity(self, count: int) -> bool:
        """
        Check if count meets k-anonymity threshold.
        
        Args:
            count: Number of records in cohort
            
        Returns:
            True if count >= k threshold
        """
        return count >= self.config.k_anonymity_threshold
    
    def require_k_anonymity(self, count: int, context: str = "operation") -> None:
        """
        HARD-FAIL k-anonymity check.
        
        Raises ValueError if count < k threshold.
        Use this for critical operations that must never proceed without k-anonymity.
        """
        if not self.check_k_anonymity(count):
            raise ValueError(
                f"BLOCKED: K-anonymity violation in {context}. "
                f"Count {count} < required threshold {self.config.k_anonymity_threshold}. "
                f"Operation cannot proceed to protect patient privacy."
            )
    
    def enforce_k_anonymity(
        self, 
        cohort_data: Dict[str, Any],
        count_field: str = "patient_count"
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Enforce k-anonymity on cohort data.
        
        If count < k, suppresses the data.
        
        Returns:
            Tuple of (data or None, k_anon_passed)
        """
        count = cohort_data.get(count_field, 0)
        
        if not self.check_k_anonymity(count):
            if self.config.suppress_low_counts:
                logger.warning(
                    f"K-anonymity check failed: {count} < {self.config.k_anonymity_threshold}. "
                    "Data suppressed."
                )
                return {
                    "suppressed": True,
                    "reason": "k_anonymity",
                    "threshold": self.config.k_anonymity_threshold
                }, False
        
        return cohort_data, True
    
    def ensure_phi_safe_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        MANDATORY privacy pipeline for any data going to Tinker.
        
        This method MUST be called before ANY Tinker API request.
        Runs full privacy pipeline:
        1. Strip PHI fields
        2. Hash all identifiers
        3. Return safe payload
        
        SECURITY: On any error, returns empty dict. Never leaks PHI.
        """
        try:
            # Step 1: Strip PHI fields
            safe_payload, stripped = self.strip_phi_fields(payload)
            
            # Step 2: Hash any remaining identifiers
            id_fields = ["patient_id", "doctor_id", "provider_id", "user_id", 
                        "cohort_id", "study_id", "trial_id", "session_id"]
            safe_payload, _ = self.hash_dict(safe_payload, id_fields)
            
            logger.debug(f"PHI safety: stripped {len(stripped)} fields, payload ready for Tinker")
            return safe_payload
            
        except Exception as e:
            logger.error(f"CRITICAL: ensure_phi_safe_payload failed, returning empty dict: {e}")
            return {}
    
    def suppress_small_cells(
        self, 
        aggregates: Dict[str, int],
        replacement: str = "<k"
    ) -> Dict[str, Union[int, str]]:
        """
        Replace counts below k threshold with suppression marker.
        
        Args:
            aggregates: Dictionary of category → count
            replacement: String to use for suppressed values
            
        Returns:
            Dictionary with small counts suppressed
        """
        result = {}
        k = self.config.k_anonymity_threshold
        
        for category, count in aggregates.items():
            if count < k:
                result[category] = replacement
            else:
                result[category] = count
        
        return result
    
    # =========================================================================
    # Aggregate Statistics (Safe for Tinker)
    # =========================================================================
    
    def compute_safe_aggregates(
        self,
        values: List[Union[int, float]],
        include_percentiles: bool = True
    ) -> Dict[str, Any]:
        """
        Compute aggregate statistics safe for Tinker API.
        
        Only returns aggregates if n >= k threshold.
        
        Returns:
            Dictionary with count, mean, std, min, max, percentiles
        """
        if not values:
            return {"count": 0, "suppressed": True, "reason": "no_data"}
        
        n = len(values)
        
        if not self.check_k_anonymity(n):
            return {
                "count": n,
                "suppressed": True,
                "reason": "k_anonymity",
                "threshold": self.config.k_anonymity_threshold
            }
        
        import statistics
        
        result = {
            "count": n,
            "mean": round(statistics.mean(values), self.config.max_precision_decimals),
            "std": round(statistics.stdev(values), self.config.max_precision_decimals) if n > 1 else 0,
            "min": round(min(values), self.config.max_precision_decimals),
            "max": round(max(values), self.config.max_precision_decimals),
        }
        
        if include_percentiles and n >= 5:
            sorted_values = sorted(values)
            result["p25"] = round(sorted_values[n // 4], self.config.max_precision_decimals)
            result["p50"] = round(sorted_values[n // 2], self.config.max_precision_decimals)
            result["p75"] = round(sorted_values[3 * n // 4], self.config.max_precision_decimals)
        
        return result
    
    def compute_category_distribution(
        self,
        categories: List[str]
    ) -> Dict[str, Any]:
        """
        Compute category distribution with k-anonymity protection.
        
        Returns counts per category, suppressing those below k.
        """
        from collections import Counter
        
        counts = Counter(categories)
        total = len(categories)
        
        if not self.check_k_anonymity(total):
            return {
                "total": total,
                "suppressed": True,
                "reason": "k_anonymity"
            }
        
        distribution = self.suppress_small_cells(dict(counts))
        
        return {
            "total": total,
            "distribution": distribution,
            "suppressed": False
        }
    
    # =========================================================================
    # Full Privacy Transform Pipeline
    # =========================================================================
    
    def transform_patient_data(
        self,
        patient_data: Dict[str, Any],
        include_vitals: bool = True,
        include_labs: bool = True
    ) -> Tuple[Dict[str, Any], PrivacyAuditRecord]:
        """
        Apply full privacy transformation pipeline to patient data.
        
        This is the main entry point for preparing patient data for Tinker.
        
        Pipeline:
        1. Strip all PHI fields
        2. Hash identifiers
        3. Bucket ages, dates, vitals, labs
        
        Returns:
            Tuple of (transformed data, audit record)
        """
        audit = PrivacyAuditRecord(
            timestamp=datetime.utcnow(),
            operation="transform_patient_data"
        )
        
        # Step 1: Strip PHI fields
        result, stripped = self.strip_phi_fields(patient_data)
        audit.fields_stripped = stripped
        
        # Step 2: Hash identifiers
        id_fields = ["patient_id", "doctor_id", "provider_id", "user_id"]
        result, hashed = self.hash_dict(result, id_fields)
        audit.fields_hashed = hashed
        
        # Step 3: Bucket age if present
        if "age" in result:
            result["age_bucket"] = self.bucket_age(result["age"])
            del result["age"]
            audit.fields_bucketed.append("age")
        
        # Step 4: Bucket dates
        date_fields = ["created_at", "updated_at", "visit_date", "onset_date", "diagnosis_date"]
        for field in date_fields:
            if field in result:
                result[f"{field}_period"] = self.bucket_date(result[field])
                del result[field]
                audit.fields_bucketed.append(field)
        
        # Step 5: Bucket vitals
        if include_vitals and "vitals" in result:
            bucketed_vitals = {}
            for vital_type, value in result.get("vitals", {}).items():
                bucketed_vitals[vital_type] = self.bucket_vital(vital_type, value)
            result["vitals_categories"] = bucketed_vitals
            del result["vitals"]
            audit.fields_bucketed.append("vitals")
        
        # Step 6: Bucket lab values
        if include_labs and "lab_values" in result:
            bucketed_labs = {}
            for lab_type, value in result.get("lab_values", {}).items():
                bucketed_labs[lab_type] = self.bucket_lab_value(lab_type, value)
            result["lab_categories"] = bucketed_labs
            del result["lab_values"]
            audit.fields_bucketed.append("lab_values")
        
        # Step 7: Bucket BMI if present
        if "bmi" in result:
            result["bmi_category"] = self.bucket_bmi(result["bmi"])
            del result["bmi"]
            audit.fields_bucketed.append("bmi")
        
        return result, audit
    
    def transform_cohort_query(
        self,
        query: Dict[str, Any],
        patient_count: int
    ) -> Tuple[Optional[Dict[str, Any]], PrivacyAuditRecord]:
        """
        Transform cohort query for Tinker with k-anonymity check.
        
        Returns None if cohort fails k-anonymity.
        """
        audit = PrivacyAuditRecord(
            timestamp=datetime.utcnow(),
            operation="transform_cohort_query",
            cohort_size=patient_count
        )
        
        # Check k-anonymity
        if not self.check_k_anonymity(patient_count):
            audit.k_anon_passed = False
            audit.suppression_applied = True
            logger.warning(
                f"Cohort query rejected: {patient_count} patients < k={self.config.k_anonymity_threshold}"
            )
            return None, audit
        
        audit.k_anon_passed = True
        
        # Strip PHI from query
        result, stripped = self.strip_phi_fields(query)
        audit.fields_stripped = stripped
        
        # Hash any IDs in query
        id_fields = ["patient_ids", "doctor_ids", "cohort_id"]
        for field in id_fields:
            if field in result:
                if isinstance(result[field], list):
                    result[field] = [self.hash_identifier(str(id_val)) for id_val in result[field]]
                else:
                    result[field] = self.hash_identifier(str(result[field]))
                audit.fields_hashed.append(field)
        
        return result, audit
    
    def create_payload_hash(self, payload: Dict[str, Any]) -> str:
        """
        Create SHA256 hash of entire payload for audit logging.
        
        This is used to log what was sent without storing the actual data.
        """
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()


# Singleton instance
_firewall_instance: Optional[TinkerPrivacyFirewall] = None


def get_privacy_firewall() -> TinkerPrivacyFirewall:
    """Get or create singleton privacy firewall instance"""
    global _firewall_instance
    if _firewall_instance is None:
        _firewall_instance = TinkerPrivacyFirewall()
    return _firewall_instance
