"""
Feature Packets Service (Phase B.22-B.23)
=========================================
Builds privacy-safe feature packets for Tinker Thinking Machine.

Tasks Covered:
- B.22: build_patient_packet(patient_id)
- B.23: Ensure packet returns ONLY allowed fields
"""

import hashlib
import logging
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set

from app.config import settings
from app.services.privacy_firewall import (
    FORBIDDEN_KEYS,
    ALLOWED_PATIENT_QUESTIONS,
    k_anon_check,
    phi_scan,
)

logger = logging.getLogger(__name__)


# =============================================================================
# B.23: Strict allowed fields for feature packets
# =============================================================================
ALLOWED_PACKET_FIELDS: Set[str] = {
    "age_bucket",
    "condition_codes",
    "risk_bucket",
    "trend_flags",
    "engagement_bucket",
    "missingness_bucket",
    "adherence_bucket",
}


# =============================================================================
# Value bucketing utilities
# =============================================================================
def _bucket_age(age: Optional[int]) -> str:
    """Convert age to 5-year bucket"""
    if age is None or age < 0:
        return "unknown"
    
    bucket_size = 5
    lower = (age // bucket_size) * bucket_size
    upper = lower + bucket_size - 1
    
    if lower >= 90:
        return "90+"
    
    return f"{lower}-{upper}"


def _bucket_risk_score(score: Optional[float]) -> str:
    """Convert risk score (0-1) to bucket"""
    if score is None:
        return "unknown"
    
    if score < 0.2:
        return "low"
    elif score < 0.5:
        return "moderate"
    elif score < 0.8:
        return "high"
    else:
        return "critical"


def _bucket_engagement(engagement_pct: Optional[float]) -> str:
    """Convert engagement percentage to bucket"""
    if engagement_pct is None:
        return "unknown"
    
    if engagement_pct >= 80:
        return "high"
    elif engagement_pct >= 50:
        return "moderate"
    elif engagement_pct >= 20:
        return "low"
    else:
        return "minimal"


def _bucket_missingness(missing_pct: Optional[float]) -> str:
    """Convert missingness percentage to bucket"""
    if missing_pct is None:
        return "unknown"
    
    if missing_pct <= 5:
        return "complete"
    elif missing_pct <= 20:
        return "minor_gaps"
    elif missing_pct <= 50:
        return "significant_gaps"
    else:
        return "sparse"


def _bucket_adherence(adherence_pct: Optional[float]) -> str:
    """Convert medication adherence percentage to bucket"""
    if adherence_pct is None:
        return "unknown"
    
    if adherence_pct >= 90:
        return "excellent"
    elif adherence_pct >= 75:
        return "good"
    elif adherence_pct >= 50:
        return "fair"
    else:
        return "poor"


def _calculate_trend_flags(data: Dict[str, Any]) -> List[str]:
    """Calculate trend flags from patient data"""
    flags = []
    
    # These are safe categorical flags, not PHI
    if data.get("pain_trend") == "increasing":
        flags.append("pain_increasing")
    elif data.get("pain_trend") == "decreasing":
        flags.append("pain_improving")
    
    if data.get("fatigue_trend") == "increasing":
        flags.append("fatigue_increasing")
    
    if data.get("mood_trend") == "declining":
        flags.append("mood_declining")
    elif data.get("mood_trend") == "improving":
        flags.append("mood_improving")
    
    if data.get("weight_change") == "significant_loss":
        flags.append("weight_loss_flagged")
    elif data.get("weight_change") == "significant_gain":
        flags.append("weight_gain_flagged")
    
    if data.get("vitals_concern"):
        flags.append("vitals_need_review")
    
    if data.get("missed_checkins", 0) >= 3:
        flags.append("engagement_declining")
    
    return flags


def _extract_condition_codes(conditions: Optional[List[str]]) -> List[str]:
    """Extract safe condition codes (ICD-10 category level only)"""
    if not conditions:
        return []
    
    safe_codes = []
    for code in conditions:
        if not isinstance(code, str):
            continue
        
        # Only use ICD-10 category level (first 3 characters)
        # This prevents identification via specific codes
        category = code[:3].upper() if len(code) >= 3 else code.upper()
        if category and not phi_scan(category):
            safe_codes.append(category)
    
    return list(set(safe_codes))  # Deduplicate


# =============================================================================
# B.22: build_patient_packet(patient_id)
# =============================================================================
def build_patient_packet(
    patient_id: str,
    patient_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    B.22: Build a privacy-safe feature packet for a patient.
    
    This function:
    1. Hashes the patient_id (never includes raw ID)
    2. Extracts only allowed bucketed features
    3. Ensures B.23 compliance: returns ONLY allowed fields
    
    Args:
        patient_id: Raw patient ID (will be hashed, not included)
        patient_data: Optional dict with patient metrics/data for feature extraction
                     If None, returns packet with default/unknown values
        
    Returns:
        Feature packet with ONLY these fields:
        - age_bucket
        - condition_codes
        - risk_bucket
        - trend_flags
        - engagement_bucket
        - missingness_bucket
        - adherence_bucket
    """
    # Hash patient_id for internal tracking (not included in packet)
    patient_hash = _compute_patient_hash(patient_id)
    
    # Default packet with unknown values
    packet: Dict[str, Any] = {
        "age_bucket": "unknown",
        "condition_codes": [],
        "risk_bucket": "unknown",
        "trend_flags": [],
        "engagement_bucket": "unknown",
        "missingness_bucket": "unknown",
        "adherence_bucket": "unknown",
    }
    
    if not patient_data:
        logger.debug(f"No patient data provided, returning default packet")
        return _enforce_allowed_fields(packet)
    
    # Extract and bucket each field
    
    # Age bucket
    age = patient_data.get("age") or patient_data.get("patient_age")
    if age is not None:
        try:
            packet["age_bucket"] = _bucket_age(int(age))
        except (ValueError, TypeError):
            pass
    
    # Condition codes (category level only)
    conditions = patient_data.get("condition_codes") or patient_data.get("diagnosis_codes") or patient_data.get("conditions")
    if conditions:
        packet["condition_codes"] = _extract_condition_codes(conditions)
    
    # Risk bucket
    risk_score = patient_data.get("risk_score") or patient_data.get("overall_risk")
    if risk_score is not None:
        try:
            packet["risk_bucket"] = _bucket_risk_score(float(risk_score))
        except (ValueError, TypeError):
            pass
    
    # Trend flags
    packet["trend_flags"] = _calculate_trend_flags(patient_data)
    
    # Engagement bucket
    engagement = patient_data.get("engagement_rate") or patient_data.get("engagement_pct")
    if engagement is not None:
        try:
            packet["engagement_bucket"] = _bucket_engagement(float(engagement))
        except (ValueError, TypeError):
            pass
    
    # Missingness bucket
    missingness = patient_data.get("data_missingness") or patient_data.get("missing_pct")
    if missingness is not None:
        try:
            packet["missingness_bucket"] = _bucket_missingness(float(missingness))
        except (ValueError, TypeError):
            pass
    
    # Adherence bucket
    adherence = patient_data.get("medication_adherence") or patient_data.get("adherence_rate")
    if adherence is not None:
        try:
            packet["adherence_bucket"] = _bucket_adherence(float(adherence))
        except (ValueError, TypeError):
            pass
    
    # B.23: Strict enforcement - return ONLY allowed fields
    return _enforce_allowed_fields(packet)


def _compute_patient_hash(patient_id: str) -> str:
    """Compute SHA256 hash of patient ID with salt"""
    salt = getattr(settings, 'TINKER_HASH_SALT', 'tinker_followup_2024')
    salted = f"{salt}:patient:{patient_id}"
    return hashlib.sha256(salted.encode('utf-8')).hexdigest()


def _enforce_allowed_fields(packet: Dict[str, Any]) -> Dict[str, Any]:
    """
    B.23: Strictly enforce that packet contains ONLY allowed fields.
    
    Removes any field not in ALLOWED_PACKET_FIELDS.
    """
    result = {}
    
    for key in ALLOWED_PACKET_FIELDS:
        if key in packet:
            value = packet[key]
            
            # Final PHI scan on string values
            if isinstance(value, str) and phi_scan(value):
                logger.warning(f"PHI detected in packet field {key}, setting to unknown")
                result[key] = "unknown"
            elif isinstance(value, list):
                # Filter list items for PHI
                safe_items = [
                    item for item in value 
                    if not (isinstance(item, str) and phi_scan(item))
                ]
                result[key] = safe_items
            else:
                result[key] = value
        else:
            # Set default values for missing allowed fields
            if key.endswith("_bucket"):
                result[key] = "unknown"
            elif key.endswith("_codes") or key.endswith("_flags"):
                result[key] = []
            else:
                result[key] = None
    
    return result


# =============================================================================
# Batch operations for cohort processing
# =============================================================================
def build_cohort_packets(
    patient_data_list: List[Dict[str, Any]],
    patient_id_field: str = "patient_id"
) -> List[Dict[str, Any]]:
    """
    Build feature packets for a cohort of patients.
    
    Args:
        patient_data_list: List of patient data dictionaries
        patient_id_field: Field name containing patient ID
        
    Returns:
        List of feature packets (one per patient)
    """
    packets = []
    
    for patient_data in patient_data_list:
        patient_id = patient_data.get(patient_id_field, "")
        packet = build_patient_packet(patient_id, patient_data)
        packets.append(packet)
    
    logger.info(f"Built {len(packets)} feature packets for cohort")
    return packets


def validate_packet(packet: Dict[str, Any]) -> bool:
    """
    Validate that a packet conforms to B.23 requirements.
    
    Returns True if packet is valid, False otherwise.
    """
    # Check only allowed fields present
    extra_fields = set(packet.keys()) - ALLOWED_PACKET_FIELDS
    if extra_fields:
        logger.error(f"Packet contains disallowed fields: {extra_fields}")
        return False
    
    # Check no PHI in values
    for key, value in packet.items():
        if isinstance(value, str) and phi_scan(value):
            logger.error(f"PHI detected in packet field: {key}")
            return False
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and phi_scan(item):
                    logger.error(f"PHI detected in packet list field: {key}")
                    return False
    
    return True


__all__ = [
    # B.22
    "build_patient_packet",
    "build_cohort_packets",
    # B.23
    "ALLOWED_PACKET_FIELDS",
    "validate_packet",
    # Utilities
    "_bucket_age",
    "_bucket_risk_score",
    "_bucket_engagement",
    "_bucket_missingness",
    "_bucket_adherence",
]
