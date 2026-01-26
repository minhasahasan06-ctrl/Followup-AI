"""
Privacy Firewall Service (Phase B.1-B.12)
==========================================
HIPAA-compliant privacy protection for Tinker Thinking Machine integration.
Implements exact specifications for Phase B tasks.

Tasks Covered:
- B.1: FORBIDDEN_KEYS list
- B.2-B.8: ALLOWED_KEYS_BY_PURPOSE dictionaries
- B.9: phi_scan(value: str) -> bool
- B.10: sanitize(purpose: str, raw: dict) -> dict
- B.11: k_anon_check(count: int, k: int) -> None
- B.12: audit_log(purpose, actor_role, payload_hash, response_hash, model_used)
"""

import hashlib
import re
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum

from app.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# B.1: FORBIDDEN_KEYS - Fields that must NEVER be sent to Tinker
# Exactly 12 keys as specified in B.1
# =============================================================================
FORBIDDEN_KEYS: Set[str] = {
    "patient_id",
    "mrn",
    "name",
    "email",
    "phone",
    "address",
    "notes",
    "symptoms_text",
    "clinician_note",
    "timestamp",
    "date",
    "dob",
}


# =============================================================================
# B.2-B.8: ALLOWED_KEYS_BY_PURPOSE - Allowlists per purpose type
# =============================================================================
class TinkerPurpose(str, Enum):
    """Tinker API request purpose types"""
    PATIENT_QUESTIONS = "patient_questions"
    PATIENT_TEMPLATES = "patient_templates"
    COHORT_BUILDER = "cohort_builder"
    STUDY_PROTOCOL = "study_protocol"
    JOB_PLANNER = "job_planner"
    MODEL_CARD = "model_card"
    DRIFT_SUMMARY = "drift_summary"


# =============================================================================
# Field Type Schemas - Define what each field can contain
# Types: "string", "string_list", "number", "bucket", "categorical"
# =============================================================================
class FieldType(str, Enum):
    """Allowed field types for schema validation"""
    STRING = "string"              # Free text (will be PHI scanned)
    STRING_LIST = "string_list"    # List of strings (each PHI scanned)
    NUMBER = "number"              # Numeric value (will be bucketized)
    BUCKET = "bucket"              # Already bucketed string
    CATEGORICAL = "categorical"    # Categorical value from allowed set


# B.2: patient_questions allowed keys
ALLOWED_PATIENT_QUESTIONS: Set[str] = {
    "age_bucket",
    "condition_codes",
    "risk_bucket",
    "trend_flags",
    "engagement_bucket",
    "missingness_bucket",
}

# B.3: patient_templates allowed keys  
ALLOWED_PATIENT_TEMPLATES: Set[str] = {
    "age_bucket",
    "condition_codes",
    "risk_bucket",
    "trend_flags",
    "engagement_bucket",
    "missingness_bucket",
    "goal_context",
    "adherence_bucket",
}

# B.4: cohort_builder allowed keys
ALLOWED_COHORT_BUILDER: Set[str] = {
    "nl_query",
    "schema_summary",
    "allowed_operators",
}

# B.5: study_protocol allowed keys
ALLOWED_STUDY_PROTOCOL: Set[str] = {
    "objective",
    "schema_summary",
    "cohort_size_range",
    "analysis_types_available",
}

# B.6: job_planner allowed keys
ALLOWED_JOB_PLANNER: Set[str] = {
    "task_type",
    "dataset_summary",
    "constraints",
}

# B.7: model_card allowed keys
ALLOWED_MODEL_CARD: Set[str] = {
    "metrics_summary",
    "feature_names",
    "training_config",
    "subgroup_metrics",
    "calibration",
}

# B.8: drift_summary allowed keys
ALLOWED_DRIFT_SUMMARY: Set[str] = {
    "drift_metrics",
    "feature_names",
    "thresholds",
}

ALLOWED_KEYS_BY_PURPOSE: Dict[str, Set[str]] = {
    TinkerPurpose.PATIENT_QUESTIONS.value: ALLOWED_PATIENT_QUESTIONS,
    TinkerPurpose.PATIENT_TEMPLATES.value: ALLOWED_PATIENT_TEMPLATES,
    TinkerPurpose.COHORT_BUILDER.value: ALLOWED_COHORT_BUILDER,
    TinkerPurpose.STUDY_PROTOCOL.value: ALLOWED_STUDY_PROTOCOL,
    TinkerPurpose.JOB_PLANNER.value: ALLOWED_JOB_PLANNER,
    TinkerPurpose.MODEL_CARD.value: ALLOWED_MODEL_CARD,
    TinkerPurpose.DRIFT_SUMMARY.value: ALLOWED_DRIFT_SUMMARY,
}

# Field schemas define the expected type for each allowed field
# Nested dicts/lists are ONLY allowed if field type is STRING_LIST
# All other nested structures are rejected
FIELD_SCHEMAS: Dict[str, Dict[str, FieldType]] = {
    TinkerPurpose.PATIENT_QUESTIONS.value: {
        "age_bucket": FieldType.BUCKET,
        "condition_codes": FieldType.STRING_LIST,
        "risk_bucket": FieldType.BUCKET,
        "trend_flags": FieldType.STRING_LIST,
        "engagement_bucket": FieldType.BUCKET,
        "missingness_bucket": FieldType.BUCKET,
    },
    TinkerPurpose.PATIENT_TEMPLATES.value: {
        "age_bucket": FieldType.BUCKET,
        "condition_codes": FieldType.STRING_LIST,
        "risk_bucket": FieldType.BUCKET,
        "trend_flags": FieldType.STRING_LIST,
        "engagement_bucket": FieldType.BUCKET,
        "missingness_bucket": FieldType.BUCKET,
        "goal_context": FieldType.STRING,
        "adherence_bucket": FieldType.BUCKET,
    },
    TinkerPurpose.COHORT_BUILDER.value: {
        "nl_query": FieldType.STRING,
        "schema_summary": FieldType.STRING,
        "allowed_operators": FieldType.STRING_LIST,
    },
    TinkerPurpose.STUDY_PROTOCOL.value: {
        "objective": FieldType.STRING,
        "schema_summary": FieldType.STRING,
        "cohort_size_range": FieldType.STRING,
        "analysis_types_available": FieldType.STRING_LIST,
    },
    TinkerPurpose.JOB_PLANNER.value: {
        "task_type": FieldType.STRING,
        "dataset_summary": FieldType.STRING,
        "constraints": FieldType.STRING,
    },
    TinkerPurpose.MODEL_CARD.value: {
        "metrics_summary": FieldType.STRING,
        "feature_names": FieldType.STRING_LIST,
        "training_config": FieldType.STRING,
        "subgroup_metrics": FieldType.STRING,
        "calibration": FieldType.STRING,
    },
    TinkerPurpose.DRIFT_SUMMARY.value: {
        "drift_metrics": FieldType.STRING,
        "feature_names": FieldType.STRING_LIST,
        "thresholds": FieldType.STRING,
    },
}


# =============================================================================
# B.9: phi_scan - Detect PHI patterns in strings
# =============================================================================
# PHI detection regex patterns
PHI_PATTERNS = {
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    "phone": re.compile(r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'),
    "ssn": re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    "date_mdy": re.compile(r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)?\d{2}\b'),
    "date_ymd": re.compile(r'\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b'),
    "address": re.compile(r'\b\d{1,5}\s+\w+\s+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|way|court|ct|boulevard|blvd)\b', re.IGNORECASE),
    "zip_code": re.compile(r'\b\d{5}(?:-\d{4})?\b'),
    "mrn": re.compile(r'\bMRN[:\s#]?\d{6,12}\b', re.IGNORECASE),
    "ip_address": re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
}


def phi_scan(value: str) -> bool:
    """
    B.9: Scan a string value for PHI patterns.
    
    Detects: emails, phones, SSNs, dates, addresses, zip codes, MRNs, IP addresses.
    
    Args:
        value: String to scan for PHI
        
    Returns:
        True if PHI detected, False otherwise
    """
    if not value or not isinstance(value, str):
        return False
    
    for pattern_name, pattern in PHI_PATTERNS.items():
        if pattern.search(value):
            logger.warning(f"PHI detected: {pattern_name} pattern found")
            return True
    
    return False


# =============================================================================
# B.10: sanitize - Clean and filter data by purpose
# =============================================================================
def _bucketize_number(value: Union[int, float], bucket_size: int = 5) -> str:
    """Convert numeric value to bucket range string"""
    if value is None:
        return "unknown"
    lower = (int(value) // bucket_size) * bucket_size
    upper = lower + bucket_size - 1
    return f"{lower}-{upper}"


def _deep_check_forbidden_keys(data: Any, path: str = "") -> None:
    """
    Recursively check for forbidden keys at ANY depth.
    Raises ValueError immediately if any forbidden key is found.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if key in FORBIDDEN_KEYS:
                raise ValueError(f"FORBIDDEN_KEYS detected at {current_path}: {key}")
            _deep_check_forbidden_keys(value, current_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _deep_check_forbidden_keys(item, f"{path}[{idx}]")


def _deep_phi_scan(data: Any, path: str = "") -> None:
    """
    Recursively scan for PHI patterns at ANY depth.
    Raises ValueError immediately if PHI is detected.
    """
    if isinstance(data, str):
        if phi_scan(data):
            raise ValueError(f"PHI detected at {path}")
    elif isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            _deep_phi_scan(value, current_path)
    elif isinstance(data, list):
        for idx, item in enumerate(data):
            _deep_phi_scan(item, f"{path}[{idx}]")


def _validate_field_value(key: str, value: Any, field_type: FieldType, path: str) -> Any:
    """
    Validate and sanitize a field value based on its schema type.
    
    REJECTS nested dicts entirely - they are NOT allowed in any field type.
    Only primitive values and lists of strings are permitted.
    
    Args:
        key: Field name
        value: Field value
        field_type: Expected type from schema
        path: Path for error messages
        
    Returns:
        Sanitized value
        
    Raises:
        ValueError: If value doesn't match expected type or contains PHI
    """
    if field_type == FieldType.BUCKET:
        # Bucket fields must be strings (already bucketed)
        if isinstance(value, str):
            if phi_scan(value):
                raise ValueError(f"PHI detected in bucket field at {path}")
            return value
        elif isinstance(value, (int, float)):
            # Auto-convert number to bucket
            return _bucketize_number(value)
        else:
            raise ValueError(f"Invalid type for bucket field at {path}: expected str, got {type(value).__name__}")
    
    elif field_type == FieldType.STRING:
        if not isinstance(value, str):
            raise ValueError(f"Invalid type for string field at {path}: expected str, got {type(value).__name__}")
        if phi_scan(value):
            raise ValueError(f"PHI detected in string field at {path}")
        return value
    
    elif field_type == FieldType.STRING_LIST:
        if not isinstance(value, list):
            raise ValueError(f"Invalid type for string_list field at {path}: expected list, got {type(value).__name__}")
        
        safe_items = []
        for idx, item in enumerate(value):
            if isinstance(item, str):
                if phi_scan(item):
                    raise ValueError(f"PHI detected in string_list at {path}[{idx}]")
                safe_items.append(item)
            elif isinstance(item, dict):
                # REJECT nested dicts in lists - not allowed
                raise ValueError(f"Nested dict not allowed in string_list at {path}[{idx}]")
            else:
                # Skip non-string items silently
                continue
        return safe_items
    
    elif field_type == FieldType.NUMBER:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Invalid type for number field at {path}: expected number, got {type(value).__name__}")
        return _bucketize_number(value)
    
    elif field_type == FieldType.CATEGORICAL:
        if not isinstance(value, str):
            raise ValueError(f"Invalid type for categorical field at {path}: expected str, got {type(value).__name__}")
        if phi_scan(value):
            raise ValueError(f"PHI detected in categorical field at {path}")
        return value
    
    else:
        raise ValueError(f"Unknown field type at {path}: {field_type}")


def sanitize(purpose: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    B.10: Sanitize data dictionary for Tinker API.
    
    SECURITY: Uses schema-based validation to ensure:
    1. Only allowed keys for the purpose are included
    2. Each field matches its expected type
    3. No nested dicts are allowed (all nested structures rejected)
    4. All strings are PHI-scanned
    5. All forbidden keys are rejected at any depth
    
    Args:
        purpose: The Tinker purpose (e.g., "patient_questions")
        raw: Raw data dictionary
        
    Returns:
        Sanitized dictionary with only allowed, safe data
        
    Raises:
        ValueError: If forbidden keys detected, PHI found, or type mismatch
    """
    if not raw:
        return {}
    
    # CRITICAL: Deep check for forbidden keys at ALL levels first
    _deep_check_forbidden_keys(raw)
    
    # CRITICAL: Deep PHI scan ALL string values at ALL levels
    _deep_phi_scan(raw)
    
    # Get allowed keys and schema for this purpose
    allowed_keys = ALLOWED_KEYS_BY_PURPOSE.get(purpose, set())
    field_schema = FIELD_SCHEMAS.get(purpose, {})
    
    if not allowed_keys:
        logger.warning(f"Unknown purpose: {purpose}, returning empty dict")
        return {}
    
    result = {}
    
    for key, value in raw.items():
        # Drop keys not in allowlist
        if key not in allowed_keys:
            logger.debug(f"Dropping unknown key: {key}")
            continue
        
        # REJECT nested dicts entirely - they are not in our schema
        if isinstance(value, dict):
            raise ValueError(f"Nested dict not allowed for field: {key}")
        
        # Get expected field type from schema
        field_type = field_schema.get(key)
        if not field_type:
            logger.warning(f"Field {key} has no schema, dropping")
            continue
        
        # Validate and sanitize based on schema
        path = key
        try:
            result[key] = _validate_field_value(key, value, field_type, path)
        except ValueError as e:
            # Re-raise with context
            raise ValueError(f"Validation failed for {key}: {e}")
    
    return result


# =============================================================================
# B.11: k_anon_check - K-anonymity enforcement
# =============================================================================
def k_anon_check(count: int, k: int = 25) -> None:
    """
    B.11: Enforce k-anonymity threshold.
    
    Args:
        count: Number of records in cohort
        k: Minimum required records (default 25)
        
    Raises:
        ValueError: If count < k (k-anonymity violation)
    """
    if k is None:
        k = getattr(settings, 'TINKER_K_ANON', 25)
    
    if count < k:
        raise ValueError(
            f"K-ANONYMITY VIOLATION: count={count} < k={k}. "
            f"Cohort too small for privacy-safe analysis."
        )
    
    logger.debug(f"K-anonymity check passed: count={count} >= k={k}")


# =============================================================================
# B.12: audit_log - Write to ai_audit_logs table
# =============================================================================
def _compute_hash(data: Any) -> str:
    """Compute SHA256 hash of data"""
    import json
    data_str = json.dumps(data, sort_keys=True, default=str) if isinstance(data, (dict, list)) else str(data)
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()


async def audit_log(
    purpose: str,
    actor_role: str,
    payload_hash: str,
    response_hash: str,
    model_used: Optional[str] = None,
    success: bool = True,
    k_anon_verified: bool = True,
    error_message: Optional[str] = None
) -> str:
    """
    B.12: Write audit entry to ai_audit_logs table.
    
    HIPAA-compliant logging - stores only hashes, never raw data.
    
    Args:
        purpose: Tinker purpose type
        actor_role: Role of actor (e.g., "doctor", "admin", "system")
        payload_hash: SHA256 hash of request payload
        response_hash: SHA256 hash of response
        model_used: Optional model identifier
        success: Whether operation succeeded
        k_anon_verified: Whether k-anonymity was verified
        error_message: Optional error message
        
    Returns:
        Audit log entry ID as string
    """
    from app.database import SessionLocal
    from app.models.tinker_models import AIAuditLog
    
    db = SessionLocal()
    try:
        audit_entry = AIAuditLog(
            purpose=purpose,
            actor_role=actor_role,
            payload_hash=payload_hash,
            response_hash=response_hash or "",
            model_used=model_used or "tinker_api",
            success=success,
            k_anon_verified=k_anon_verified,
            error_code=error_message[:50] if error_message else None,
            tinker_mode="NON_BAA"
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        
        logger.info(f"Audit log created: id={audit_entry.id}, purpose={purpose}, success={success}")
        return str(audit_entry.id)
    
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to write audit log: {e}")
        raise
    finally:
        db.close()


def audit_log_sync(
    purpose: str,
    actor_role: str,
    payload_hash: str,
    response_hash: str,
    model_used: Optional[str] = None,
    success: bool = True,
    k_anon_verified: bool = True,
    error_message: Optional[str] = None
) -> str:
    """Synchronous version of audit_log for non-async contexts"""
    from app.database import SessionLocal
    from app.models.tinker_models import AIAuditLog
    
    db = SessionLocal()
    try:
        audit_entry = AIAuditLog(
            purpose=purpose,
            actor_role=actor_role,
            payload_hash=payload_hash,
            response_hash=response_hash or "",
            model_used=model_used or "tinker_api",
            success=success,
            k_anon_verified=k_anon_verified,
            error_code=error_message[:50] if error_message else None,
            tinker_mode="NON_BAA"
        )
        db.add(audit_entry)
        db.commit()
        db.refresh(audit_entry)
        
        logger.info(f"Audit log created: id={audit_entry.id}, purpose={purpose}, success={success}")
        return str(audit_entry.id)
    
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to write audit log: {e}")
        raise
    finally:
        db.close()


# =============================================================================
# Utility exports
# =============================================================================
__all__ = [
    # B.1
    "FORBIDDEN_KEYS",
    # B.2-B.8
    "TinkerPurpose",
    "ALLOWED_KEYS_BY_PURPOSE",
    "ALLOWED_PATIENT_QUESTIONS",
    "ALLOWED_PATIENT_TEMPLATES",
    "ALLOWED_COHORT_BUILDER",
    "ALLOWED_STUDY_PROTOCOL",
    "ALLOWED_JOB_PLANNER",
    "ALLOWED_MODEL_CARD",
    "ALLOWED_DRIFT_SUMMARY",
    # B.9
    "phi_scan",
    "PHI_PATTERNS",
    # B.10
    "sanitize",
    # B.11
    "k_anon_check",
    # B.12
    "audit_log",
    "audit_log_sync",
]
