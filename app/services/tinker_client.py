"""
Tinker Client Service (Phase B.13-B.15)
=======================================
HTTP client for Tinker Thinking Machine API.

Tasks Covered:
- B.13: call_tinker(purpose, payload, actor_role)
- B.14: Stub responses when TINKER_ENABLED=false
- B.15: Live Tinker API call with requests.post, timeout=15s, retries=2
"""

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.config import settings
from app.services.privacy_firewall import (
    sanitize,
    k_anon_check,
    audit_log_sync,
    ALLOWED_KEYS_BY_PURPOSE,
    TinkerPurpose,
)

logger = logging.getLogger(__name__)


# =============================================================================
# B.14: Stub Responses for TINKER_ENABLED=false
# =============================================================================
STUB_RESPONSES: Dict[str, Dict[str, Any]] = {
    TinkerPurpose.PATIENT_QUESTIONS.value: {
        "questions": [
            {"id": "Q001", "text": "How are you feeling today?", "type": "scale"},
            {"id": "Q002", "text": "Any new symptoms?", "type": "boolean"},
        ],
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
    TinkerPurpose.PATIENT_TEMPLATES.value: {
        "templates": [
            {"id": "T001", "message": "Hello {patient_name}, time for your check-in!"},
            {"id": "T002", "message": "Your daily reminder to log your symptoms."},
        ],
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
    TinkerPurpose.COHORT_BUILDER.value: {
        "cohort_definition": {
            "filters": [],
            "estimated_size": 100,
        },
        "sql_preview": "SELECT * FROM patients WHERE ...",
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
    TinkerPurpose.STUDY_PROTOCOL.value: {
        "protocol": {
            "name": "Stub Study",
            "description": "Generated when Tinker is disabled",
            "cohort_size": "50-200",
            "analysis_types": ["cohort_comparison"],
        },
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
    TinkerPurpose.JOB_PLANNER.value: {
        "jobs": [
            {"type": "data_quality", "priority": "high", "estimated_duration": "5m"},
            {"type": "feature_extraction", "priority": "medium", "estimated_duration": "15m"},
        ],
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
    TinkerPurpose.MODEL_CARD.value: {
        "model_card": {
            "name": "Stub Model",
            "version": "1.0.0",
            "metrics": {"accuracy": 0.85, "auc": 0.90},
            "feature_importance": {},
        },
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
    TinkerPurpose.DRIFT_SUMMARY.value: {
        "drift_detected": False,
        "drift_score": 0.02,
        "features_affected": [],
        "recommendations": ["No action needed"],
        "generated_at": "stub",
        "model": "stub_model",
        "k_anon_verified": True,
    },
}


def _get_stub_response(purpose: str) -> Dict[str, Any]:
    """B.14: Return deterministic stub response for purpose"""
    response = STUB_RESPONSES.get(purpose, {"status": "stub", "purpose": purpose})
    response["generated_at"] = datetime.utcnow().isoformat()
    response["is_stub"] = True
    return response


# =============================================================================
# B.15: HTTP Session with retry configuration
# =============================================================================
def _create_session() -> requests.Session:
    """Create requests session with retry config: timeout=15s, retries=2"""
    session = requests.Session()
    
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session


def _compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Compute SHA256 hash of payload"""
    payload_str = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(payload_str.encode('utf-8')).hexdigest()


def _compute_response_hash(response: Any) -> str:
    """Compute SHA256 hash of response"""
    if isinstance(response, dict):
        response_str = json.dumps(response, sort_keys=True, default=str)
    else:
        response_str = str(response)
    return hashlib.sha256(response_str.encode('utf-8')).hexdigest()


# =============================================================================
# B.13: Main call_tinker function
# =============================================================================
def call_tinker(
    purpose: str,
    payload: Dict[str, Any],
    actor_role: str,
    cohort_count: Optional[int] = None
) -> Tuple[Dict[str, Any], bool]:
    """
    B.13: Main entry point for Tinker API calls.
    
    Handles:
    1. Sanitization via privacy_firewall.sanitize()
    2. K-anonymity check if cohort_count provided
    3. Stub response if TINKER_ENABLED=false (B.14)
    4. Live API call with timeout=15s, retries=2 (B.15)
    5. Audit logging via privacy_firewall.audit_log()
    
    Args:
        purpose: Tinker purpose (e.g., "patient_questions")
        payload: Raw payload dictionary
        actor_role: Role of caller (e.g., "doctor", "admin")
        cohort_count: Optional cohort size for k-anonymity check
        
    Returns:
        Tuple of (response_dict, success_bool)
    """
    start_time = time.time()
    
    # Validate purpose
    if purpose not in ALLOWED_KEYS_BY_PURPOSE:
        logger.error(f"Invalid purpose: {purpose}")
        return {"error": f"Invalid purpose: {purpose}"}, False
    
    # Sanitize payload (B.10)
    try:
        safe_payload = sanitize(purpose, payload)
    except ValueError as e:
        logger.error(f"Sanitization failed: {e}")
        audit_log_sync(
            purpose=purpose,
            actor_role=actor_role,
            payload_hash=_compute_payload_hash(payload),
            response_hash="",
            success=False,
            error_message=str(e)
        )
        return {"error": str(e)}, False
    
    # K-anonymity check (B.11) if cohort_count provided
    if cohort_count is not None:
        try:
            k_anon_check(cohort_count)
        except ValueError as e:
            logger.error(f"K-anonymity check failed: {e}")
            audit_log_sync(
                purpose=purpose,
                actor_role=actor_role,
                payload_hash=_compute_payload_hash(safe_payload),
                response_hash="",
                success=False,
                k_anon_verified=False,
                error_message=str(e)
            )
            return {"error": str(e)}, False
    
    payload_hash = _compute_payload_hash(safe_payload)
    
    # Check if Tinker is enabled
    tinker_enabled = getattr(settings, 'TINKER_ENABLED', False)
    tinker_api_key = getattr(settings, 'TINKER_API_KEY', None)
    
    if not tinker_enabled or not tinker_api_key:
        # B.14: Return stub response
        logger.info(f"Tinker disabled, returning stub response for purpose={purpose}")
        response = _get_stub_response(purpose)
        
        audit_log_sync(
            purpose=purpose,
            actor_role=actor_role,
            payload_hash=payload_hash,
            response_hash=_compute_response_hash(response),
            model_used="stub",
            success=True,
            k_anon_verified=True
        )
        
        return response, True
    
    # B.15: Live Tinker API call
    try:
        response = _call_tinker_live(purpose, safe_payload, actor_role)
        response_hash = _compute_response_hash(response)
        
        duration_ms = (time.time() - start_time) * 1000
        logger.info(f"Tinker API call completed: purpose={purpose}, duration={duration_ms:.1f}ms")
        
        audit_log_sync(
            purpose=purpose,
            actor_role=actor_role,
            payload_hash=payload_hash,
            response_hash=response_hash,
            model_used=response.get("model", "tinker_api"),
            success=True,
            k_anon_verified=True
        )
        
        return response, True
    
    except Exception as e:
        logger.error(f"Tinker API call failed: {e}")
        
        audit_log_sync(
            purpose=purpose,
            actor_role=actor_role,
            payload_hash=payload_hash,
            response_hash="",
            success=False,
            error_message=str(e)
        )
        
        return {"error": str(e)}, False


def _call_tinker_live(
    purpose: str,
    payload: Dict[str, Any],
    actor_role: str
) -> Dict[str, Any]:
    """
    B.15: Execute live Tinker API call with timeout=15s, retries=2.
    
    Args:
        purpose: Tinker purpose
        payload: Sanitized payload
        actor_role: Caller role
        
    Returns:
        Response dictionary from Tinker API
        
    Raises:
        Exception: On API call failure
    """
    base_url = getattr(settings, 'TINKER_API_URL', 'https://api.tinker.ai')
    api_key = getattr(settings, 'TINKER_API_KEY', '')
    
    url = f"{base_url}/v1/{purpose}"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Client": "followup-ai",
        "X-Mode": "NON_BAA",
        "X-Actor-Role": actor_role,
    }
    
    request_body = {
        "purpose": purpose,
        "payload": payload,
        "timestamp": datetime.utcnow().isoformat(),
    }
    
    session = _create_session()
    
    try:
        response = session.post(
            url,
            json=request_body,
            headers=headers,
            timeout=15  # B.15: timeout=15s
        )
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        raise Exception("Tinker API request timed out (15s)")
    
    except requests.exceptions.RequestException as e:
        raise Exception(f"Tinker API request failed: {e}")
    
    finally:
        session.close()


# =============================================================================
# Async version for FastAPI endpoints
# =============================================================================
async def call_tinker_async(
    purpose: str,
    payload: Dict[str, Any],
    actor_role: str,
    cohort_count: Optional[int] = None
) -> Tuple[Dict[str, Any], bool]:
    """Async wrapper for call_tinker using run_in_executor"""
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        call_tinker, 
        purpose, 
        payload, 
        actor_role, 
        cohort_count
    )


__all__ = [
    "call_tinker",
    "call_tinker_async",
    "STUB_RESPONSES",
    "TinkerPurpose",
]
