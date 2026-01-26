"""
Tinker API Client
=================
Production-grade HTTP client for Tinker Thinking Machine API.
Implements retry logic, circuit breaker, and comprehensive audit logging.

CRITICAL: All requests must pass through TinkerPrivacyFirewall first.
This client only sends hashed/anonymized data - never raw PHI.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import httpx

from app.config import settings
from app.services.tinker_privacy_firewall import get_privacy_firewall, TinkerPrivacyFirewall

logger = logging.getLogger(__name__)


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.
    
    Prevents cascading failures by stopping requests to failing service.
    """
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0
    
    def record_success(self):
        """Record successful call"""
        self.failure_count = 0
        self.half_open_calls = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker: Recovered, state → CLOSED")
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker: Failed in HALF_OPEN, state → OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker: {self.failure_count} failures, state → OPEN")
    
    def can_execute(self) -> bool:
        """Check if request can proceed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            if self.last_failure_time:
                elapsed = (datetime.utcnow() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("Circuit breaker: Recovery timeout elapsed, state → HALF_OPEN")
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False


@dataclass
class TinkerRequestMetrics:
    """Metrics for a single Tinker API request"""
    endpoint: str
    method: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status_code: Optional[int] = None
    success: bool = False
    retry_count: int = 0
    payload_hash: str = ""
    response_hash: str = ""
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0


class TinkerAPIClient:
    """
    Production HTTP client for Tinker API.
    
    Features:
    - Exponential backoff retry
    - Circuit breaker pattern
    - Request/response hash logging (no raw data)
    - Timeout management
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        self.base_url = base_url or settings.TINKER_API_URL
        self.api_key = api_key or settings.TINKER_API_KEY
        self.timeout = timeout or settings.TINKER_TIMEOUT_SECONDS
        self.max_retries = max_retries or settings.TINKER_MAX_RETRIES
        
        self.circuit_breaker = CircuitBreaker()
        self.firewall = get_privacy_firewall()
        
        self._validate_config()
        
        logger.info(f"TinkerAPIClient initialized: {self.base_url}")
    
    def _validate_config(self):
        """Validate configuration"""
        if not settings.is_tinker_enabled():
            logger.warning("Tinker API is not enabled or API key not set")
        
        settings.validate_tinker_non_baa()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with auth"""
        return {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json",
            "X-Client": "followup-ai",
            "X-Mode": "NON_BAA",
        }
    
    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Create SHA256 hash of payload for audit logging"""
        return self.firewall.create_payload_hash(payload)
    
    def _hash_response(self, response: Any) -> str:
        """Create SHA256 hash of response for audit logging"""
        if isinstance(response, dict):
            response_str = json.dumps(response, sort_keys=True, default=str)
        else:
            response_str = str(response)
        return hashlib.sha256(response_str.encode('utf-8')).hexdigest()
    
    async def _execute_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        skip_phi_check: bool = False
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """
        Execute request with retry logic and circuit breaker.
        
        SECURITY: All payloads pass through ensure_phi_safe_payload unless
        skip_phi_check=True (only for already-processed data).
        
        Returns:
            Tuple of (response_data, metrics)
        """
        # MANDATORY: Run payload through privacy firewall
        if payload and not skip_phi_check:
            payload = self.firewall.ensure_phi_safe_payload(payload)
        
        metrics = TinkerRequestMetrics(
            endpoint=endpoint,
            method=method,
            start_time=datetime.utcnow(),
            payload_hash=self._hash_payload(payload) if payload else ""
        )
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            metrics.error_message = "Circuit breaker OPEN - request rejected"
            metrics.end_time = datetime.utcnow()
            logger.warning(f"Request rejected by circuit breaker: {endpoint}")
            return None, metrics
        
        url = f"{self.base_url}{endpoint}"
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            metrics.retry_count = attempt
            
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(
                            url,
                            headers=self._get_headers(),
                            params=params
                        )
                    elif method.upper() == "POST":
                        response = await client.post(
                            url,
                            headers=self._get_headers(),
                            json=payload
                        )
                    elif method.upper() == "PUT":
                        response = await client.put(
                            url,
                            headers=self._get_headers(),
                            json=payload
                        )
                    elif method.upper() == "DELETE":
                        response = await client.delete(
                            url,
                            headers=self._get_headers(),
                            params=params
                        )
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    metrics.status_code = response.status_code
                    
                    if response.status_code == 200:
                        result = response.json()
                        metrics.success = True
                        metrics.end_time = datetime.utcnow()
                        metrics.response_hash = self._hash_response(result)
                        self.circuit_breaker.record_success()
                        return result, metrics
                    
                    elif response.status_code == 429:
                        # Rate limited - wait and retry
                        retry_after = int(response.headers.get("Retry-After", 5))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    elif response.status_code >= 500:
                        # Server error - retry with backoff
                        last_error = f"Server error: {response.status_code}"
                        await self._exponential_backoff(attempt)
                        continue
                    
                    else:
                        # Client error - don't retry
                        metrics.error_message = f"Client error: {response.status_code} - {response.text}"
                        metrics.end_time = datetime.utcnow()
                        self.circuit_breaker.record_failure()
                        return None, metrics
                    
            except httpx.TimeoutException as e:
                last_error = f"Timeout: {str(e)}"
                logger.warning(f"Request timeout (attempt {attempt + 1}): {endpoint}")
                await self._exponential_backoff(attempt)
                
            except httpx.ConnectError as e:
                last_error = f"Connection error: {str(e)}"
                logger.warning(f"Connection error (attempt {attempt + 1}): {endpoint}")
                await self._exponential_backoff(attempt)
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error: {e}")
                break
        
        # All retries exhausted
        metrics.error_message = last_error
        metrics.end_time = datetime.utcnow()
        self.circuit_breaker.record_failure()
        
        return None, metrics
    
    async def _exponential_backoff(self, attempt: int):
        """Wait with exponential backoff"""
        wait_time = min(2 ** attempt, 30)  # Max 30 seconds
        await asyncio.sleep(wait_time)
    
    # =========================================================================
    # Cohort Analysis Endpoints
    # =========================================================================
    
    async def analyze_cohort(
        self,
        cohort_query: Dict[str, Any],
        patient_count: int
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """
        Analyze a patient cohort.
        
        Applies privacy firewall before sending.
        """
        # Apply privacy firewall
        safe_query, audit = self.firewall.transform_cohort_query(
            cohort_query, patient_count
        )
        
        if safe_query is None:
            metrics = TinkerRequestMetrics(
                endpoint="/cohorts/analyze",
                method="POST",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                error_message="K-anonymity check failed"
            )
            return None, metrics
        
        return await self._execute_with_retry(
            "POST",
            "/cohorts/analyze",
            payload=safe_query
        )
    
    async def get_cohort_insights(
        self,
        cohort_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get insights for a cohort (by hashed ID)"""
        hashed_id = self.firewall.hash_identifier(cohort_id, "cohort")
        
        return await self._execute_with_retry(
            "GET",
            f"/cohorts/{hashed_id}/insights"
        )
    
    # =========================================================================
    # Study & Protocol Endpoints
    # =========================================================================
    
    async def create_study(
        self,
        study_config: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Create a new study"""
        # Strip any PHI from config
        safe_config, _ = self.firewall.strip_phi_fields(study_config)
        
        return await self._execute_with_retry(
            "POST",
            "/studies",
            payload=safe_config
        )
    
    async def get_study(
        self,
        study_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get study by ID"""
        return await self._execute_with_retry(
            "GET",
            f"/studies/{study_id}"
        )
    
    async def create_protocol(
        self,
        protocol_config: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Create a study protocol"""
        safe_config, _ = self.firewall.strip_phi_fields(protocol_config)
        
        return await self._execute_with_retry(
            "POST",
            "/protocols",
            payload=safe_config
        )
    
    async def get_protocol(
        self,
        protocol_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get protocol by ID"""
        return await self._execute_with_retry(
            "GET",
            f"/protocols/{protocol_id}"
        )
    
    # =========================================================================
    # Trial Execution Endpoints
    # =========================================================================
    
    async def run_trial(
        self,
        trial_spec: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Execute a trial run"""
        safe_spec, _ = self.firewall.strip_phi_fields(trial_spec)
        
        return await self._execute_with_retry(
            "POST",
            "/trials/run",
            payload=safe_spec
        )
    
    async def get_trial_status(
        self,
        trial_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get trial execution status"""
        return await self._execute_with_retry(
            "GET",
            f"/trials/{trial_id}/status"
        )
    
    async def get_trial_results(
        self,
        trial_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get trial results"""
        return await self._execute_with_retry(
            "GET",
            f"/trials/{trial_id}/results"
        )
    
    # =========================================================================
    # Drift Detection Endpoints
    # =========================================================================
    
    async def check_drift(
        self,
        model_id: str,
        feature_packet: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """
        Check for model drift.
        
        Feature packet must already be anonymized.
        """
        payload = {
            "model_id": model_id,
            "features": feature_packet
        }
        
        return await self._execute_with_retry(
            "POST",
            "/drift/check",
            payload=payload
        )
    
    async def get_drift_report(
        self,
        model_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get drift report for a model"""
        params = {"model_id": model_id}
        if start_date:
            params["start_date"] = self.firewall.bucket_date(start_date)
        if end_date:
            params["end_date"] = self.firewall.bucket_date(end_date)
        
        return await self._execute_with_retry(
            "GET",
            "/drift/report",
            params=params
        )
    
    # =========================================================================
    # Model Metrics Endpoints
    # =========================================================================
    
    async def submit_model_metrics(
        self,
        model_id: str,
        metrics: Dict[str, Any]
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Submit model performance metrics"""
        payload = {
            "model_id": model_id,
            "metrics": metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return await self._execute_with_retry(
            "POST",
            "/models/metrics",
            payload=payload
        )
    
    async def get_model_metrics(
        self,
        model_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get metrics for a model"""
        return await self._execute_with_retry(
            "GET",
            f"/models/{model_id}/metrics"
        )
    
    async def get_calibration_curves(
        self,
        model_id: str
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get calibration curves for a model"""
        return await self._execute_with_retry(
            "GET",
            f"/models/{model_id}/calibration"
        )
    
    async def get_threshold_recommendations(
        self,
        model_id: str,
        optimization_target: str = "f1"
    ) -> Tuple[Optional[Dict[str, Any]], TinkerRequestMetrics]:
        """Get threshold optimization recommendations"""
        return await self._execute_with_retry(
            "GET",
            f"/models/{model_id}/thresholds",
            params={"target": optimization_target}
        )
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def health_check(self) -> Tuple[bool, TinkerRequestMetrics]:
        """Check if Tinker API is healthy"""
        result, metrics = await self._execute_with_retry(
            "GET",
            "/health"
        )
        
        return result is not None and result.get("status") == "healthy", metrics


# Singleton instance
_client_instance: Optional[TinkerAPIClient] = None


def get_tinker_client() -> TinkerAPIClient:
    """Get or create singleton Tinker API client"""
    global _client_instance
    if _client_instance is None:
        _client_instance = TinkerAPIClient()
    return _client_instance
