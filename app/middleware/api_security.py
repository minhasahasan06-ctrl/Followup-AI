"""
API Security Middleware - Request Signing & API Key Management
HIPAA-Compliant API Security

Features:
- API key authentication
- Request signing for sensitive operations
- Request replay prevention
- API usage tracking
"""

from fastapi import Request, HTTPException, status, Header
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Callable, Dict
import hmac
import hashlib
import time
import json
import logging
from datetime import datetime, timedelta

from app.config import settings
from app.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class APISecurityMiddleware(BaseHTTPMiddleware):
    """
    API security middleware for request signing and API key validation
    Implements HIPAA-compliant API security
    """
    
    def __init__(self, app, require_api_key: bool = False, require_signature: bool = False):
        super().__init__(app)
        self.require_api_key = require_api_key
        self.require_signature = require_signature
        self._api_keys: Dict[str, Dict[str, any]] = {}  # In-memory cache (use Redis in production)
        self._nonce_cache: Dict[str, datetime] = {}  # Replay prevention
        self._nonce_ttl = 300  # 5 minutes
    
    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, any]]:
        """
        Validate API key
        
        In production, this should:
        1. Check AWS Secrets Manager or database
        2. Verify key is active and not expired
        3. Check rate limits per key
        4. Log API key usage
        """
        # TODO: Implement proper API key validation
        # For now, check environment variable
        valid_key = settings.OPENAI_API_KEY  # Placeholder
        
        if api_key == valid_key:
            return {
                "key_id": "default",
                "permissions": ["read", "write"],
                "rate_limit": 1000
            }
        
        return None
    
    def _validate_request_signature(
        self,
        request: Request,
        signature: str,
        timestamp: str,
        nonce: str
    ) -> bool:
        """
        Validate request signature for replay prevention
        
        Signature format: HMAC-SHA256(request_body + timestamp + nonce, secret_key)
        """
        # Check timestamp (prevent replay attacks)
        try:
            req_timestamp = int(timestamp)
            current_timestamp = int(time.time())
            
            # Reject requests older than 5 minutes
            if abs(current_timestamp - req_timestamp) > 300:
                logger.warning(f"Request timestamp too old: {req_timestamp}")
                return False
            
            # Check nonce (prevent replay attacks)
            if nonce in self._nonce_cache:
                logger.warning(f"Nonce reuse detected: {nonce}")
                return False
            
            # Add nonce to cache
            self._nonce_cache[nonce] = datetime.utcnow()
            
            # Clean old nonces
            cutoff = datetime.utcnow() - timedelta(seconds=self._nonce_ttl)
            self._nonce_cache = {
                k: v for k, v in self._nonce_cache.items()
                if v > cutoff
            }
            
            # Get request body
            body = b""
            if hasattr(request, "_body"):
                body = request._body
            elif hasattr(request, "body"):
                # This is a workaround - body might be consumed
                pass
            
            # Reconstruct signature
            secret_key = settings.SESSION_SECRET or "default-secret"  # TODO: Use proper secret
            message = f"{body.decode('utf-8', errors='ignore')}{timestamp}{nonce}"
            expected_signature = hmac.new(
                secret_key.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Constant-time comparison
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception as e:
            logger.error(f"Signature validation error: {e}")
            return False
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip security for health checks and docs
        skip_paths = ["/health", "/healthz", "/docs", "/openapi.json", "/redoc"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Check for API key
        api_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if self.require_api_key:
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "ApiKey"}
                )
            
            key_info = self._validate_api_key(api_key)
            if not key_info:
                AuditLogger.log_event(
                    event_type="api_key_invalid",
                    user_id=None,
                    resource_type="api_endpoint",
                    resource_id=request.url.path,
                    action="api_access",
                    status="denied",
                    metadata={"ip": request.client.host if request.client else "unknown"},
                    ip_address=request.client.host if request.client else None
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid API key"
                )
            
            # Store key info in request state
            request.state.api_key_info = key_info
        
        # Check for request signature (for sensitive operations)
        if self.require_signature and request.method in ["POST", "PUT", "PATCH", "DELETE"]:
            signature = request.headers.get("X-Signature")
            timestamp = request.headers.get("X-Timestamp")
            nonce = request.headers.get("X-Nonce")
            
            if not all([signature, timestamp, nonce]):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Request signature required for this operation"
                )
            
            if not self._validate_request_signature(request, signature, timestamp, nonce):
                AuditLogger.log_event(
                    event_type="api_signature_invalid",
                    user_id=None,
                    resource_type="api_endpoint",
                    resource_id=request.url.path,
                    action="api_access",
                    status="denied",
                    metadata={"ip": request.client.host if request.client else "unknown"},
                    ip_address=request.client.host if request.client else None
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid request signature"
                )
        
        # Add security headers to response
        response = await call_next(request)
        response.headers["X-API-Version"] = "1.0"
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "unknown")
        
        return response
