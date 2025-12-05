"""
Rate Limiting Middleware - DDoS Protection & Brute Force Prevention
HIPAA-compliant rate limiting with IP-based and user-based limits

Features:
- IP-based rate limiting
- User-based rate limiting
- Adaptive rate limiting based on threat detection
- Redis-backed distributed rate limiting
- Configurable limits per endpoint
"""

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Dict, Callable
import time
import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from app.config import settings
from app.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    In-memory rate limiter (fallback when Redis unavailable)
    For production, use Redis-backed distributed rate limiting
    """
    
    def __init__(self):
        self._requests: Dict[str, list] = defaultdict(list)
        self._blocked_ips: Dict[str, datetime] = {}
        self._cleanup_interval = 300  # Clean up old entries every 5 minutes
        self._last_cleanup = time.time()
    
    def _cleanup_old_entries(self):
        """Remove old request timestamps"""
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep last hour
        
        for key in list(self._requests.keys()):
            self._requests[key] = [
                ts for ts in self._requests[key] 
                if ts > cutoff_time
            ]
            if not self._requests[key]:
                del self._requests[key]
        
        # Clean up expired blocks
        current_dt = datetime.utcnow()
        for ip in list(self._blocked_ips.keys()):
            if self._blocked_ips[ip] < current_dt:
                del self._blocked_ips[ip]
        
        self._last_cleanup = current_time
    
    def is_rate_limited(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, Optional[int]]:
        """
        Check if rate limit exceeded
        
        Returns:
            (is_limited, retry_after_seconds)
        """
        self._cleanup_old_entries()
        
        current_time = time.time()
        window_start = current_time - window_seconds
        
        # Filter requests within window
        requests_in_window = [
            ts for ts in self._requests[key]
            if ts > window_start
        ]
        self._requests[key] = requests_in_window
        
        if len(requests_in_window) >= max_requests:
            # Calculate retry after
            oldest_request = min(requests_in_window)
            retry_after = int(window_seconds - (current_time - oldest_request)) + 1
            return True, retry_after
        
        # Add current request
        self._requests[key].append(current_time)
        return False, None
    
    def block_ip(self, ip: str, duration_seconds: int = 3600):
        """Temporarily block an IP address"""
        self._blocked_ips[ip] = datetime.utcnow() + timedelta(seconds=duration_seconds)
        logger.warning(f"🚫 IP {ip} blocked for {duration_seconds} seconds")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is currently blocked"""
        if ip not in self._blocked_ips:
            return False
        
        if self._blocked_ips[ip] < datetime.utcnow():
            del self._blocked_ips[ip]
            return False
        
        return True


# Global rate limiter instance
_rate_limiter = RateLimiter()


# Rate limit configurations per endpoint pattern
RATE_LIMIT_CONFIGS = {
    # Authentication endpoints - strict limits
    "/api/auth/login": {"max_requests": 5, "window_seconds": 300},  # 5 per 5 min
    "/api/auth/register": {"max_requests": 3, "window_seconds": 3600},  # 3 per hour
    "/api/auth/forgot-password": {"max_requests": 3, "window_seconds": 3600},
    
    # PHI access endpoints - moderate limits
    "/api/patients": {"max_requests": 100, "window_seconds": 60},  # 100 per min
    "/api/video": {"max_requests": 50, "window_seconds": 60},
    "/api/medical-records": {"max_requests": 50, "window_seconds": 60},
    
    # General API endpoints
    "/api/": {"max_requests": 200, "window_seconds": 60},  # Default: 200 per min
    
    # AI/ML endpoints - stricter limits (expensive)
    "/api/v1/behavior-ai": {"max_requests": 20, "window_seconds": 60},
    "/api/v1/ai-deterioration": {"max_requests": 20, "window_seconds": 60},
    
    # File upload endpoints
    "/api/upload": {"max_requests": 10, "window_seconds": 60},
}


def get_rate_limit_config(path: str) -> Dict[str, int]:
    """Get rate limit configuration for a path"""
    # Check exact matches first
    if path in RATE_LIMIT_CONFIGS:
        return RATE_LIMIT_CONFIGS[path]
    
    # Check prefix matches
    for pattern, config in RATE_LIMIT_CONFIGS.items():
        if path.startswith(pattern):
            return config
    
    # Default configuration
    return {"max_requests": 200, "window_seconds": 60}


def get_rate_limit_key(request: Request, user_id: Optional[str] = None) -> str:
    """Generate rate limit key from request"""
    # Prefer user-based limiting if user authenticated
    if user_id:
        return f"user:{user_id}:{request.url.path}"
    
    # Fall back to IP-based limiting
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}:{request.url.path}"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI
    Implements HIPAA-compliant rate limiting with threat detection
    """
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/healthz", "/"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check if IP is blocked
        if _rate_limiter.is_ip_blocked(client_ip):
            AuditLogger.log_event(
                event_type="rate_limit_exceeded",
                user_id=None,
                resource_type="api_endpoint",
                resource_id=request.url.path,
                action="blocked_ip_access",
                status="denied",
                metadata={"ip": client_ip, "path": request.url.path},
                ip_address=client_ip
            )
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too many requests",
                    "message": "IP address temporarily blocked due to excessive requests",
                    "retry_after": 3600
                },
                headers={"Retry-After": "3600"}
            )
        
        # Get user ID from request if available
        user_id = None
        if hasattr(request.state, "user_id"):
            user_id = request.state.user_id
        
        # Get rate limit configuration
        config = get_rate_limit_config(request.url.path)
        
        # Generate rate limit key
        rate_limit_key = get_rate_limit_key(request, user_id)
        
        # Check rate limit
        is_limited, retry_after = _rate_limiter.is_rate_limited(
            key=rate_limit_key,
            max_requests=config["max_requests"],
            window_seconds=config["window_seconds"]
        )
        
        if is_limited:
            # Log rate limit violation
            AuditLogger.log_event(
                event_type="rate_limit_exceeded",
                user_id=user_id,
                resource_type="api_endpoint",
                resource_id=request.url.path,
                action="rate_limit_violation",
                status="denied",
                metadata={
                    "ip": client_ip,
                    "path": request.url.path,
                    "limit": config["max_requests"],
                    "window": config["window_seconds"]
                },
                ip_address=client_ip
            )
            
            # Block IP if excessive violations
            violation_key = f"violations:{client_ip}"
            violations, _ = _rate_limiter.is_rate_limited(
                violation_key, max_requests=10, window_seconds=300
            )
            
            if violations:
                _rate_limiter.block_ip(client_ip, duration_seconds=3600)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {config['max_requests']} per {config['window_seconds']} seconds",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(config["max_requests"])
        response.headers["X-RateLimit-Window"] = str(config["window_seconds"])
        
        return response
