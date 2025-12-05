"""
Input Validation & Sanitization Middleware - HIPAA Security
Prevents injection attacks, XSS, and data corruption
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import re
import html
import json
import logging
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class InputValidationMiddleware(BaseHTTPMiddleware):
    """
    Validates and sanitizes all input to prevent injection attacks
    Implements HIPAA-compliant input validation
    """
    
    # Dangerous patterns to detect
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE|UNION|SCRIPT)\b)",
        r"(--|#|/\*|\*/|;|\||&)",
        r"(\bOR\b.*=.*|\bAND\b.*=.*)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}]",
        r"\b(cat|ls|pwd|whoami|id|uname|wget|curl|nc|netcat)\b",
    ]
    
    def __init__(self, app):
        super().__init__(app)
        self.sql_pattern = re.compile("|".join(self.SQL_INJECTION_PATTERNS), re.IGNORECASE)
        self.xss_pattern = re.compile("|".join(self.XSS_PATTERNS), re.IGNORECASE)
        self.cmd_pattern = re.compile("|".join(self.COMMAND_INJECTION_PATTERNS), re.IGNORECASE)
    
    def sanitize_string(self, value: str) -> str:
        """Sanitize string input"""
        if not isinstance(value, str):
            return value
        
        # HTML escape
        sanitized = html.escape(value)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized
    
    def validate_json(self, data: dict) -> dict:
        """Recursively validate and sanitize JSON data"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            sanitized_key = self.sanitize_string(str(key))
            
            # Validate and sanitize value
            if isinstance(value, str):
                sanitized[sanitized_key] = self.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[sanitized_key] = self.validate_json(value)
            elif isinstance(value, list):
                sanitized[sanitized_key] = [
                    self.sanitize_string(str(v)) if isinstance(v, str) else v
                    for v in value
                ]
            else:
                sanitized[sanitized_key] = value
        
        return sanitized
    
    def detect_malicious_patterns(self, value: str) -> list[str]:
        """Detect malicious patterns in input"""
        threats = []
        
        if self.sql_pattern.search(value):
            threats.append("sql_injection")
        
        if self.xss_pattern.search(value):
            threats.append("xss")
        
        if self.cmd_pattern.search(value):
            threats.append("command_injection")
        
        return threats
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip validation for certain endpoints
        skip_paths = ["/health", "/healthz", "/docs", "/openapi.json", "/redoc"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Validate query parameters
        for param_name, param_value in request.query_params.items():
            if isinstance(param_value, str):
                threats = self.detect_malicious_patterns(param_value)
                if threats:
                    logger.warning(
                        f"🚨 Malicious pattern detected in query param {param_name}: {threats}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid input detected in parameter: {param_name}"
                    )
        
        # Validate path parameters
        for param_name, param_value in request.path_params.items():
            if isinstance(param_value, str):
                threats = self.detect_malicious_patterns(param_value)
                if threats:
                    logger.warning(
                        f"🚨 Malicious pattern detected in path param {param_name}: {threats}"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Invalid input detected in path parameter: {param_name}"
                    )
        
        # Validate request body for POST/PUT/PATCH
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    # Try to parse as JSON
                    try:
                        json_data = json.loads(body.decode('utf-8'))
                        # Validate JSON structure
                        if isinstance(json_data, dict):
                            threats_found = []
                            for key, value in json_data.items():
                                if isinstance(value, str):
                                    threats = self.detect_malicious_patterns(value)
                                    if threats:
                                        threats_found.extend(threats)
                            
                            if threats_found:
                                logger.warning(
                                    f"🚨 Malicious pattern detected in request body: {threats_found}"
                                )
                                raise HTTPException(
                                    status_code=status.HTTP_400_BAD_REQUEST,
                                    detail="Invalid input detected in request body"
                                )
                    except json.JSONDecodeError:
                        # Not JSON, validate as string
                        body_str = body.decode('utf-8', errors='ignore')
                        threats = self.detect_malicious_patterns(body_str)
                        if threats:
                            logger.warning(
                                f"🚨 Malicious pattern detected in request body: {threats}"
                            )
                            raise HTTPException(
                                status_code=status.HTTP_400_BAD_REQUEST,
                                detail="Invalid input detected in request body"
                            )
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                logger.error(f"Error validating request body: {e}")
        
        return await call_next(request)
