"""
Error Handling & Sanitization - HIPAA-Compliant
Prevents information leakage through error messages

SECURITY REQUIREMENTS:
- No sensitive data in error responses
- Generic error messages for users
- Detailed errors only in secure logs
- Consistent error format
"""

import logging
import traceback
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import SecureLogger, log_error

logger = logging.getLogger(__name__)


class ErrorSanitizer:
    """Sanitizes errors to prevent information leakage"""
    
    # Error messages that are safe to expose
    SAFE_ERROR_MESSAGES = {
        "authentication_required": "Authentication required",
        "invalid_credentials": "Invalid authentication credentials",
        "access_denied": "Access denied",
        "resource_not_found": "Resource not found",
        "validation_error": "Validation error",
        "rate_limit_exceeded": "Rate limit exceeded",
        "service_unavailable": "Service temporarily unavailable",
    }
    
    @staticmethod
    def sanitize_error(error: Exception, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Sanitize error for client response
        
        Args:
            error: Exception instance
            context: Additional context
            
        Returns:
            Sanitized error dictionary
        """
        error_type = type(error).__name__
        error_message = str(error)
        
        # Check if it's a known safe error
        if isinstance(error, HTTPException):
            return {
                "error": error.detail,
                "status_code": error.status_code,
                "type": "http_exception"
            }
        
        # Check for sensitive patterns
        error_lower = error_message.lower()
        sensitive_patterns = [
            'password', 'secret', 'token', 'key', 'credential',
            'database', 'connection', 'sql', 'query', 'stack',
            'traceback', 'file', 'path', 'internal', 'server'
        ]
        
        if any(pattern in error_lower for pattern in sensitive_patterns):
            # Use generic error message
            return {
                "error": "An error occurred processing your request",
                "status_code": 500,
                "type": "internal_error",
                "error_id": ErrorSanitizer._generate_error_id()
            }
        
        # For known error types, provide safe messages
        if error_type in ["ValidationError", "ValueError"]:
            return {
                "error": "Validation error",
                "status_code": 400,
                "type": "validation_error"
            }
        
        if error_type in ["PermissionError", "Forbidden"]:
            return {
                "error": "Access denied",
                "status_code": 403,
                "type": "access_denied"
            }
        
        if error_type in ["NotFound", "FileNotFoundError"]:
            return {
                "error": "Resource not found",
                "status_code": 404,
                "type": "not_found"
            }
        
        # Default generic error
        return {
            "error": "An error occurred processing your request",
            "status_code": 500,
            "type": "internal_error",
            "error_id": ErrorSanitizer._generate_error_id()
        }
    
    @staticmethod
    def _generate_error_id() -> str:
        """Generate a unique error ID for tracking"""
        import uuid
        return str(uuid.uuid4())[:8]


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and sanitize all errors
    Prevents information leakage while maintaining audit trail
    """
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            # SECURITY: Sanitize HTTPException.detail to prevent information leakage
            # Even HTTPExceptions can contain sensitive information (PHI, database errors, etc.)
            # Log full details securely, but return sanitized version
            error_id = ErrorSanitizer._generate_error_id()
            log_error(
                f"HTTPException [{error_id}]: {e.status_code} - {e.detail}",
                logger_name="error_handler",
                exc_info=False  # HTTPExceptions are expected, don't need full traceback
            )
            
            # Sanitize the HTTPException detail using ErrorSanitizer
            # Create a temporary exception to leverage sanitization logic
            temp_exception = Exception(str(e.detail))
            sanitized = ErrorSanitizer.sanitize_error(temp_exception)
            
            # Preserve the original status code from HTTPException
            sanitized["status_code"] = e.status_code
            sanitized["error_id"] = error_id
            sanitized["type"] = "http_exception"
            
            return JSONResponse(
                status_code=e.status_code,
                content=sanitized
            )
        except Exception as e:
            # Log full error details securely
            error_id = ErrorSanitizer._generate_error_id()
            log_error(
                f"Unhandled exception [{error_id}]: {type(e).__name__}: {str(e)}",
                logger_name="error_handler",
                exc_info=True
            )
            
            # Return sanitized error
            sanitized = ErrorSanitizer.sanitize_error(e)
            sanitized["error_id"] = error_id
            
            return JSONResponse(
                status_code=sanitized["status_code"],
                content=sanitized
            )


def create_error_response(
    error: Exception,
    status_code: int = 500,
    context: Optional[str] = None
) -> JSONResponse:
    """
    Create a sanitized error response
    
    Args:
        error: Exception instance
        status_code: HTTP status code
        context: Additional context
        
    Returns:
        JSONResponse with sanitized error
    """
    sanitized = ErrorSanitizer.sanitize_error(error, context)
    sanitized["status_code"] = status_code
    
    # Log error securely
    log_error(
        f"Error response: {type(error).__name__}",
        logger_name="error_handler",
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status_code,
        content=sanitized
    )
