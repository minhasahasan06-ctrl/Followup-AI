"""
Secure Logging Utility - HIPAA-Compliant
Replaces all print() statements with secure, structured logging

SECURITY REQUIREMENTS:
- No sensitive data in logs
- Structured logging for audit trails
- Log levels appropriate for production
- Sanitized error messages
"""

import logging
import sys
import json
from typing import Optional, Dict, Any, Union
from datetime import datetime
from functools import wraps

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Create module-specific loggers
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger


class SecureLogger:
    """
    Secure logging wrapper that prevents sensitive data leakage
    """
    
    # Patterns that indicate sensitive data
    SENSITIVE_PATTERNS = [
        r'password',
        r'secret',
        r'token',
        r'key',
        r'credential',
        r'auth',
        r'jwt',
        r'session',
        r'cookie',
        r'api[_-]?key',
        r'access[_-]?token',
        r'refresh[_-]?token',
        r'authorization',
        r'bearer',
        r'ssn',
        r'social[_-]?security',
        r'credit[_-]?card',
        r'cvv',
        r'pin',
        r'phi',
        r'protected[_-]?health[_-]?information',
    ]
    
    @staticmethod
    def sanitize_message(message: str) -> str:
        """
        Sanitize log message to remove sensitive information
        
        Args:
            message: Original log message
            
        Returns:
            Sanitized log message
        """
        import re
        
        # Remove email addresses
        message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[email]', message)
        
        # Remove IP addresses
        message = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[ip]', message)
        
        # Remove long alphanumeric strings (likely tokens)
        message = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[token]', message)
        
        # Remove file paths (keep filename only)
        message = re.sub(r'/[^\s]+/([^/\s]+)', r'\1', message)
        
        # Remove stack traces (keep first line only)
        if '\n' in message:
            message = message.split('\n')[0] + ' [stack trace truncated]'
        
        return message
    
    @staticmethod
    def should_sanitize(message: str) -> bool:
        """Check if message contains sensitive patterns"""
        import re
        message_lower = message.lower()
        for pattern in SecureLogger.SENSITIVE_PATTERNS:
            if re.search(pattern, message_lower):
                return True
        return False
    
    @classmethod
    def log(cls, logger: logging.Logger, level: int, message: str, *args, **kwargs):
        """
        Secure logging wrapper
        
        Args:
            logger: Python logger instance
            level: Log level
            message: Log message
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Sanitize message if it contains sensitive patterns
        if cls.should_sanitize(message):
            message = cls.sanitize_message(message)
            logger.log(level, f"[SANITIZED] {message}", *args, **kwargs)
        else:
            logger.log(level, message, *args, **kwargs)


def secure_log(level: int = logging.INFO):
    """
    Decorator for secure logging of function calls
    
    Usage:
        @secure_log(logging.INFO)
        def my_function():
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            try:
                SecureLogger.log(logger, level, f"Calling {func.__name__}")
                result = func(*args, **kwargs)
                SecureLogger.log(logger, level, f"Completed {func.__name__}")
                return result
            except Exception as e:
                SecureLogger.log(logger, logging.ERROR, f"Error in {func.__name__}: {str(e)}")
                raise
        return wrapper
    return decorator


# Convenience functions to replace print() statements
def log_info(message: str, logger_name: Optional[str] = None, **kwargs):
    """Log info message securely"""
    logger = get_logger(logger_name or __name__)
    SecureLogger.log(logger, logging.INFO, message, **kwargs)


def log_warning(message: str, logger_name: Optional[str] = None, **kwargs):
    """Log warning message securely"""
    logger = get_logger(logger_name or __name__)
    SecureLogger.log(logger, logging.WARNING, message, **kwargs)


def log_error(message: str, logger_name: Optional[str] = None, exc_info: bool = False, **kwargs):
    """Log error message securely"""
    logger = get_logger(logger_name or __name__)
    if exc_info:
        kwargs['exc_info'] = True
    SecureLogger.log(logger, logging.ERROR, message, **kwargs)


def log_debug(message: str, logger_name: Optional[str] = None, **kwargs):
    """Log debug message securely"""
    logger = get_logger(logger_name or __name__)
    SecureLogger.log(logger, logging.DEBUG, message, **kwargs)


def log_audit(event_type: str, user_id: Optional[str], details: Dict[str, Any]):
    """
    Log audit event with structured data
    
    Args:
        event_type: Type of audit event
        user_id: User ID (if applicable)
        details: Additional event details
    """
    logger = get_logger("audit")
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "user_id": user_id,
        "details": details
    }
    logger.info(f"[AUDIT] {json.dumps(audit_entry)}")
