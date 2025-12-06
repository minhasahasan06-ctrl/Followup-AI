"""
Security utilities - DEPRECATED
Use app.core.authentication instead for JWT verification
This module is kept for backward compatibility
"""

from typing import Optional, Dict, Any
from passlib.context import CryptContext
from app.core.authentication import verify_cognito_token as _verify_cognito_token
from app.core.logging import log_warning

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def verify_cognito_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify Cognito token - DEPRECATED
    Use app.core.authentication.verify_cognito_token instead
    """
    log_warning(
        "app.utils.security.verify_cognito_token is deprecated. "
        "Use app.core.authentication.verify_cognito_token instead",
        logger_name="security"
    )
    try:
        return _verify_cognito_token(token)
    except Exception:
        return None


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify token - DEPRECATED, use app.core.authentication instead"""
    return verify_cognito_token(token)
