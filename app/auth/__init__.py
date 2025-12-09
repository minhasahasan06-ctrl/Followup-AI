"""
Authentication Module for Followup AI
Provides Auth0-based JWT authentication for the FastAPI backend
"""

from app.auth.auth0 import (
    get_current_token,
    get_current_user_id,
    get_optional_token,
    verify_auth0_token,
    authenticate_websocket,
    is_auth0_configured,
    get_auth_status,
    require_permissions,
    TokenPayload,
)
from app.dependencies import get_current_user

__all__ = [
    "get_current_token",
    "get_current_user",
    "get_current_user_id",
    "get_optional_token",
    "verify_auth0_token",
    "authenticate_websocket",
    "is_auth0_configured",
    "get_auth_status",
    "require_permissions",
    "TokenPayload",
]
