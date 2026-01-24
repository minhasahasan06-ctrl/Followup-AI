"""
Authentication Module for Followup AI

MIGRATION STATUS:
- Auth0 JWT authentication is DEPRECATED (legacy)
- Stytch M2M is the PRIMARY auth method for Python backend
- See app/dependencies.py for Stytch M2M validation

Auth flow:
1. Express backend authenticates users via Stytch (magic links, sessions)
2. Express calls Python FastAPI using Stytch M2M tokens
3. Python validates M2M tokens via app/dependencies.py

Legacy Auth0 exports below are for backward compatibility only.
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
