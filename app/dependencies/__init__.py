"""
FastAPI Dependencies Module

Contains authentication, authorization, and other shared dependencies.
"""

from .stytch_auth import (
    M2MTokenPayload,
    StytchM2MValidator,
    get_stytch_validator,
    require_m2m_auth,
    require_scopes,
    require_read_users,
    require_write_users,
    require_read_health,
    require_write_health,
    require_read_ml,
    require_write_ml,
    require_admin,
    optional_m2m_auth,
)

__all__ = [
    "M2MTokenPayload",
    "StytchM2MValidator",
    "get_stytch_validator",
    "require_m2m_auth",
    "require_scopes",
    "require_read_users",
    "require_write_users",
    "require_read_health",
    "require_write_health",
    "require_read_ml",
    "require_write_ml",
    "require_admin",
    "optional_m2m_auth",
]
