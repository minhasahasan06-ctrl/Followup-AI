"""
Authentication and authorization utilities for HIPAA-compliant video exam system

DEPRECATED: This module is deprecated. Use app.core.authentication instead.
This file is kept for backward compatibility but redirects to the new module.
"""

from typing import Optional
from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import User
from app.core.authentication import (
    get_current_user as _get_current_user,
    get_current_doctor as _get_current_doctor,
    get_current_patient as _get_current_patient,
    require_role as _require_role
)

security = HTTPBearer(auto_error=False)

# Re-export for backward compatibility
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Backward compatibility wrapper - use app.core.authentication.get_current_user
    
    This wrapper properly extracts credentials from FastAPI's dependency injection
    and passes them to the new authentication function, maintaining backward
    compatibility with existing code that uses this module.
    
    Note: With HTTPBearer(auto_error=False), credentials will be None if no
    Authorization header is present, which the underlying function handles correctly.
    """
    return await _get_current_user(request, credentials, db)


async def get_current_doctor(
    current_user: User = Depends(get_current_user)
) -> User:
    """Backward compatibility wrapper"""
    return await _get_current_doctor(current_user)


async def get_current_patient(
    current_user: User = Depends(get_current_user)
) -> User:
    """Backward compatibility wrapper"""
    return await _get_current_patient(current_user)


def require_role(role: str):
    """Backward compatibility wrapper"""
    return _require_role(role)
