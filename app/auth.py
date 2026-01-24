"""
Authentication and authorization utilities for HIPAA-compliant video exam system

This module uses JWT tokens signed with DEV_MODE_SECRET for Express-to-Python authentication.
For production authentication, use Stytch M2M tokens - see app/dependencies.py
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import os

from app.database import get_db
from app.models import User

DEV_MODE_SECRET = os.getenv("DEV_MODE_SECRET")
SESSION_SECRET = os.getenv("SESSION_SECRET")

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Verify JWT token and return current user.
    
    Uses DEV_MODE_SECRET or SESSION_SECRET to verify HS256-signed JWT tokens
    from the Express server.
    
    Args:
        credentials: Bearer token from Authorization header
        db: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: 401 if authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        secret = DEV_MODE_SECRET or SESSION_SECRET or "dev-secret-key-for-testing"
        
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in database"
            )
        
        return user
        
    except JWTError as e:
        print(f"[AUTH ERROR] JWT validation failed: {str(e)}")
        raise credentials_exception
    except Exception as e:
        print(f"[AUTH ERROR] Authentication error: {str(e)}")
        raise credentials_exception
