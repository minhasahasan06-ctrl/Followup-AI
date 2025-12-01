"""
Authentication and authorization utilities for HIPAA-compliant video exam system

SECURITY WARNING: This module currently uses DEVELOPMENT-ONLY mode with disabled JWT verification.
In production, this MUST be replaced with proper AWS Cognito JWT verification using JWKS.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import os
import requests
from functools import lru_cache

from app.database import get_db
from app.models import User

# AWS Cognito configuration
COGNITO_REGION = os.getenv("AWS_COGNITO_REGION", "us-east-1")
COGNITO_USER_POOL_ID = os.getenv("AWS_COGNITO_USER_POOL_ID")
COGNITO_CLIENT_ID = os.getenv("AWS_COGNITO_CLIENT_ID")
DEV_MODE = os.getenv("DEV_MODE_SECRET") is not None  # Development mode flag

security = HTTPBearer()


@lru_cache(maxsize=1)
def get_cognito_jwks():
    """
    Fetch AWS Cognito JWKS (JSON Web Key Set) for JWT verification
    Cached to avoid repeated network calls
    """
    if not COGNITO_USER_POOL_ID or not COGNITO_REGION:
        return None
    
    jwks_url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json"
    
    try:
        response = requests.get(jwks_url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[AUTH ERROR] Failed to fetch JWKS: {str(e)}")
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Verify JWT token and return current user.
    
    PRODUCTION MODE: Verifies JWT signature using AWS Cognito JWKS
    DEVELOPMENT MODE: Skips signature verification (DEV_MODE_SECRET must be set)
    
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
        
        # PRODUCTION: Verify JWT with AWS Cognito JWKS
        if not DEV_MODE:
            jwks = get_cognito_jwks()
            if not jwks:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication service unavailable"
                )
            
            # Verify JWT signature and claims
            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=COGNITO_CLIENT_ID,
                options={
                    "verify_signature": True,
                    "verify_aud": True,
                    "verify_exp": True
                }
            )
        else:
            # DEVELOPMENT ONLY: Skip signature verification
            print("[AUTH WARNING] Development mode - JWT signature verification disabled")
            payload = jwt.decode(
                token,
                "",  # Empty key when not verifying signature
                options={"verify_signature": False}
            )
        
        # Extract user ID from token
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Verify user exists in database
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
