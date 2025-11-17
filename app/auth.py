"""
Authentication and authorization utilities
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import os

from app.database import get_db
from app.models import User

# AWS Cognito configuration
COGNITO_REGION = os.getenv("AWS_COGNITO_REGION", "us-east-1")
COGNITO_USER_POOL_ID = os.getenv("AWS_COGNITO_USER_POOL_ID")
COGNITO_CLIENT_ID = os.getenv("AWS_COGNITO_CLIENT_ID")

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    Verify JWT token and return current user.
    In production, this would verify against AWS Cognito.
    For development, we use a simplified approach.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        
        # TODO: In production, verify JWT with AWS Cognito JWKS
        # For now, decode without verification (DEVELOPMENT ONLY)
        payload = jwt.decode(
            token,
            options={"verify_signature": False}  # DEVELOPMENT ONLY
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise credentials_exception
        
        return user
        
    except JWTError:
        raise credentials_exception
