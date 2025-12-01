"""
Authentication API Router
Handles user authentication and profile management with Auth0
"""

import os
import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.auth.auth0 import (
    get_current_token,
    get_optional_token,
    get_auth_status,
    TokenPayload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class UserProfile(BaseModel):
    """User profile response"""
    id: str
    email: Optional[str]
    name: Optional[str]
    picture: Optional[str]
    role: str
    email_verified: bool = False
    created_at: Optional[datetime] = None


class UserRegistration(BaseModel):
    """User registration request"""
    email: Optional[EmailStr] = None
    name: Optional[str] = None
    picture: Optional[str] = None


class AuthStatusResponse(BaseModel):
    """Authentication status response"""
    provider: str
    configured: bool
    domain: Optional[str]
    dev_mode: bool


@router.get("/status", response_model=AuthStatusResponse)
async def auth_status():
    """
    Get authentication configuration status
    Useful for debugging auth issues
    """
    return get_auth_status()


@router.get("/me", response_model=UserProfile)
async def get_current_user(
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Get the current authenticated user's profile
    
    Requires valid Auth0 JWT token in Authorization header
    """
    # Try to find user in database
    user = db.query(User).filter(User.id == token.user_id).first()
    
    if not user:
        # Try by email
        if token.email:
            user = db.query(User).filter(User.email == token.email).first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found. Please register first."
        )
    
    user_email = getattr(user, 'email', None)
    user_name = getattr(user, 'name', None) or getattr(user, 'first_name', None)
    user_picture = getattr(user, 'picture', None) or getattr(user, 'profile_image', None)
    user_role = getattr(user, 'role', 'patient')
    
    return UserProfile(
        id=str(user.id),
        email=str(user_email) if user_email is not None else None,
        name=str(user_name) if user_name else '',
        picture=str(user_picture) if user_picture else None,
        role=str(user_role) if user_role else 'patient',
        email_verified=token.email_verified,
        created_at=getattr(user, 'created_at', None),
    )


@router.post("/register", response_model=UserProfile)
async def register_user(
    registration: UserRegistration,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Register a new user from Auth0 authentication
    
    Called automatically when a user first authenticates
    Creates a local user record linked to Auth0 identity
    """
    # Check if user already exists
    existing_user = db.query(User).filter(User.id == token.user_id).first()
    if existing_user:
        ex_email = getattr(existing_user, 'email', None)
        ex_name = getattr(existing_user, 'name', None) or getattr(existing_user, 'first_name', None)
        ex_picture = getattr(existing_user, 'picture', None)
        ex_role = getattr(existing_user, 'role', 'patient')
        return UserProfile(
            id=str(existing_user.id),
            email=str(ex_email) if ex_email is not None else None,
            name=str(ex_name) if ex_name else '',
            picture=str(ex_picture) if ex_picture else None,
            role=str(ex_role) if ex_role else 'patient',
            email_verified=token.email_verified,
            created_at=getattr(existing_user, 'created_at', None),
        )
    
    # Also check by email
    if token.email:
        existing_by_email = db.query(User).filter(User.email == token.email).first()
        if existing_by_email:
            # Link existing user to Auth0 identity via metadata update
            # Note: We don't change the primary key, just associate the Auth0 sub
            setattr(existing_by_email, 'auth0_id', token.user_id)
            db.commit()
            db.refresh(existing_by_email)
            eb_email = getattr(existing_by_email, 'email', None)
            eb_name = getattr(existing_by_email, 'name', None)
            eb_picture = getattr(existing_by_email, 'picture', None)
            eb_role = getattr(existing_by_email, 'role', 'patient')
            return UserProfile(
                id=str(existing_by_email.id),
                email=str(eb_email) if eb_email is not None else None,
                name=str(eb_name) if eb_name else '',
                picture=str(eb_picture) if eb_picture else None,
                role=str(eb_role) if eb_role else 'patient',
                email_verified=token.email_verified,
                created_at=getattr(existing_by_email, 'created_at', None),
            )
    
    # Create new user
    try:
        new_user = User(
            id=token.user_id,
            email=registration.email or token.email,
            name=registration.name or token.name,
            picture=registration.picture or token.picture,
            role='patient',  # Default role
            email_verified=token.email_verified,
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        logger.info(f"[Auth] Created new user: {new_user.id}")
        
        new_email = getattr(new_user, 'email', None)
        new_name = getattr(new_user, 'name', None)
        new_picture = getattr(new_user, 'picture', None)
        new_role = getattr(new_user, 'role', 'patient')
        return UserProfile(
            id=str(new_user.id),
            email=str(new_email) if new_email is not None else None,
            name=str(new_name) if new_name else '',
            picture=str(new_picture) if new_picture else None,
            role=str(new_role) if new_role else 'patient',
            email_verified=token.email_verified,
            created_at=getattr(new_user, 'created_at', None),
        )
    except Exception as e:
        db.rollback()
        logger.error(f"[Auth] Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )


@router.get("/check")
async def check_auth(
    token: Optional[TokenPayload] = Depends(get_optional_token)
):
    """
    Check if user is authenticated
    Returns auth status without requiring authentication
    """
    if token:
        return {
            "authenticated": True,
            "user_id": token.user_id,
            "email": token.email,
        }
    return {
        "authenticated": False,
        "user_id": None,
        "email": None,
    }


@router.post("/logout")
async def logout(token: TokenPayload = Depends(get_current_token)):
    """
    Server-side logout handler
    Note: Auth0 logout should be done on the client side
    This endpoint is for any server-side cleanup needed
    """
    logger.info(f"[Auth] User logged out: {token.user_id}")
    return {"message": "Logged out successfully"}
