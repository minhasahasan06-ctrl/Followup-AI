"""
Unified Authentication & Authorization Module - HIPAA-Compliant
Consolidates all authentication logic into a single secure module

SECURITY REQUIREMENTS:
- JWT verification using AWS Cognito JWKS
- No development mode bypasses in production
- Comprehensive audit logging
- Secure error handling without information leakage
"""

import logging
import os
from typing import Optional, Dict, Any
from functools import lru_cache
from datetime import datetime

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from jose import JWTError, jwt, jwk
from jose.utils import base64url_decode
import requests

from app.database import get_db
from app.models import User
from app.services.audit_logger import AuditLogger, AuditEvent
from app.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)


class AuthenticationError(Exception):
    """Base exception for authentication errors - sanitized for security"""
    pass


class TokenVerificationError(AuthenticationError):
    """Token verification failed"""
    pass


class AuthorizationError(AuthenticationError):
    """Authorization failed"""
    pass


class CognitoJWKSManager:
    """
    Manages AWS Cognito JWKS (JSON Web Key Set) for JWT verification
    Implements caching and secure key retrieval
    """
    
    def __init__(self):
        self.region = settings.AWS_COGNITO_REGION or "us-east-1"
        self.user_pool_id = settings.AWS_COGNITO_USER_POOL_ID
        self.client_id = settings.AWS_COGNITO_CLIENT_ID
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_timestamp: float = 0
        self._jwks_cache_ttl = 3600  # 1 hour
    
    def _get_jwks_url(self) -> str:
        """Construct JWKS URL from configuration"""
        if not self.user_pool_id or not self.region:
            raise ValueError("AWS Cognito configuration missing")
        return f"https://cognito-idp.{self.region}.amazonaws.com/{self.user_pool_id}/.well-known/jwks.json"
    
    def get_jwks(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get JWKS with caching
        
        Args:
            force_refresh: Force refresh of cache
            
        Returns:
            JWKS dictionary
        """
        import time
        current_time = time.time()
        cache_expired = (current_time - self._jwks_cache_timestamp) > self._jwks_cache_ttl
        
        if not force_refresh and self._jwks_cache is not None and not cache_expired:
            return self._jwks_cache
        
        try:
            jwks_url = self._get_jwks_url()
            response = requests.get(jwks_url, timeout=10)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cache_timestamp = current_time
            logger.debug(f"JWKS cache refreshed from {jwks_url}")
            return self._jwks_cache
        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}", exc_info=True)
            if self._jwks_cache is not None:
                logger.warning("Using stale JWKS cache")
                return self._jwks_cache
            raise
    
    def get_signing_key(self, kid: str) -> Any:
        """
        Get signing key for a given key ID
        
        Args:
            kid: Key ID from JWT header
            
        Returns:
            Public key for verification
        """
        jwks = self.get_jwks()
        keys = jwks.get('keys', [])
        
        for key in keys:
            if key.get('kid') == kid:
                return jwk.construct(key)
        
        # Try refreshing cache
        logger.warning(f"Key {kid} not found in cache, refreshing JWKS")
        jwks = self.get_jwks(force_refresh=True)
        keys = jwks.get('keys', [])
        
        for key in keys:
            if key.get('kid') == kid:
                return jwk.construct(key)
        
        raise ValueError(f"Public key not found for kid: {kid}")


# Global JWKS manager instance
_jwks_manager: Optional[CognitoJWKSManager] = None


def get_jwks_manager() -> CognitoJWKSManager:
    """Get or create JWKS manager instance"""
    global _jwks_manager
    if _jwks_manager is None:
        _jwks_manager = CognitoJWKSManager()
    return _jwks_manager


def verify_cognito_token(token: str, request: Optional[Request] = None) -> Dict[str, Any]:
    """
    Verify AWS Cognito JWT token with comprehensive security checks
    
    Args:
        token: JWT token string
        request: FastAPI request object for audit logging
        
    Returns:
        Decoded token payload
        
    Raises:
        TokenVerificationError: If token verification fails
    """
    # Check configuration
    if not settings.AWS_COGNITO_REGION or not settings.AWS_COGNITO_USER_POOL_ID:
        if settings.ENVIRONMENT == "production":
            logger.error("AWS Cognito not configured in production - authentication blocked")
            raise TokenVerificationError("Authentication service unavailable")
        
        # Development mode - require secure secret
        if not settings.is_dev_mode_enabled():
            logger.error("Development mode requires DEV_MODE_SECRET (min 32 chars)")
            raise TokenVerificationError("Authentication configuration invalid")
        
        logger.warning("Using development mode authentication")
        return _verify_dev_mode_token(token)
    
    try:
        # Get token header to extract key ID
        headers = jwt.get_unverified_headers(token)
        kid = headers.get('kid')
        
        if not kid:
            logger.warning("Token missing 'kid' header")
            raise TokenVerificationError("Invalid token format")
        
        # Get signing key
        jwks_manager = get_jwks_manager()
        public_key = jwks_manager.get_signing_key(kid)
        
        # Verify signature
        message, encoded_signature = token.rsplit('.', 1)
        decoded_signature = base64url_decode(encoded_signature.encode())
        
        if not public_key.verify(message.encode(), decoded_signature):
            logger.warning("Token signature verification failed")
            raise TokenVerificationError("Invalid token signature")
        
        # Get claims (without verification yet)
        claims = jwt.get_unverified_claims(token)
        
        # Verify issuer
        expected_issuer = f"https://cognito-idp.{settings.AWS_COGNITO_REGION}.amazonaws.com/{settings.AWS_COGNITO_USER_POOL_ID}"
        if claims.get('iss') != expected_issuer:
            logger.warning(f"Invalid issuer: expected {expected_issuer}, got {claims.get('iss')}")
            raise TokenVerificationError("Invalid token issuer")
        
        # Verify token use
        token_use = claims.get('token_use')
        if token_use not in ['id', 'access']:
            logger.warning(f"Invalid token use: {token_use}")
            raise TokenVerificationError("Invalid token type")
        
        # Verify audience/client ID
        if settings.AWS_COGNITO_CLIENT_ID:
            client_id = claims.get('client_id') or claims.get('aud')
            if client_id != settings.AWS_COGNITO_CLIENT_ID:
                logger.warning(f"Invalid client_id: expected {settings.AWS_COGNITO_CLIENT_ID}, got {client_id}")
                raise TokenVerificationError("Invalid token audience")
        
        # Verify expiration
        exp = claims.get('exp', 0)
        if exp < datetime.utcnow().timestamp():
            logger.warning(f"Token expired at {datetime.fromtimestamp(exp)}")
            raise TokenVerificationError("Token expired")
        
        return claims
        
    except TokenVerificationError:
        raise
    except Exception as e:
        logger.error(f"Token verification error: {e}", exc_info=True)
        raise TokenVerificationError("Token verification failed")


def _verify_dev_mode_token(token: str) -> Dict[str, Any]:
    """
    Verify token in development mode (requires DEV_MODE_SECRET)
    SECURITY: Only available when DEV_MODE_SECRET is set (min 32 chars)
    """
    if not settings.DEV_MODE_SECRET or len(settings.DEV_MODE_SECRET) < 32:
        raise TokenVerificationError("Development mode requires secure secret")
    
    try:
        payload = jwt.decode(
            token,
            settings.DEV_MODE_SECRET,
            algorithms=["HS256"]
        )
        return payload
    except JWTError as e:
        logger.warning(f"Development mode token verification failed: {e}")
        raise TokenVerificationError("Invalid development token")


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency to get current authenticated user
    
    SECURITY FEATURES:
    - JWT verification with AWS Cognito JWKS
    - Comprehensive audit logging
    - Secure error handling
    - User existence verification
    
    Args:
        request: FastAPI request object
        credentials: Bearer token credentials
        db: Database session
        
    Returns:
        Authenticated User object
        
    Raises:
        HTTPException: 401 if authentication fails
    """
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    
    # Check for token
    if not credentials:
        AuditLogger.log_event(
            event_type=AuditEvent.AUTH_FAILED,
            user_id=None,
            resource_type="api_endpoint",
            resource_id=request.url.path,
            action="authenticate",
            status="denied",
            metadata={"reason": "missing_token"},
            ip_address=ip_address,
            user_agent=user_agent
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        token = credentials.credentials
        
        # Verify token
        payload = verify_cognito_token(token, request)
        
        # Extract user ID
        user_id: str = payload.get("sub")
        if not user_id or not isinstance(user_id, str):
            raise TokenVerificationError("Invalid user ID in token")
        
        # Verify user exists in database
        user = db.query(User).filter(User.id == user_id).first()
        if user is None:
            AuditLogger.log_event(
                event_type=AuditEvent.AUTH_FAILED,
                user_id=user_id,
                resource_type="user",
                resource_id=user_id,
                action="authenticate",
                status="denied",
                metadata={"reason": "user_not_found"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Log successful authentication
        AuditLogger.log_event(
            event_type=AuditEvent.USER_LOGIN,
            user_id=user_id,
            resource_type="user",
            resource_id=user_id,
            action="authenticate",
            status="success",
            metadata={"role": user.role},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return user
        
    except TokenVerificationError as e:
        logger.warning(f"Token verification failed: {e}")
        AuditLogger.log_event(
            event_type=AuditEvent.AUTH_FAILED,
            user_id=None,
            resource_type="api_endpoint",
            resource_id=request.url.path,
            action="authenticate",
            status="denied",
            metadata={"reason": "token_verification_failed"},
            ip_address=ip_address,
            user_agent=user_agent
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}", exc_info=True)
        AuditLogger.log_event(
            event_type=AuditEvent.AUTH_FAILED,
            user_id=None,
            resource_type="api_endpoint",
            resource_id=request.url.path,
            action="authenticate",
            status="denied",
            metadata={"reason": "system_error"},
            ip_address=ip_address,
            user_agent=user_agent
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_doctor(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    FastAPI dependency to ensure current user is a doctor
    
    Returns:
        User object with doctor role
        
    Raises:
        HTTPException: 403 if user is not a doctor
    """
    user_role = str(current_user.role) if current_user.role else ""
    if user_role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Doctor access required"
        )
    return current_user


async def get_current_patient(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    FastAPI dependency to ensure current user is a patient
    
    Returns:
        User object with patient role
        
    Raises:
        HTTPException: 403 if user is not a patient
    """
    user_role = str(current_user.role) if current_user.role else ""
    if user_role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Patient access required"
        )
    return current_user


def require_role(role: str):
    """
    Dependency factory for role-based access control
    
    Args:
        role: Required role ('doctor', 'patient', etc.)
        
    Returns:
        FastAPI dependency function
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = str(current_user.role) if current_user.role else ""
        if user_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{role.title()} access required"
            )
        return current_user
    return role_checker
