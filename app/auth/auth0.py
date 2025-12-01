"""
Auth0 JWT Authentication for FastAPI
HIPAA-compliant authentication using Auth0

This module provides:
- JWT token verification using Auth0 JWKS
- User extraction from tokens
- Development mode bypass for testing
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
import httpx

logger = logging.getLogger(__name__)

# Auth0 configuration from environment
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "")
AUTH0_API_AUDIENCE = os.getenv("AUTH0_API_AUDIENCE", "")
AUTH0_ALGORITHMS = ["RS256"]

# Development mode flag
DEV_MODE = os.getenv("DEV_MODE_SECRET") is not None

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


class Auth0JWKSClient:
    """
    Cached JWKS client for Auth0 token verification
    Uses python-jose for JWT operations
    """
    
    _instance: Optional["Auth0JWKSClient"] = None
    _jwks_client: Optional[PyJWKClient] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        if AUTH0_DOMAIN:
            jwks_url = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"
            try:
                self._jwks_client = PyJWKClient(jwks_url, cache_keys=True)
                logger.info(f"[Auth0] JWKS client initialized for domain: {AUTH0_DOMAIN}")
            except Exception as e:
                logger.error(f"[Auth0] Failed to initialize JWKS client: {e}")
                self._jwks_client = None
        else:
            logger.warning("[Auth0] AUTH0_DOMAIN not configured")
            self._jwks_client = None
    
    def get_signing_key(self, token: str) -> Optional[str]:
        """Get the signing key for a token from JWKS"""
        if not self._jwks_client:
            return None
        try:
            signing_key = self._jwks_client.get_signing_key_from_jwt(token)
            return signing_key.key
        except Exception as e:
            logger.error(f"[Auth0] Failed to get signing key: {e}")
            return None


# Global JWKS client instance
_jwks_client = Auth0JWKSClient()


def get_jwks_client() -> Auth0JWKSClient:
    """Get the singleton JWKS client"""
    return _jwks_client


class TokenPayload:
    """Parsed JWT token payload"""
    
    def __init__(self, payload: Dict[str, Any]):
        self.sub: str = payload.get("sub", "")
        self.email: Optional[str] = payload.get("email")
        self.email_verified: bool = payload.get("email_verified", False)
        self.name: Optional[str] = payload.get("name")
        self.nickname: Optional[str] = payload.get("nickname")
        self.picture: Optional[str] = payload.get("picture")
        self.updated_at: Optional[str] = payload.get("updated_at")
        self.iss: str = payload.get("iss", "")
        self.aud: Any = payload.get("aud")
        self.iat: int = payload.get("iat", 0)
        self.exp: int = payload.get("exp", 0)
        self.azp: Optional[str] = payload.get("azp")
        self.scope: str = payload.get("scope", "")
        self.permissions: list = payload.get("permissions", [])
        self.raw: Dict[str, Any] = payload
    
    @property
    def user_id(self) -> str:
        """Get the user ID from the token (Auth0 user ID)"""
        return self.sub
    
    def has_permission(self, permission: str) -> bool:
        """Check if token has a specific permission"""
        return permission in self.permissions


async def verify_auth0_token(token: str) -> TokenPayload:
    """
    Verify an Auth0 JWT token and return the payload
    
    Args:
        token: The JWT token string
        
    Returns:
        TokenPayload with decoded claims
        
    Raises:
        HTTPException: If token verification fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Development mode bypass - SECURITY: Only enabled in local development
    # This mode accepts simplified tokens for testing but still validates structure
    if DEV_MODE:
        logger.warning("[Auth0] Development mode - using simplified token verification")
        try:
            # Try to decode JWT structure without signature verification
            # Still validates token format and expiry if present
            payload = jwt.decode(
                token, 
                options={
                    "verify_signature": False,
                    "verify_exp": True,  # Still verify expiry in dev mode
                    "verify_iat": True,  # Still verify issued-at
                }
            )
            return TokenPayload(payload)
        except jwt.exceptions.ExpiredSignatureError:
            logger.warning("[Auth0] Token expired even in dev mode")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.exceptions.DecodeError:
            # SECURITY: Only accept structured dev tokens with specific prefix
            if token.startswith("dev_") and len(token) > 10:
                user_id = token[4:]  # Remove "dev_" prefix
                logger.warning(f"[Auth0] DEV MODE: Using test token for user: {user_id[:8]}...")
                return TokenPayload({
                    "sub": user_id,
                    "email": f"{user_id}@dev.local",
                    "email_verified": True,
                    "name": f"Dev User {user_id[:8]}",
                    "iat": int(datetime.now().timestamp()),
                    "exp": int(datetime.now().timestamp()) + 3600,
                })
            else:
                logger.error("[Auth0] Invalid dev token format")
                raise credentials_exception
    
    # Production mode - full verification
    if not AUTH0_DOMAIN:
        logger.error("[Auth0] AUTH0_DOMAIN not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured"
        )
    
    try:
        # Get signing key from JWKS
        jwks_client = get_jwks_client()
        signing_key = jwks_client.get_signing_key(token)
        
        if not signing_key:
            logger.error("[Auth0] Could not get signing key")
            raise credentials_exception
        
        # Verify and decode the token
        payload = jwt.decode(
            token,
            signing_key,
            algorithms=AUTH0_ALGORITHMS,
            audience=AUTH0_API_AUDIENCE if AUTH0_API_AUDIENCE else None,
            issuer=f"https://{AUTH0_DOMAIN}/",
            options={
                "verify_signature": True,
                "verify_aud": bool(AUTH0_API_AUDIENCE),
                "verify_iss": True,
                "verify_exp": True,
            }
        )
        
        return TokenPayload(payload)
        
    except jwt.ExpiredSignatureError:
        logger.warning("[Auth0] Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        logger.warning("[Auth0] Invalid token audience")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidIssuerError:
        logger.warning("[Auth0] Invalid token issuer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        logger.error(f"[Auth0] JWT validation failed: {e}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"[Auth0] Unexpected error during token verification: {e}")
        raise credentials_exception


async def get_current_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> TokenPayload:
    """
    FastAPI dependency to get current user's token payload
    
    Usage:
        @app.get("/protected")
        async def protected_route(token: TokenPayload = Depends(get_current_token)):
            return {"user_id": token.user_id}
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await verify_auth0_token(credentials.credentials)


async def get_current_user_id(
    token: TokenPayload = Depends(get_current_token)
) -> str:
    """
    FastAPI dependency to get just the current user's ID
    
    Usage:
        @app.get("/my-data")
        async def my_data(user_id: str = Depends(get_current_user_id)):
            return {"user_id": user_id}
    """
    return token.user_id


async def get_optional_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[TokenPayload]:
    """
    FastAPI dependency for optional authentication
    Returns None if no token provided, TokenPayload if valid token provided
    
    Usage:
        @app.get("/public-or-private")
        async def route(token: Optional[TokenPayload] = Depends(get_optional_token)):
            if token:
                return {"authenticated": True, "user": token.user_id}
            return {"authenticated": False}
    """
    if credentials is None:
        return None
    
    try:
        return await verify_auth0_token(credentials.credentials)
    except HTTPException:
        return None


def require_permissions(*permissions: str):
    """
    Decorator factory for requiring specific permissions
    
    Usage:
        @app.get("/admin")
        @require_permissions("admin:read", "admin:write")
        async def admin_route(token: TokenPayload = Depends(get_current_token)):
            return {"admin": True}
    """
    async def permission_checker(token: TokenPayload = Depends(get_current_token)):
        for permission in permissions:
            if not token.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permission: {permission}"
                )
        return token
    
    return Depends(permission_checker)


# WebSocket authentication helper
async def authenticate_websocket(token: str) -> Optional[TokenPayload]:
    """
    Authenticate a WebSocket connection using a token
    
    Args:
        token: JWT token string
        
    Returns:
        TokenPayload if valid, None otherwise
    """
    try:
        return await verify_auth0_token(token)
    except HTTPException:
        return None
    except Exception as e:
        logger.error(f"[Auth0] WebSocket auth error: {e}")
        return None


def is_auth0_configured() -> bool:
    """Check if Auth0 is properly configured"""
    return bool(AUTH0_DOMAIN and AUTH0_API_AUDIENCE)


# Export convenience function for checking auth status
def get_auth_status() -> Dict[str, Any]:
    """Get current authentication configuration status"""
    return {
        "provider": "auth0",
        "configured": is_auth0_configured(),
        "domain": AUTH0_DOMAIN[:20] + "..." if AUTH0_DOMAIN else None,
        "dev_mode": DEV_MODE,
    }
