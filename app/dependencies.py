from fastapi import Depends, HTTPException, status, WebSocket
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import Optional, List
from pydantic import BaseModel
from app.database import get_db
from app.utils.security import verify_token
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# ============================================================================
# TOKEN PAYLOAD MODEL (Auth0-compatible)
# ============================================================================

class TokenPayload(BaseModel):
    """JWT token payload - compatible with Auth0/Stytch tokens"""
    user_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    picture: Optional[str] = None
    role: str = "patient"
    permissions: List[str] = []
    email_verified: bool = False


# ============================================================================
# TOKEN-BASED AUTHENTICATION DEPENDENCIES
# ============================================================================

async def get_current_token(
    token: Optional[str] = Depends(oauth2_scheme),
) -> TokenPayload:
    """
    FastAPI dependency that extracts and validates JWT token.
    Returns TokenPayload with user info extracted from token claims.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user ID",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return TokenPayload(
        user_id=str(user_id),
        email=payload.get("email"),
        name=payload.get("name"),
        picture=payload.get("picture"),
        role=payload.get("role", "patient"),
        permissions=payload.get("permissions", []),
        email_verified=payload.get("email_verified", False),
    )


async def get_optional_token(
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[TokenPayload]:
    """
    Same as get_current_token but returns None if no token provided.
    """
    if not token:
        return None
    
    payload = verify_token(token)
    if payload is None:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    return TokenPayload(
        user_id=str(user_id),
        email=payload.get("email"),
        name=payload.get("name"),
        picture=payload.get("picture"),
        role=payload.get("role", "patient"),
        permissions=payload.get("permissions", []),
        email_verified=payload.get("email_verified", False),
    )


async def authenticate_websocket(token: Optional[str]) -> Optional[TokenPayload]:
    """
    Validates websocket token and returns TokenPayload.
    Used for WebSocket authentication.
    """
    if not token:
        return None
    
    payload = verify_token(token)
    if payload is None:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    return TokenPayload(
        user_id=str(user_id),
        email=payload.get("email"),
        name=payload.get("name"),
        picture=payload.get("picture"),
        role=payload.get("role", "patient"),
        permissions=payload.get("permissions", []),
        email_verified=payload.get("email_verified", False),
    )


def is_auth0_configured() -> bool:
    """Returns False - Auth0 is not used, we use Stytch."""
    return False


def get_auth_status() -> dict:
    """Returns auth configuration status."""
    return {
        "provider": "stytch",
        "configured": True,
        "dev_mode": True,
        "domain": None,
    }


def require_permissions(*required_permissions: str):
    """
    Dependency factory for checking permissions.
    Usage: token = Depends(require_permissions("read:users", "write:users"))
    """
    async def permission_checker(
        token: TokenPayload = Depends(get_current_token),
    ) -> TokenPayload:
        if required_permissions:
            missing = [p for p in required_permissions if p not in token.permissions]
            if missing:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing required permissions: {', '.join(missing)}"
                )
        return token
    return permission_checker


# ============================================================================
# USER-BASED AUTHENTICATION DEPENDENCIES (existing)
# ============================================================================

oauth2_scheme_required = OAuth2PasswordBearer(tokenUrl="token")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_token(token)
    if payload is None:
        raise credentials_exception
    
    user_id = payload.get("sub")
    if user_id is None or not isinstance(user_id, str):
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_doctor(
    current_user: User = Depends(get_current_user)
) -> User:
    user_role = str(current_user.role) if current_user.role else ""
    if user_role != "doctor":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access this resource"
        )
    return current_user


async def get_current_patient(
    current_user: User = Depends(get_current_user)
) -> User:
    user_role = str(current_user.role) if current_user.role else ""
    if user_role != "patient":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only patients can access this resource"
        )
    return current_user


def require_role(role: str):
    """
    Dependency factory that creates a role-checking dependency.
    Usage: current_user = Depends(require_role("patient"))
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_role = str(current_user.role) if current_user.role else ""
        if user_role != role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Only {role}s can access this resource"
            )
        return current_user
    return role_checker


# ============================================================================
# STYTCH M2M AUTHENTICATION FOR SERVICE-TO-SERVICE COMMUNICATION
# ============================================================================

import os
import httpx
from typing import Optional, List
from functools import lru_cache
from fastapi import Header
from pydantic import BaseModel

STYTCH_PROJECT_ID = os.getenv("STYTCH_PROJECT_ID")
STYTCH_SECRET = os.getenv("STYTCH_SECRET")
DEV_MODE_SECRET = os.getenv("DEV_MODE_SECRET", "dev-secret-key")


class M2MTokenPayload(BaseModel):
    client_id: str
    scopes: List[str]
    custom_claims: Optional[dict] = None


class StytchM2MValidator:
    def __init__(self):
        self.project_id = STYTCH_PROJECT_ID
        self.secret = STYTCH_SECRET
        self.base_url = "https://api.stytch.com/v1"
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.project_id and self.secret)

    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                auth=(self.project_id, self.secret),
                timeout=10.0,
            )
        return self._client

    async def validate_token(
        self,
        access_token: str,
        required_scopes: List[str] = None,
    ) -> M2MTokenPayload:
        if not self.is_configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Stytch authentication not configured",
            )

        client = await self.get_client()
        payload = {"access_token": access_token}
        if required_scopes:
            payload["required_scopes"] = required_scopes

        try:
            response = await client.post("/m2m/authenticate", json=payload)
            if response.status_code == 200:
                data = response.json()
                return M2MTokenPayload(
                    client_id=data.get("client_id", ""),
                    scopes=data.get("scopes", []),
                    custom_claims=data.get("custom_claims"),
                )
            elif response.status_code == 401:
                raise HTTPException(status_code=401, detail="Invalid M2M token")
            elif response.status_code == 403:
                raise HTTPException(status_code=403, detail="Insufficient scopes")
            else:
                raise HTTPException(status_code=502, detail=f"Stytch error: {response.text}")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Stytch unavailable: {str(e)}")

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()


@lru_cache()
def get_stytch_validator() -> StytchM2MValidator:
    return StytchM2MValidator()


def extract_bearer_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[7:]


async def require_m2m_auth(
    authorization: Optional[str] = Header(None),
    validator: StytchM2MValidator = Depends(get_stytch_validator),
) -> M2MTokenPayload:
    token = extract_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if token == DEV_MODE_SECRET:
        return M2MTokenPayload(
            client_id="express-backend",
            scopes=["read:users", "write:users", "read:health", "write:health", "read:ml", "write:ml"],
        )
    return await validator.validate_token(token)


def require_scopes(*scopes: str):
    async def dependency(
        authorization: Optional[str] = Header(None),
        validator: StytchM2MValidator = Depends(get_stytch_validator),
    ) -> M2MTokenPayload:
        token = extract_bearer_token(authorization)
        if not token:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        if token == DEV_MODE_SECRET:
            return M2MTokenPayload(client_id="express-backend", scopes=list(scopes))
        return await validator.validate_token(token, required_scopes=list(scopes))
    return dependency


require_read_users = require_scopes("read:users")
require_write_users = require_scopes("write:users")
require_read_health = require_scopes("read:health")
require_write_health = require_scopes("write:health")
require_read_ml = require_scopes("read:ml")
require_write_ml = require_scopes("write:ml")
require_admin = require_scopes("admin:all")


async def optional_m2m_auth(
    authorization: Optional[str] = Header(None),
    validator: StytchM2MValidator = Depends(get_stytch_validator),
) -> Optional[M2MTokenPayload]:
    token = extract_bearer_token(authorization)
    if not token:
        return None
    if token == DEV_MODE_SECRET:
        return M2MTokenPayload(
            client_id="express-backend",
            scopes=["read:users", "write:users", "read:health", "write:health", "read:ml", "write:ml"],
        )
    try:
        return await validator.validate_token(token)
    except HTTPException:
        return None
