"""
Stytch M2M Authentication Dependency for FastAPI

This module provides authentication and authorization for M2M (Machine-to-Machine)
requests from the Express backend to FastAPI endpoints using Stytch tokens.
"""

import os
import httpx
from typing import Optional, List
from functools import lru_cache
from fastapi import Depends, HTTPException, Header
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
                status_code=503,
                detail="Stytch authentication not configured",
            )

        client = await self.get_client()

        payload = {
            "access_token": access_token,
        }
        if required_scopes:
            payload["required_scopes"] = required_scopes

        try:
            response = await client.post(
                "/m2m/authenticate",
                json=payload,
            )

            if response.status_code == 200:
                data = response.json()
                return M2MTokenPayload(
                    client_id=data.get("client_id", ""),
                    scopes=data.get("scopes", []),
                    custom_claims=data.get("custom_claims"),
                )
            elif response.status_code == 401:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid M2M token",
                )
            elif response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient scopes for this operation",
                )
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Stytch authentication error: {response.text}",
                )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Stytch service unavailable: {str(e)}",
            )

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

@lru_cache()
def get_stytch_validator() -> StytchM2MValidator:
    return StytchM2MValidator()

def extract_bearer_token(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization:
        return None
    if not authorization.startswith("Bearer "):
        return None
    return authorization[7:]

async def require_m2m_auth(
    authorization: Optional[str] = Header(None),
    validator: StytchM2MValidator = Depends(get_stytch_validator),
) -> M2MTokenPayload:
    token = extract_bearer_token(authorization)

    if not token:
        if not validator.is_configured and token == DEV_MODE_SECRET:
            return M2MTokenPayload(
                client_id="dev-client",
                scopes=["admin:all"],
            )
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

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
            raise HTTPException(
                status_code=401,
                detail="Missing Authorization header",
            )

        if token == DEV_MODE_SECRET:
            return M2MTokenPayload(
                client_id="express-backend",
                scopes=list(scopes),
            )

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
