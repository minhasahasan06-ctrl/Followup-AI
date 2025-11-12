from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt, jwk
from jose.utils import base64url_decode
from passlib.context import CryptContext
from app.config import settings
import requests
import time

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

_jwks_cache: Optional[Dict[str, Any]] = None
_jwks_cache_timestamp: float = 0
JWKS_CACHE_TTL = 3600


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_cognito_jwks(force_refresh: bool = False):
    global _jwks_cache, _jwks_cache_timestamp
    
    current_time = time.time()
    cache_expired = (current_time - _jwks_cache_timestamp) > JWKS_CACHE_TTL
    
    if not force_refresh and _jwks_cache is not None and not cache_expired:
        return _jwks_cache
    
    if not settings.AWS_COGNITO_REGION or not settings.AWS_COGNITO_USER_POOL_ID:
        raise ValueError("AWS Cognito configuration missing")
    
    jwks_url = f"https://cognito-idp.{settings.AWS_COGNITO_REGION}.amazonaws.com/{settings.AWS_COGNITO_USER_POOL_ID}/.well-known/jwks.json"
    
    try:
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_timestamp = current_time
        return _jwks_cache
    except Exception as e:
        print(f"Error fetching JWKS: {e}")
        if _jwks_cache is not None:
            print("Using stale JWKS cache")
            return _jwks_cache
        raise


def verify_cognito_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        if not settings.AWS_COGNITO_REGION or not settings.AWS_COGNITO_USER_POOL_ID:
            if settings.ENVIRONMENT == "production":
                print("ERROR: AWS Cognito not configured in production. Authentication failed.")
                return None
            
            if not settings.is_dev_mode_enabled():
                print("ERROR: Development mode requires DEV_MODE_SECRET (min 32 chars). Authentication failed.")
                return None
            
            print("WARNING: Using development mode authentication (DEV_MODE_SECRET)")
            return verify_token_dev_mode(token)
        
        headers = jwt.get_unverified_headers(token)
        kid = headers.get('kid')
        
        if not kid:
            print("Token missing 'kid' header")
            return None
        
        jwks = get_cognito_jwks()
        
        key = None
        for k in jwks.get('keys', []):
            if k.get('kid') == kid:
                key = k
                break
        
        if not key:
            print(f"Public key not found for kid: {kid}. Refreshing JWKS cache...")
            jwks = get_cognito_jwks(force_refresh=True)
            for k in jwks.get('keys', []):
                if k.get('kid') == kid:
                    key = k
                    break
            
            if not key:
                print(f"Public key still not found after cache refresh for kid: {kid}")
                return None
        
        public_key = jwk.construct(key)
        
        message, encoded_signature = token.rsplit('.', 1)
        decoded_signature = base64url_decode(encoded_signature.encode())
        
        if not public_key.verify(message.encode(), decoded_signature):
            print("Signature verification failed")
            return None
        
        claims = jwt.get_unverified_claims(token)
        
        issuer = f"https://cognito-idp.{settings.AWS_COGNITO_REGION}.amazonaws.com/{settings.AWS_COGNITO_USER_POOL_ID}"
        if claims.get('iss') != issuer:
            print(f"Invalid issuer: expected {issuer}, got {claims.get('iss')}")
            return None
        
        if claims.get('token_use') != 'access':
            print(f"Invalid token use: expected 'access', got {claims.get('token_use')}")
            return None
        
        if settings.AWS_COGNITO_CLIENT_ID:
            client_id = claims.get('client_id') or claims.get('aud')
            if client_id != settings.AWS_COGNITO_CLIENT_ID:
                print(f"Invalid client_id/audience: expected {settings.AWS_COGNITO_CLIENT_ID}, got {client_id}")
                return None
        
        exp = claims.get('exp', 0)
        if exp < datetime.utcnow().timestamp():
            print(f"Token expired at {datetime.fromtimestamp(exp)}")
            return None
        
        return claims
        
    except Exception as e:
        print(f"Error verifying Cognito token: {e}")
        return None


def verify_token_dev_mode(token: str) -> Optional[Dict[str, Any]]:
    try:
        if not settings.DEV_MODE_SECRET:
            print("ERROR: DEV_MODE_SECRET not configured")
            return None
        
        payload = jwt.decode(token, settings.DEV_MODE_SECRET, algorithms=["HS256"])
        return payload
    except JWTError as e:
        print(f"Development mode token verification failed: {e}")
        return None


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    return verify_cognito_token(token)
