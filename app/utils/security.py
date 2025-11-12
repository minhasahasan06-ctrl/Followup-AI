from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt, jwk
from jose.utils import base64url_decode
from passlib.context import CryptContext
from app.config import settings
import requests
import json

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

_jwks_cache: Optional[Dict[str, Any]] = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def get_cognito_jwks():
    global _jwks_cache
    
    if _jwks_cache is not None:
        return _jwks_cache
    
    if not settings.AWS_COGNITO_REGION or not settings.AWS_COGNITO_USER_POOL_ID:
        raise ValueError("AWS Cognito configuration missing")
    
    jwks_url = f"https://cognito-idp.{settings.AWS_COGNITO_REGION}.amazonaws.com/{settings.AWS_COGNITO_USER_POOL_ID}/.well-known/jwks.json"
    
    try:
        response = requests.get(jwks_url)
        _jwks_cache = response.json()
        return _jwks_cache
    except Exception as e:
        print(f"Error fetching JWKS: {e}")
        raise


def verify_cognito_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        if not settings.AWS_COGNITO_REGION or not settings.AWS_COGNITO_USER_POOL_ID:
            print("WARNING: AWS Cognito not configured. Using development mode authentication.")
            return verify_token_dev_mode(token)
        
        headers = jwt.get_unverified_headers(token)
        kid = headers['kid']
        
        jwks = get_cognito_jwks()
        
        key = None
        for k in jwks['keys']:
            if k['kid'] == kid:
                key = k
                break
        
        if not key:
            print(f"Public key not found for kid: {kid}")
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
            print(f"Invalid issuer: {claims.get('iss')}")
            return None
        
        if claims.get('token_use') != 'access':
            print(f"Invalid token use: {claims.get('token_use')}")
            return None
        
        if claims.get('exp', 0) < datetime.utcnow().timestamp():
            print("Token expired")
            return None
        
        return claims
        
    except Exception as e:
        print(f"Error verifying Cognito token: {e}")
        return None


def verify_token_dev_mode(token: str) -> Optional[Dict[str, Any]]:
    try:
        payload = jwt.decode(token, settings.SESSION_SECRET, algorithms=["HS256"])
        return payload
    except JWTError as e:
        print(f"Development mode token verification failed: {e}")
        return None


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    return verify_cognito_token(token)
