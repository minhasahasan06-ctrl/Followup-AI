from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from app.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    secret = settings.DEV_MODE_SECRET or settings.SESSION_SECRET or "fallback-secret-key"
    encoded_jwt = jwt.encode(to_encode, secret, algorithm="HS256")
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify JWT token using DEV_MODE_SECRET.
    This is used for Express-to-Python backend authentication.
    Auth0 tokens are verified separately via app/auth/auth0.py
    """
    try:
        secret = settings.DEV_MODE_SECRET
        if not secret:
            secret = settings.SESSION_SECRET
        if not secret:
            # Use same fallback as Express for dev mode
            secret = "dev-secret-key-for-testing"
        
        payload = jwt.decode(token, secret, algorithms=["HS256"])
        return payload
    except JWTError as e:
        print(f"Token verification failed: {e}")
        return None
