"""
Application Configuration
Centralized settings for the entire application
"""

import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # AWS Cognito
    AWS_COGNITO_REGION: str = os.getenv("AWS_COGNITO_REGION", "us-east-1")
    AWS_COGNITO_USER_POOL_ID: str = os.getenv("AWS_COGNITO_USER_POOL_ID", "")
    AWS_COGNITO_CLIENT_ID: str = os.getenv("AWS_COGNITO_CLIENT_ID", "")
    AWS_COGNITO_CLIENT_SECRET: str = os.getenv("AWS_COGNITO_CLIENT_SECRET", "")
    
    # Session
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "")
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BAA_SIGNED: bool = os.getenv("OPENAI_BAA_SIGNED", "false").lower() == "true"
    
    # Redis (for ML caching)
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    
    # ML Model Paths
    ML_MODELS_DIR: str = os.getenv("ML_MODELS_DIR", "./ml_models")
    
    # CORS - configurable via environment variable (comma-separated list)
    # In production, set CORS_ORIGINS to your frontend domain(s)
    CORS_ORIGINS: list = [
        origin.strip() 
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:5000,http://localhost:3000").split(",")
        if origin.strip()
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
