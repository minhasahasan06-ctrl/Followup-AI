import os
from pydantic_settings import BaseSettings
from typing import Optional
from openai import OpenAI


class Settings(BaseSettings):
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_COGNITO_USER_POOL_ID: Optional[str] = os.getenv("AWS_COGNITO_USER_POOL_ID")
    AWS_COGNITO_CLIENT_ID: Optional[str] = os.getenv("AWS_COGNITO_CLIENT_ID")
    AWS_COGNITO_CLIENT_SECRET: Optional[str] = os.getenv("AWS_COGNITO_CLIENT_SECRET")
    AWS_COGNITO_REGION: Optional[str] = os.getenv("AWS_COGNITO_REGION")
    AWS_COGNITO_DOMAIN: Optional[str] = os.getenv("AWS_COGNITO_DOMAIN")
    AWS_S3_BUCKET_NAME: Optional[str] = os.getenv("AWS_S3_BUCKET_NAME")
    
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OPENAI_BAA_SIGNED: bool = os.getenv("OPENAI_BAA_SIGNED", "false").lower() == "true"
    OPENAI_ZDR_ENABLED: bool = os.getenv("OPENAI_ZDR_ENABLED", "false").lower() == "true"
    OPENAI_ENTERPRISE: bool = os.getenv("OPENAI_ENTERPRISE", "false").lower() == "true"
    
    TWILIO_ACCOUNT_SID: Optional[str] = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: Optional[str] = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER: Optional[str] = os.getenv("TWILIO_PHONE_NUMBER")
    
    DEV_MODE_SECRET: Optional[str] = os.getenv("DEV_MODE_SECRET")
    SESSION_SECRET: Optional[str] = os.getenv("SESSION_SECRET")
    
    CORS_ORIGINS: list = ["http://localhost:5000", "http://127.0.0.1:5000"]
    
    ENVIRONMENT: str = os.getenv("NODE_ENV", "development")
    
    def validate_database_url(self):
        if not self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL environment variable is required for database operations. "
                "Please set it to your PostgreSQL connection string."
            )
    
    def is_dev_mode_enabled(self) -> bool:
        return self.DEV_MODE_SECRET is not None and len(self.DEV_MODE_SECRET) >= 32
    
    class Config:
        env_file = ".env"


settings = Settings()


def get_openai_client() -> OpenAI:
    """
    Get configured OpenAI client instance
    HIPAA Compliance: Uses API key from environment
    """
    if not settings.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def check_openai_baa_compliance():
    """Check OpenAI BAA compliance - uses secure logging"""
    import logging
    logger = logging.getLogger(__name__)
    
    if not settings.OPENAI_BAA_SIGNED:
        logger.warning("HIPAA COMPLIANCE WARNINGS:")
        logger.warning("CRITICAL: Business Associate Agreement (BAA) with OpenAI NOT signed. AI features BLOCKED.")
        logger.warning("Set OPENAI_BAA_SIGNED=true after signing BAA.")
        if not settings.OPENAI_ZDR_ENABLED:
            logger.warning("IMPORTANT: Zero Data Retention (ZDR) not enabled. Set OPENAI_ZDR_ENABLED=true for HIPAA compliance.")
        if not settings.OPENAI_ENTERPRISE:
            logger.warning("NOTICE: OpenAI Enterprise plan recommended for HIPAA compliance. Set OPENAI_ENTERPRISE=true.")
        logger.warning("AI FEATURES BLOCKED until BAA is signed. Visit: https://openai.com/enterprise")
        return False
    return True
