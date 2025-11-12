import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.DATABASE_URL:
            raise ValueError(
                "DATABASE_URL environment variable is required. "
                "Please set it to your PostgreSQL connection string."
            )
    
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
    
    SESSION_SECRET: str = os.getenv("SESSION_SECRET", "dev-secret-key-change-in-production")
    
    CORS_ORIGINS: list = ["http://localhost:5000", "http://127.0.0.1:5000"]
    
    ENVIRONMENT: str = os.getenv("NODE_ENV", "development")
    
    class Config:
        env_file = ".env"


settings = Settings()


def check_openai_baa_compliance():
    if not settings.OPENAI_BAA_SIGNED:
        print("⚠️  HIPAA COMPLIANCE WARNINGS:")
        print("   - CRITICAL: Business Associate Agreement (BAA) with OpenAI NOT signed. AI features BLOCKED.")
        print("   - Set OPENAI_BAA_SIGNED=true after signing BAA.")
        if not settings.OPENAI_ZDR_ENABLED:
            print("   - IMPORTANT: Zero Data Retention (ZDR) not enabled. Set OPENAI_ZDR_ENABLED=true for HIPAA compliance.")
        if not settings.OPENAI_ENTERPRISE:
            print("   - NOTICE: OpenAI Enterprise plan recommended for HIPAA compliance. Set OPENAI_ENTERPRISE=true.")
        print("   🚫 AI FEATURES BLOCKED until BAA is signed.")
        print("   Visit: https://openai.com/enterprise")
        return False
    return True
