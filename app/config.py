import os
from pydantic_settings import BaseSettings
from typing import Optional
from openai import OpenAI


class Settings(BaseSettings):
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")
    
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
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
    
    # Daily.co Video Configuration
    DAILY_API_KEY: Optional[str] = os.getenv("DAILY_API_KEY")
    DAILY_DOMAIN: str = os.getenv("DAILY_DOMAIN", "followupai.daily.co")
    DAILY_WEBHOOK_SECRET: Optional[str] = os.getenv("DAILY_WEBHOOK_SECRET")
    DAILY_RATE_USD: str = os.getenv("DAILY_RATE_USD", "0.004")
    
    # Video Billing Plan Configuration
    OVERAGE_RATE_USD: str = os.getenv("OVERAGE_RATE_USD", "0.008")
    PLAN_TRIAL_INCLUDED_PM: int = int(os.getenv("PLAN_TRIAL_INCLUDED_PM", "300"))
    PLAN_PRO_INCLUDED_PM: int = int(os.getenv("PLAN_PRO_INCLUDED_PM", "3000"))
    PLAN_CLINIC_INCLUDED_PM: int = int(os.getenv("PLAN_CLINIC_INCLUDED_PM", "20000"))
    
    # Application URLs
    APP_BASE_URL: str = os.getenv("APP_BASE_URL", "http://localhost:8000")
    FRONTEND_BASE_URL: str = os.getenv("FRONTEND_BASE_URL", "http://localhost:5000")
    
    CORS_ORIGINS: list = ["http://localhost:5000", "http://127.0.0.1:5000"]
    
    ENVIRONMENT: str = os.getenv("NODE_ENV", "development")
    
    # Tinker Thinking Machine Configuration (NON-BAA Mode)
    # CRITICAL: Tinker does NOT have a BAA - never send PHI
    TINKER_API_KEY: Optional[str] = os.getenv("TINKER_API_KEY")
    TINKER_ENABLED: bool = os.getenv("TINKER_ENABLED", "false").lower() == "true"
    TINKER_MODE: str = os.getenv("TINKER_MODE", "NON_BAA")
    TINKER_K_ANON: int = int(os.getenv("TINKER_K_ANON", "25"))
    TINKER_API_URL: str = os.getenv("TINKER_API_URL", "https://api.tinker.ai/v1")
    TINKER_TIMEOUT_SECONDS: int = int(os.getenv("TINKER_TIMEOUT_SECONDS", "15"))
    TINKER_MAX_RETRIES: int = int(os.getenv("TINKER_MAX_RETRIES", "2"))
    
    def is_tinker_enabled(self) -> bool:
        """Check if Tinker integration is enabled and configured"""
        return self.TINKER_ENABLED and self.TINKER_API_KEY is not None
    
    def validate_tinker_non_baa(self) -> bool:
        """Validate Tinker is in NON-BAA mode (required - no PHI allowed)"""
        if self.TINKER_MODE != "NON_BAA":
            raise ValueError("TINKER_MODE must be 'NON_BAA' - Tinker does not have a BAA")
        return True
    
    def get_plan_included_minutes(self, plan: str) -> int:
        """Get included participant minutes for a plan"""
        plan_map = {
            "TRIAL": self.PLAN_TRIAL_INCLUDED_PM,
            "PRO": self.PLAN_PRO_INCLUDED_PM,
            "CLINIC": self.PLAN_CLINIC_INCLUDED_PM,
        }
        return plan_map.get(plan.upper(), self.PLAN_TRIAL_INCLUDED_PM)
    
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
