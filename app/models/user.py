from sqlalchemy import Column, String, Boolean, DateTime, Integer
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    role = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    profile_image_url = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    
    organization = Column(String, nullable=True)
    medical_license_number = Column(String, nullable=True)
    license_country = Column(String, nullable=True)
    license_verified = Column(Boolean, default=False)
    kyc_photo_url = Column(String, nullable=True)
    
    ehr_platform = Column(String, nullable=True)
    ehr_import_method = Column(String, nullable=True)
    
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String, nullable=True)
    verification_token_expires = Column(DateTime, nullable=True)
    reset_token = Column(String, nullable=True)
    reset_token_expires = Column(DateTime, nullable=True)
    
    phone_verified = Column(Boolean, default=False)
    phone_verification_code = Column(String, nullable=True)
    phone_verification_expires = Column(DateTime, nullable=True)
    
    google_drive_application_url = Column(String, nullable=True)
    
    admin_verified = Column(Boolean, default=False)
    admin_verified_at = Column(DateTime, nullable=True)
    admin_verified_by = Column(String, nullable=True)
    
    sms_notifications_enabled = Column(Boolean, default=True)
    sms_medication_reminders = Column(Boolean, default=True)
    sms_appointment_reminders = Column(Boolean, default=True)
    sms_daily_followups = Column(Boolean, default=True)
    sms_health_alerts = Column(Boolean, default=True)
    
    terms_accepted = Column(Boolean, default=False)
    terms_accepted_at = Column(DateTime, nullable=True)
    
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    subscription_status = Column(String, nullable=True)
    trial_ends_at = Column(DateTime, nullable=True)
    credit_balance = Column(Integer, default=0)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
