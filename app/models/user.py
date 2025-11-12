from sqlalchemy import Column, String, Boolean, DateTime, Integer
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    cognito_sub = Column(String, unique=True, nullable=True)
    role = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    phone_number = Column(String, nullable=True)
    date_of_birth = Column(DateTime, nullable=True)
    gender = Column(String, nullable=True)
    
    is_email_verified = Column(Boolean, default=False)
    is_phone_verified = Column(Boolean, default=False)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String, nullable=True)
    
    medical_license_number = Column(String, nullable=True)
    specialty = Column(String, nullable=True)
    years_of_experience = Column(Integer, nullable=True)
    
    linkedin_url = Column(String, nullable=True)
    hospital_id = Column(Integer, nullable=True)
    hospital_name = Column(String, nullable=True)
    location_city = Column(String, nullable=True, index=True)
    location_state = Column(String, nullable=True, index=True)
    location_country = Column(String, nullable=True, index=True)
    availability_status = Column(String, default="available")
    bio = Column(String, nullable=True)
    
    immune_system_status = Column(String, nullable=True)
    medical_conditions = Column(String, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
