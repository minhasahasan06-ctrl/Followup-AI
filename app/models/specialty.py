from sqlalchemy import Column, String, Integer, Text, DateTime
from sqlalchemy.sql import func
from app.database import Base


class Specialty(Base):
    __tablename__ = "specialties"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DoctorSpecialty(Base):
    """Many-to-many relationship between doctors and specialties"""
    __tablename__ = "doctor_specialties"

    id = Column(Integer, primary_key=True, autoincrement=True)
    doctor_id = Column(String, nullable=False, index=True)
    specialty_id = Column(Integer, nullable=False, index=True)
    
    is_primary = Column(String, default="false")
    
    created_at = Column(DateTime, server_default=func.now())
