"""
SQLAlchemy models for video examination system
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    """User model - simplified for auth reference"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    first_name = Column(String)
    last_name = Column(String)
    role = Column(String, default="patient")


class VideoExamSession(Base):
    """Video examination session - tracks overall guided exam workflow"""
    __tablename__ = "video_exam_sessions"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Session timing
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Combined recording storage
    combined_s3_key = Column(String)
    combined_s3_bucket = Column(String)
    combined_kms_key_id = Column(String)
    combined_file_size_bytes = Column(Integer)
    
    # Combined analysis reference
    combined_analysis_id = Column(String)
    
    # Session status: 'in_progress', 'completed', 'abandoned'
    status = Column(String, nullable=False, default="in_progress")
    
    # Metadata
    total_segments = Column(Integer, default=0)
    completed_segments = Column(Integer, default=0)
    skipped_segments = Column(Integer, default=0)
    total_duration_seconds = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class VideoExamSegment(Base):
    """Video examination segment - individual exam recordings"""
    __tablename__ = "video_exam_segments"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    session_id = Column(String, ForeignKey("video_exam_sessions.id", ondelete="CASCADE"), nullable=False)
    
    # Exam type: 'respiratory', 'skin_pallor', 'eye_sclera', 'swelling', 'tremor', 'tongue', 'custom'
    exam_type = Column(String, nullable=False)
    sequence_order = Column(Integer, nullable=False)
    
    # Recording status
    skipped = Column(Boolean, default=False)
    prep_duration_seconds = Column(Integer, default=30)
    
    # Timing
    capture_started_at = Column(DateTime(timezone=True))
    capture_ended_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Integer)
    
    # S3 storage (encrypted with KMS)
    s3_key = Column(String)
    s3_bucket = Column(String)
    kms_key_id = Column(String)
    file_size_bytes = Column(Integer)
    
    # AI Analysis reference
    analysis_id = Column(String)
    
    # Processing status: 'pending', 'processing', 'completed', 'failed', 'skipped'
    status = Column(String, nullable=False, default="pending")
    
    # Custom abnormality description (for examType='custom')
    custom_location = Column(Text)
    custom_description = Column(Text)
    
    # Audit metadata
    uploaded_by = Column(String, ForeignKey("users.id"))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
