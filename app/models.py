"""
SQLAlchemy models for video examination system
"""

from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, ForeignKey, Float, JSON, Index
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

class RespiratoryBaseline(Base):
    """Patient respiratory baseline - normal RR for anomaly detection"""
    __tablename__ = "respiratory_baselines"
    
    patient_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    # Baseline statistics
    baseline_rr_bpm = Column(Float, nullable=False)  # Mean RR
    baseline_rr_std = Column(Float, nullable=False)  # Standard deviation
    sample_size = Column(Integer, nullable=False, default=0)  # Number of sessions used
    confidence = Column(Float, nullable=False, default=0.0)  # Confidence score 0-1
    
    # Metadata
    source = Column(String, default="auto")  # 'auto' or 'manual' (clinician override)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RespiratoryMetric(Base):
    """Time-series respiratory metrics with rolling statistics"""
    __tablename__ = "respiratory_metrics"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("video_exam_sessions.id", ondelete="CASCADE"))
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Core respiratory metrics
    rr_bpm = Column(Float)  # Respiratory rate (breaths per minute)
    rr_confidence = Column(Float, default=0.0)  # Detection confidence 0-1
    
    # Advanced metrics
    breath_interval_std = Column(Float)  # Variability in breath intervals
    variability_index = Column(Float)  # Coefficient of variation
    accessory_muscle_score = Column(Float)  # Neck muscle use indicator
    chest_expansion_amplitude = Column(Float)  # Chest movement amplitude
    gasping_detected = Column(Boolean, default=False)  # Irregular gasping pattern
    chest_shape_asymmetry = Column(Float)  # Barrel chest / asymmetry score
    thoracoabdominal_synchrony = Column(Float)  # Breathing synchrony (0-1)
    
    # Temporal analytics
    z_score_vs_baseline = Column(Float)  # Anomaly score vs patient baseline
    rolling_daily_avg = Column(Float)  # Mean RR last 24 hours
    rolling_three_day_slope = Column(Float)  # Trend: positive=worsening, negative=improving
    
    # Raw data for recomputation
    metadata = Column(JSON)  # Stores raw chest movement trace, fps, etc.
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indices for efficient queries
    __table_args__ = (
        Index('idx_respiratory_patient_time', 'patient_id', 'recorded_at'),
    )
