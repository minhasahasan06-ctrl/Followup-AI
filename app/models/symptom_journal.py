"""
Symptom Journal Models - HIPAA-compliant visual symptom tracking
Monitors changes in legs, face, eyes, and respiratory patterns
WITHOUT making medical diagnoses
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from app.database import Base
import enum


class BodyArea(str, enum.Enum):
    """Body areas that can be monitored"""
    LEGS = "legs"
    FACE = "face"
    EYES = "eyes"
    CHEST = "chest"  # For respiratory rate monitoring


class ChangeType(str, enum.Enum):
    """Types of changes detected"""
    COLOR = "color"
    SWELLING = "swelling"
    RESPIRATORY_RATE = "respiratory_rate"
    GENERAL = "general"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels"""
    INFO = "info"
    CAUTION = "caution"
    ATTENTION = "attention"  # Requires clinical follow-up


class SymptomImage(Base):
    """
    Stores references to symptom images uploaded to S3
    Images are encrypted at rest and in transit (HIPAA compliant)
    """
    __tablename__ = "symptom_images"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    body_area = Column(SQLEnum(BodyArea), nullable=False)
    
    # S3 storage references
    s3_bucket = Column(String, nullable=False)
    s3_key = Column(String, nullable=False)
    s3_url = Column(String)  # Pre-signed URL (temporary)
    
    # Image metadata
    file_size = Column(Integer)  # bytes
    mime_type = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    
    # Capture metadata
    capture_type = Column(String)  # 'photo' or 'video'
    duration_seconds = Column(Float)  # For videos
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SymptomMeasurement(Base):
    """
    Stores visual symptom measurements and AI-detected changes
    IMPORTANT: This tracks changes, NOT diagnoses
    """
    __tablename__ = "symptom_measurements"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    body_area = Column(SQLEnum(BodyArea), nullable=False)
    
    # Link to uploaded image
    image_id = Column(Integer, ForeignKey("symptom_images.id"))
    
    # Color metrics (RGB analysis, brightness/contrast)
    avg_red = Column(Float)
    avg_green = Column(Float)
    avg_blue = Column(Float)
    brightness = Column(Float)
    contrast = Column(Float)
    
    # Change detection (compared to previous measurement)
    color_change_percent = Column(Float)
    brightness_change_percent = Column(Float)
    
    # Swelling metrics (area-based, NOT medical grading)
    roi_area_pixels = Column(Integer)  # Region of interest area
    area_change_percent = Column(Float)  # Compared to baseline/previous
    
    # Respiratory rate (for chest/breathing area only)
    respiratory_rate_bpm = Column(Float)  # Breaths per minute (estimated)
    rr_confidence = Column(Float)  # 0-1 confidence score
    
    # AI-generated observations (OpenAI Vision API)
    ai_observations = Column(Text)  # Structured, non-diagnostic observations
    detected_changes = Column(JSON)  # List of detected changes
    
    # Comparison metadata
    compared_to_measurement_id = Column(Integer)  # Previous measurement for comparison
    days_since_baseline = Column(Integer)
    
    # Patient-reported context
    patient_notes = Column(Text)
    symptoms_reported = Column(JSON)  # Patient-described symptoms
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class SymptomAlert(Base):
    """
    Early warning alerts when significant pattern changes are detected
    Language emphasizes monitoring, NOT diagnosis
    """
    __tablename__ = "symptom_alerts"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    measurement_id = Column(Integer, ForeignKey("symptom_measurements.id"))
    
    severity = Column(SQLEnum(AlertSeverity), nullable=False)
    change_type = Column(SQLEnum(ChangeType), nullable=False)
    body_area = Column(SQLEnum(BodyArea), nullable=False)
    
    # Alert message (non-diagnostic language)
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    
    # Change details
    change_percent = Column(Float)
    days_of_trend = Column(Integer)
    
    # Alert status
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime(timezone=True))
    
    # Doctor notification
    doctor_notified = Column(Boolean, default=False)
    doctor_notified_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class WeeklySummary(Base):
    """
    Weekly aggregated summary for doctor review
    Structured symptom timeline with visual change tracking
    """
    __tablename__ = "weekly_summaries"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Time period
    week_start = Column(DateTime(timezone=True), nullable=False)
    week_end = Column(DateTime(timezone=True), nullable=False)
    
    # Measurement counts by body area
    measurements_count = Column(JSON)  # {"legs": 5, "face": 7, "chest": 4}
    
    # Detected changes summary
    significant_changes = Column(JSON)  # List of significant changes
    alert_count = Column(Integer, default=0)
    
    # Trend analysis
    legs_trend = Column(String)  # "improving", "stable", "worsening", "no_data"
    face_trend = Column(String)
    eyes_trend = Column(String)
    respiratory_trend = Column(String)
    
    # Averages and ranges
    avg_respiratory_rate = Column(Float)
    respiratory_rate_range = Column(JSON)  # {"min": 12, "max": 18}
    
    color_change_summary = Column(Text)
    swelling_change_summary = Column(Text)
    
    # AI-generated summary for doctors
    clinician_summary = Column(Text)  # Structured, actionable summary
    
    # PDF report
    pdf_generated = Column(Boolean, default=False)
    pdf_s3_key = Column(String)  # S3 location of PDF report
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
