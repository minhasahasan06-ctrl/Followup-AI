from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON, Boolean
from sqlalchemy.sql import func
from app.database import Base


class PainMeasurement(Base):
    """
    Stores daily facial analysis measurements for pain detection.
    Each record represents one 10-second camera session.
    """
    __tablename__ = "pain_measurements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True, nullable=False)
    
    # Facial metrics - eyebrow angles
    left_eyebrow_angle = Column(Float, nullable=True)
    right_eyebrow_angle = Column(Float, nullable=True)
    eyebrow_asymmetry = Column(Float, nullable=True)
    
    # Nasolabial fold tension (0-1 scale)
    left_nasolabial_tension = Column(Float, nullable=True)
    right_nasolabial_tension = Column(Float, nullable=True)
    
    # Micro facial contractions count during 10-second session
    forehead_contractions = Column(Integer, default=0)
    eye_contractions = Column(Integer, default=0)
    mouth_contractions = Column(Integer, default=0)
    
    # Grimacing patterns
    grimace_intensity = Column(Float, nullable=True)  # 0-1 scale
    grimace_duration_ms = Column(Integer, default=0)
    
    # Overall pain indicators
    facial_stress_score = Column(Float, nullable=True)  # 0-100 scale
    pain_severity_estimate = Column(String, nullable=True)  # "low", "moderate", "severe"
    
    # Comparison with previous measurement
    change_from_previous = Column(Float, nullable=True)  # Percentage change
    
    # Raw landmark data (JSON) for detailed analysis
    facial_landmarks = Column(JSON, nullable=True)
    
    # Session metadata
    recording_duration_ms = Column(Integer, default=10000)
    recording_quality = Column(String, nullable=True)  # "good", "fair", "poor"
    
    created_at = Column(DateTime, server_default=func.now(), index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PainQuestionnaire(Base):
    """
    Stores patient responses to pain-related questions.
    Asked after each facial analysis session.
    """
    __tablename__ = "pain_questionnaires"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True, nullable=False)
    measurement_id = Column(Integer, nullable=True)  # Links to PainMeasurement
    
    # Pain assessment questions
    pain_level_self_reported = Column(Integer, nullable=True)  # 0-10 scale
    pain_location = Column(String, nullable=True)
    pain_type = Column(String, nullable=True)  # "sharp", "dull", "throbbing", etc.
    pain_duration = Column(String, nullable=True)
    pain_triggers = Column(Text, nullable=True)
    
    # Activity impact
    affects_sleep = Column(Boolean, default=False)
    affects_daily_activities = Column(Boolean, default=False)
    affects_mood = Column(Boolean, default=False)
    
    # Medication tracking
    pain_medication_taken = Column(Boolean, default=False)
    medication_names = Column(String, nullable=True)
    medication_effectiveness = Column(String, nullable=True)  # "effective", "somewhat", "not effective"
    
    # Additional notes
    additional_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PainTrendSummary(Base):
    """
    Stores weekly/monthly aggregated pain trends for clinician review.
    Generated automatically from daily measurements.
    """
    __tablename__ = "pain_trend_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, index=True, nullable=False)
    
    period_type = Column(String, nullable=False)  # "weekly", "monthly"
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    
    # Aggregated metrics
    average_stress_score = Column(Float, nullable=True)
    max_stress_score = Column(Float, nullable=True)
    min_stress_score = Column(Float, nullable=True)
    
    stress_score_change_percent = Column(Float, nullable=True)
    days_with_high_pain = Column(Integer, default=0)
    days_with_moderate_pain = Column(Integer, default=0)
    days_with_low_pain = Column(Integer, default=0)
    
    # Trends
    trend_direction = Column(String, nullable=True)  # "improving", "stable", "worsening"
    
    # Alert flags
    requires_physician_attention = Column(Boolean, default=False)
    alert_reason = Column(Text, nullable=True)
    
    # PDF report
    pdf_report_url = Column(String, nullable=True)
    pdf_generated_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
