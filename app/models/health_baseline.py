"""
Health Baseline Models for Deterioration Prediction

Stores per-patient baseline metrics calculated from 7-day rolling windows.
Used for change detection and deviation analysis (NOT diagnosis).
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Index, Boolean
from sqlalchemy.sql import func
from app.database import Base


class HealthBaseline(Base):
    """
    Stores 7-day rolling baseline statistics for each patient's health metrics.
    Used for wellness monitoring and change detection (NOT medical diagnosis).
    
    Baselines are recalculated daily to track patient's normal patterns.
    Deviations from baseline trigger wellness alerts for discussion with healthcare provider.
    """
    __tablename__ = "health_baselines"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Baseline calculation window
    baseline_start_date = Column(DateTime, nullable=False)  # Start of 7-day window
    baseline_end_date = Column(DateTime, nullable=False)    # End of 7-day window
    data_points_count = Column(Integer, nullable=False)     # Number of measurements used
    
    # Pain Baselines (facial stress score 0-100)
    pain_facial_mean = Column(Float, nullable=True)
    pain_facial_std = Column(Float, nullable=True)
    pain_facial_min = Column(Float, nullable=True)
    pain_facial_max = Column(Float, nullable=True)
    
    # Pain Self-Reported Baselines (0-10 scale)
    pain_self_reported_mean = Column(Float, nullable=True)
    pain_self_reported_std = Column(Float, nullable=True)
    pain_self_reported_min = Column(Float, nullable=True)
    pain_self_reported_max = Column(Float, nullable=True)
    
    # Respiratory Rate Baselines (breaths per minute)
    respiratory_rate_mean = Column(Float, nullable=True)
    respiratory_rate_std = Column(Float, nullable=True)
    respiratory_rate_min = Column(Float, nullable=True)
    respiratory_rate_max = Column(Float, nullable=True)
    
    # Symptom Severity Baselines (0-10 scale from SymptomLog)
    symptom_severity_mean = Column(Float, nullable=True)
    symptom_severity_std = Column(Float, nullable=True)
    symptom_severity_min = Column(Float, nullable=True)
    symptom_severity_max = Column(Float, nullable=True)
    
    # Activity Impact Baseline (percentage of days affected)
    activity_impact_rate = Column(Float, nullable=True)  # 0-1 scale
    
    # Metadata for tracking baseline quality
    baseline_quality = Column(String, nullable=True)  # "excellent", "good", "fair", "poor"
    baseline_quality_notes = Column(String, nullable=True)
    
    # Raw data for debugging (JSON of daily values)
    raw_daily_values = Column(JSON, nullable=True)
    
    # Auto-update tracking
    is_current = Column(Boolean, default=True)  # Latest baseline for this patient
    calculation_method = Column(String, default="rolling_7day")
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Index for efficient queries
    __table_args__ = (
        Index('ix_health_baselines_patient_current', 'patient_id', 'is_current'),
    )


class BaselineDeviation(Base):
    """
    Stores detected deviations from patient baseline (NOT diagnosis).
    Used for wellness monitoring and change detection alerts.
    
    Z-scores > 2 or < -1.5 trigger wellness alerts for healthcare provider discussion.
    """
    __tablename__ = "baseline_deviations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, nullable=False, index=True)
    baseline_id = Column(Integer, nullable=False)  # Reference to HealthBaseline
    
    # Measurement details
    metric_name = Column(String, nullable=False)  # "pain_facial", "respiratory_rate", etc.
    measurement_value = Column(Float, nullable=False)
    measurement_date = Column(DateTime, nullable=False, index=True)
    
    # Deviation analysis
    z_score = Column(Float, nullable=False)  # Standard deviations from baseline mean
    percent_change = Column(Float, nullable=False)  # Percentage change from baseline mean
    baseline_mean = Column(Float, nullable=False)
    baseline_std = Column(Float, nullable=False)
    
    # Trend analysis (3-day and 7-day slopes)
    trend_3day_slope = Column(Float, nullable=True)  # Slope over last 3 days
    trend_7day_slope = Column(Float, nullable=True)  # Slope over last 7 days
    trend_direction = Column(String, nullable=True)  # "improving", "stable", "worsening"
    
    # Deviation classification
    deviation_type = Column(String, nullable=False)  # "above_threshold", "below_threshold", "normal"
    severity_level = Column(String, nullable=False)  # "critical", "moderate", "mild", "normal"
    
    # Alert flag for wellness monitoring
    alert_triggered = Column(Boolean, default=False)
    alert_message = Column(String, nullable=True)  # Non-diagnostic alert message
    
    # Metadata
    source_measurement_id = Column(Integer, nullable=True)  # Link to original measurement
    source_table = Column(String, nullable=True)  # "pain_measurements", "symptom_measurements", etc.
    
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    
    # Index for efficient trend queries
    __table_args__ = (
        Index('ix_baseline_deviations_patient_date', 'patient_id', 'measurement_date'),
    )
