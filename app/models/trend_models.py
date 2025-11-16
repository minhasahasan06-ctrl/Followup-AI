"""
Trend Prediction Engine Database Models
Time-series risk modeling, anomaly detection, Bayesian updates
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class TrendSnapshot(Base):
    """Store patient health trend snapshots for time-series analysis"""
    __tablename__ = "trend_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Risk assessment (main output)
    risk_level = Column(String, nullable=False)  # "green", "yellow", "red"
    risk_score = Column(Float, nullable=False)  # 0-100 composite risk score
    confidence_score = Column(Float, nullable=False)  # 0-1 confidence in prediction
    
    # What changed (explainability)
    changed_metrics = Column(JSON)  # List of metrics that changed significantly
    primary_concern = Column(String)  # Main driver of risk increase
    secondary_concerns = Column(JSON)  # Other contributing factors
    
    # Why risk increased (detailed reasoning)
    risk_explanation = Column(Text)  # Human-readable explanation
    deviation_from_baseline = Column(JSON)  # Specific baseline deviations
    anomaly_flags = Column(JSON)  # Detected anomalies
    
    # Time-series data
    rolling_average_7day = Column(JSON)  # 7-day rolling averages
    rolling_average_14day = Column(JSON)  # 14-day rolling averages
    rolling_average_30day = Column(JSON)  # 30-day rolling averages
    
    # Baseline comparison
    baseline_metrics = Column(JSON)  # Patient's baseline values
    current_metrics = Column(JSON)  # Current measurement values
    z_scores = Column(JSON)  # Z-scores for each metric vs baseline
    
    # Anomaly detection
    anomaly_detected = Column(Boolean, default=False)
    anomaly_type = Column(String)  # "spike", "trend", "pattern_change", "outlier"
    anomaly_severity = Column(Float)  # 0-100
    anomaly_metrics = Column(JSON)  # Which metrics are anomalous
    
    # Bayesian risk update
    prior_risk = Column(Float)  # Previous risk estimate
    posterior_risk = Column(Float)  # Updated risk after new evidence
    likelihood_ratio = Column(Float)  # Evidence strength
    bayesian_update_params = Column(JSON)  # Alpha, beta parameters
    
    # Patient-specific personalization
    patient_baseline_id = Column(Integer, ForeignKey("patient_baselines.id"))
    personalization_factors = Column(JSON)  # Patient-specific adjustments
    historical_pattern_match = Column(Float)  # Similarity to past patterns
    
    # Trend direction
    trend_direction = Column(String)  # "improving", "stable", "worsening", "fluctuating"
    trend_velocity = Column(Float)  # Rate of change (per day)
    trend_acceleration = Column(Float)  # Change in rate of change
    
    # Contributing data sources
    video_metrics_id = Column(Integer, ForeignKey("video_metrics.id"))
    audio_metrics_id = Column(Integer, ForeignKey("audio_metrics.id"))
    symptom_data_ids = Column(JSON)  # List of symptom log IDs
    vital_signs_ids = Column(JSON)  # List of vital sign measurement IDs
    
    # Model metadata
    model_version = Column(String)  # Trend engine version
    algorithm_used = Column(String)  # "bayesian", "time_series", "ml_ensemble"
    processing_time_ms = Column(Float)
    
    # Timestamps
    snapshot_date = Column(DateTime(timezone=True), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_trend_patient_date', 'patient_id', 'snapshot_date'),
        Index('idx_trend_risk_level', 'risk_level'),
        Index('idx_trend_anomaly', 'anomaly_detected'),
    )


class RiskEvent(Base):
    """Track significant risk level changes and transitions"""
    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Event type
    event_type = Column(String, nullable=False)  # "risk_increase", "risk_decrease", "level_transition", "anomaly"
    event_severity = Column(String)  # "low", "medium", "high", "critical"
    
    # Risk transition
    previous_risk_level = Column(String)  # "green", "yellow", "red"
    new_risk_level = Column(String)  # "green", "yellow", "red"
    previous_risk_score = Column(Float)
    new_risk_score = Column(Float)
    risk_delta = Column(Float)  # Change in risk score
    
    # Trigger details
    trigger_metric = Column(String)  # Primary metric that triggered event
    trigger_value = Column(Float)  # Value of trigger metric
    trigger_threshold = Column(Float)  # Threshold that was crossed
    
    # Context
    trend_snapshot_id = Column(Integer, ForeignKey("trend_snapshots.id"))
    explanation = Column(Text)  # Why this event occurred
    recommended_actions = Column(JSON)  # Suggested interventions
    
    # Alert status
    alert_generated = Column(Boolean, default=False)
    alert_ids = Column(JSON)  # List of alert IDs generated from this event
    
    # Clinician review
    reviewed_by_clinician = Column(Boolean, default=False)
    reviewed_at = Column(DateTime(timezone=True))
    reviewed_by = Column(String)  # Clinician user ID
    clinician_notes = Column(Text)
    
    # Timestamps
    occurred_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    __table_args__ = (
        Index('idx_risk_event_patient', 'patient_id', 'occurred_at'),
        Index('idx_risk_event_severity', 'event_severity'),
        Index('idx_risk_event_level', 'new_risk_level'),
    )


class PatientBaseline(Base):
    """Store patient-specific baseline values for personalization"""
    __tablename__ = "patient_baselines"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, unique=True, index=True)
    
    # Baseline metrics (calculated from first 7-14 days)
    respiratory_rate_baseline = Column(Float)
    respiratory_rate_std = Column(Float)
    heart_rate_baseline = Column(Float)
    heart_rate_std = Column(Float)
    pain_score_baseline = Column(Float)
    pain_score_std = Column(Float)
    
    # Video metrics baselines
    skin_pallor_baseline = Column(Float)
    facial_symmetry_baseline = Column(Float)
    head_stability_baseline = Column(Float)
    
    # Audio metrics baselines
    voice_pitch_baseline = Column(Float)
    speech_pace_baseline = Column(Float)
    cough_frequency_baseline = Column(Float)
    
    # Symptom baselines
    symptom_severity_baseline = Column(Float)
    symptom_frequency_baseline = Column(Float)
    
    # Full baseline data
    all_baselines = Column(JSON)  # Complete baseline metrics
    
    # Baseline calculation metadata
    baseline_period_start = Column(DateTime(timezone=True))
    baseline_period_end = Column(DateTime(timezone=True))
    baseline_data_points = Column(Integer)  # Number of measurements used
    baseline_quality_score = Column(Float)  # 0-100 (data quality)
    
    # Last recalculation
    last_recalculated_at = Column(DateTime(timezone=True))
    recalculation_frequency_days = Column(Integer, default=30)  # How often to update
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
