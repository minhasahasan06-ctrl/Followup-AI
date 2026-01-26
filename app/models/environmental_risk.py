"""
Environmental Risk Map Models
SQLAlchemy models for the comprehensive environmental health intelligence system.
"""

from sqlalchemy import Column, String, Integer, Boolean, Text, DECIMAL, ForeignKey, Index, JSON
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from sqlalchemy.sql import func
from app.database import Base


class PatientEnvironmentProfile(Base):
    """Links patients to their location and condition-specific environmental triggers."""
    __tablename__ = "patient_environment_profiles"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    zip_code = Column(String(10), nullable=False)
    city = Column(String)
    state = Column(String(2))
    timezone = Column(String, default="America/New_York")
    
    chronic_conditions = Column(JSONB)
    allergies = Column(JSONB)
    
    alerts_enabled = Column(Boolean, default=True)
    alert_thresholds = Column(JSONB)
    
    push_notifications = Column(Boolean, default=True)
    sms_notifications = Column(Boolean, default=False)
    email_digest = Column(Boolean, default=True)
    digest_frequency = Column(String, default="daily")
    
    correlation_consent_given = Column(Boolean, default=False)
    correlation_consent_at = Column(TIMESTAMP)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("env_profile_patient_idx", "patient_id"),
        Index("env_profile_zip_idx", "zip_code"),
    )


class ConditionTriggerMapping(Base):
    """Maps chronic conditions to their environmental factor sensitivities."""
    __tablename__ = "condition_trigger_mappings"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    condition_code = Column(String, nullable=False)
    condition_name = Column(String, nullable=False)
    
    factor_type = Column(String, nullable=False)
    factor_name = Column(String, nullable=False)
    
    trigger_threshold = Column(DECIMAL(10, 4))
    critical_threshold = Column(DECIMAL(10, 4))
    
    base_weight = Column(DECIMAL(5, 4), default=0.5)
    
    impact_direction = Column(String, nullable=False)
    
    clinical_evidence = Column(Text)
    recommendations = Column(JSONB)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("trigger_condition_idx", "condition_code"),
        Index("trigger_factor_idx", "factor_type"),
    )


class PatientTriggerWeight(Base):
    """Personalized trigger weights learned from patient-specific correlations."""
    __tablename__ = "patient_trigger_weights"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    factor_type = Column(String, nullable=False)
    
    personalized_weight = Column(DECIMAL(5, 4), nullable=False)
    confidence_score = Column(DECIMAL(5, 4))
    
    source = Column(String, nullable=False)
    
    correlation_coefficient = Column(DECIMAL(5, 4))
    p_value = Column(DECIMAL(8, 6))
    sample_size = Column(Integer)
    
    last_updated_at = Column(TIMESTAMP, server_default=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("patient_trigger_idx", "patient_id", "factor_type"),
    )


class EnvironmentalDataSnapshot(Base):
    """Time-series environmental readings by ZIP code."""
    __tablename__ = "environmental_data_snapshots"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    zip_code = Column(String(10), nullable=False)
    
    measured_at = Column(TIMESTAMP, nullable=False)
    
    temperature = Column(DECIMAL(6, 2))
    feels_like = Column(DECIMAL(6, 2))
    humidity = Column(DECIMAL(5, 2))
    pressure = Column(DECIMAL(7, 2))
    wind_speed = Column(DECIMAL(6, 2))
    wind_direction = Column(Integer)
    precipitation = Column(DECIMAL(6, 2))
    uv_index = Column(DECIMAL(4, 2))
    cloud_cover = Column(Integer)
    visibility = Column(DECIMAL(8, 2))
    
    aqi = Column(Integer)
    aqi_category = Column(String)
    pm25 = Column(DECIMAL(7, 3))
    pm10 = Column(DECIMAL(7, 3))
    ozone = Column(DECIMAL(7, 3))
    no2 = Column(DECIMAL(7, 3))
    so2 = Column(DECIMAL(7, 3))
    co = Column(DECIMAL(8, 3))
    
    pollen_tree_count = Column(Integer)
    pollen_grass_count = Column(Integer)
    pollen_weed_count = Column(Integer)
    pollen_overall = Column(Integer)
    pollen_category = Column(String)
    mold_spore_count = Column(Integer)
    mold_category = Column(String)
    
    active_hazards = Column(JSONB)
    
    weather_source = Column(String)
    aqi_source = Column(String)
    pollen_source = Column(String)
    hazard_source = Column(String)
    
    data_quality_score = Column(DECIMAL(4, 2))
    missing_fields = Column(JSONB)
    
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("env_snapshot_zip_time_idx", "zip_code", "measured_at"),
        Index("env_snapshot_measured_idx", "measured_at"),
    )


class PatientEnvironmentRiskScore(Base):
    """Computed personalized risk scores for patients."""
    __tablename__ = "patient_environment_risk_scores"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    snapshot_id = Column(String, ForeignKey("environmental_data_snapshots.id"))
    
    computed_at = Column(TIMESTAMP, nullable=False)
    
    composite_risk_score = Column(DECIMAL(5, 2), nullable=False)
    risk_level = Column(String, nullable=False)
    
    weather_risk_score = Column(DECIMAL(5, 2))
    air_quality_risk_score = Column(DECIMAL(5, 2))
    allergen_risk_score = Column(DECIMAL(5, 2))
    hazard_risk_score = Column(DECIMAL(5, 2))
    
    trend_24hr = Column(DECIMAL(6, 3))
    trend_48hr = Column(DECIMAL(6, 3))
    trend_72hr = Column(DECIMAL(6, 3))
    
    volatility_score = Column(DECIMAL(5, 2))
    
    factor_contributions = Column(JSONB)
    top_risk_factors = Column(JSONB)
    
    scoring_version = Column(String, default="1.0")
    
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("risk_score_patient_time_idx", "patient_id", "computed_at"),
        Index("risk_score_level_idx", "risk_level"),
    )


class EnvironmentalForecast(Base):
    """ML-predicted future environmental risk."""
    __tablename__ = "environmental_forecasts"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    generated_at = Column(TIMESTAMP, nullable=False)
    
    forecast_horizon = Column(String, nullable=False)
    forecast_target_time = Column(TIMESTAMP, nullable=False)
    
    predicted_risk_score = Column(DECIMAL(5, 2), nullable=False)
    predicted_risk_level = Column(String, nullable=False)
    confidence_interval = Column(JSONB)
    
    predicted_weather_risk = Column(DECIMAL(5, 2))
    predicted_air_quality_risk = Column(DECIMAL(5, 2))
    predicted_allergen_risk = Column(DECIMAL(5, 2))
    
    predicted_values = Column(JSONB)
    
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    feature_importance = Column(JSONB)
    
    actual_risk_score = Column(DECIMAL(5, 2))
    forecast_error = Column(DECIMAL(5, 2))
    
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("forecast_patient_time_idx", "patient_id", "generated_at"),
        Index("forecast_horizon_idx", "forecast_horizon"),
    )


class SymptomEnvironmentCorrelation(Base):
    """Learned correlations between symptoms and environmental factors."""
    __tablename__ = "symptom_environment_correlations"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    symptom_type = Column(String, nullable=False)
    symptom_severity_metric = Column(String, nullable=False)
    
    environmental_factor = Column(String, nullable=False)
    
    correlation_type = Column(String, nullable=False)
    correlation_coefficient = Column(DECIMAL(6, 4), nullable=False)
    p_value = Column(DECIMAL(10, 8))
    is_statistically_significant = Column(Boolean, default=False)
    
    optimal_lag = Column(Integer)
    lag_correlation = Column(DECIMAL(6, 4))
    
    sample_size = Column(Integer, nullable=False)
    data_window_days = Column(Integer)
    
    relationship_strength = Column(String)
    relationship_direction = Column(String)
    interpretation = Column(Text)
    
    confidence_score = Column(DECIMAL(5, 4))
    data_quality_score = Column(DECIMAL(5, 4))
    
    last_analyzed_at = Column(TIMESTAMP, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("correlation_patient_symptom_idx", "patient_id", "symptom_type"),
        Index("correlation_significant_idx", "is_statistically_significant"),
    )


class EnvironmentalAlert(Base):
    """Environmental alert history and active alerts."""
    __tablename__ = "environmental_alerts"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    alert_type = Column(String, nullable=False)
    triggered_by = Column(String, nullable=False)
    
    severity = Column(String, nullable=False)
    priority = Column(Integer, nullable=False)
    
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    recommendations = Column(JSONB)
    
    trigger_value = Column(DECIMAL(10, 4))
    threshold_value = Column(DECIMAL(10, 4))
    percent_over_threshold = Column(DECIMAL(6, 2))
    
    risk_score_id = Column(String, ForeignKey("patient_environment_risk_scores.id"))
    forecast_id = Column(String, ForeignKey("environmental_forecasts.id"))
    snapshot_id = Column(String, ForeignKey("environmental_data_snapshots.id"))
    
    status = Column(String, default="active")
    acknowledged_at = Column(TIMESTAMP)
    resolved_at = Column(TIMESTAMP)
    expires_at = Column(TIMESTAMP)
    
    push_notification_sent = Column(Boolean, default=False)
    sms_notification_sent = Column(Boolean, default=False)
    email_notification_sent = Column(Boolean, default=False)
    notification_sent_at = Column(TIMESTAMP)
    
    was_helpful = Column(Boolean)
    user_feedback = Column(Text)
    
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("env_alert_patient_status_idx", "patient_id", "status"),
        Index("env_alert_severity_idx", "severity"),
        Index("env_alert_created_idx", "created_at"),
    )


class EnvironmentalPipelineJob(Base):
    """Track background job execution for environmental data pipeline."""
    __tablename__ = "environmental_pipeline_jobs"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    
    job_type = Column(String, nullable=False)
    
    target_zip_codes = Column(JSONB)
    target_patient_ids = Column(JSONB)
    
    status = Column(String, nullable=False)
    started_at = Column(TIMESTAMP)
    completed_at = Column(TIMESTAMP)
    
    records_processed = Column(Integer, default=0)
    records_created = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    alerts_generated = Column(Integer, default=0)
    
    error_message = Column(Text)
    error_stack = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    
    execution_time_ms = Column(Integer)
    
    trigger_source = Column(String)
    job_metadata = Column(JSONB)
    
    created_at = Column(TIMESTAMP, server_default=func.now())
    
    __table_args__ = (
        Index("pipeline_job_type_status_idx", "job_type", "status"),
        Index("pipeline_job_created_idx", "created_at"),
    )
