"""
Behavior AI Database Models (Python/SQLAlchemy)
===============================================

SQLAlchemy models for behavior analysis tables.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, JSON, Text, Index, DECIMAL
from sqlalchemy.sql import func
from app.database import Base


class BehaviorCheckin(Base):
    """Daily check-ins for behavioral pattern tracking"""
    __tablename__ = "behavior_checkins"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False, index=True)
    
    # Check-in timing
    scheduled_time = Column(DateTime)
    completed_at = Column(DateTime, index=True)
    response_latency_minutes = Column(Integer)
    skipped = Column(Boolean, default=False, index=True)
    skip_reason = Column(Text)
    
    # Self-reported data
    symptom_severity = Column(Integer)
    symptom_description = Column(Text)
    pain_level = Column(Integer)
    medication_taken = Column(Boolean, default=False)
    medication_skipped_reason = Column(Text)
    
    # Engagement
    session_duration_seconds = Column(Integer)
    interaction_count = Column(Integer)
    
    # Avoidance patterns
    avoidance_language_detected = Column(Boolean, default=False)
    avoidance_phrases = Column(JSON)
    
    # Sentiment
    sentiment_polarity = Column(DECIMAL(5, 3))
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class BehaviorMetric(Base):
    """Aggregated behavioral metrics (daily rollup)"""
    __tablename__ = "behavior_metrics"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Check-in consistency
    checkin_time_consistency_score = Column(DECIMAL(5, 3))
    checkin_completion_rate = Column(DECIMAL(5, 3))
    avg_response_latency_minutes = Column(DECIMAL(8, 2))
    skipped_checkins_count = Column(Integer, default=0)
    
    # Routine
    routine_deviation_score = Column(DECIMAL(5, 3))
    
    # Medication
    medication_adherence_rate = Column(DECIMAL(5, 3))
    medication_skips_count = Column(Integer, default=0)
    
    # Engagement
    app_engagement_duration_minutes = Column(DECIMAL(8, 2))
    app_open_count = Column(Integer, default=0)
    
    # Avoidance
    avoidance_patterns_detected = Column(Boolean, default=False)
    avoidance_count = Column(Integer, default=0)
    avoidance_phrases = Column(JSON)
    
    # Sentiment
    avg_sentiment_polarity = Column(DECIMAL(5, 3))
    sentiment_trend_slope = Column(DECIMAL(8, 5))
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('behavior_metrics_patient_date_idx', 'patient_id', 'date'),
    )


class DigitalBiomarker(Base):
    """Digital biomarkers from phone/wearable data"""
    __tablename__ = "digital_biomarkers"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    date = Column(DateTime, nullable=False)
    
    # Activity
    daily_step_count = Column(Integer)
    step_trend_7day = Column(DECIMAL(8, 2))
    activity_burst_count = Column(Integer)
    sedentary_duration_minutes = Column(Integer)
    movement_variability_score = Column(DECIMAL(5, 3))
    
    # Circadian
    circadian_rhythm_stability = Column(DECIMAL(5, 3))
    sleep_wake_irregularity_minutes = Column(Integer)
    daily_peak_activity_time = Column(String)
    
    # Phone usage
    phone_usage_irregularity = Column(DECIMAL(5, 3))
    night_phone_interaction_count = Column(Integer)
    screen_on_duration_minutes = Column(Integer)
    
    # Mobility
    mobility_drop_detected = Column(Boolean, default=False, index=True)
    mobility_change_percent = Column(DECIMAL(6, 2))
    
    # Accelerometer
    accelerometer_std_dev = Column(DECIMAL(10, 5))
    accelerometer_mean_magnitude = Column(DECIMAL(10, 5))
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('digital_biomarkers_patient_date_idx', 'patient_id', 'date'),
    )


class CognitiveTest(Base):
    """Cognitive test results (weekly micro-tests)"""
    __tablename__ = "cognitive_tests"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    test_type = Column(String, nullable=False)
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    duration_seconds = Column(Integer)
    
    # Performance
    reaction_time_ms = Column(Integer)
    tapping_speed = Column(DECIMAL(6, 2))
    error_rate = Column(DECIMAL(5, 3))
    memory_score = Column(DECIMAL(5, 3))
    pattern_recall_accuracy = Column(DECIMAL(5, 3))
    instruction_accuracy = Column(DECIMAL(5, 3))
    
    # Results
    raw_results = Column(JSON)
    
    # Drift
    baseline_deviation = Column(DECIMAL(6, 3))
    anomaly_detected = Column(Boolean, default=False, index=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('cognitive_tests_patient_type_idx', 'patient_id', 'test_type'),
    )


class SentimentAnalysis(Base):
    """Sentiment/language analysis from text inputs"""
    __tablename__ = "sentiment_analysis"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    source_type = Column(String, nullable=False)
    source_id = Column(String)
    
    # Text
    text_content = Column(Text, nullable=False)
    analyzed_at = Column(DateTime, nullable=False)
    
    # Sentiment
    sentiment_polarity = Column(DECIMAL(5, 3), index=True)
    sentiment_label = Column(String)
    sentiment_confidence = Column(DECIMAL(5, 3))
    
    # Language biomarkers
    message_length_chars = Column(Integer)
    word_count = Column(Integer)
    lexical_complexity = Column(DECIMAL(5, 3))
    negativity_ratio = Column(DECIMAL(5, 3))
    
    # Keywords
    stress_keyword_count = Column(Integer, default=0)
    stress_keywords = Column(JSON)
    help_seeking_detected = Column(Boolean, default=False)
    help_seeking_phrases = Column(JSON)
    
    # Hesitation
    hesitation_count = Column(Integer, default=0)
    hesitation_markers = Column(JSON)
    
    # Model
    model_version = Column(String, default="distilbert-base-uncased-finetuned-sst-2-english")
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('sentiment_analysis_patient_source_idx', 'patient_id', 'source_type'),
    )


class BehaviorRiskScore(Base):
    """Multi-modal risk scores"""
    __tablename__ = "behavior_risk_scores"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    calculated_at = Column(DateTime, nullable=False)
    
    # Component risks (0-100)
    behavioral_risk = Column(DECIMAL(5, 2), nullable=False)
    digital_biomarker_risk = Column(DECIMAL(5, 2), nullable=False)
    cognitive_risk = Column(DECIMAL(5, 2), nullable=False)
    sentiment_risk = Column(DECIMAL(5, 2), nullable=False)
    
    # Composite
    composite_risk = Column(DECIMAL(5, 2), nullable=False)
    risk_level = Column(String, nullable=False, index=True)
    
    # Model
    model_type = Column(String, default="transformer_xgboost_ensemble")
    model_version = Column(String)
    feature_contributions = Column(JSON)
    
    # Risk factors
    top_risk_factors = Column(JSON)
    
    # Confidence
    prediction_confidence = Column(DECIMAL(5, 3))
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('behavior_risk_scores_patient_calc_idx', 'patient_id', 'calculated_at'),
    )


class DeteriorationTrend(Base):
    """Deterioration trend detection"""
    __tablename__ = "deterioration_trends"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    detected_at = Column(DateTime, nullable=False)
    
    # Trend
    trend_type = Column(String, nullable=False)
    severity = Column(String, nullable=False, index=True)
    
    # Temporal
    trend_start_date = Column(DateTime)
    trend_duration_days = Column(Integer)
    trend_slope = Column(DECIMAL(10, 5))
    
    # Statistics
    z_score = Column(DECIMAL(6, 3))
    p_value = Column(DECIMAL(10, 8))
    confidence_level = Column(DECIMAL(5, 3))
    
    # Metrics
    affected_metrics = Column(JSON)
    metric_values = Column(JSON)
    
    # Clinical
    clinical_significance = Column(Text)
    recommended_actions = Column(JSON)
    
    # Alert
    alert_generated = Column(Boolean, default=False, index=True)
    alert_id = Column(String)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('deterioration_trends_patient_type_idx', 'patient_id', 'trend_type'),
    )


class BehaviorAlert(Base):
    """Alerts for behavior AI system"""
    __tablename__ = "behavior_alerts"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, nullable=False)
    triggered_at = Column(DateTime, nullable=False)
    
    # Alert
    alert_type = Column(String, nullable=False, index=True)
    severity = Column(String, nullable=False)
    priority = Column(Integer, nullable=False)
    
    # Content
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    
    # Sources
    source_risk_score_id = Column(String)
    source_trend_id = Column(String)
    
    # Delivery
    email_sent = Column(Boolean, default=False)
    email_sent_at = Column(DateTime)
    sms_sent = Column(Boolean, default=False)
    sms_sent_at = Column(DateTime)
    dashboard_notified = Column(Boolean, default=True)
    
    # Resolution
    acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime)
    acknowledged_by = Column(String)
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('behavior_alerts_patient_severity_idx', 'patient_id', 'severity'),
    )
