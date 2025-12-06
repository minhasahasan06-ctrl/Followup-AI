"""
SQLAlchemy models for Daily Symptom Check-In system.

Tables:
- symptom_checkins: Structured daily check-in data
- chat_symptoms: Symptoms extracted from Agent Clona conversations
- passive_metrics: Device-collected wellness metrics
- trend_reports: ML-generated trend analysis reports

HIPAA Compliance: All data is patient-owned and encrypted
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, Text, TIMESTAMP, ARRAY, JSON, ForeignKey
from sqlalchemy.sql import func
from app.database import Base


class SymptomCheckin(Base):
    """Daily symptom check-in with structured patient-reported metrics"""
    __tablename__ = "symptom_checkins"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    timestamp = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Structured symptom metrics (0-10 scales for self-reported patient data)
    pain_level = Column(Integer)  # 0-10 VAS scale (self-reported)
    fatigue_level = Column(Integer)  # 0-10 scale
    breathlessness_level = Column(Integer)  # 0-10 scale
    sleep_quality = Column(Integer)  # 0-10 scale
    mood = Column(String)  # 'great', 'good', 'okay', 'low', 'very_low'
    mobility_score = Column(Integer)  # 0-10 scale
    
    # Medication adherence
    medications_taken = Column(Boolean)  # Did patient take meds as prescribed?
    
    # Free-form selections
    triggers = Column(ARRAY(String), nullable=False, server_default='{}')  # Possible triggers
    symptoms = Column(ARRAY(String), nullable=False, server_default='{}')  # Symptoms experienced
    note = Column(Text)  # Patient notes
    
    # Metadata
    source = Column(String, default="app")  # 'app', 'voice', 'chat'
    device_type = Column(String)  # 'ios', 'android', 'web'
    voice_note_url = Column(String)  # S3 URL if voice note attached
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())


class ChatSymptom(Base):
    """Symptoms extracted from Agent Clona conversations using GPT-4o"""
    __tablename__ = "chat_symptoms"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(String, nullable=False)  # Chat session ID
    message_id = Column(String)  # Specific message ID if available
    timestamp = Column(TIMESTAMP, nullable=False, server_default=func.now())
    
    # Extracted structured data (JSON from GPT-4o)
    extracted_json = Column(JSON, nullable=False)  # Full GPT extraction result
    confidence = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # Extracted fields (denormalized for querying)
    locations = Column(ARRAY(String))  # Body locations mentioned
    symptom_types = Column(ARRAY(String))  # Symptom types
    intensity_mentions = Column(ARRAY(String))  # Intensity descriptors
    temporal_info = Column(Text)  # When/duration
    aggravating_factors = Column(ARRAY(String))
    relieving_factors = Column(ARRAY(String))
    
    # Metadata
    extraction_model = Column(String, default="gpt-4o")
    created_at = Column(TIMESTAMP, server_default=func.now())


class PassiveMetric(Base):
    """Passive wellness metrics from wearables and phone sensors"""
    __tablename__ = "passive_metrics"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    date = Column(TIMESTAMP, nullable=False)  # Date of metric collection
    
    # Activity metrics
    steps = Column(Integer)  # Daily steps
    active_minutes = Column(Integer)  # Active minutes
    calories_burned = Column(Integer)  # Calories
    
    # Sleep metrics
    sleep_minutes = Column(Integer)  # Total sleep time
    deep_sleep_minutes = Column(Integer)
    rem_sleep_minutes = Column(Integer)
    sleep_efficiency = Column(Float)  # 0.0 to 1.0
    
    # Heart metrics
    hr_mean = Column(Integer)  # Average heart rate (bpm)
    hr_min = Column(Integer)  # Min heart rate
    hr_max = Column(Integer)  # Max heart rate
    hrv = Column(Integer)  # Heart rate variability (ms)
    resting_hr = Column(Integer)  # Resting heart rate
    
    # Respiratory metrics
    respiratory_rate_mean = Column(Float)  # Average breaths per minute
    spo2_mean = Column(Integer)  # Average SpO2 (%)
    spo2_min = Column(Integer)  # Min SpO2
    
    # Metadata
    source = Column(String)  # 'fitbit', 'apple_health', 'google_fit', 'manual'
    data_quality = Column(String)  # 'high', 'medium', 'low'
    created_at = Column(TIMESTAMP, server_default=func.now())


class TrendReport(Base):
    """ML-generated trend analysis reports for clinician review"""
    __tablename__ = "trend_reports"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    report_type = Column(String, nullable=False)  # '3day', '7day', '15day', '30day'
    period_start = Column(TIMESTAMP, nullable=False)
    period_end = Column(TIMESTAMP, nullable=False)
    
    # Aggregated metrics (JSON)
    aggregated_metrics = Column(JSON, nullable=False)  # Summary statistics
    
    # Anomalies detected (JSON array)
    anomalies = Column(JSON)  # Observational anomalies
    
    # Correlations found (JSON array)
    correlations = Column(JSON)  # Observational correlations
    
    # Clinician summary (plain English)
    clinician_summary = Column(Text, nullable=False)  # Human-readable summary
    
    # Metadata
    data_points_analyzed = Column(Integer, nullable=False)  # Number of check-ins included
    confidence_score = Column(Float)  # Overall confidence (0.0 to 1.0)
    generated_at = Column(TIMESTAMP, server_default=func.now())
    created_at = Column(TIMESTAMP, server_default=func.now())
