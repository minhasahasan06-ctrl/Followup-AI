"""
Mental Health Questionnaire Database Models
Stores standardized mental health assessment responses and AI-powered pattern analysis
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class MentalHealthResponse(Base):
    """Store mental health questionnaire responses (PHQ-9, GAD-7, PSS-10, etc.)"""
    __tablename__ = "mental_health_responses"

    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    # Questionnaire identification
    questionnaire_type = Column(String, nullable=False, index=True)  # 'PHQ9', 'GAD7', 'PSS10'
    questionnaire_version = Column(String, default="1.0")
    
    # Response data
    responses = Column(JSON, nullable=False)  # Array of {questionId, questionText, response, responseText}
    
    # Scoring results
    total_score = Column(Integer)
    max_score = Column(Integer)
    severity_level = Column(String, index=True)  # 'minimal', 'mild', 'moderate', 'moderately_severe', 'severe'
    
    # Symptom cluster scores (domain-specific subscales)
    cluster_scores = Column(JSON)  # {cluster_name: {score, maxScore, label, items}}
    
    # Crisis flags
    crisis_detected = Column(Boolean, default=False, index=True)
    crisis_question_ids = Column(JSON)  # Array of question IDs that triggered crisis
    crisis_responses = Column(JSON)  # Array of {questionId, questionText, response}
    
    # Completion metadata
    completed_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    duration_seconds = Column(Integer)  # Time taken to complete
    
    # Privacy and consent
    allow_storage = Column(Boolean, default=True)
    allow_clinical_sharing = Column(Boolean, default=False)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship to pattern analysis
    pattern_analyses = relationship("MentalHealthPatternAnalysis", back_populates="response")
    
    __table_args__ = (
        Index('mh_responses_patient_type_idx', 'patient_id', 'questionnaire_type'),
        Index('mh_responses_severity_idx', 'severity_level'),
        Index('mh_responses_crisis_idx', 'crisis_detected'),
        Index('mh_responses_completed_idx', 'completed_at'),
    )


class MentalHealthPatternAnalysis(Base):
    """AI-powered pattern analysis for mental health questionnaires"""
    __tablename__ = "mental_health_pattern_analysis"

    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    response_id = Column(String, ForeignKey("mental_health_responses.id"), index=True)
    
    # Analysis type
    analysis_type = Column(String, nullable=False, index=True)  # 'symptom_clustering', 'temporal_trends', 'llm_insights'
    
    # Pattern detection results
    patterns = Column(JSON)  # Array of {patternType, patternName, description, severity, confidence}
    
    # Symptom clusters identified by LLM
    symptom_clusters = Column(JSON)  # {cluster_name: {clusterName, symptoms, frequency, severity, neutralDescription}}
    
    # Temporal trends (changes over time)
    temporal_trends = Column(JSON)  # Array of {metric, direction, magnitude, timeframe, dataPoints}
    
    # LLM-generated neutral summary (NO diagnostic language)
    neutral_summary = Column(Text)
    
    # Key observations (non-diagnostic)
    key_observations = Column(JSON)  # Array of strings
    
    # Suggested actions (general wellness, not treatment)
    suggested_actions = Column(JSON)  # Array of {category, action, priority}
    
    # LLM model information
    llm_model = Column(String, default="gpt-4o")
    llm_tokens_used = Column(Integer)
    
    # Analysis metadata
    analysis_version = Column(String, default="1.0")
    analysis_completed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Relationship back to response
    response = relationship("MentalHealthResponse", back_populates="pattern_analyses")
    
    __table_args__ = (
        Index('mh_analysis_patient_idx', 'patient_id'),
        Index('mh_analysis_response_idx', 'response_id'),
        Index('mh_analysis_type_idx', 'analysis_type'),
    )
