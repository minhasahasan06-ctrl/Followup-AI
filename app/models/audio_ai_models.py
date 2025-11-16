"""
Audio AI Engine Database Models
Tracks audio analysis, breath detection, speech patterns, cough detection
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class AudioMetrics(Base):
    """Store audio AI analysis results (10+ metrics)"""
    __tablename__ = "audio_metrics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("media_sessions.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Breath cycle detection
    breath_cycles_detected = Column(Integer)  # Number of breath cycles
    breath_rate_per_minute = Column(Float)  # Respiratory rate from audio
    breath_pattern = Column(String)  # "regular", "irregular", "rapid", "slow"
    inhale_duration_avg = Column(Float)  # Average inhale duration (seconds)
    exhale_duration_avg = Column(Float)  # Average exhale duration (seconds)
    breath_depth_variation = Column(Float)  # Std dev of breath intensity
    
    # Speech analysis
    speech_segments_count = Column(Integer)  # Number of speech segments
    speech_total_duration = Column(Float)  # Total speech time (seconds)
    speech_pace_words_per_minute = Column(Float)  # Speaking rate
    speech_pace_variability = Column(Float)  # Std dev of pace
    syllables_per_second = Column(Float)  # Articulation rate
    
    # Pause analysis
    pause_count = Column(Integer)  # Number of pauses
    pause_frequency_per_minute = Column(Float)  # Pauses per minute
    pause_duration_avg = Column(Float)  # Average pause length (seconds)
    pause_duration_max = Column(Float)  # Longest pause (seconds)
    unusual_pause_pattern = Column(Boolean)  # Abnormal pausing detected
    
    # Cough detection
    cough_count = Column(Integer)  # Number of coughs detected
    cough_frequency_per_minute = Column(Float)  # Coughs per minute
    cough_intensity_avg = Column(Float)  # Average cough intensity (0-100)
    cough_type = Column(String)  # "dry", "wet", "productive", "mixed"
    cough_duration_avg = Column(Float)  # Average cough duration (seconds)
    coughing_fit_detected = Column(Boolean)  # Multiple rapid coughs
    
    # Hoarseness / vocal fatigue analysis
    hoarseness_score = Column(Float)  # 0-100 (higher = more hoarse)
    vocal_fatigue_score = Column(Float)  # 0-100 (vocal strain indicator)
    voice_pitch_avg_hz = Column(Float)  # Average fundamental frequency
    voice_pitch_variability = Column(Float)  # Pitch variation
    voice_intensity_avg_db = Column(Float)  # Average loudness
    voice_quality_score = Column(Float)  # 0-100 (clarity/smoothness)
    
    # Wheeze detection (frequency signatures)
    wheeze_detected = Column(Boolean)  # Wheeze present
    wheeze_count = Column(Integer)  # Number of wheeze events
    wheeze_frequency_hz = Column(Float)  # Dominant wheeze frequency
    wheeze_intensity = Column(Float)  # Wheeze loudness (0-100)
    wheeze_type = Column(String)  # "inspiratory", "expiratory", "both"
    stridor_detected = Column(Boolean)  # High-pitched breathing sound
    
    # Background noise analysis
    background_noise_level_db = Column(Float)  # Ambient noise level
    noise_removed_db = Column(Float)  # Noise reduction applied
    signal_to_noise_ratio = Column(Float)  # SNR (higher = cleaner)
    noise_type = Column(String)  # "white", "pink", "environmental", "clean"
    
    # Audio quality scoring
    audio_quality_score = Column(Float)  # 0-100 overall quality
    sample_rate_hz = Column(Integer)  # Audio sample rate
    bit_depth = Column(Integer)  # Audio bit depth
    clipping_detected = Column(Boolean)  # Audio distortion
    clipping_percent = Column(Float)  # % of audio clipped
    silence_percent = Column(Float)  # % of audio that is silence
    
    # Vocal characteristics (extended)
    jitter_percent = Column(Float)  # Voice instability
    shimmer_percent = Column(Float)  # Amplitude variation
    harmonic_to_noise_ratio = Column(Float)  # Voice clarity metric
    formant_dispersion = Column(Float)  # Vocal tract characteristics
    
    # Processing metadata
    audio_duration_seconds = Column(Float)
    processing_time_seconds = Column(Float)
    model_version = Column(String)
    denoising_applied = Column(Boolean)
    
    # Full metrics JSON (for extensibility)
    raw_metrics = Column(JSON)  # Store all raw spectral data
    spectral_features = Column(JSON)  # MFCC, mel spectrogram data
    
    # Timestamps
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("MediaSession", back_populates="audio_metrics")
    
    __table_args__ = (
        Index('idx_audio_patient_date', 'patient_id', 'analyzed_at'),
        Index('idx_audio_session', 'session_id'),
    )
