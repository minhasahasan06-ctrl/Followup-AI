"""
Video AI Engine Database Models
Tracks video analysis sessions, metrics, and quality scoring
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class MediaSession(Base):
    """Track video/audio recording sessions"""
    __tablename__ = "media_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    session_type = Column(String, nullable=False)  # "video", "audio", "combined"
    
    # S3 storage (encrypted)
    video_s3_uri = Column(String)  # s3://bucket/patient_id/video_uuid.mp4
    audio_s3_uri = Column(String)  # s3://bucket/patient_id/audio_uuid.wav
    thumbnail_s3_uri = Column(String)  # Preview thumbnail
    
    # Encryption metadata
    kms_key_id = Column(String)  # AWS KMS key used for encryption
    encrypted_at_rest = Column(Boolean, default=True)
    
    # Session metadata
    duration_seconds = Column(Float)
    file_size_bytes = Column(Integer)
    resolution = Column(String)  # "1920x1080"
    fps = Column(Float)  # Frames per second
    codec = Column(String)  # "h264", "vp9", etc.
    
    # Processing status
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    # Quality scores
    video_quality_score = Column(Float)  # 0-100
    audio_quality_score = Column(Float)  # 0-100
    overall_quality_score = Column(Float)  # 0-100
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    uploaded_by = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    
    # Relationships
    video_metrics = relationship("VideoMetrics", back_populates="session", cascade="all, delete-orphan")
    audio_metrics = relationship("AudioMetrics", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_media_patient_date', 'patient_id', 'created_at'),
        Index('idx_media_status', 'processing_status'),
    )


class VideoMetrics(Base):
    """Store video AI analysis results (10+ metrics)"""
    __tablename__ = "video_metrics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("media_sessions.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Respiratory metrics
    respiratory_rate_bpm = Column(Float)  # Breaths per minute
    respiratory_rate_confidence = Column(Float)  # 0-1
    breathing_pattern = Column(String)  # "regular", "irregular", "shallow", "labored"
    chest_movement_amplitude = Column(Float)  # Pixels of movement
    
    # Skin pallor detection (HSV color space)
    skin_pallor_score = Column(Float)  # 0-100 (higher = more pale)
    face_brightness_avg = Column(Float)  # Average face brightness
    face_saturation_avg = Column(Float)  # Average face saturation
    pallor_confidence = Column(Float)  # 0-1
    
    # Eye sclera color (jaundice detection)
    sclera_yellowness_score = Column(Float)  # 0-100 (higher = more yellow)
    sclera_red_component = Column(Float)  # RGB red channel
    sclera_green_component = Column(Float)  # RGB green channel
    sclera_blue_component = Column(Float)  # RGB blue channel
    jaundice_risk_level = Column(String)  # "low", "medium", "high"
    
    # Facial swelling detection
    facial_swelling_score = Column(Float)  # 0-100
    left_cheek_distance = Column(Float)  # Landmark distance
    right_cheek_distance = Column(Float)
    eye_puffiness_left = Column(Float)  # 0-100
    eye_puffiness_right = Column(Float)  # 0-100
    facial_asymmetry_score = Column(Float)  # Deviation from baseline
    
    # Head movement / stability
    head_movement_total = Column(Float)  # Total movement in pixels
    head_stability_score = Column(Float)  # 0-100 (higher = more stable)
    head_tilt_angle = Column(Float)  # Degrees
    tremor_detected = Column(Boolean)
    tremor_frequency_hz = Column(Float)
    
    # Lighting quality
    lighting_quality_score = Column(Float)  # 0-100
    lighting_uniformity = Column(Float)  # Std dev of brightness
    shadows_detected = Column(Boolean)
    overexposure_percent = Column(Float)  # % of frame overexposed
    underexposure_percent = Column(Float)  # % of frame underexposed
    
    # Face detection quality
    face_detection_confidence = Column(Float)  # 0-1
    face_occlusion_percent = Column(Float)  # % of face occluded
    face_size_pixels = Column(Integer)  # Face bounding box area
    multiple_faces_detected = Column(Boolean)
    
    # Overall video quality
    frame_quality_avg = Column(Float)  # 0-100
    blur_score = Column(Float)  # Higher = more blurred
    noise_level = Column(Float)  # Image noise
    compression_artifacts = Column(Float)  # JPEG/video compression issues
    
    # Processing metadata
    frames_analyzed = Column(Integer)
    processing_time_seconds = Column(Float)
    model_version = Column(String)
    
    # Full metrics JSON (for extensibility)
    raw_metrics = Column(JSON)  # Store all raw data
    
    # Timestamps
    analyzed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("MediaSession", back_populates="video_metrics")
    
    __table_args__ = (
        Index('idx_video_patient_date', 'patient_id', 'analyzed_at'),
        Index('idx_video_session', 'session_id'),
    )
