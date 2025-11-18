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


class FacialPuffinessMetric(Base):
    """Time-series facial puffiness score (FPS) metrics using MediaPipe Face Mesh"""
    __tablename__ = "facial_puffiness_metrics"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("video_exam_sessions.id", ondelete="CASCADE"))
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Composite Facial Puffiness Score (FPS)
    facial_puffiness_score = Column(Float)  # Weighted composite score (0-100+)
    fps_risk_level = Column(String)  # 'low', 'medium', 'high'
    
    # Regional FPS scores (% expansion from baseline)
    fps_periorbital = Column(Float)  # Eye region (30% weight)
    fps_cheek = Column(Float)  # Cheek region (30% weight)
    fps_jawline = Column(Float)  # Jawline region (20% weight)
    fps_forehead = Column(Float)  # Forehead region (10% weight)
    fps_overall_contour = Column(Float)  # Overall facial contour (10% weight)
    
    # Asymmetry detection
    facial_asymmetry_score = Column(Float)  # % difference left/right eye areas
    asymmetry_detected = Column(Boolean, default=False)  # Asymmetry >20%
    
    # Raw measurements (for baseline calculation & validation)
    raw_eye_area = Column(Float)  # Current average eye area
    raw_cheek_width = Column(Float)  # Current cheek width
    raw_cheek_projection = Column(Float)  # Current cheek projection
    raw_jawline_width = Column(Float)  # Current jawline width
    raw_forehead_width = Column(Float)  # Current forehead width
    raw_face_perimeter = Column(Float)  # Current face perimeter
    
    # Detection quality
    detection_confidence = Column(Float, default=0.0)  # 0-1
    frames_analyzed = Column(Integer, default=0)  # Number of frames with face detected
    
    # Metadata for recomputation & debugging
    metrics_metadata = Column(JSON)  # Stores full landmark tracking, frame-by-frame measurements
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indices for efficient time-series queries
    __table_args__ = (
        Index('idx_fps_patient_time', 'patient_id', 'recorded_at'),
    )


class FacialPuffinessBaseline(Base):
    """Patient facial baseline measurements for FPS comparison"""
    __tablename__ = "facial_puffiness_baselines"
    
    patient_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    # Baseline facial measurements (from healthy state videos)
    baseline_eye_area = Column(Float, nullable=False)
    baseline_cheek_width = Column(Float, nullable=False)
    baseline_cheek_projection = Column(Float, nullable=False)
    baseline_jawline_width = Column(Float, nullable=False)
    baseline_forehead_width = Column(Float, nullable=False)
    baseline_face_perimeter = Column(Float, nullable=False)
    
    # Baseline quality metrics
    sample_size = Column(Integer, nullable=False, default=0)  # Number of videos used
    confidence = Column(Float, nullable=False, default=0.0)  # 0-1
    
    # Metadata
    source = Column(String, default="auto")  # 'auto' or 'manual' (clinician override)
    last_calibration_at = Column(DateTime(timezone=True))  # Last time baseline was recalibrated
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class RespiratoryMetric(Base):
    """Time-series respiratory metrics with advanced temporal analytics"""
    __tablename__ = "respiratory_metrics"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    patient_id = Column(String, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, ForeignKey("video_exam_sessions.id", ondelete="CASCADE"))
    
    # Timestamp
    recorded_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Core respiratory rate
    rr_bpm = Column(Float)  # Breaths per minute
    rr_confidence = Column(Float, default=0.0)  # 0-1
    
    # Variability metrics
    breath_interval_std = Column(Float)  # Standard deviation of breath intervals
    variability_index = Column(Float)  # Respiratory Variability Index (RVI)
    
    # Pattern detection
    accessory_muscle_score = Column(Float)  # 0-1 (use of accessory muscles)
    chest_expansion_amplitude = Column(Float)  # Standard deviation of chest movements
    gasping_detected = Column(Boolean, default=False)  # Irregular breathing with deep gasps
    chest_shape_asymmetry = Column(Float)  # Chest width variability
    thoracoabdominal_synchrony = Column(Float)  # 0-1 (0=asynchronous, 1=synchronous)
    
    # Baseline comparison
    z_score_vs_baseline = Column(Float)  # Z-score anomaly detection
    
    # Rolling statistics (computed after insert)
    rolling_daily_avg = Column(Float)  # 24-hour rolling average RR
    rolling_three_day_slope = Column(Float)  # 3-day trend slope (linear regression)
    
    # Metadata
    metrics_metadata = Column(JSON)  # Chest movements sample, FPS, duration
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_resp_patient_time', 'patient_id', 'recorded_at'),
    )


class RespiratoryBaseline(Base):
    """Patient respiratory baseline for Z-score anomaly detection"""
    __tablename__ = "respiratory_baselines"
    
    patient_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    # Baseline respiratory rate
    baseline_rr_bpm = Column(Float, nullable=False)
    baseline_rr_std = Column(Float, nullable=False, default=2.0)  # Guards against divide-by-zero
    
    # Baseline quality
    sample_size = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=False, default=0.0)  # 0-1
    
    # Metadata
    source = Column(String, default="auto")  # 'auto' or 'manual'
    last_calibration_at = Column(DateTime(timezone=True))
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SkinAnalysisMetric(Base):
    """
    Comprehensive skin analysis metrics from video examination
    Tracks perfusion, color changes, capillary refill, and nailbed health
    """
    __tablename__ = "skin_analysis_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    session_id = Column(String, ForeignKey("video_exam_sessions.id", ondelete="CASCADE"), nullable=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False)
    
    # === LAB Color Space Metrics (L*=lightness, a*=red-green, b*=yellow-blue) ===
    
    # Facial region (pallor, cyanosis, jaundice detection)
    facial_l_lightness = Column(Float)  # 0-100 (darker=pallor/cyanosis)
    facial_a_red_green = Column(Float)  # -128 to 127 (negative=cyanosis, positive=healthy pink)
    facial_b_yellow_blue = Column(Float)  # -128 to 127 (positive=jaundice)
    facial_perfusion_index = Column(Float)  # 0-100 composite (low=poor perfusion)
    
    # Palmar region (pallor, anemia detection)
    palmar_l_lightness = Column(Float)
    palmar_a_red_green = Column(Float)
    palmar_b_yellow_blue = Column(Float)
    palmar_perfusion_index = Column(Float)  # 0-100
    
    # Nailbed region (cyanosis, anemia, clubbing)
    nailbed_l_lightness = Column(Float)
    nailbed_a_red_green = Column(Float)
    nailbed_b_yellow_blue = Column(Float)
    nailbed_color_index = Column(Float)  # 0-100 (low=cyanosis/anemia)
    
    # === Clinical Color Changes ===
    pallor_detected = Column(Boolean, default=False)  # Low L* + low a*
    pallor_severity = Column(Float)  # 0-1 (0=none, 1=severe)
    pallor_region = Column(String)  # 'facial', 'palmar', 'nailbed', 'generalized'
    
    cyanosis_detected = Column(Boolean, default=False)  # Low L* + negative a* + negative b*
    cyanosis_severity = Column(Float)  # 0-1
    cyanosis_region = Column(String)  # 'perioral', 'peripheral', 'central'
    
    jaundice_detected = Column(Boolean, default=False)  # High b* (yellow)
    jaundice_severity = Column(Float)  # 0-1
    jaundice_region = Column(String)  # 'sclera', 'facial', 'generalized'
    
    # === Capillary Refill Proxy ===
    capillary_refill_time_sec = Column(Float)  # Time to 90% recovery after finger press
    capillary_refill_method = Column(String)  # 'guided_press', 'passive_observation', 'not_measured'
    capillary_refill_quality = Column(Float)  # 0-1 (confidence in measurement)
    capillary_refill_abnormal = Column(Boolean, default=False)  # >2 seconds = abnormal
    
    # === Nailbed Analysis ===
    nail_clubbing_detected = Column(Boolean, default=False)  # Distal phalanx bulging
    nail_clubbing_severity = Column(Float)  # 0-1 (Schamroth window test proxy)
    nail_pitting_detected = Column(Boolean, default=False)  # Small depressions in nail plate
    nail_pitting_count = Column(Integer)  # Number of pits detected
    nail_abnormalities = Column(JSON)  # Array of {type, location, severity} for leukonychia, splinter hemorrhages, etc.
    
    # === Texture & Temperature Proxies ===
    skin_texture_score = Column(Float)  # 0-1 (Laplacian variance - low=smooth, high=rough/dry)
    hydration_status = Column(String)  # 'dry', 'normal', 'moist'
    temperature_proxy = Column(String)  # 'cool' (pallor), 'normal', 'warm' (redness/inflammation)
    
    # === Rash/Lesion Detection ===
    rash_detected = Column(Boolean, default=False)
    rash_distribution = Column(String)  # 'localized', 'scattered', 'generalized'
    lesions_bruises_detected = Column(Boolean, default=False)
    lesion_details = Column(JSON)  # Array of {type: 'bruise'|'lesion'|'ulcer', location, size_mm, color}
    
    # === Baseline Comparison ===
    z_score_perfusion_vs_baseline = Column(Float)  # Facial perfusion deviation
    z_score_capillary_vs_baseline = Column(Float)  # Capillary refill time deviation
    z_score_nailbed_vs_baseline = Column(Float)  # Nailbed color deviation
    
    # === Rolling Temporal Analytics ===
    rolling_24hr_avg_perfusion = Column(Float)  # 24-hour rolling average
    rolling_3day_perfusion_slope = Column(Float)  # 3-day trend (positive=improving, negative=worsening)
    rolling_24hr_avg_capillary_refill = Column(Float)
    
    # === Detection Quality ===
    detection_confidence = Column(Float, default=0.0)  # 0-1
    frames_analyzed = Column(Integer, default=0)
    facial_roi_detected = Column(Boolean, default=False)
    palmar_roi_detected = Column(Boolean, default=False)
    nailbed_roi_detected = Column(Boolean, default=False)
    
    # Metadata for recomputation & debugging
    metrics_metadata = Column(JSON)  # Raw LAB histograms, landmark coords, video segment timestamps
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Indices for efficient time-series queries
    __table_args__ = (
        Index('idx_skin_patient_time', 'patient_id', 'recorded_at'),
        Index('idx_skin_session', 'session_id'),
    )


class SkinBaseline(Base):
    """Patient skin baseline measurements for personalized deviation detection"""
    __tablename__ = "skin_baselines"
    
    patient_id = Column(String, ForeignKey("users.id"), primary_key=True)
    
    # === Baseline LAB Color Vectors ===
    
    # Facial baseline
    baseline_facial_l = Column(Float, nullable=False)
    baseline_facial_a = Column(Float, nullable=False)
    baseline_facial_b = Column(Float, nullable=False)
    baseline_facial_perfusion = Column(Float, nullable=False)
    
    # Palmar baseline
    baseline_palmar_l = Column(Float, nullable=False)
    baseline_palmar_a = Column(Float, nullable=False)
    baseline_palmar_b = Column(Float, nullable=False)
    baseline_palmar_perfusion = Column(Float, nullable=False)
    
    # Nailbed baseline
    baseline_nailbed_l = Column(Float, nullable=False)
    baseline_nailbed_a = Column(Float, nullable=False)
    baseline_nailbed_b = Column(Float, nullable=False)
    baseline_nailbed_color_index = Column(Float, nullable=False)
    
    # === Capillary Refill Baseline ===
    baseline_capillary_refill_sec = Column(Float, nullable=False, default=1.5)  # Normal <2 sec
    baseline_capillary_refill_std = Column(Float, nullable=False, default=0.3)
    
    # === Texture/Hydration Baseline ===
    baseline_texture_score = Column(Float)
    baseline_hydration_status = Column(String)  # Patient's typical hydration
    
    # === Baseline Quality & EMA Parameters ===
    sample_size = Column(Integer, nullable=False, default=0)
    confidence = Column(Float, nullable=False, default=0.0)  # 0-1
    
    # Metadata
    source = Column(String, default="auto")  # 'auto' or 'manual' (dermatologist-calibrated)
    last_calibration_at = Column(DateTime(timezone=True))
    
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
