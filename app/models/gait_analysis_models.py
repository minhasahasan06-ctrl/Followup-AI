"""
Gait Analysis Database Models
HAR (Human Activity Recognition) based gait pattern tracking
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, JSON, Text, ForeignKey, Index
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class GaitSession(Base):
    """Track gait analysis video sessions"""
    __tablename__ = "gait_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Video storage
    video_s3_uri = Column(String)  # s3://bucket/patient_id/gait_uuid.mp4
    thumbnail_s3_uri = Column(String)
    kms_key_id = Column(String)
    encrypted_at_rest = Column(Boolean, default=True)
    
    # Video metadata
    duration_seconds = Column(Float)
    fps = Column(Float)
    total_frames = Column(Integer)
    resolution = Column(String)  # "1920x1080"
    
    # Analysis status
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    processing_started_at = Column(DateTime(timezone=True))
    processing_completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    
    # Quality scores
    pose_detection_confidence = Column(Float)  # Average MediaPipe confidence (0-1)
    lighting_quality_score = Column(Float)  # 0-100
    camera_stability_score = Column(Float)  # 0-100
    overall_quality_score = Column(Float)  # 0-100
    
    # Gait analysis summary
    total_strides_detected = Column(Integer, default=0)
    walking_detected = Column(Boolean, default=False)
    gait_abnormality_detected = Column(Boolean, default=False)
    gait_abnormality_score = Column(Float)  # 0-100, higher = more abnormal
    
    # Audit
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    uploaded_by = Column(String)
    device_info = Column(JSON)
    
    # Relationships
    gait_metrics = relationship("GaitMetrics", back_populates="session", cascade="all, delete-orphan")
    gait_patterns = relationship("GaitPattern", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_gait_session_patient_date', 'patient_id', 'created_at'),
        Index('idx_gait_session_status', 'processing_status'),
    )


class GaitMetrics(Base):
    """
    Comprehensive gait parameters extracted from video
    Uses MediaPipe Pose (33 landmarks) + HAR analysis
    """
    __tablename__ = "gait_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("gait_sessions.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # === TEMPORAL PARAMETERS (time-based) ===
    # Primary timing metrics
    stride_time_left_sec = Column(Float)  # Time for one full left stride (sec)
    stride_time_right_sec = Column(Float)  # Time for one full right stride (sec)
    stride_time_avg_sec = Column(Float)  # Average stride time
    step_time_left_sec = Column(Float)  # Time for left step (sec)
    step_time_right_sec = Column(Float)  # Time for right step (sec)
    step_time_avg_sec = Column(Float)  # Average step time
    
    # Phase durations
    stance_time_left_sec = Column(Float)  # Left foot ground contact time
    stance_time_right_sec = Column(Float)  # Right foot ground contact time
    swing_time_left_sec = Column(Float)  # Left foot swing phase
    swing_time_right_sec = Column(Float)  # Right foot swing phase
    double_support_time_sec = Column(Float)  # Both feet on ground
    single_support_time_sec = Column(Float)  # One foot on ground
    
    # Cadence & speed
    cadence_steps_per_min = Column(Float)  # Steps per minute (normal: 90-120)
    walking_speed_m_per_sec = Column(Float)  # Estimated walking speed
    
    # === SPATIAL PARAMETERS (distance-based) ===
    stride_length_left_cm = Column(Float)  # Left stride length (cm)
    stride_length_right_cm = Column(Float)  # Right stride length (cm)
    stride_length_avg_cm = Column(Float)  # Average stride length
    step_length_left_cm = Column(Float)  # Left step length
    step_length_right_cm = Column(Float)  # Right step length
    step_length_avg_cm = Column(Float)  # Average step length
    step_width_cm = Column(Float)  # Lateral distance between feet (normal: 5-13cm)
    
    # === JOINT ANGLES (kinematics) ===
    # Hip angles (flexion/extension)
    hip_flexion_angle_left_deg = Column(Float)  # Max left hip flexion (deg)
    hip_flexion_angle_right_deg = Column(Float)  # Max right hip flexion (deg)
    hip_extension_angle_left_deg = Column(Float)  # Max left hip extension (deg)
    hip_extension_angle_right_deg = Column(Float)  # Max right hip extension (deg)
    hip_range_of_motion_left_deg = Column(Float)  # Left hip ROM
    hip_range_of_motion_right_deg = Column(Float)  # Right hip ROM
    
    # Knee angles (flexion/extension)
    knee_flexion_angle_left_deg = Column(Float)  # Max left knee flexion at swing
    knee_flexion_angle_right_deg = Column(Float)  # Max right knee flexion at swing
    knee_extension_angle_left_deg = Column(Float)  # Knee extension at heel strike
    knee_extension_angle_right_deg = Column(Float)  # Knee extension at heel strike
    knee_range_of_motion_left_deg = Column(Float)  # Left knee ROM
    knee_range_of_motion_right_deg = Column(Float)  # Right knee ROM
    
    # Ankle angles (dorsiflexion/plantarflexion)
    ankle_dorsiflexion_angle_left_deg = Column(Float)  # Left ankle dorsiflexion
    ankle_dorsiflexion_angle_right_deg = Column(Float)  # Right ankle dorsiflexion
    ankle_plantarflexion_angle_left_deg = Column(Float)  # Left ankle plantarflexion
    ankle_plantarflexion_angle_right_deg = Column(Float)  # Right ankle plantarflexion
    ankle_range_of_motion_left_deg = Column(Float)  # Left ankle ROM
    ankle_range_of_motion_right_deg = Column(Float)  # Right ankle ROM
    
    # === SYMMETRY & STABILITY ===
    # Symmetry indices (0-1, 1 = perfect symmetry)
    temporal_symmetry_index = Column(Float)  # Timing symmetry (stride time L vs R)
    spatial_symmetry_index = Column(Float)  # Distance symmetry (stride length L vs R)
    joint_angle_symmetry_index = Column(Float)  # Joint angle symmetry
    overall_gait_symmetry_index = Column(Float)  # Composite symmetry score
    
    # Stability metrics
    trunk_sway_lateral_cm = Column(Float)  # Trunk lateral sway (cm)
    trunk_sway_anterior_posterior_cm = Column(Float)  # Trunk AP sway (cm)
    head_stability_score = Column(Float)  # 0-100, higher = more stable
    balance_confidence_score = Column(Float)  # 0-100, ML-based balance estimate
    
    # Gait variability (consistency)
    stride_time_variability_percent = Column(Float)  # CV% of stride time
    stride_length_variability_percent = Column(Float)  # CV% of stride length
    step_width_variability_percent = Column(Float)  # CV% of step width
    
    # === HAR ACTIVITY CLASSIFICATION ===
    # Activity recognition scores (0-1 probability)
    walking_confidence = Column(Float)  # Confidence this is normal walking
    shuffling_confidence = Column(Float)  # Shuffling gait (Parkinson's indicator)
    limping_confidence = Column(Float)  # Asymmetric gait (injury indicator)
    unsteady_confidence = Column(Float)  # Balance issues
    primary_activity_detected = Column(String)  # "walking", "shuffling", "limping", "unsteady"
    
    # === CLINICAL RISK FLAGS ===
    fall_risk_score = Column(Float)  # 0-100, ML-based fall risk
    parkinson_gait_indicators = Column(JSON)  # {shuffling: bool, reduced_arm_swing: bool, freezing: bool}
    neuropathy_indicators = Column(JSON)  # {foot_drop: bool, slapping_gait: bool}
    pain_gait_indicators = Column(JSON)  # {antalgic_gait: bool, reduced_rom: bool}
    
    # === BASELINE COMPARISON ===
    has_baseline = Column(Boolean, default=False)
    baseline_gait_metrics_id = Column(Integer, nullable=True)  # Reference to baseline
    deviation_from_baseline_percent = Column(Float)  # Overall deviation %
    significant_deterioration_detected = Column(Boolean, default=False)
    
    # Metadata
    analysis_method = Column(String, default="mediapipe_pose_har")  # mediapipe_pose_har, opengait
    model_version = Column(String)  # "mediapipe_v0.9.3"
    frames_analyzed = Column(Integer)
    landmarks_detected_percent = Column(Float)  # % of frames with valid pose
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("GaitSession", back_populates="gait_metrics")
    
    __table_args__ = (
        Index('idx_gait_metrics_patient_date', 'patient_id', 'created_at'),
        Index('idx_gait_metrics_abnormality', 'gait_abnormality_detected'),
        Index('idx_gait_metrics_fall_risk', 'fall_risk_score'),
    )


class GaitPattern(Base):
    """
    Time-series gait pattern data for detailed analysis
    Stores stride-by-stride breakdown
    """
    __tablename__ = "gait_patterns"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("gait_sessions.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Stride identification
    stride_number = Column(Integer)  # Stride sequence number (1, 2, 3...)
    side = Column(String)  # "left", "right"
    
    # Temporal
    stride_start_frame = Column(Integer)  # Frame number of heel strike
    stride_end_frame = Column(Integer)  # Frame number of next heel strike
    stride_duration_sec = Column(Float)
    
    # Spatial
    stride_length_cm = Column(Float)
    step_width_cm = Column(Float)
    
    # Joint angles at key events
    hip_angle_at_heel_strike_deg = Column(Float)
    knee_angle_at_heel_strike_deg = Column(Float)
    ankle_angle_at_heel_strike_deg = Column(Float)
    hip_angle_at_toe_off_deg = Column(Float)
    knee_angle_at_toe_off_deg = Column(Float)
    ankle_angle_at_toe_off_deg = Column(Float)
    
    # Gait events (frame numbers)
    heel_strike_frame = Column(Integer)
    toe_off_frame = Column(Integer)
    mid_swing_frame = Column(Integer)
    
    # Quality
    detection_confidence = Column(Float)  # 0-1
    landmarks_complete = Column(Boolean, default=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("GaitSession", back_populates="gait_patterns")
    
    __table_args__ = (
        Index('idx_gait_pattern_session_stride', 'session_id', 'stride_number'),
    )


class GaitBaseline(Base):
    """
    Patient baseline gait metrics for longitudinal tracking
    Updated weekly with rolling 7-day average
    """
    __tablename__ = "gait_baselines"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, unique=True, index=True)
    
    # Baseline temporal parameters
    baseline_stride_time_sec = Column(Float)
    baseline_cadence_steps_per_min = Column(Float)
    baseline_walking_speed_m_per_sec = Column(Float)
    
    # Baseline spatial parameters
    baseline_stride_length_cm = Column(Float)
    baseline_step_width_cm = Column(Float)
    
    # Baseline symmetry
    baseline_symmetry_index = Column(Float)
    
    # Baseline stability
    baseline_trunk_sway_cm = Column(Float)
    baseline_balance_score = Column(Float)
    
    # Variability (normal range for patient)
    baseline_stride_time_cv_percent = Column(Float)
    baseline_stride_length_cv_percent = Column(Float)
    
    # Fall risk baseline
    baseline_fall_risk_score = Column(Float)
    
    # Baseline metadata
    baseline_established_date = Column(DateTime(timezone=True))
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    sessions_used_for_baseline = Column(Integer, default=0)  # Number of sessions averaged
    baseline_quality_score = Column(Float)  # Confidence in baseline (0-100)
    
    # Disease-specific personalization
    disease_conditions = Column(JSON)  # ["parkinsons", "neuropathy", "arthritis"]
    expected_gait_patterns = Column(JSON)  # Disease-specific expected patterns
    
    __table_args__ = (
        Index('idx_gait_baseline_patient', 'patient_id'),
    )
