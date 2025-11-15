"""
Home Clinical Exam Coach (HCEC) Models
AI-powered system that teaches patients proper self-examination techniques
with real-time coaching and quality feedback
"""

from sqlalchemy import Column, String, Integer, Float, DateTime, Text, Boolean, JSON, ForeignKey, Enum as SQLEnum
from sqlalchemy.sql import func
from app.database import Base
import enum


class ExamType(str, enum.Enum):
    """Types of standardized home examinations"""
    SKIN = "skin"  # Skin inspection for lesions, rashes, changes
    THROAT = "throat"  # Throat examination for inflammation, tonsils
    LEGS = "legs"  # Leg examination for edema, swelling
    ROM = "rom"  # Range of Motion for joints
    ABDOMEN = "abdomen"  # Abdomen palpation guidance
    NEURO = "neuro"  # Brief neurological exam
    RESPIRATORY = "respiratory"  # Respiratory effort assessment
    SINUS = "sinus"  # Sinus pressure test


class SessionStatus(str, enum.Enum):
    """Status of an exam coaching session"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


class QualityLevel(str, enum.Enum):
    """Image/video quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"


class FeedbackType(str, enum.Enum):
    """Types of coaching feedback"""
    LIGHTING = "lighting"
    ANGLE = "angle"
    DISTANCE = "distance"
    POSITIONING = "positioning"
    VISIBILITY = "visibility"
    READINESS = "readiness"


class ExamSession(Base):
    """
    Tracks individual exam coaching sessions
    Each session guides the patient through a specific exam type
    """
    __tablename__ = "exam_sessions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    exam_type = Column(SQLEnum(ExamType), nullable=False)
    
    # Session metadata
    status = Column(SQLEnum(SessionStatus), default=SessionStatus.IN_PROGRESS)
    total_steps = Column(Integer, default=0)
    completed_steps = Column(Integer, default=0)
    
    # Quality assessment
    overall_quality = Column(SQLEnum(QualityLevel))
    quality_score = Column(Float)  # 0-1 score
    
    # Coaching metrics
    coaching_interactions = Column(Integer, default=0)  # Number of coaching prompts given
    average_step_duration = Column(Float)  # Seconds per step
    
    # Pre-consultation flag
    is_pre_consultation = Column(Boolean, default=False)  # True if preparing for doctor visit
    consultation_date = Column(DateTime(timezone=True))
    
    # Timestamps
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ExamStep(Base):
    """
    Individual steps within an exam session
    Each step has coaching feedback and quality assessment
    """
    __tablename__ = "exam_steps"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False, index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Step details
    step_number = Column(Integer, nullable=False)
    step_instruction = Column(Text, nullable=False)  # e.g., "Position camera to view throat"
    step_type = Column(String)  # "photo", "video", "palpation_guidance"
    
    # Media references
    s3_bucket = Column(String)
    s3_key = Column(String)
    s3_url = Column(String)  # Pre-signed URL
    
    # Quality metrics
    lighting_quality = Column(SQLEnum(QualityLevel))
    angle_quality = Column(SQLEnum(QualityLevel))
    visibility_quality = Column(SQLEnum(QualityLevel))
    overall_quality = Column(SQLEnum(QualityLevel))
    
    # Coaching feedback
    coaching_feedback = Column(JSON)  # List of coaching prompts given
    feedback_count = Column(Integer, default=0)  # Number of prompts
    
    # AI analysis
    ai_assessment = Column(Text)  # AI evaluation of this step
    visibility_confirmed = Column(Boolean, default=False)  # "Lesion visible", "Ear canal centered"
    
    # Readiness confirmation
    is_ready = Column(Boolean, default=False)  # Patient confirmed ready
    readiness_message = Column(String)  # "Good lighting", "Angle optimal"
    
    # Timing
    duration_seconds = Column(Float)
    attempts = Column(Integer, default=1)  # Number of retakes
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    completed_at = Column(DateTime(timezone=True))


class CoachingFeedback(Base):
    """
    Real-time coaching feedback given during examination
    Tracks all prompts and guidance provided
    """
    __tablename__ = "coaching_feedback"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), nullable=False, index=True)
    step_id = Column(Integer, ForeignKey("exam_steps.id"), index=True)
    patient_id = Column(String, nullable=False, index=True)
    
    # Feedback details
    feedback_type = Column(SQLEnum(FeedbackType), nullable=False)
    feedback_message = Column(Text, nullable=False)  # "Tilt camera closer", "Turn on a light"
    
    # Voice guidance
    voice_guidance = Column(Text)  # Text-to-speech message
    was_spoken = Column(Boolean, default=False)
    
    # Impact
    issue_resolved = Column(Boolean, default=False)
    resolution_time_seconds = Column(Float)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ExamPacket(Base):
    """
    Generated exam packet for doctor review
    Compiles all exam images, videos, and assessments
    """
    __tablename__ = "exam_packets"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("exam_sessions.id"), index=True)
    
    # Packet type
    packet_type = Column(String, nullable=False)  # "daily_followup", "pre_consultation"
    exam_date = Column(DateTime(timezone=True), nullable=False)
    
    # Exam summary
    exams_included = Column(JSON)  # List of exam types included
    total_images = Column(Integer, default=0)
    total_videos = Column(Integer, default=0)
    
    # Quality summary
    overall_quality = Column(SQLEnum(QualityLevel))
    lighting_adequate = Column(Boolean, default=False)
    all_areas_visible = Column(Boolean, default=False)
    
    # Patient description
    patient_description = Column(Text)  # Patient's description of findings
    concerns_noted = Column(JSON)  # List of patient concerns
    
    # Media compilation
    compiled_images = Column(JSON)  # List of S3 keys for all images
    compiled_videos = Column(JSON)  # List of S3 keys for all videos
    
    # Doctor notes (filled during review)
    doctor_reviewed = Column(Boolean, default=False)
    doctor_id = Column(String, index=True)
    doctor_notes = Column(Text)
    reviewed_at = Column(DateTime(timezone=True))
    
    # PDF report
    pdf_generated = Column(Boolean, default=False)
    pdf_s3_key = Column(String)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class ExamProtocol(Base):
    """
    Standardized protocols for each exam type
    Defines step-by-step instructions and quality criteria
    """
    __tablename__ = "exam_protocols"

    id = Column(Integer, primary_key=True, index=True)
    exam_type = Column(SQLEnum(ExamType), unique=True, nullable=False, index=True)
    
    # Protocol details
    name = Column(String, nullable=False)
    description = Column(Text)
    estimated_duration_minutes = Column(Integer)
    
    # Steps (JSON array of step instructions)
    steps = Column(JSON, nullable=False)
    # Example structure:
    # [
    #   {
    #     "step_number": 1,
    #     "instruction": "Position camera to view throat",
    #     "type": "photo",
    #     "coaching_hints": ["Ensure good lighting", "Open mouth wide"],
    #     "quality_criteria": {"lighting": "good", "angle": "centered"}
    #   }
    # ]
    
    # Quality requirements
    min_quality_score = Column(Float, default=0.7)
    required_images = Column(Integer, default=1)
    required_videos = Column(Integer, default=0)
    
    # Safety notes
    safety_instructions = Column(Text)
    contraindications = Column(Text)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
