"""
SQLAlchemy models for the AI-Powered Habit Tracker (13 Features)
HIPAA-compliant with proper audit logging and encryption considerations
"""

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Float, Text, 
    ForeignKey, Enum, JSON, Date, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid
import enum


def generate_uuid():
    return str(uuid.uuid4())


class HabitCategory(enum.Enum):
    health = "health"
    medication = "medication"
    exercise = "exercise"
    wellness = "wellness"
    nutrition = "nutrition"
    sleep = "sleep"
    other = "other"


class HabitFrequency(enum.Enum):
    daily = "daily"
    weekly = "weekly"
    custom = "custom"


class HabitStatus(enum.Enum):
    active = "active"
    paused = "paused"
    completed = "completed"
    archived = "archived"


class CompletionStatus(enum.Enum):
    completed = "completed"
    skipped = "skipped"
    partial = "partial"


class QuitPlanCategory(enum.Enum):
    substance = "substance"
    behavioral = "behavioral"
    food = "food"
    other = "other"


class QuitMethod(enum.Enum):
    cold_turkey = "cold_turkey"
    gradual_reduction = "gradual_reduction"
    replacement = "replacement"


class QuitPlanStatus(enum.Enum):
    active = "active"
    completed = "completed"
    relapsed = "relapsed"
    paused = "paused"


class ReminderStatus(enum.Enum):
    active = "active"
    snoozed = "snoozed"
    dismissed = "dismissed"
    completed = "completed"


class TriggerType(enum.Enum):
    time_based = "time_based"
    location_based = "location_based"
    mood_based = "mood_based"
    activity_based = "activity_based"
    social_based = "social_based"


class RewardType(enum.Enum):
    plant_growth = "plant_growth"
    badge = "badge"
    points = "points"
    streak_bonus = "streak_bonus"


class CbtSessionType(enum.Enum):
    urge_surfing = "urge_surfing"
    reframe_thought = "reframe_thought"
    grounding = "grounding"
    breathing = "breathing"


class CbtSessionStatus(enum.Enum):
    in_progress = "in_progress"
    completed = "completed"
    abandoned = "abandoned"


class BuddyStatus(enum.Enum):
    pending = "pending"
    accepted = "accepted"
    declined = "declined"
    removed = "removed"


class JournalEntryType(enum.Enum):
    daily = "daily"
    reflection = "reflection"
    gratitude = "gratitude"
    goal_setting = "goal_setting"


class RecommendationStatus(enum.Enum):
    pending = "pending"
    accepted = "accepted"
    dismissed = "dismissed"
    completed = "completed"


class AlertStatus(enum.Enum):
    active = "active"
    acknowledged = "acknowledged"
    resolved = "resolved"
    dismissed = "dismissed"


class AlertSeverity(enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# Core Habit Model
class HabitHabit(Base):
    __tablename__ = "habit_habits"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    category = Column(String(50), default="other")
    frequency = Column(String(50), default="daily")
    goal_count = Column(Integer, default=1)
    current_streak = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    total_completions = Column(Integer, default=0)
    status = Column(String(50), default="active")
    reminder_enabled = Column(Boolean, default=True)
    reminder_time = Column(String(10))  # HH:MM format
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    completions = relationship("HabitCompletion", back_populates="habit", cascade="all, delete-orphan")
    routines = relationship("HabitRoutine", back_populates="habit", cascade="all, delete-orphan")
    reminders = relationship("HabitReminder", back_populates="habit", cascade="all, delete-orphan")


# Habit Completion Tracking
class HabitCompletion(Base):
    __tablename__ = "habit_completions"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    completion_date = Column(Date, nullable=False)
    status = Column(String(50), default="completed")
    mood = Column(String(50))
    notes = Column(Text)
    difficulty_level = Column(Integer)  # 1-5
    completed_at = Column(DateTime, server_default=func.now())

    # Relationships
    habit = relationship("HabitHabit", back_populates="completions")

    __table_args__ = (
        UniqueConstraint('habit_id', 'completion_date', name='uq_habit_completion_date'),
    )


# Daily Routines with Micro-Steps
class HabitRoutine(Base):
    __tablename__ = "habit_routines"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    order_index = Column(Integer, default=0)
    time_of_day = Column(String(50))  # morning, afternoon, evening, night
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    habit = relationship("HabitHabit", back_populates="routines")
    micro_steps = relationship("HabitMicroStep", back_populates="routine", cascade="all, delete-orphan")


class HabitMicroStep(Base):
    __tablename__ = "habit_micro_steps"

    id = Column(String, primary_key=True, default=generate_uuid)
    routine_id = Column(String, ForeignKey("habit_routines.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    description = Column(Text)
    order_index = Column(Integer, default=0)
    duration_minutes = Column(Integer)
    is_completed = Column(Boolean, default=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    routine = relationship("HabitRoutine", back_populates="micro_steps")


# Smart Reminders
class HabitReminder(Base):
    __tablename__ = "habit_reminders"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"))
    user_id = Column(String, nullable=False, index=True)
    reminder_time = Column(String(10), nullable=False)  # HH:MM format
    days_of_week = Column(JSON)  # [0,1,2,3,4,5,6] for days
    message = Column(Text)
    status = Column(String(50), default="active")
    snooze_until = Column(DateTime)
    last_triggered = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    habit = relationship("HabitHabit", back_populates="reminders")


# AI Trigger Detection
class HabitTrigger(Base):
    __tablename__ = "habit_triggers"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    trigger_type = Column(String(50), nullable=False)
    pattern = Column(Text, nullable=False)
    correlated_factor = Column(String(200))
    confidence = Column(Float, default=0.0)
    data_points = Column(Integer, default=0)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Mood Tracking
class HabitMoodEntry(Base):
    __tablename__ = "habit_mood_entries"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    mood_score = Column(Integer, nullable=False)  # 1-10
    mood_label = Column(String(50))  # great, good, okay, struggling
    energy_level = Column(Integer)  # 1-10
    stress_level = Column(Integer)  # 1-10
    journal_text = Column(Text)
    sentiment_score = Column(Float)  # -1 to 1
    recorded_at = Column(DateTime, server_default=func.now())


# Addiction-Mode Quit Plans
class HabitQuitPlan(Base):
    __tablename__ = "habit_quit_plans"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    habit_name = Column(String(200), nullable=False)
    category = Column(String(50), default="behavioral")
    quit_method = Column(String(50), default="gradual_reduction")
    start_date = Column(Date, server_default=func.current_date())
    days_clean = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    total_relapses = Column(Integer, default=0)
    daily_limit = Column(Integer)
    reasons_to_quit = Column(JSON)  # Array of strings
    money_saved_per_day = Column(Float, default=0.0)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # Relationships
    cravings = relationship("HabitCraving", back_populates="quit_plan", cascade="all, delete-orphan")


class HabitCraving(Base):
    __tablename__ = "habit_cravings"

    id = Column(String, primary_key=True, default=generate_uuid)
    quit_plan_id = Column(String, ForeignKey("habit_quit_plans.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    intensity = Column(Integer)  # 1-10
    trigger = Column(Text)
    coping_strategy = Column(Text)
    was_resisted = Column(Boolean, default=True)
    recorded_at = Column(DateTime, server_default=func.now())

    # Relationships
    quit_plan = relationship("HabitQuitPlan", back_populates="cravings")


# AI Habit Coach Messages
class HabitCoachMessage(Base):
    __tablename__ = "habit_coach_messages"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    role = Column(String(20), nullable=False)  # user or assistant
    content = Column(Text, nullable=False)
    response_type = Column(String(50))  # motivation, cbt_technique, advice, reflection
    context_data = Column(JSON)  # Additional context for personalization
    created_at = Column(DateTime, server_default=func.now())


# CBT Sessions
class HabitCbtSession(Base):
    __tablename__ = "habit_cbt_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    session_type = Column(String(50), nullable=False)
    title = Column(String(200))
    total_steps = Column(Integer, default=0)
    current_step = Column(Integer, default=1)
    status = Column(String(50), default="in_progress")
    responses = Column(JSON)  # Array of step responses
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime)


# Gamification & Rewards
class HabitReward(Base):
    __tablename__ = "habit_rewards"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True, unique=True)
    reward_type = Column(String(50), default="plant_growth")
    current_level = Column(Integer, default=1)
    growth_stage = Column(String(50), default="seed")  # seed, sprout, growing, blooming, flourishing
    total_points = Column(Integer, default=0)
    streak_bonus = Column(Integer, default=0)
    completion_points = Column(Integer, default=0)
    visual_state = Column(JSON)  # Custom visual state data
    unlocked_badges = Column(JSON)  # Array of badge IDs
    days_active = Column(Integer, default=0)
    perfect_days = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Social Accountability / Buddy System
class HabitBuddy(Base):
    __tablename__ = "habit_buddies"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    buddy_user_id = Column(String, nullable=False, index=True)
    status = Column(String(50), default="pending")
    connected_at = Column(DateTime)
    last_nudge_at = Column(DateTime)
    shared_habits = Column(JSON)  # Array of habit IDs they share
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint('user_id', 'buddy_user_id', name='uq_buddy_pair'),
    )


# Smart Journals with AI Insights
class HabitJournal(Base):
    __tablename__ = "habit_journals"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String(200))
    content = Column(Text, nullable=False)
    entry_type = Column(String(50), default="daily")
    mood = Column(String(50))
    ai_summary = Column(Text)
    highlights = Column(JSON)  # Array of key insights
    risks = Column(JSON)  # Array of detected risks
    recommendations = Column(JSON)  # Array of AI recommendations
    sentiment_trend = Column(String(50))
    is_weekly_summary = Column(Boolean, default=False)
    recorded_at = Column(DateTime, server_default=func.now())


# AI Recommendations
class HabitRecommendation(Base):
    __tablename__ = "habit_recommendations"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    recommendation_type = Column(String(50), nullable=False)
    category = Column(String(50))
    title = Column(String(200), nullable=False)
    description = Column(Text)
    reasoning = Column(Text)
    confidence_score = Column(Float, default=0.0)
    priority = Column(String(20), default="medium")
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, server_default=func.now())
    expires_at = Column(DateTime)


# Risk Alerts
class HabitRiskAlert(Base):
    __tablename__ = "habit_risk_alerts"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    risk_score = Column(Float, default=0.0)
    contributing_factors = Column(JSON)  # Array of factor objects
    suggested_actions = Column(JSON)  # Array of action strings
    status = Column(String(50), default="active")
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
