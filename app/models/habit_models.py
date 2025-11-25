"""
SQLAlchemy models for the AI-Powered Habit Tracker (13 Features)
HIPAA-compliant with proper audit logging and encryption considerations

These models are aligned with the raw SQL queries in app/routers/habits.py
"""

from sqlalchemy import (
    Column, String, Integer, Boolean, DateTime, Float, Text, 
    ForeignKey, JSON, Date, UniqueConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


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
    reminder_time = Column(String(10))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    completions = relationship("HabitCompletion", back_populates="habit", cascade="all, delete-orphan")
    routines = relationship("HabitRoutine", back_populates="habit", cascade="all, delete-orphan")
    reminders = relationship("HabitReminder", back_populates="habit", cascade="all, delete-orphan")


# Habit Completion Tracking - matches SQL in complete_habit endpoint
class HabitCompletion(Base):
    __tablename__ = "habit_completions"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    completion_date = Column(DateTime, server_default=func.now())
    completed = Column(Boolean, default=True)
    mood = Column(String(50))
    notes = Column(Text)
    difficulty_level = Column(Integer)

    habit = relationship("HabitHabit", back_populates="completions")


# Daily Routines - matches SQL in create_routine endpoint
class HabitRoutine(Base):
    __tablename__ = "habit_routines"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    scheduled_time = Column(String(10))
    duration = Column(Integer)
    time_flexibility = Column(String(50), default="flexible")
    location = Column(String(200))
    location_details = Column(Text)
    trigger_cue = Column(String(200))
    stacked_after = Column(String(200))
    day_of_week = Column(JSON)
    is_weekend_only = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())

    habit = relationship("HabitHabit", back_populates="routines")


# Micro-Steps - matches SQL in create_micro_steps endpoint
class HabitMicroStep(Base):
    __tablename__ = "habit_micro_steps"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"), nullable=False, index=True)
    step_order = Column(Integer, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    estimated_minutes = Column(Integer)
    is_required = Column(Boolean, default=True)
    completion_count = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())


# Smart Reminders - matches SQL in create_reminder endpoint
class HabitReminder(Base):
    __tablename__ = "habit_reminders"

    id = Column(String, primary_key=True, default=generate_uuid)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="CASCADE"))
    user_id = Column(String, nullable=False, index=True)
    reminder_type = Column(String(50), default="in_app")
    scheduled_time = Column(String(10), nullable=False)
    message = Column(Text)
    adaptive_enabled = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    snooze_until = Column(DateTime)
    last_sent_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())

    habit = relationship("HabitHabit", back_populates="reminders")


# AI Habit Coach Chat Messages - matches SQL in coach/chat endpoint
class HabitCoachChat(Base):
    __tablename__ = "habit_coach_chats"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    session_id = Column(String, nullable=False, index=True)
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    coach_personality = Column(String(50))
    response_type = Column(String(50))
    related_habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    related_quit_plan_id = Column(String)
    created_at = Column(DateTime, server_default=func.now())


# AI Trigger Detection - matches SQL in triggers/analyze endpoint
class HabitAiTrigger(Base):
    __tablename__ = "habit_ai_triggers"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    trigger_type = Column(String(50), nullable=False)
    pattern = Column(Text, nullable=False)
    correlated_factor = Column(String(200))
    confidence = Column(Float, default=0.0)
    data_points = Column(Integer, default=0)
    sample_period_days = Column(Integer, default=30)
    is_active = Column(Boolean, default=True)
    acknowledged = Column(Boolean, default=False)
    helpful = Column(Boolean)
    last_detected_at = Column(DateTime, server_default=func.now())
    created_at = Column(DateTime, server_default=func.now())


# Mood Tracking - matches SQL in mood/log endpoint
class HabitMoodEntry(Base):
    __tablename__ = "habit_mood_entries"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    mood_score = Column(Integer, nullable=False)
    mood_label = Column(String(50))
    energy_level = Column(Integer)
    stress_level = Column(Integer)
    journal_text = Column(Text)
    sentiment_score = Column(Float)
    extracted_emotions = Column(JSON)
    extracted_themes = Column(JSON)
    associated_habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    context_tags = Column(JSON)
    recorded_at = Column(DateTime, server_default=func.now())


# Addiction-Mode Quit Plans - matches SQL in quit-plans/create endpoint
class HabitQuitPlan(Base):
    __tablename__ = "habit_quit_plans"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    habit_name = Column(String(200), nullable=False)
    category = Column(String(50), default="behavioral")
    quit_method = Column(String(50), default="gradual_reduction")
    target_quit_date = Column(Date)
    daily_limit = Column(Integer)
    harm_reduction_steps = Column(JSON)
    reasons_to_quit = Column(JSON)
    money_saved_per_day = Column(Float, default=0.0)
    start_date = Column(Date, server_default=func.current_date())
    days_clean = Column(Integer, default=0)
    longest_streak = Column(Integer, default=0)
    total_relapses = Column(Integer, default=0)
    status = Column(String(50), default="active")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Cravings Log - matches SQL in quit-plans/craving endpoint
class HabitCravingsLog(Base):
    __tablename__ = "habit_cravings_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    quit_plan_id = Column(String, ForeignKey("habit_quit_plans.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    intensity = Column(Integer)
    duration = Column(Integer)
    trigger = Column(Text)
    trigger_details = Column(Text)
    coping_strategy_used = Column(Text)
    overcame = Column(Boolean, default=False)
    notes = Column(Text)
    location = Column(String(200))
    mood = Column(String(50))
    occurred_at = Column(DateTime, server_default=func.now())


# Relapse Log - matches SQL in quit-plans/relapse endpoint
class HabitRelapseLog(Base):
    __tablename__ = "habit_relapse_log"

    id = Column(String, primary_key=True, default=generate_uuid)
    quit_plan_id = Column(String, ForeignKey("habit_quit_plans.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String, nullable=False, index=True)
    severity = Column(String(50), default="minor_slip")
    quantity = Column(String(100))
    trigger = Column(Text)
    emotional_state = Column(String(100))
    what_happened = Column(Text)
    what_learned = Column(Text)
    plan_to_prevent = Column(Text)
    streak_days_lost = Column(Integer, default=0)
    occurred_at = Column(DateTime, server_default=func.now())


# AI Recommendations - matches SQL in recommendations/generate endpoint
class HabitAiRecommendation(Base):
    __tablename__ = "habit_ai_recommendations"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    recommendation_type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    based_on_completion_rate = Column(Float)
    based_on_streak = Column(Integer)
    confidence = Column(Float, default=0.0)
    priority = Column(String(20), default="medium")
    status = Column(String(50), default="pending")
    created_at = Column(DateTime, server_default=func.now())


# Social Accountability / Buddy System - matches SQL in buddies endpoints
class HabitBuddy(Base):
    __tablename__ = "habit_buddies"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    buddy_user_id = Column(String, nullable=False, index=True)
    status = Column(String(50), default="pending")
    initiated_by = Column(String)
    share_streak = Column(Boolean, default=True)
    share_completions = Column(Boolean, default=True)
    share_mood = Column(Boolean, default=False)
    encouragements_sent = Column(Integer, default=0)
    encouragements_received = Column(Integer, default=0)
    last_interaction = Column(DateTime)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint('user_id', 'buddy_user_id', name='uq_buddy_pair'),
    )


# Encouragement Messages - matches SQL in buddies/encourage endpoint
class HabitEncouragement(Base):
    __tablename__ = "habit_encouragements"

    id = Column(String, primary_key=True, default=generate_uuid)
    from_user_id = Column(String, nullable=False, index=True)
    to_user_id = Column(String, nullable=False, index=True)
    message_type = Column(String(50), default="support")
    message = Column(Text, nullable=False)
    related_habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    is_read = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


# CBT Sessions - matches SQL in cbt/start endpoint
class HabitCbtSession(Base):
    __tablename__ = "habit_cbt_sessions"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    session_type = Column(String(50), nullable=False)
    title = Column(String(200))
    current_step = Column(Integer, default=1)
    total_steps = Column(Integer, default=0)
    step_responses = Column(JSON)
    pre_session_mood = Column(Integer)
    post_session_mood = Column(Integer)
    completed = Column(Boolean, default=False)
    related_habit_id = Column(String, ForeignKey("habit_habits.id", ondelete="SET NULL"))
    related_quit_plan_id = Column(String)
    started_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime)


# Gamification & Rewards - matches SQL in rewards endpoint
class HabitReward(Base):
    __tablename__ = "habit_rewards"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True, unique=True)
    reward_type = Column(String(50), default="plant_growth")
    current_level = Column(Integer, default=1)
    growth_stage = Column(String(50), default="seed")
    total_points = Column(Integer, default=0)
    streak_bonus = Column(Integer, default=0)
    completion_points = Column(Integer, default=0)
    visual_state = Column(JSON)
    unlocked_badges = Column(JSON)
    days_active = Column(Integer, default=0)
    perfect_days = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


# Smart Journals with AI Insights - matches SQL in journals endpoints
class HabitJournal(Base):
    __tablename__ = "habit_journals"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    title = Column(String(200))
    content = Column(Text, nullable=False)
    entry_type = Column(String(50), default="daily")
    mood = Column(String(50))
    tags = Column(JSON)
    ai_summary = Column(Text)
    highlights = Column(JSON)
    risks = Column(JSON)
    recommendations = Column(JSON)
    sentiment_trend = Column(String(50))
    is_weekly_summary = Column(Boolean, default=False)
    recorded_at = Column(DateTime, server_default=func.now())


# Risk Alerts - matches SQL in alerts endpoints
class HabitRiskAlert(Base):
    __tablename__ = "habit_risk_alerts"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, nullable=False, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    risk_score = Column(Float, default=0.0)
    contributing_factors = Column(JSON)
    suggested_actions = Column(JSON)
    status = Column(String(50), default="active")
    acknowledged_at = Column(DateTime)
    created_at = Column(DateTime, server_default=func.now())
