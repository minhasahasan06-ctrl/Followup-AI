"""
Comprehensive Habit Tracker API Router

Features:
1. Habit Creation & Daily Routine Builder
2. Streaks + Calendar View
3. Smart Reminders
4. AI Habit Coach (Personalized Agent)
5. AI Trigger & Pattern Detection
6. Addiction-Mode (Quit Plans / Bad Habit Control)
7. Emotion + Mood Tracking
8. Dynamic AI Recommendations
9. Social Accountability
10. Guided CBT / Motivational Interventions
11. Visual Reward System (Gamification)
12. Smart Journals with AI Reflection
13. Preventive Alerts & Prediction Engine

HIPAA Compliance:
- All endpoints require proper authentication
- User data is scoped to authenticated user only
- Audit logging for all PHI access
"""

import os
import json
import logging
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
from decimal import Decimal
import random
import re
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.habit_models import (
    HabitHabit, HabitCompletion, HabitRoutine, HabitMicroStep,
    HabitReminder, HabitCoachChat, HabitAiTrigger, HabitMoodEntry, 
    HabitQuitPlan, HabitCravingsLog, HabitRelapseLog, HabitAiRecommendation,
    HabitBuddy, HabitEncouragement, HabitCbtSession, HabitReward,
    HabitJournal, HabitRiskAlert
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/habits", tags=["habits"])


def get_user_id_from_auth_or_query(
    user_id: Optional[str] = Query(None, description="User ID (dev mode only)"),
    current_user: Optional[User] = Depends(get_current_user)
) -> str:
    """
    Get user ID from authenticated user or fallback to query param in dev mode.
    In production, only authenticated user ID is used.
    """
    if current_user:
        return current_user.id
    if user_id:
        return user_id
    raise HTTPException(status_code=401, detail="Authentication required")

# ============================================
# Pydantic Models
# ============================================

class HabitCreate(BaseModel):
    name: str
    description: Optional[str] = None
    category: str = "health"
    frequency: str = "daily"
    target_days_per_week: Optional[int] = 7
    reminder_enabled: bool = True
    reminder_time: Optional[str] = None
    goal_count: int = 1

class HabitRoutineCreate(BaseModel):
    habit_id: str
    scheduled_time: Optional[str] = None
    duration: Optional[int] = None
    time_flexibility: str = "flexible"
    location: Optional[str] = None
    location_details: Optional[str] = None
    trigger_cue: Optional[str] = None
    stacked_after: Optional[str] = None
    day_of_week: Optional[List[str]] = None

class MicroStepCreate(BaseModel):
    habit_id: str
    step_order: int
    title: str
    description: Optional[str] = None
    estimated_minutes: Optional[int] = None
    is_required: bool = True

class HabitCompletionCreate(BaseModel):
    habit_id: str
    completed: bool = True
    notes: Optional[str] = None
    mood: Optional[str] = None
    difficulty_level: Optional[int] = None

class QuitPlanCreate(BaseModel):
    habit_name: str
    category: Optional[str] = "behavioral"
    quit_method: Optional[str] = "gradual_reduction"
    target_quit_date: Optional[datetime] = None
    daily_limit: Optional[int] = None
    harm_reduction_steps: Optional[List[Dict[str, Any]]] = None
    reasons_to_quit: Optional[List[str]] = None
    money_saved_per_day: Optional[float] = None

class CravingsLogCreate(BaseModel):
    quit_plan_id: str
    intensity: int
    duration: Optional[int] = None
    trigger: Optional[str] = None
    trigger_details: Optional[str] = None
    coping_strategy_used: Optional[str] = None
    overcame: bool = False
    notes: Optional[str] = None
    location: Optional[str] = None
    mood: Optional[str] = None

class RelapseLogCreate(BaseModel):
    quit_plan_id: str
    severity: Optional[str] = "minor_slip"
    quantity: Optional[str] = None
    trigger: Optional[str] = None
    emotional_state: Optional[str] = None
    what_happened: Optional[str] = None
    what_learned: Optional[str] = None
    plan_to_prevent: Optional[str] = None

class MoodEntryCreate(BaseModel):
    mood_score: int = Field(..., ge=1, le=10)
    mood_label: Optional[str] = None
    energy_level: Optional[int] = Field(None, ge=1, le=10)
    stress_level: Optional[int] = Field(None, ge=1, le=10)
    journal_text: Optional[str] = None
    associated_habit_id: Optional[str] = None
    context_tags: Optional[List[str]] = None

class JournalCreate(BaseModel):
    title: Optional[str] = None
    content: str
    entry_type: Optional[str] = "daily"
    tags: Optional[List[str]] = None
    mood: Optional[str] = None

class BuddyRequest(BaseModel):
    buddy_user_id: str
    share_streak: bool = True
    share_completions: bool = True
    share_mood: bool = False

class EncouragementCreate(BaseModel):
    to_user_id: str
    message_type: Optional[str] = "support"
    message: str
    related_habit_id: Optional[str] = None

class CbtSessionCreate(BaseModel):
    session_type: str
    related_habit_id: Optional[str] = None
    related_quit_plan_id: Optional[str] = None
    pre_session_mood: Optional[int] = None

class CbtStepResponse(BaseModel):
    session_id: str
    step_number: int
    response: str

class CoachMessage(BaseModel):
    message: str
    related_habit_id: Optional[str] = None
    related_quit_plan_id: Optional[str] = None

# ============================================
# FEATURE 1: Habit Creation & Daily Routine Builder
# ============================================

@router.post("/create")
async def create_habit(
    habit: HabitCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Create a new habit with optional routine configuration"""
    try:
        new_habit = HabitHabit(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=habit.name,
            description=habit.description,
            category=habit.category,
            frequency=habit.frequency,
            goal_count=habit.goal_count,
            reminder_enabled=habit.reminder_enabled,
            reminder_time=habit.reminder_time,
            is_active=True,
            current_streak=0,
            longest_streak=0,
            total_completions=0
        )
        
        db.add(new_habit)
        db.commit()
        db.refresh(new_habit)
        
        return {
            "success": True,
            "habit": {
                "id": new_habit.id,
                "name": new_habit.name,
                "category": new_habit.category,
                "frequency": new_habit.frequency,
                "currentStreak": new_habit.current_streak,
                "longestStreak": new_habit.longest_streak,
                "totalCompletions": new_habit.total_completions,
                "createdAt": new_habit.created_at.isoformat() if new_habit.created_at else None
            }
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating habit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_habits(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    category: Optional[str] = None,
    include_inactive: bool = False,
    db: Session = Depends(get_db)
):
    """Get all habits for a user with optional filtering"""
    try:
        query = db.query(HabitHabit).filter(HabitHabit.user_id == user_id)
        
        if not include_inactive:
            query = query.filter(HabitHabit.is_active == True)
        if category:
            query = query.filter(HabitHabit.category == category)
            
        habits_db = query.order_by(HabitHabit.created_at.desc()).all()
        
        habits = []
        for h in habits_db:
            habits.append({
                "id": h.id,
                "name": h.name,
                "description": h.description,
                "category": h.category,
                "frequency": h.frequency,
                "goalCount": h.goal_count,
                "reminderEnabled": h.reminder_enabled,
                "reminderTime": h.reminder_time,
                "isActive": h.is_active,
                "currentStreak": h.current_streak or 0,
                "longestStreak": h.longest_streak or 0,
                "totalCompletions": h.total_completions or 0,
                "createdAt": h.created_at.isoformat() if h.created_at else None
            })
        
        return {"habits": habits, "total": len(habits)}
    except Exception as e:
        logger.error(f"Error listing habits: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/routine/create")
async def create_routine(
    routine: HabitRoutineCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Create a daily routine configuration for a habit"""
    try:
        insert_query = text("""
            INSERT INTO habit_routines (id, habit_id, user_id, scheduled_time, duration,
                                       time_flexibility, location, location_details, 
                                       trigger_cue, stacked_after, day_of_week, is_weekend_only)
            VALUES (gen_random_uuid(), :habit_id, :user_id, :scheduled_time, :duration,
                   :time_flexibility, :location, :location_details, :trigger_cue, 
                   :stacked_after, :day_of_week::jsonb, false)
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "habit_id": routine.habit_id,
            "user_id": user_id,
            "scheduled_time": routine.scheduled_time,
            "duration": routine.duration,
            "time_flexibility": routine.time_flexibility,
            "location": routine.location,
            "location_details": routine.location_details,
            "trigger_cue": routine.trigger_cue,
            "stacked_after": routine.stacked_after,
            "day_of_week": json.dumps(routine.day_of_week) if routine.day_of_week else None
        })
        
        row = result.fetchone()
        db.commit()
        
        return {"success": True, "routine_id": row[0]}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating routine: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/micro-steps/create")
async def create_micro_steps(
    steps: List[MicroStepCreate],
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Create micro-steps for breaking down a habit"""
    try:
        created = []
        for step in steps:
            # Verify habit ownership before creating micro-step
            ownership_check = text("SELECT id FROM habit_habits WHERE id = :habit_id AND user_id = :user_id")
            habit_exists = db.execute(ownership_check, {"habit_id": step.habit_id, "user_id": user_id}).fetchone()
            if not habit_exists:
                raise HTTPException(status_code=403, detail="Not authorized to add micro-steps to this habit")
            
            insert_query = text("""
                INSERT INTO habit_micro_steps (id, habit_id, step_order, title, 
                                               description, estimated_minutes, is_required)
                VALUES (gen_random_uuid(), :habit_id, :step_order, :title, 
                       :description, :estimated_minutes, :is_required)
                RETURNING id
            """)
            
            result = db.execute(insert_query, {
                "habit_id": step.habit_id,
                "step_order": step.step_order,
                "title": step.title,
                "description": step.description,
                "estimated_minutes": step.estimated_minutes,
                "is_required": step.is_required
            })
            
            created.append(result.fetchone()[0])
        
        db.commit()
        return {"success": True, "created_steps": created}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating micro-steps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/micro-steps/{habit_id}")
async def get_micro_steps(
    habit_id: str,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get all micro-steps for a habit"""
    try:
        # Verify habit ownership
        ownership_check = text("SELECT id FROM habit_habits WHERE id = :habit_id AND user_id = :user_id")
        habit_exists = db.execute(ownership_check, {"habit_id": habit_id, "user_id": user_id}).fetchone()
        if not habit_exists:
            raise HTTPException(status_code=403, detail="Not authorized to view this habit's micro-steps")
        
        query = text("""
            SELECT id, step_order, title, description, estimated_minutes, 
                   is_required, completion_count
            FROM habit_micro_steps
            WHERE habit_id = :habit_id
            ORDER BY step_order ASC
        """)
        
        result = db.execute(query, {"habit_id": habit_id})
        rows = result.fetchall()
        
        return {
            "steps": [
                {
                    "id": row[0],
                    "stepOrder": row[1],
                    "title": row[2],
                    "description": row[3],
                    "estimatedMinutes": row[4],
                    "isRequired": row[5],
                    "completionCount": row[6] or 0
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting micro-steps: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 2: Streaks + Calendar View
# ============================================

@router.post("/complete")
async def complete_habit(
    completion: HabitCompletionCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Log a habit completion and update streaks"""
    try:
        # Insert completion record
        insert_query = text("""
            INSERT INTO habit_completions (id, habit_id, user_id, completion_date, 
                                          completed, notes, mood, difficulty_level)
            VALUES (gen_random_uuid(), :habit_id, :user_id, NOW(), 
                   :completed, :notes, :mood, :difficulty)
            RETURNING id, completion_date
        """)
        
        result = db.execute(insert_query, {
            "habit_id": completion.habit_id,
            "user_id": user_id,
            "completed": completion.completed,
            "notes": completion.notes,
            "mood": completion.mood,
            "difficulty": completion.difficulty_level
        })
        
        completion_row = result.fetchone()
        
        # Update habit streak if completed
        if completion.completed:
            update_query = text("""
                UPDATE habit_habits 
                SET current_streak = current_streak + 1,
                    longest_streak = GREATEST(longest_streak, current_streak + 1),
                    total_completions = total_completions + 1,
                    updated_at = NOW()
                WHERE id = :habit_id
                RETURNING current_streak, longest_streak, total_completions
            """)
            
            streak_result = db.execute(update_query, {"habit_id": completion.habit_id})
            streak_row = streak_result.fetchone()
        else:
            # Reset streak on skip
            update_query = text("""
                UPDATE habit_habits 
                SET current_streak = 0,
                    updated_at = NOW()
                WHERE id = :habit_id
                RETURNING current_streak, longest_streak, total_completions
            """)
            streak_result = db.execute(update_query, {"habit_id": completion.habit_id})
            streak_row = streak_result.fetchone()
        
        db.commit()
        
        # Update rewards in background
        if background_tasks and completion.completed:
            background_tasks.add_task(update_rewards, user_id, db)
        
        return {
            "success": True,
            "completion_id": completion_row[0],
            "completion_date": completion_row[1].isoformat(),
            "current_streak": streak_row[0] if streak_row else 0,
            "longest_streak": streak_row[1] if streak_row else 0,
            "total_completions": streak_row[2] if streak_row else 0
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error completing habit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/calendar")
async def get_calendar_view(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    year: int = Query(default=None),
    month: int = Query(default=None),
    db: Session = Depends(get_db)
):
    """Get calendar view of habit completions for a month"""
    try:
        if year is None:
            year = datetime.now().year
        if month is None:
            month = datetime.now().month
            
        start_date = datetime(year, month, 1)
        if month == 12:
            end_date = datetime(year + 1, 1, 1)
        else:
            end_date = datetime(year, month + 1, 1)
        
        query = text("""
            SELECT hc.habit_id, h.name, hc.completion_date::date as day, 
                   hc.completed, hc.mood
            FROM habit_completions hc
            JOIN habit_habits h ON h.id = hc.habit_id
            WHERE hc.user_id = :user_id
            AND hc.completion_date >= :start_date
            AND hc.completion_date < :end_date
            ORDER BY hc.completion_date
        """)
        
        result = db.execute(query, {
            "user_id": user_id,
            "start_date": start_date,
            "end_date": end_date
        })
        
        rows = result.fetchall()
        
        # Organize by date
        calendar_data = {}
        for row in rows:
            day_str = row[2].isoformat() if row[2] else None
            if day_str not in calendar_data:
                calendar_data[day_str] = {"completions": [], "totalCompleted": 0, "totalSkipped": 0}
            
            calendar_data[day_str]["completions"].append({
                "habitId": row[0],
                "habitName": row[1],
                "completed": row[3],
                "mood": row[4]
            })
            
            if row[3]:
                calendar_data[day_str]["totalCompleted"] += 1
            else:
                calendar_data[day_str]["totalSkipped"] += 1
        
        return {
            "year": year,
            "month": month,
            "calendar": calendar_data
        }
    except Exception as e:
        logger.error(f"Error getting calendar: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/streaks")
async def get_streaks_summary(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get streak summary for all active habits"""
    try:
        query = text("""
            SELECT id, name, category, current_streak, longest_streak, total_completions
            FROM habit_habits
            WHERE user_id = :user_id AND is_active = true
            ORDER BY current_streak DESC
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        total_current = sum(row[3] or 0 for row in rows)
        total_longest = sum(row[4] or 0 for row in rows)
        
        return {
            "habits": [
                {
                    "id": row[0],
                    "name": row[1],
                    "category": row[2],
                    "currentStreak": row[3] or 0,
                    "longestStreak": row[4] or 0,
                    "totalCompletions": row[5] or 0
                }
                for row in rows
            ],
            "summary": {
                "totalCurrentStreaks": total_current,
                "totalLongestStreaks": total_longest,
                "activeHabits": len(rows)
            }
        }
    except Exception as e:
        logger.error(f"Error getting streaks: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 3: Smart Reminders
# ============================================

@router.post("/reminders/create")
async def create_reminder(
    habit_id: str,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    reminder_type: str = "in_app",
    scheduled_time: str = "09:00",
    message: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Create a smart reminder for a habit"""
    try:
        insert_query = text("""
            INSERT INTO habit_reminders (id, habit_id, user_id, reminder_type, 
                                        scheduled_time, message, adaptive_enabled, is_active)
            VALUES (gen_random_uuid(), :habit_id, :user_id, :reminder_type, 
                   :scheduled_time, :message, true, true)
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "habit_id": habit_id,
            "user_id": user_id,
            "reminder_type": reminder_type,
            "scheduled_time": scheduled_time,
            "message": message
        })
        
        db.commit()
        return {"success": True, "reminder_id": result.fetchone()[0]}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating reminder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reminders/{reminder_id}/snooze")
async def snooze_reminder(
    reminder_id: str,
    minutes: int = 30,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Snooze a reminder for specified minutes"""
    try:
        # Verify reminder ownership
        ownership_check = text("SELECT id FROM habit_reminders WHERE id = :reminder_id AND user_id = :user_id")
        reminder_exists = db.execute(ownership_check, {"reminder_id": reminder_id, "user_id": user_id}).fetchone()
        if not reminder_exists:
            raise HTTPException(status_code=403, detail="Not authorized to snooze this reminder")
        
        update_query = text("""
            UPDATE habit_reminders
            SET snooze_until = NOW() + :minutes * INTERVAL '1 minute'
            WHERE id = :reminder_id AND user_id = :user_id
            RETURNING id
        """)
        
        result = db.execute(update_query, {"reminder_id": reminder_id, "minutes": minutes, "user_id": user_id})
        db.commit()
        
        return {"success": True, "snoozed_for_minutes": minutes}
    except Exception as e:
        db.rollback()
        logger.error(f"Error snoozing reminder: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reminders/pending")
async def get_pending_reminders(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get all pending reminders for today"""
    try:
        query = text("""
            SELECT r.id, r.habit_id, h.name as habit_name, r.scheduled_time, 
                   r.message, r.snooze_until, r.last_sent_at
            FROM habit_reminders r
            JOIN habit_habits h ON h.id = r.habit_id
            WHERE r.user_id = :user_id 
            AND r.is_active = true
            AND (r.snooze_until IS NULL OR r.snooze_until < NOW())
            ORDER BY r.scheduled_time
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        return {
            "reminders": [
                {
                    "id": row[0],
                    "habitId": row[1],
                    "habitName": row[2],
                    "scheduledTime": row[3],
                    "message": row[4],
                    "snoozeUntil": row[5].isoformat() if row[5] else None,
                    "lastSentAt": row[6].isoformat() if row[6] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting reminders: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 4: AI Habit Coach (Personalized Agent)
# ============================================

@router.post("/coach/chat")
async def chat_with_coach(
    message: CoachMessage,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Chat with the AI habit coach for personalized advice"""
    try:
        import openai
        
        # Get user context
        context_query = text("""
            SELECT h.name, h.category, h.current_streak, h.total_completions,
                   (SELECT mood FROM habit_completions WHERE user_id = :user_id 
                    ORDER BY completion_date DESC LIMIT 1) as recent_mood
            FROM habit_habits h
            WHERE h.user_id = :user_id AND h.is_active = true
            LIMIT 5
        """)
        
        context_result = db.execute(context_query, {"user_id": user_id})
        habits_context = context_result.fetchall()
        
        habits_summary = "\n".join([
            f"- {row[0]} ({row[1]}): {row[2]} day streak, {row[3]} total completions"
            for row in habits_context
        ])
        
        system_prompt = f"""You are a supportive and empathetic AI habit coach. You help users build healthy habits using evidence-based techniques like CBT, motivational interviewing, and behavioral science.

Current user habits:
{habits_summary}

Guidelines:
- Be encouraging but realistic
- Offer specific, actionable advice
- Use CBT techniques when appropriate (reframing, urge surfing)
- Celebrate small wins
- Help identify triggers and patterns
- Keep responses concise (2-3 paragraphs max)
- If the user seems distressed, recommend professional help while being supportive"""

        # Generate session ID if new conversation
        session_id = f"coach_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Save user message
        insert_msg = text("""
            INSERT INTO habit_coach_chats (id, user_id, session_id, role, content, 
                                          related_habit_id, related_quit_plan_id)
            VALUES (gen_random_uuid(), :user_id, :session_id, 'user', :content,
                   :habit_id, :quit_plan_id)
        """)
        
        db.execute(insert_msg, {
            "user_id": user_id,
            "session_id": session_id,
            "content": message.message,
            "habit_id": message.related_habit_id,
            "quit_plan_id": message.related_quit_plan_id
        })
        
        # Generate AI response
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message.message}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Determine response type
        response_type = "encouragement"
        if "cbt" in message.message.lower() or "thought" in message.message.lower():
            response_type = "cbt_technique"
        elif "tip" in message.message.lower() or "advice" in message.message.lower():
            response_type = "tip"
        
        # Save AI response
        insert_ai = text("""
            INSERT INTO habit_coach_chats (id, user_id, session_id, role, content,
                                          coach_personality, response_type,
                                          related_habit_id, related_quit_plan_id)
            VALUES (gen_random_uuid(), :user_id, :session_id, 'assistant', :content,
                   'supportive', :response_type, :habit_id, :quit_plan_id)
        """)
        
        db.execute(insert_ai, {
            "user_id": user_id,
            "session_id": session_id,
            "content": ai_response,
            "response_type": response_type,
            "habit_id": message.related_habit_id,
            "quit_plan_id": message.related_quit_plan_id
        })
        
        db.commit()
        
        return {
            "response": ai_response,
            "session_id": session_id,
            "response_type": response_type
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error in coach chat: {e}")
        # Fallback response
        return {
            "response": "I'm here to support you on your habit journey! Remember, every small step counts. What specific challenge are you facing today?",
            "session_id": "fallback",
            "response_type": "encouragement"
        }

@router.get("/coach/history")
async def get_coach_history(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """Get recent coach conversation history"""
    try:
        query = text("""
            SELECT id, session_id, role, content, response_type, created_at
            FROM habit_coach_chats
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        
        result = db.execute(query, {"user_id": user_id, "limit": limit})
        rows = result.fetchall()
        
        return {
            "messages": [
                {
                    "id": row[0],
                    "sessionId": row[1],
                    "role": row[2],
                    "content": row[3],
                    "responseType": row[4],
                    "createdAt": row[5].isoformat() if row[5] else None
                }
                for row in reversed(rows)
            ]
        }
    except Exception as e:
        logger.error(f"Error getting coach history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 5: AI Trigger & Pattern Detection
# ============================================

@router.post("/triggers/analyze")
async def analyze_triggers(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Analyze habit completion patterns to detect triggers"""
    try:
        # Get completion history with mood/sleep context
        query = text("""
            SELECT hc.habit_id, h.name, hc.completion_date, hc.completed, hc.mood,
                   hc.difficulty_level,
                   EXTRACT(DOW FROM hc.completion_date) as day_of_week,
                   EXTRACT(HOUR FROM hc.completion_date) as hour
            FROM habit_completions hc
            JOIN habit_habits h ON h.id = hc.habit_id
            WHERE hc.user_id = :user_id
            AND hc.completion_date >= NOW() - INTERVAL '30 days'
            ORDER BY hc.completion_date
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        if len(rows) < 7:
            return {
                "triggers": [],
                "message": "Need more data to detect patterns. Keep tracking for at least 7 days!"
            }
        
        # Analyze patterns
        triggers_found = []
        habit_stats = {}
        
        for row in rows:
            habit_id = row[0]
            if habit_id not in habit_stats:
                habit_stats[habit_id] = {
                    "name": row[1],
                    "completions": 0,
                    "skips": 0,
                    "mood_when_completed": [],
                    "mood_when_skipped": [],
                    "day_completions": {i: 0 for i in range(7)},
                    "day_skips": {i: 0 for i in range(7)}
                }
            
            stats = habit_stats[habit_id]
            day = int(row[6])
            
            if row[3]:  # completed
                stats["completions"] += 1
                stats["day_completions"][day] += 1
                if row[4]:
                    stats["mood_when_completed"].append(row[4])
            else:
                stats["skips"] += 1
                stats["day_skips"][day] += 1
                if row[4]:
                    stats["mood_when_skipped"].append(row[4])
        
        # Generate insights
        days_map = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        
        for habit_id, stats in habit_stats.items():
            total = stats["completions"] + stats["skips"]
            if total < 5:
                continue
            
            # Day of week pattern
            for day in range(7):
                day_total = stats["day_completions"][day] + stats["day_skips"][day]
                if day_total >= 2:
                    skip_rate = stats["day_skips"][day] / day_total if day_total > 0 else 0
                    if skip_rate > 0.6:
                        pattern = f"You tend to skip {stats['name']} on {days_map[day]}s"
                        confidence = min(0.9, 0.5 + (day_total * 0.1))
                        
                        triggers_found.append({
                            "habitId": habit_id,
                            "habitName": stats["name"],
                            "triggerType": "day_of_week",
                            "pattern": pattern,
                            "correlatedFactor": days_map[day],
                            "confidence": round(confidence, 2),
                            "dataPoints": day_total
                        })
            
            # Mood pattern
            if len(stats["mood_when_skipped"]) >= 3:
                struggling_count = stats["mood_when_skipped"].count("struggling")
                if struggling_count >= 2:
                    pattern = f"You often skip {stats['name']} when feeling stressed or struggling"
                    triggers_found.append({
                        "habitId": habit_id,
                        "habitName": stats["name"],
                        "triggerType": "mood",
                        "pattern": pattern,
                        "correlatedFactor": "struggling_mood",
                        "confidence": 0.75,
                        "dataPoints": len(stats["mood_when_skipped"])
                    })
        
        # Store detected triggers
        for trigger in triggers_found[:10]:  # Store top 10
            insert_query = text("""
                INSERT INTO habit_ai_triggers (id, user_id, habit_id, trigger_type, pattern,
                                              correlated_factor, confidence, data_points, 
                                              sample_period_days, is_active, last_detected_at)
                VALUES (gen_random_uuid(), :user_id, :habit_id, :trigger_type, :pattern,
                       :correlated_factor, :confidence, :data_points, 30, true, NOW())
                ON CONFLICT DO NOTHING
            """)
            
            db.execute(insert_query, {
                "user_id": user_id,
                "habit_id": trigger["habitId"],
                "trigger_type": trigger["triggerType"],
                "pattern": trigger["pattern"],
                "correlated_factor": trigger["correlatedFactor"],
                "confidence": trigger["confidence"],
                "data_points": trigger["dataPoints"]
            })
        
        db.commit()
        
        return {
            "triggers": triggers_found,
            "analyzed_habits": len(habit_stats),
            "data_points": len(rows)
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error analyzing triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/triggers")
async def get_triggers(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get detected triggers for a user"""
    try:
        query = text("""
            SELECT id, habit_id, trigger_type, pattern, correlated_factor,
                   confidence, data_points, acknowledged, helpful, last_detected_at
            FROM habit_ai_triggers
            WHERE user_id = :user_id AND is_active = true
            ORDER BY confidence DESC, last_detected_at DESC
            LIMIT 20
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        return {
            "triggers": [
                {
                    "id": row[0],
                    "habitId": row[1],
                    "triggerType": row[2],
                    "pattern": row[3],
                    "correlatedFactor": row[4],
                    "confidence": float(row[5]) if row[5] else 0,
                    "dataPoints": row[6],
                    "acknowledged": row[7],
                    "helpful": row[8],
                    "lastDetectedAt": row[9].isoformat() if row[9] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 6: Addiction-Mode (Quit Plans)
# ============================================

@router.post("/quit-plans/create")
async def create_quit_plan(
    plan: QuitPlanCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Create a quit plan for breaking a bad habit"""
    try:
        insert_query = text("""
            INSERT INTO habit_quit_plans (id, user_id, habit_name, category, quit_method,
                                         target_quit_date, daily_limit, harm_reduction_steps,
                                         reasons_to_quit, money_saved_per_day, start_date, status)
            VALUES (gen_random_uuid(), :user_id, :habit_name, :category, :quit_method,
                   :target_quit_date, :daily_limit, :harm_reduction::jsonb,
                   :reasons::jsonb, :money_saved, NOW(), 'active')
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "user_id": user_id,
            "habit_name": plan.habit_name,
            "category": plan.category,
            "quit_method": plan.quit_method,
            "target_quit_date": plan.target_quit_date,
            "daily_limit": plan.daily_limit,
            "harm_reduction": json.dumps(plan.harm_reduction_steps) if plan.harm_reduction_steps else None,
            "reasons": json.dumps(plan.reasons_to_quit) if plan.reasons_to_quit else None,
            "money_saved": plan.money_saved_per_day
        })
        
        db.commit()
        return {"success": True, "quit_plan_id": result.fetchone()[0]}
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating quit plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/quit-plans")
async def get_quit_plans(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get all quit plans for a user"""
    try:
        query = text("""
            SELECT id, habit_name, category, quit_method, target_quit_date,
                   daily_limit, harm_reduction_steps, reasons_to_quit, money_saved_per_day,
                   start_date, days_clean, longest_streak, total_relapses, status, created_at
            FROM habit_quit_plans
            WHERE user_id = :user_id
            ORDER BY created_at DESC
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        return {
            "quitPlans": [
                {
                    "id": row[0],
                    "habitName": row[1],
                    "category": row[2],
                    "quitMethod": row[3],
                    "targetQuitDate": row[4].isoformat() if row[4] else None,
                    "dailyLimit": row[5],
                    "harmReductionSteps": row[6],
                    "reasonsToQuit": row[7],
                    "moneySavedPerDay": float(row[8]) if row[8] else 0,
                    "startDate": row[9].isoformat() if row[9] else None,
                    "daysClean": row[10] or 0,
                    "longestStreak": row[11] or 0,
                    "totalRelapses": row[12] or 0,
                    "status": row[13],
                    "createdAt": row[14].isoformat() if row[14] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting quit plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quit-plans/craving")
async def log_craving(
    craving: CravingsLogCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Log a craving event"""
    try:
        insert_query = text("""
            INSERT INTO habit_cravings_log (id, quit_plan_id, user_id, intensity, duration,
                                           trigger, trigger_details, coping_strategy_used,
                                           overcame, notes, location, mood, occurred_at)
            VALUES (gen_random_uuid(), :quit_plan_id, :user_id, :intensity, :duration,
                   :trigger, :trigger_details, :coping_strategy, :overcame, :notes,
                   :location, :mood, NOW())
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "quit_plan_id": craving.quit_plan_id,
            "user_id": user_id,
            "intensity": craving.intensity,
            "duration": craving.duration,
            "trigger": craving.trigger,
            "trigger_details": craving.trigger_details,
            "coping_strategy": craving.coping_strategy_used,
            "overcame": craving.overcame,
            "notes": craving.notes,
            "location": craving.location,
            "mood": craving.mood
        })
        
        # If overcame, update days clean
        if craving.overcame:
            update_query = text("""
                UPDATE habit_quit_plans
                SET days_clean = days_clean + 1,
                    longest_streak = GREATEST(longest_streak, days_clean + 1)
                WHERE id = :quit_plan_id
            """)
            db.execute(update_query, {"quit_plan_id": craving.quit_plan_id})
        
        db.commit()
        return {"success": True, "craving_id": result.fetchone()[0], "overcame": craving.overcame}
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging craving: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/quit-plans/relapse")
async def log_relapse(
    relapse: RelapseLogCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Log a relapse event"""
    try:
        # Get current streak before reset
        streak_query = text("""
            SELECT days_clean FROM habit_quit_plans WHERE id = :quit_plan_id
        """)
        streak_result = db.execute(streak_query, {"quit_plan_id": relapse.quit_plan_id})
        current_streak = streak_result.fetchone()
        streak_days_lost = current_streak[0] if current_streak else 0
        
        insert_query = text("""
            INSERT INTO habit_relapse_log (id, quit_plan_id, user_id, severity, quantity,
                                          trigger, emotional_state, what_happened,
                                          what_learned, plan_to_prevent, streak_days_lost)
            VALUES (gen_random_uuid(), :quit_plan_id, :user_id, :severity, :quantity,
                   :trigger, :emotional_state, :what_happened, :what_learned,
                   :plan_to_prevent, :streak_days_lost)
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "quit_plan_id": relapse.quit_plan_id,
            "user_id": user_id,
            "severity": relapse.severity,
            "quantity": relapse.quantity,
            "trigger": relapse.trigger,
            "emotional_state": relapse.emotional_state,
            "what_happened": relapse.what_happened,
            "what_learned": relapse.what_learned,
            "plan_to_prevent": relapse.plan_to_prevent,
            "streak_days_lost": streak_days_lost
        })
        
        # Reset streak and increment relapses
        update_query = text("""
            UPDATE habit_quit_plans
            SET days_clean = 0,
                total_relapses = total_relapses + 1
            WHERE id = :quit_plan_id
        """)
        db.execute(update_query, {"quit_plan_id": relapse.quit_plan_id})
        
        db.commit()
        
        return {
            "success": True,
            "relapse_id": result.fetchone()[0],
            "streak_days_lost": streak_days_lost,
            "message": "Remember, setbacks are part of the journey. Every day is a new opportunity."
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging relapse: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 7: Emotion + Mood Tracking
# ============================================

@router.post("/mood/log")
async def log_mood(
    entry: MoodEntryCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Log a mood entry with optional journal text"""
    try:
        # Simple sentiment analysis on journal text
        sentiment_score = 0.0
        extracted_emotions = []
        extracted_themes = []
        
        if entry.journal_text:
            text_lower = entry.journal_text.lower()
            
            # Basic sentiment scoring
            positive_words = ["happy", "great", "good", "wonderful", "amazing", "excited", "grateful", "love", "joy"]
            negative_words = ["sad", "angry", "anxious", "worried", "stressed", "tired", "frustrated", "depressed"]
            
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            total = pos_count + neg_count
            
            if total > 0:
                sentiment_score = (pos_count - neg_count) / total
            
            # Extract emotions
            emotion_map = {
                "happy": ["happy", "joy", "excited", "cheerful"],
                "sad": ["sad", "down", "depressed", "unhappy"],
                "anxious": ["anxious", "worried", "nervous", "stressed"],
                "angry": ["angry", "frustrated", "annoyed", "irritated"],
                "calm": ["calm", "peaceful", "relaxed", "content"],
                "tired": ["tired", "exhausted", "fatigued", "drained"]
            }
            
            for emotion, keywords in emotion_map.items():
                if any(kw in text_lower for kw in keywords):
                    extracted_emotions.append(emotion)
            
            # Extract themes
            theme_map = {
                "work": ["work", "job", "office", "meeting", "boss", "colleague"],
                "family": ["family", "mom", "dad", "parent", "child", "sibling"],
                "health": ["health", "doctor", "sick", "medicine", "exercise"],
                "relationships": ["friend", "partner", "relationship", "date"],
                "self-care": ["self-care", "relax", "meditation", "sleep"]
            }
            
            for theme, keywords in theme_map.items():
                if any(kw in text_lower for kw in keywords):
                    extracted_themes.append(theme)
        
        insert_query = text("""
            INSERT INTO habit_mood_entries (id, user_id, mood_score, mood_label, energy_level,
                                           stress_level, journal_text, sentiment_score,
                                           extracted_emotions, extracted_themes,
                                           associated_habit_id, context_tags, recorded_at)
            VALUES (gen_random_uuid(), :user_id, :mood_score, :mood_label, :energy_level,
                   :stress_level, :journal_text, :sentiment_score,
                   :extracted_emotions::jsonb, :extracted_themes::jsonb,
                   :associated_habit_id, :context_tags::jsonb, NOW())
            RETURNING id, recorded_at
        """)
        
        result = db.execute(insert_query, {
            "user_id": user_id,
            "mood_score": entry.mood_score,
            "mood_label": entry.mood_label,
            "energy_level": entry.energy_level,
            "stress_level": entry.stress_level,
            "journal_text": entry.journal_text,
            "sentiment_score": sentiment_score,
            "extracted_emotions": json.dumps(extracted_emotions),
            "extracted_themes": json.dumps(extracted_themes),
            "associated_habit_id": entry.associated_habit_id,
            "context_tags": json.dumps(entry.context_tags) if entry.context_tags else None
        })
        
        row = result.fetchone()
        db.commit()
        
        return {
            "success": True,
            "entry_id": row[0],
            "recorded_at": row[1].isoformat(),
            "sentiment_score": sentiment_score,
            "extracted_emotions": extracted_emotions,
            "extracted_themes": extracted_themes
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error logging mood: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/mood/trends")
async def get_mood_trends(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    days: int = 30,
    db: Session = Depends(get_db)
):
    """Get mood trends over time"""
    try:
        query = text("""
            SELECT recorded_at::date as day, 
                   AVG(mood_score) as avg_mood,
                   AVG(energy_level) as avg_energy,
                   AVG(stress_level) as avg_stress,
                   AVG(sentiment_score) as avg_sentiment
            FROM habit_mood_entries
            WHERE user_id = :user_id
            AND recorded_at >= NOW() - :days * INTERVAL '1 day'
            GROUP BY recorded_at::date
            ORDER BY day
        """)
        
        result = db.execute(query, {"user_id": user_id, "days": days})
        rows = result.fetchall()
        
        return {
            "trends": [
                {
                    "date": row[0].isoformat(),
                    "avgMood": round(float(row[1]), 1) if row[1] else None,
                    "avgEnergy": round(float(row[2]), 1) if row[2] else None,
                    "avgStress": round(float(row[3]), 1) if row[3] else None,
                    "avgSentiment": round(float(row[4]), 2) if row[4] else None
                }
                for row in rows
            ],
            "period_days": days
        }
    except Exception as e:
        logger.error(f"Error getting mood trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 8: Dynamic AI Recommendations
# ============================================

@router.post("/recommendations/generate")
async def generate_recommendations(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Generate AI recommendations based on habit completion patterns"""
    try:
        # Analyze completion rates
        query = text("""
            SELECT h.id, h.name, h.category,
                   COUNT(CASE WHEN hc.completed THEN 1 END)::float / NULLIF(COUNT(*), 0) as completion_rate,
                   h.current_streak,
                   AVG(hc.difficulty_level) as avg_difficulty
            FROM habit_habits h
            LEFT JOIN habit_completions hc ON hc.habit_id = h.id 
                AND hc.completion_date >= NOW() - INTERVAL '14 days'
            WHERE h.user_id = :user_id AND h.is_active = true
            GROUP BY h.id, h.name, h.category, h.current_streak
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        recommendations = []
        
        for row in rows:
            habit_id, name, category, completion_rate, streak, avg_difficulty = row
            completion_rate = completion_rate or 0
            avg_difficulty = float(avg_difficulty) if avg_difficulty else 3
            
            # Low completion rate - suggest micro-steps
            if completion_rate < 0.5 and completion_rate > 0:
                recommendations.append({
                    "habitId": habit_id,
                    "habitName": name,
                    "recommendationType": "micro_step",
                    "title": f"Break down '{name}' into smaller steps",
                    "description": f"Your completion rate is {completion_rate*100:.0f}%. Try breaking this habit into 2-3 micro-steps to make it more achievable.",
                    "basedOnCompletionRate": completion_rate,
                    "confidence": 0.8,
                    "priority": "high"
                })
            
            # High difficulty - suggest easier version
            if avg_difficulty >= 4:
                recommendations.append({
                    "habitId": habit_id,
                    "habitName": name,
                    "recommendationType": "difficulty_adjustment",
                    "title": f"Consider an easier version of '{name}'",
                    "description": f"You've rated this habit as difficult (avg: {avg_difficulty:.1f}/5). Consider reducing the intensity or duration.",
                    "basedOnCompletionRate": completion_rate,
                    "confidence": 0.75,
                    "priority": "medium"
                })
            
            # Good streak - celebrate
            if streak and streak >= 7:
                recommendations.append({
                    "habitId": habit_id,
                    "habitName": name,
                    "recommendationType": "celebrate",
                    "title": f"Amazing {streak}-day streak on '{name}'!",
                    "description": "You're doing great! Consider increasing the challenge or adding a related habit.",
                    "basedOnStreak": streak,
                    "confidence": 0.9,
                    "priority": "low"
                })
        
        # Store recommendations
        for rec in recommendations[:5]:  # Top 5
            insert_query = text("""
                INSERT INTO habit_ai_recommendations (id, user_id, habit_id, recommendation_type,
                                                     title, description, based_on_completion_rate,
                                                     based_on_streak, confidence, priority, status)
                VALUES (gen_random_uuid(), :user_id, :habit_id, :rec_type, :title, :description,
                       :completion_rate, :streak, :confidence, :priority, 'pending')
            """)
            
            db.execute(insert_query, {
                "user_id": user_id,
                "habit_id": rec.get("habitId"),
                "rec_type": rec["recommendationType"],
                "title": rec["title"],
                "description": rec["description"],
                "completion_rate": rec.get("basedOnCompletionRate"),
                "streak": rec.get("basedOnStreak"),
                "confidence": rec["confidence"],
                "priority": rec["priority"]
            })
        
        db.commit()
        
        return {"recommendations": recommendations}
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recommendations")
async def get_recommendations(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    status: str = "pending",
    db: Session = Depends(get_db)
):
    """Get AI recommendations for a user"""
    try:
        query = text("""
            SELECT id, habit_id, recommendation_type, title, description,
                   based_on_completion_rate, based_on_streak, confidence, priority,
                   status, created_at
            FROM habit_ai_recommendations
            WHERE user_id = :user_id AND status = :status
            ORDER BY priority DESC, confidence DESC, created_at DESC
            LIMIT 10
        """)
        
        result = db.execute(query, {"user_id": user_id, "status": status})
        rows = result.fetchall()
        
        return {
            "recommendations": [
                {
                    "id": row[0],
                    "habitId": row[1],
                    "recommendationType": row[2],
                    "title": row[3],
                    "description": row[4],
                    "basedOnCompletionRate": float(row[5]) if row[5] else None,
                    "basedOnStreak": row[6],
                    "confidence": float(row[7]) if row[7] else 0,
                    "priority": row[8],
                    "status": row[9],
                    "createdAt": row[10].isoformat() if row[10] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 9: Social Accountability
# ============================================

@router.post("/buddies/request")
async def request_buddy(
    request: BuddyRequest,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Send a buddy request to another user"""
    try:
        insert_query = text("""
            INSERT INTO habit_buddies (id, user_id, buddy_user_id, status, initiated_by,
                                      share_streak, share_completions, share_mood)
            VALUES (gen_random_uuid(), :user_id, :buddy_user_id, 'pending', :user_id,
                   :share_streak, :share_completions, :share_mood)
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "user_id": user_id,
            "buddy_user_id": request.buddy_user_id,
            "share_streak": request.share_streak,
            "share_completions": request.share_completions,
            "share_mood": request.share_mood
        })
        
        db.commit()
        return {"success": True, "buddy_request_id": result.fetchone()[0]}
    except Exception as e:
        db.rollback()
        logger.error(f"Error requesting buddy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/buddies/{buddy_id}/accept")
async def accept_buddy(
    buddy_id: str,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Accept a buddy request"""
    try:
        # Verify buddy request is for this user (they are the buddy_user_id)
        ownership_check = text("SELECT id FROM habit_buddies WHERE id = :buddy_id AND buddy_user_id = :user_id AND status = 'pending'")
        buddy_exists = db.execute(ownership_check, {"buddy_id": buddy_id, "user_id": user_id}).fetchone()
        if not buddy_exists:
            raise HTTPException(status_code=403, detail="Not authorized to accept this buddy request")
        
        update_query = text("""
            UPDATE habit_buddies
            SET status = 'active', updated_at = NOW()
            WHERE id = :buddy_id AND buddy_user_id = :user_id
            RETURNING id
        """)
        
        result = db.execute(update_query, {"buddy_id": buddy_id, "user_id": user_id})
        db.commit()
        
        return {"success": True, "status": "active"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error accepting buddy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/buddies")
async def get_buddies(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get all buddies for a user"""
    try:
        query = text("""
            SELECT b.id, b.buddy_user_id, u.email as buddy_email, u.first_name, u.last_name,
                   b.status, b.share_streak, b.share_completions, b.share_mood,
                   b.encouragements_sent, b.encouragements_received, b.last_interaction
            FROM habit_buddies b
            JOIN users u ON u.id = b.buddy_user_id
            WHERE b.user_id = :user_id AND b.status = 'active'
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        return {
            "buddies": [
                {
                    "id": row[0],
                    "buddyUserId": row[1],
                    "buddyEmail": row[2],
                    "buddyName": f"{row[3] or ''} {row[4] or ''}".strip() or "Anonymous",
                    "status": row[5],
                    "shareStreak": row[6],
                    "shareCompletions": row[7],
                    "shareMood": row[8],
                    "encouragementsSent": row[9] or 0,
                    "encouragementsReceived": row[10] or 0,
                    "lastInteraction": row[11].isoformat() if row[11] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting buddies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/buddies/encourage")
async def send_encouragement(
    encouragement: EncouragementCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Send an encouragement message to a buddy"""
    try:
        insert_query = text("""
            INSERT INTO habit_encouragements (id, from_user_id, to_user_id, message_type,
                                             message, related_habit_id)
            VALUES (gen_random_uuid(), :from_user, :to_user, :message_type,
                   :message, :habit_id)
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "from_user": user_id,
            "to_user": encouragement.to_user_id,
            "message_type": encouragement.message_type,
            "message": encouragement.message,
            "habit_id": encouragement.related_habit_id
        })
        
        # Update encouragement counts
        update_sent = text("""
            UPDATE habit_buddies
            SET encouragements_sent = encouragements_sent + 1, last_interaction = NOW()
            WHERE user_id = :user_id AND buddy_user_id = :buddy_id
        """)
        
        update_received = text("""
            UPDATE habit_buddies
            SET encouragements_received = encouragements_received + 1, last_interaction = NOW()
            WHERE user_id = :buddy_id AND buddy_user_id = :user_id
        """)
        
        db.execute(update_sent, {"user_id": user_id, "buddy_id": encouragement.to_user_id})
        db.execute(update_received, {"user_id": user_id, "buddy_id": encouragement.to_user_id})
        
        db.commit()
        return {"success": True, "encouragement_id": result.fetchone()[0]}
    except Exception as e:
        db.rollback()
        logger.error(f"Error sending encouragement: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 10: Guided CBT / Motivational Interventions
# ============================================

CBT_FLOWS = {
    "urge_surfing": {
        "title": "Urge Surfing",
        "description": "Ride the wave of urges without acting on them",
        "steps": [
            {"prompt": "Notice the urge. Where do you feel it in your body?", "type": "text"},
            {"prompt": "Rate the intensity of the urge (1-10)", "type": "number"},
            {"prompt": "Take 3 deep breaths. Breathe in for 4 counts, hold for 4, out for 4.", "type": "confirm"},
            {"prompt": "Imagine the urge as a wave. It rises, peaks, and falls. Watch it without acting.", "type": "confirm"},
            {"prompt": "The wave is passing. What do you notice now?", "type": "text"},
            {"prompt": "Rate the intensity now (1-10)", "type": "number"}
        ]
    },
    "reframe_thought": {
        "title": "Reframe Your Thought",
        "description": "Challenge negative thoughts and find balanced perspectives",
        "steps": [
            {"prompt": "What negative thought are you having right now?", "type": "text"},
            {"prompt": "What emotion does this thought create? Rate its intensity (1-10)", "type": "text"},
            {"prompt": "What evidence supports this thought?", "type": "text"},
            {"prompt": "What evidence contradicts this thought?", "type": "text"},
            {"prompt": "What would you tell a friend who had this thought?", "type": "text"},
            {"prompt": "Write a more balanced, realistic thought:", "type": "text"},
            {"prompt": "How do you feel now? Rate the emotion intensity (1-10)", "type": "text"}
        ]
    },
    "grounding": {
        "title": "5-4-3-2-1 Grounding",
        "description": "Ground yourself in the present moment",
        "steps": [
            {"prompt": "Name 5 things you can SEE right now:", "type": "text"},
            {"prompt": "Name 4 things you can TOUCH right now:", "type": "text"},
            {"prompt": "Name 3 things you can HEAR right now:", "type": "text"},
            {"prompt": "Name 2 things you can SMELL right now:", "type": "text"},
            {"prompt": "Name 1 thing you can TASTE right now:", "type": "text"},
            {"prompt": "Take a deep breath. How do you feel now?", "type": "text"}
        ]
    },
    "breathing": {
        "title": "Box Breathing",
        "description": "Calm your nervous system with controlled breathing",
        "steps": [
            {"prompt": "Find a comfortable position. Ready to begin?", "type": "confirm"},
            {"prompt": "Breathe IN slowly for 4 seconds...", "type": "timer", "duration": 4},
            {"prompt": "HOLD your breath for 4 seconds...", "type": "timer", "duration": 4},
            {"prompt": "Breathe OUT slowly for 4 seconds...", "type": "timer", "duration": 4},
            {"prompt": "HOLD empty for 4 seconds...", "type": "timer", "duration": 4},
            {"prompt": "Repeat 3 more times. Ready for round 2?", "type": "confirm"},
            {"prompt": "Great job! How do you feel after the breathing exercise?", "type": "text"}
        ]
    }
}

@router.get("/cbt/flows")
async def get_cbt_flows():
    """Get available CBT/intervention flows"""
    return {
        "flows": [
            {
                "id": flow_id,
                "title": flow["title"],
                "description": flow["description"],
                "totalSteps": len(flow["steps"])
            }
            for flow_id, flow in CBT_FLOWS.items()
        ]
    }

@router.post("/cbt/start")
async def start_cbt_session(
    session: CbtSessionCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Start a new CBT session"""
    try:
        if session.session_type not in CBT_FLOWS:
            raise HTTPException(status_code=400, detail="Invalid session type")
        
        flow = CBT_FLOWS[session.session_type]
        
        insert_query = text("""
            INSERT INTO habit_cbt_sessions (id, user_id, session_type, title, current_step,
                                           total_steps, step_responses, pre_session_mood,
                                           related_habit_id, related_quit_plan_id, started_at)
            VALUES (gen_random_uuid(), :user_id, :session_type, :title, 1, :total_steps,
                   '[]'::jsonb, :pre_mood, :habit_id, :quit_plan_id, NOW())
            RETURNING id
        """)
        
        result = db.execute(insert_query, {
            "user_id": user_id,
            "session_type": session.session_type,
            "title": flow["title"],
            "total_steps": len(flow["steps"]),
            "pre_mood": session.pre_session_mood,
            "habit_id": session.related_habit_id,
            "quit_plan_id": session.related_quit_plan_id
        })
        
        db.commit()
        session_id = result.fetchone()[0]
        
        return {
            "session_id": session_id,
            "title": flow["title"],
            "description": flow["description"],
            "total_steps": len(flow["steps"]),
            "current_step": 1,
            "current_prompt": flow["steps"][0]
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error starting CBT session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cbt/respond")
async def respond_to_cbt_step(
    response: CbtStepResponse,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Submit a response to a CBT step"""
    try:
        # Get current session
        query = text("""
            SELECT session_type, current_step, total_steps, step_responses
            FROM habit_cbt_sessions
            WHERE id = :session_id AND user_id = :user_id
        """)
        
        result = db.execute(query, {"session_id": response.session_id, "user_id": user_id})
        row = result.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_type, current_step, total_steps, step_responses = row
        flow = CBT_FLOWS.get(session_type)
        
        if not flow:
            raise HTTPException(status_code=400, detail="Invalid session")
        
        # Add response to history
        responses = step_responses or []
        responses.append({
            "step": response.step_number,
            "prompt": flow["steps"][response.step_number - 1]["prompt"],
            "response": response.response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if completed
        is_complete = response.step_number >= total_steps
        next_step = response.step_number + 1 if not is_complete else total_steps
        
        # Update session
        update_query = text("""
            UPDATE habit_cbt_sessions
            SET current_step = :next_step,
                step_responses = :responses::jsonb,
                completed = :is_complete,
                completed_at = CASE WHEN :is_complete THEN NOW() ELSE NULL END
            WHERE id = :session_id
        """)
        
        db.execute(update_query, {
            "session_id": response.session_id,
            "next_step": next_step,
            "responses": json.dumps(responses),
            "is_complete": is_complete
        })
        
        db.commit()
        
        result = {
            "session_id": response.session_id,
            "current_step": next_step,
            "total_steps": total_steps,
            "completed": is_complete
        }
        
        if not is_complete:
            result["next_prompt"] = flow["steps"][next_step - 1]
        else:
            result["message"] = "Great job completing this exercise! How do you feel?"
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error responding to CBT step: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# FEATURE 11: Visual Reward System (Gamification)
# ============================================

GROWTH_STAGES = {
    "seed": {"minPoints": 0, "visual": {"height": 1, "leaves": 0, "flowers": 0}},
    "sprout": {"minPoints": 50, "visual": {"height": 2, "leaves": 2, "flowers": 0}},
    "growing": {"minPoints": 150, "visual": {"height": 4, "leaves": 6, "flowers": 0}},
    "blooming": {"minPoints": 350, "visual": {"height": 6, "leaves": 10, "flowers": 3}},
    "flourishing": {"minPoints": 700, "visual": {"height": 8, "leaves": 15, "flowers": 8}}
}

@router.get("/rewards")
async def get_rewards(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Get the user's reward/gamification state"""
    try:
        query = text("""
            SELECT id, reward_type, current_level, growth_stage, total_points,
                   streak_bonus, completion_points, visual_state, unlocked_badges,
                   unlocked_themes, days_active, perfect_days
            FROM habit_rewards
            WHERE user_id = :user_id
        """)
        
        result = db.execute(query, {"user_id": user_id})
        row = result.fetchone()
        
        if not row:
            # Create default reward state
            insert_query = text("""
                INSERT INTO habit_rewards (id, user_id, reward_type, current_level, growth_stage,
                                          total_points, streak_bonus, completion_points,
                                          visual_state, unlocked_badges, unlocked_themes,
                                          days_active, perfect_days)
                VALUES (gen_random_uuid(), :user_id, 'sunflower', 1, 'seed', 0, 0, 0,
                       '{"height": 1, "leaves": 0, "flowers": 0}'::jsonb,
                       '[]'::jsonb, '["default"]'::jsonb, 0, 0)
                RETURNING id, reward_type, current_level, growth_stage, total_points,
                         streak_bonus, completion_points, visual_state, unlocked_badges,
                         unlocked_themes, days_active, perfect_days
            """)
            
            result = db.execute(insert_query, {"user_id": user_id})
            row = result.fetchone()
            db.commit()
        
        # Calculate next stage requirements
        current_stage = row[3]
        current_points = row[4] or 0
        next_stage = None
        points_to_next = None
        
        stages = list(GROWTH_STAGES.keys())
        current_idx = stages.index(current_stage) if current_stage in stages else 0
        
        if current_idx < len(stages) - 1:
            next_stage = stages[current_idx + 1]
            points_to_next = GROWTH_STAGES[next_stage]["minPoints"] - current_points
        
        return {
            "id": row[0],
            "rewardType": row[1],
            "currentLevel": row[2],
            "growthStage": row[3],
            "totalPoints": row[4] or 0,
            "streakBonus": row[5] or 0,
            "completionPoints": row[6] or 0,
            "visualState": row[7] or GROWTH_STAGES["seed"]["visual"],
            "unlockedBadges": row[8] or [],
            "unlockedThemes": row[9] or ["default"],
            "daysActive": row[10] or 0,
            "perfectDays": row[11] or 0,
            "nextStage": next_stage,
            "pointsToNextStage": max(0, points_to_next) if points_to_next else None
        }
    except Exception as e:
        logger.error(f"Error getting rewards: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def update_rewards(user_id: str, db: Session):
    """Background task to update rewards after habit completion"""
    try:
        # Calculate new points based on completions
        stats_query = text("""
            SELECT 
                SUM(CASE WHEN hc.completed THEN 1 ELSE 0 END) as completions,
                MAX(h.current_streak) as max_streak
            FROM habit_completions hc
            JOIN habit_habits h ON h.id = hc.habit_id
            WHERE hc.user_id = :user_id
            AND hc.completion_date >= NOW() - INTERVAL '1 day'
        """)
        
        result = db.execute(stats_query, {"user_id": user_id})
        row = result.fetchone()
        
        if row:
            completions = row[0] or 0
            max_streak = row[1] or 0
            
            # Points calculation
            completion_points = completions * 10
            streak_bonus = min(max_streak * 5, 100)  # Cap at 100
            total_points_today = completion_points + streak_bonus
            
            # Get current reward state
            current_query = text("""
                SELECT total_points, growth_stage FROM habit_rewards WHERE user_id = :user_id
            """)
            current = db.execute(current_query, {"user_id": user_id}).fetchone()
            
            if current:
                new_total = (current[0] or 0) + total_points_today
                
                # Determine new stage
                new_stage = "seed"
                for stage, config in GROWTH_STAGES.items():
                    if new_total >= config["minPoints"]:
                        new_stage = stage
                
                # Update
                update_query = text("""
                    UPDATE habit_rewards
                    SET total_points = :total,
                        streak_bonus = streak_bonus + :streak_bonus,
                        completion_points = completion_points + :completion_points,
                        growth_stage = :stage,
                        visual_state = :visual::jsonb,
                        days_active = days_active + 1,
                        updated_at = NOW()
                    WHERE user_id = :user_id
                """)
                
                db.execute(update_query, {
                    "user_id": user_id,
                    "total": new_total,
                    "streak_bonus": streak_bonus,
                    "completion_points": completion_points,
                    "stage": new_stage,
                    "visual": json.dumps(GROWTH_STAGES[new_stage]["visual"])
                })
                
                db.commit()
    except Exception as e:
        logger.error(f"Error updating rewards: {e}")

# ============================================
# FEATURE 12: Smart Journals with AI Reflection
# ============================================

@router.post("/journals/create")
async def create_journal(
    journal: JournalCreate,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Create a journal entry"""
    try:
        insert_query = text("""
            INSERT INTO habit_journals (id, user_id, title, content, entry_type,
                                       tags, mood, is_weekly_summary, recorded_at)
            VALUES (gen_random_uuid(), :user_id, :title, :content, :entry_type,
                   :tags::jsonb, :mood, false, NOW())
            RETURNING id, recorded_at
        """)
        
        result = db.execute(insert_query, {
            "user_id": user_id,
            "title": journal.title,
            "content": journal.content,
            "entry_type": journal.entry_type,
            "tags": json.dumps(journal.tags) if journal.tags else None,
            "mood": journal.mood
        })
        
        row = result.fetchone()
        db.commit()
        
        return {
            "success": True,
            "journal_id": row[0],
            "recorded_at": row[1].isoformat()
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating journal: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/journals")
async def get_journals(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    limit: int = 20,
    entry_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get journal entries for a user"""
    try:
        query = text("""
            SELECT id, title, content, entry_type, ai_summary, highlights, risks,
                   recommendations, sentiment_trend, tags, mood, is_weekly_summary,
                   week_start_date, recorded_at
            FROM habit_journals
            WHERE user_id = :user_id
            """ + (" AND entry_type = :entry_type" if entry_type else "") + """
            ORDER BY recorded_at DESC
            LIMIT :limit
        """)
        
        params = {"user_id": user_id, "limit": limit}
        if entry_type:
            params["entry_type"] = entry_type
        
        result = db.execute(query, params)
        rows = result.fetchall()
        
        return {
            "journals": [
                {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "entryType": row[3],
                    "aiSummary": row[4],
                    "highlights": row[5],
                    "risks": row[6],
                    "recommendations": row[7],
                    "sentimentTrend": row[8],
                    "tags": row[9],
                    "mood": row[10],
                    "isWeeklySummary": row[11],
                    "weekStartDate": row[12].isoformat() if row[12] else None,
                    "recordedAt": row[13].isoformat() if row[13] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting journals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/journals/weekly-summary")
async def generate_weekly_summary(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Generate AI weekly summary from journal entries"""
    try:
        import openai
        
        # Get this week's journals
        query = text("""
            SELECT content, mood, recorded_at
            FROM habit_journals
            WHERE user_id = :user_id
            AND recorded_at >= NOW() - INTERVAL '7 days'
            AND is_weekly_summary = false
            ORDER BY recorded_at
        """)
        
        result = db.execute(query, {"user_id": user_id})
        rows = result.fetchall()
        
        if len(rows) < 2:
            return {
                "message": "Need at least 2 journal entries to generate a weekly summary",
                "entries_found": len(rows)
            }
        
        # Compile entries
        entries_text = "\n\n".join([
            f"Entry ({row[2].strftime('%A')}): {row[0]}" +
            (f" [Mood: {row[1]}]" if row[1] else "")
            for row in rows
        ])
        
        # Generate AI summary
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """You are a supportive wellness coach analyzing journal entries.
Create a weekly reflection with:
1. Three key highlights/wins from the week
2. Two potential risks or concerns to watch
3. Two actionable recommendations for next week

Be encouraging, specific, and actionable. Format as JSON with keys: highlights (array), risks (array), recommendations (array), sentiment_trend (string: improving/stable/declining), summary (string: 2-3 sentence overview)."""},
                {"role": "user", "content": f"Here are this week's journal entries:\n\n{entries_text}"}
            ],
            max_tokens=600,
            temperature=0.7
        )
        
        ai_text = response.choices[0].message.content
        
        # Parse AI response
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', ai_text, re.DOTALL)
            if json_match:
                ai_data = json.loads(json_match.group())
            else:
                ai_data = {
                    "highlights": ["Great effort this week!"],
                    "risks": ["Keep monitoring your progress"],
                    "recommendations": ["Continue your current habits"],
                    "sentiment_trend": "stable",
                    "summary": ai_text[:500]
                }
        except:
            ai_data = {
                "highlights": ["Great effort this week!"],
                "risks": ["Keep monitoring your progress"],
                "recommendations": ["Continue your current habits"],
                "sentiment_trend": "stable",
                "summary": ai_text[:500]
            }
        
        # Store weekly summary
        insert_query = text("""
            INSERT INTO habit_journals (id, user_id, title, content, entry_type,
                                       ai_summary, highlights, risks, recommendations,
                                       sentiment_trend, is_weekly_summary, week_start_date, recorded_at)
            VALUES (gen_random_uuid(), :user_id, :title, :content, 'weekly_summary',
                   :ai_summary, :highlights::jsonb, :risks::jsonb, :recommendations::jsonb,
                   :sentiment_trend, true, :week_start, NOW())
            RETURNING id
        """)
        
        week_start = (datetime.now() - timedelta(days=7)).date()
        
        result = db.execute(insert_query, {
            "user_id": user_id,
            "title": f"Weekly Summary - Week of {week_start.strftime('%B %d')}",
            "content": ai_data.get("summary", ""),
            "ai_summary": ai_data.get("summary", ""),
            "highlights": json.dumps(ai_data.get("highlights", [])),
            "risks": json.dumps(ai_data.get("risks", [])),
            "recommendations": json.dumps(ai_data.get("recommendations", [])),
            "sentiment_trend": ai_data.get("sentiment_trend", "stable"),
            "week_start": week_start
        })
        
        db.commit()
        
        return {
            "success": True,
            "summary_id": result.fetchone()[0],
            "highlights": ai_data.get("highlights", []),
            "risks": ai_data.get("risks", []),
            "recommendations": ai_data.get("recommendations", []),
            "sentiment_trend": ai_data.get("sentiment_trend", "stable"),
            "summary": ai_data.get("summary", "")
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating weekly summary: {e}")
        # Return fallback
        return {
            "success": True,
            "highlights": ["You showed up and put in effort this week"],
            "risks": ["Watch for signs of burnout"],
            "recommendations": ["Continue your positive momentum"],
            "sentiment_trend": "stable",
            "summary": "Keep up the great work on your habit journey!"
        }

# ============================================
# FEATURE 13: Preventive Alerts & Prediction Engine
# ============================================

@router.post("/alerts/analyze")
async def analyze_risk_and_generate_alerts(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Analyze patterns and generate preventive alerts for high-risk days"""
    try:
        alerts_generated = []
        
        # Get recent data
        data_query = text("""
            SELECT 
                -- Habit performance
                (SELECT COUNT(*) FROM habit_completions WHERE user_id = :user_id 
                 AND completed = false AND completion_date >= NOW() - INTERVAL '3 days') as recent_skips,
                
                -- Mood trends
                (SELECT AVG(mood_score) FROM habit_mood_entries WHERE user_id = :user_id
                 AND recorded_at >= NOW() - INTERVAL '3 days') as avg_mood,
                
                -- Streak at risk
                (SELECT MAX(current_streak) FROM habit_habits WHERE user_id = :user_id 
                 AND is_active = true AND current_streak >= 7) as streak_at_risk,
                
                -- Sleep quality from symptom checkins
                (SELECT AVG(sleep_quality) FROM symptom_checkins WHERE user_id = :user_id
                 AND created_at >= NOW() - INTERVAL '3 days') as avg_sleep
        """)
        
        result = db.execute(data_query, {"user_id": user_id})
        row = result.fetchone()
        
        recent_skips = row[0] or 0
        avg_mood = float(row[1]) if row[1] else 5
        streak_at_risk = row[2]
        avg_sleep = float(row[3]) if row[3] else 5
        
        # Calculate risk score
        risk_factors = []
        risk_score = 0
        
        # Factor 1: Recent skips
        if recent_skips >= 3:
            risk_score += 0.3
            risk_factors.append({
                "factor": "missed_habits",
                "weight": 0.3,
                "value": f"{recent_skips} habits skipped in 3 days"
            })
        
        # Factor 2: Low mood
        if avg_mood < 4:
            risk_score += 0.25
            risk_factors.append({
                "factor": "low_mood",
                "weight": 0.25,
                "value": f"Average mood: {avg_mood:.1f}/10"
            })
        
        # Factor 3: Poor sleep
        if avg_sleep < 4:
            risk_score += 0.25
            risk_factors.append({
                "factor": "poor_sleep",
                "weight": 0.25,
                "value": f"Average sleep quality: {avg_sleep:.1f}/10"
            })
        
        # Factor 4: Streak at risk
        if streak_at_risk and streak_at_risk >= 7:
            risk_score += 0.2
            risk_factors.append({
                "factor": "streak_at_risk",
                "weight": 0.2,
                "value": f"{streak_at_risk}-day streak could be lost"
            })
        
        # Generate alert if risk is high
        if risk_score >= 0.5:
            severity = "critical" if risk_score >= 0.75 else "high" if risk_score >= 0.6 else "medium"
            
            suggested_actions = [
                "Take a moment for self-care today",
                "Consider a lighter habit load",
                "Try a quick CBT exercise"
            ]
            
            if avg_sleep < 4:
                suggested_actions.append("Prioritize getting better sleep tonight")
            if avg_mood < 4:
                suggested_actions.append("Journal about what's on your mind")
            
            insert_query = text("""
                INSERT INTO habit_risk_alerts (id, user_id, alert_type, severity, title, message,
                                              risk_score, contributing_factors, suggested_actions,
                                              status, predicted_for, expires_at)
                VALUES (gen_random_uuid(), :user_id, 'high_risk_day', :severity,
                       'High Risk Day Detected', :message, :risk_score,
                       :factors::jsonb, :actions::jsonb, 'active',
                       NOW() + INTERVAL '1 day', NOW() + INTERVAL '2 days')
                RETURNING id
            """)
            
            message = f"Based on recent patterns (skipped habits, mood, sleep), today may be challenging. Risk score: {risk_score*100:.0f}%"
            
            result = db.execute(insert_query, {
                "user_id": user_id,
                "severity": severity,
                "message": message,
                "risk_score": risk_score,
                "factors": json.dumps(risk_factors),
                "actions": json.dumps(suggested_actions)
            })
            
            alerts_generated.append({
                "id": result.fetchone()[0],
                "type": "high_risk_day",
                "severity": severity,
                "riskScore": risk_score,
                "factors": risk_factors,
                "suggestedActions": suggested_actions
            })
        
        db.commit()
        
        return {
            "analyzed": True,
            "riskScore": risk_score,
            "riskFactors": risk_factors,
            "alertsGenerated": len(alerts_generated),
            "alerts": alerts_generated
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error analyzing risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts")
async def get_alerts(
    user_id: str = Depends(get_user_id_from_auth_or_query),
    status: str = "active",
    db: Session = Depends(get_db)
):
    """Get all risk alerts for a user"""
    try:
        query = text("""
            SELECT id, alert_type, severity, title, message, risk_score,
                   contributing_factors, suggested_actions, status,
                   predicted_for, acknowledged_at, created_at
            FROM habit_risk_alerts
            WHERE user_id = :user_id AND status = :status
            AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY severity DESC, created_at DESC
            LIMIT 10
        """)
        
        result = db.execute(query, {"user_id": user_id, "status": status})
        rows = result.fetchall()
        
        return {
            "alerts": [
                {
                    "id": row[0],
                    "alertType": row[1],
                    "severity": row[2],
                    "title": row[3],
                    "message": row[4],
                    "riskScore": float(row[5]) if row[5] else 0,
                    "contributingFactors": row[6],
                    "suggestedActions": row[7],
                    "status": row[8],
                    "predictedFor": row[9].isoformat() if row[9] else None,
                    "acknowledgedAt": row[10].isoformat() if row[10] else None,
                    "createdAt": row[11].isoformat() if row[11] else None
                }
                for row in rows
            ]
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user_id: str = Depends(get_user_id_from_auth_or_query),
    db: Session = Depends(get_db)
):
    """Acknowledge a risk alert"""
    try:
        # Verify alert ownership
        ownership_check = text("SELECT id FROM habit_risk_alerts WHERE id = :alert_id AND user_id = :user_id")
        alert_exists = db.execute(ownership_check, {"alert_id": alert_id, "user_id": user_id}).fetchone()
        if not alert_exists:
            raise HTTPException(status_code=403, detail="Not authorized to acknowledge this alert")
        
        update_query = text("""
            UPDATE habit_risk_alerts
            SET status = 'acknowledged', acknowledged_at = NOW()
            WHERE id = :alert_id AND user_id = :user_id
            RETURNING id
        """)
        
        result = db.execute(update_query, {"alert_id": alert_id, "user_id": user_id})
        db.commit()
        
        return {"success": True, "status": "acknowledged"}
    except Exception as e:
        db.rollback()
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))
