"""
Habit Coaching Service - Production-Grade AI Coaching
======================================================

Provides personalized AI coaching for habit formation using OpenAI GPT-4o.
Features:
- Context-aware coaching with full user history
- Multiple coaching personalities
- Session continuity and memory
- Proactive intervention triggers
- HIPAA-compliant audit logging
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import OpenAI

from app.services.access_control import HIPAAAuditLogger, AccessScope, PHICategory

logger = logging.getLogger(__name__)


class CoachingPersonality(str, Enum):
    SUPPORTIVE = "supportive"
    MOTIVATIONAL = "motivational"
    ANALYTICAL = "analytical"
    TOUGH_LOVE = "tough_love"
    MINDFUL = "mindful"


PERSONALITY_PROMPTS = {
    CoachingPersonality.SUPPORTIVE: """You are a warm, empathetic coach who prioritizes emotional support. 
You validate feelings, celebrate small wins, and gently guide users through challenges. 
Use phrases like "I understand", "It's okay to struggle", "You're making progress".""",
    
    CoachingPersonality.MOTIVATIONAL: """You are an energetic, inspiring coach focused on momentum and action.
You use power phrases, celebrate victories enthusiastically, and reframe setbacks as learning.
Use phrases like "You've got this!", "Every day is a new opportunity", "Push through!".""",
    
    CoachingPersonality.ANALYTICAL: """You are a data-driven coach who uses patterns and evidence.
You analyze trends, provide specific statistics, and offer actionable insights based on data.
Use phrases like "Based on your patterns", "The data shows", "Optimizing your approach".""",
    
    CoachingPersonality.TOUGH_LOVE: """You are a direct, no-nonsense coach who holds users accountable.
You're honest about challenges, set high standards, and push users to their potential.
Use phrases like "Let's be real", "You can do better", "No excuses - let's go".""",
    
    CoachingPersonality.MINDFUL: """You are a calm, centered coach focused on mindfulness and self-compassion.
You emphasize process over outcomes, present-moment awareness, and self-kindness.
Use phrases like "Be present", "Notice without judgment", "Self-compassion matters"."""
}


class HabitCoachingService:
    """Production-grade AI coaching service for habit formation"""
    
    def __init__(self, db: Session):
        self.db = db
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "gpt-4o"
        self.max_tokens = 600
        self.temperature = 0.7
    
    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Aggregate comprehensive user context for personalized coaching"""
        context = {
            "habits": [],
            "streaks": {},
            "mood_history": [],
            "recent_completions": [],
            "quit_plans": [],
            "triggers": [],
            "coaching_history": []
        }
        
        try:
            habits_query = text("""
                SELECT id, name, category, current_streak, longest_streak, 
                       total_completions, goal_count, frequency
                FROM habit_habits
                WHERE user_id = :user_id AND is_active = true
                ORDER BY current_streak DESC
                LIMIT 10
            """)
            habits = self.db.execute(habits_query, {"user_id": user_id}).fetchall()
            
            for h in habits:
                context["habits"].append({
                    "id": h[0], "name": h[1], "category": h[2],
                    "current_streak": h[3] or 0, "longest_streak": h[4] or 0,
                    "total_completions": h[5] or 0, "goal_count": h[6], "frequency": h[7]
                })
                context["streaks"][h[1]] = h[3] or 0
            
            mood_query = text("""
                SELECT mood_score, mood_label, energy_level, stress_level, recorded_at
                FROM habit_mood_entries
                WHERE user_id = :user_id
                ORDER BY recorded_at DESC
                LIMIT 7
            """)
            moods = self.db.execute(mood_query, {"user_id": user_id}).fetchall()
            
            for m in moods:
                context["mood_history"].append({
                    "score": m[0], "label": m[1], "energy": m[2], 
                    "stress": m[3], "date": m[4].isoformat() if m[4] else None
                })
            
            completions_query = text("""
                SELECT h.name, hc.completed, hc.mood, hc.completion_date
                FROM habit_completions hc
                JOIN habit_habits h ON h.id = hc.habit_id
                WHERE hc.user_id = :user_id
                ORDER BY hc.completion_date DESC
                LIMIT 14
            """)
            completions = self.db.execute(completions_query, {"user_id": user_id}).fetchall()
            
            for c in completions:
                context["recent_completions"].append({
                    "habit": c[0], "completed": c[1], "mood": c[2],
                    "date": c[3].isoformat() if c[3] else None
                })
            
            quit_query = text("""
                SELECT habit_name, quit_method, days_clean, total_relapses, status
                FROM habit_quit_plans
                WHERE user_id = :user_id AND status = 'active'
                LIMIT 3
            """)
            quits = self.db.execute(quit_query, {"user_id": user_id}).fetchall()
            
            for q in quits:
                context["quit_plans"].append({
                    "habit": q[0], "method": q[1], "days_clean": q[2] or 0,
                    "relapses": q[3] or 0, "status": q[4]
                })
            
            triggers_query = text("""
                SELECT trigger_type, pattern, confidence
                FROM habit_ai_triggers
                WHERE user_id = :user_id AND is_active = true
                ORDER BY confidence DESC
                LIMIT 5
            """)
            triggers = self.db.execute(triggers_query, {"user_id": user_id}).fetchall()
            
            for t in triggers:
                context["triggers"].append({
                    "type": t[0], "pattern": t[1], "confidence": float(t[2]) if t[2] else 0
                })
            
            history_query = text("""
                SELECT role, content, response_type, created_at
                FROM habit_coach_chats
                WHERE user_id = :user_id
                ORDER BY created_at DESC
                LIMIT 10
            """)
            history = self.db.execute(history_query, {"user_id": user_id}).fetchall()
            
            for h in reversed(list(history)):
                context["coaching_history"].append({
                    "role": h[0], "content": h[1][:200], "type": h[2],
                    "date": h[3].isoformat() if h[3] else None
                })
                
        except Exception as e:
            logger.error(f"Error fetching user context: {e}")
        
        return context
    
    def _build_system_prompt(
        self, 
        context: Dict[str, Any], 
        personality: CoachingPersonality
    ) -> str:
        """Build comprehensive system prompt with user context"""
        
        habits_summary = ""
        if context["habits"]:
            habits_summary = "Current habits:\n" + "\n".join([
                f"- {h['name']} ({h['category']}): {h['current_streak']}-day streak, "
                f"{h['total_completions']} total completions"
                for h in context["habits"][:5]
            ])
        else:
            habits_summary = "No active habits yet."
        
        mood_summary = ""
        if context["mood_history"]:
            avg_mood = sum(m["score"] for m in context["mood_history"]) / len(context["mood_history"])
            avg_stress = sum(m["stress"] or 5 for m in context["mood_history"]) / len(context["mood_history"])
            mood_summary = f"Recent mood: avg {avg_mood:.1f}/10, stress {avg_stress:.1f}/10"
        
        quit_summary = ""
        if context["quit_plans"]:
            quit_summary = "Active quit plans:\n" + "\n".join([
                f"- Quitting {q['habit']}: {q['days_clean']} days clean"
                for q in context["quit_plans"]
            ])
        
        triggers_summary = ""
        if context["triggers"]:
            triggers_summary = "Known triggers:\n" + "\n".join([
                f"- {t['pattern']} (confidence: {t['confidence']:.0%})"
                for t in context["triggers"][:3]
            ])
        
        personality_prompt = PERSONALITY_PROMPTS.get(
            personality, PERSONALITY_PROMPTS[CoachingPersonality.SUPPORTIVE]
        )
        
        return f"""You are an expert AI habit coach helping users build healthy habits and break unhealthy ones.
You use evidence-based techniques: CBT (cognitive behavioral therapy), motivational interviewing, 
behavioral science, and habit stacking.

{personality_prompt}

USER CONTEXT:
{habits_summary}

{mood_summary}

{quit_summary}

{triggers_summary}

COACHING GUIDELINES:
1. Be concise (2-3 paragraphs max)
2. Offer specific, actionable advice
3. Reference their actual habits and progress
4. Celebrate wins, normalize setbacks
5. Use CBT techniques when appropriate (thought reframing, urge surfing)
6. If user seems distressed, recommend professional help while being supportive
7. Never make medical diagnoses or give medical advice

SAFETY: If user mentions self-harm, suicide, or severe mental health crisis, 
provide crisis resources (988 Suicide & Crisis Lifeline) and strongly encourage professional help."""
    
    def get_or_create_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Get existing session or create new one"""
        if session_id:
            check_query = text("""
                SELECT session_id FROM habit_coach_chats 
                WHERE user_id = :user_id AND session_id = :session_id
                LIMIT 1
            """)
            exists = self.db.execute(check_query, {
                "user_id": user_id, "session_id": session_id
            }).fetchone()
            
            if exists:
                return session_id
        
        return f"coach_{user_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    
    def get_session_messages(self, user_id: str, session_id: str, limit: int = 20) -> List[Dict]:
        """Get conversation history for session continuity"""
        query = text("""
            SELECT role, content
            FROM habit_coach_chats
            WHERE user_id = :user_id AND session_id = :session_id
            ORDER BY created_at ASC
            LIMIT :limit
        """)
        
        rows = self.db.execute(query, {
            "user_id": user_id, "session_id": session_id, "limit": limit
        }).fetchall()
        
        return [{"role": row[0], "content": row[1]} for row in rows]
    
    def chat(
        self,
        user_id: str,
        message: str,
        personality: CoachingPersonality = CoachingPersonality.SUPPORTIVE,
        session_id: Optional[str] = None,
        related_habit_id: Optional[str] = None,
        related_quit_plan_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send message to AI coach and get response"""
        
        session_id = self.get_or_create_session(user_id, session_id)
        context = self.get_user_context(user_id)
        system_prompt = self._build_system_prompt(context, personality)
        
        messages = [{"role": "system", "content": system_prompt}]
        
        session_history = self.get_session_messages(user_id, session_id, limit=10)
        messages.extend(session_history)
        
        messages.append({"role": "user", "content": message})
        
        self._save_message(
            user_id=user_id,
            session_id=session_id,
            role="user",
            content=message,
            personality=personality.value,
            related_habit_id=related_habit_id,
            related_quit_plan_id=related_quit_plan_id
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            ai_response = response.choices[0].message.content
            response_type = self._classify_response(message, ai_response)
            
            self._save_message(
                user_id=user_id,
                session_id=session_id,
                role="assistant",
                content=ai_response,
                personality=personality.value,
                response_type=response_type,
                related_habit_id=related_habit_id,
                related_quit_plan_id=related_quit_plan_id
            )
            
            self.db.commit()
            
            HIPAAAuditLogger.log_access(
                user_id=user_id,
                user_role="patient",
                action="habit_coach_chat",
                resource_type="HabitCoachChat",
                resource_id=session_id,
                access_reason="ai_coaching_session",
                was_successful=True
            )
            
            return {
                "response": ai_response,
                "session_id": session_id,
                "response_type": response_type,
                "personality": personality.value,
                "context_summary": {
                    "active_habits": len(context["habits"]),
                    "total_streak_days": sum(h["current_streak"] for h in context["habits"]),
                    "active_quit_plans": len(context["quit_plans"])
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI coaching error: {e}")
            self.db.rollback()
            
            fallback = self._get_fallback_response(message, context)
            return {
                "response": fallback,
                "session_id": session_id,
                "response_type": "fallback",
                "personality": personality.value,
                "error": True
            }
    
    def _save_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        personality: str,
        response_type: Optional[str] = None,
        related_habit_id: Optional[str] = None,
        related_quit_plan_id: Optional[str] = None
    ):
        """Save chat message to database"""
        query = text("""
            INSERT INTO habit_coach_chats 
            (id, user_id, session_id, role, content, coach_personality, 
             response_type, related_habit_id, related_quit_plan_id)
            VALUES (gen_random_uuid(), :user_id, :session_id, :role, :content, 
                   :personality, :response_type, :habit_id, :quit_plan_id)
        """)
        
        self.db.execute(query, {
            "user_id": user_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "personality": personality,
            "response_type": response_type,
            "habit_id": related_habit_id,
            "quit_plan_id": related_quit_plan_id
        })
    
    def _classify_response(self, user_message: str, ai_response: str) -> str:
        """Classify the type of coaching response"""
        user_lower = user_message.lower()
        response_lower = ai_response.lower()
        
        if any(w in user_lower for w in ["cbt", "thought", "reframe", "cognitive"]):
            return "cbt_technique"
        elif any(w in user_lower for w in ["craving", "urge", "relapse", "slip"]):
            return "craving_support"
        elif any(w in user_lower for w in ["tip", "advice", "how to", "suggest"]):
            return "actionable_tip"
        elif any(w in user_lower for w in ["stuck", "failing", "can't", "hard"]):
            return "motivation"
        elif any(w in response_lower for w in ["congratulations", "amazing", "proud"]):
            return "celebration"
        else:
            return "general_coaching"
    
    def _get_fallback_response(self, message: str, context: Dict) -> str:
        """Generate fallback response when OpenAI fails"""
        
        if context["habits"]:
            best_streak = max(context["habits"], key=lambda h: h["current_streak"])
            if best_streak["current_streak"] > 0:
                return f"I'm here to support your habit journey! I noticed you have a {best_streak['current_streak']}-day streak on '{best_streak['name']}' - that's excellent progress. What specific challenge can I help you with today?"
        
        return "I'm here to support you on your habit journey! Remember, every small step counts. Building lasting habits takes time and patience. What specific challenge are you facing today?"
    
    def get_proactive_coaching(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Generate proactive coaching intervention based on patterns"""
        
        context = self.get_user_context(user_id)
        interventions = []
        
        for habit in context["habits"]:
            if habit["current_streak"] == 0 and habit["longest_streak"] > 7:
                interventions.append({
                    "type": "streak_recovery",
                    "priority": "high",
                    "habit": habit["name"],
                    "message": f"I noticed your '{habit['name']}' streak ended. You had {habit['longest_streak']} days before - you can absolutely get back on track! Want to talk about what happened?"
                })
            
            if habit["current_streak"] in [7, 14, 21, 30, 60, 90, 100]:
                interventions.append({
                    "type": "milestone",
                    "priority": "medium",
                    "habit": habit["name"],
                    "message": f"Congratulations! You've reached {habit['current_streak']} days on '{habit['name']}'! This is a significant milestone. How are you feeling about your progress?"
                })
        
        if context["mood_history"]:
            recent_moods = [m["score"] for m in context["mood_history"][:3]]
            if all(m <= 4 for m in recent_moods):
                interventions.append({
                    "type": "mood_support",
                    "priority": "high",
                    "message": "I've noticed your mood has been low lately. Remember, it's okay to have tough days. Would you like to talk about what's been challenging?"
                })
        
        for qp in context["quit_plans"]:
            if qp["days_clean"] in [1, 3, 7, 14, 30]:
                interventions.append({
                    "type": "quit_milestone",
                    "priority": "high",
                    "habit": qp["habit"],
                    "message": f"Amazing! {qp['days_clean']} days clean from {qp['habit']}! Each day is a victory. How are you handling cravings?"
                })
        
        if interventions:
            interventions.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x["priority"]])
            return interventions[0]
        
        return None
