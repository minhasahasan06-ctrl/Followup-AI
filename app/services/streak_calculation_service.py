"""
Streak Calculation Service - Production-Grade Streak Management
================================================================

Complete streak management system with:
- Daily streak validation and recalculation
- Streak freeze tokens for protection
- Streak recovery within grace period
- Milestone detection and celebration
- APScheduler job for nightly validation
- HIPAA-compliant audit logging
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.access_control import HIPAAAuditLogger, AccessScope, PHICategory

logger = logging.getLogger(__name__)


STREAK_MILESTONES = [3, 7, 14, 21, 30, 45, 60, 75, 90, 100, 150, 200, 300, 365]


class StreakCalculationService:
    """Production-grade streak management for habit tracking"""
    
    def __init__(self, db: Session):
        self.db = db
        self.grace_period_hours = 24
        self.max_freeze_tokens = 3
    
    def calculate_streak(self, user_id: str, habit_id: str) -> Dict[str, Any]:
        """Calculate current streak for a habit based on completion history"""
        
        query = text("""
            SELECT DATE(completion_date) as comp_date, completed
            FROM habit_completions
            WHERE habit_id = :habit_id AND user_id = :user_id
            ORDER BY completion_date DESC
            LIMIT 365
        """)
        
        completions = self.db.execute(query, {
            "habit_id": habit_id, "user_id": user_id
        }).fetchall()
        
        if not completions:
            return {"current_streak": 0, "longest_streak": 0, "streak_days": []}
        
        completed_dates = set()
        for row in completions:
            if row[1]:
                completed_dates.add(row[0])
        
        streak = 0
        streak_days = []
        check_date = date.today()
        
        while check_date in completed_dates:
            streak += 1
            streak_days.append(check_date.isoformat())
            check_date -= timedelta(days=1)
        
        if streak == 0:
            yesterday = date.today() - timedelta(days=1)
            if yesterday in completed_dates:
                check_date = yesterday
                while check_date in completed_dates:
                    streak += 1
                    streak_days.append(check_date.isoformat())
                    check_date -= timedelta(days=1)
        
        longest_streak = self._calculate_longest_streak(completed_dates)
        
        return {
            "current_streak": streak,
            "longest_streak": longest_streak,
            "streak_days": streak_days[:30],
            "last_completion": max(completed_dates).isoformat() if completed_dates else None
        }
    
    def _calculate_longest_streak(self, completed_dates: set) -> int:
        """Calculate longest streak from set of completion dates"""
        if not completed_dates:
            return 0
        
        sorted_dates = sorted(completed_dates)
        longest = 1
        current = 1
        
        for i in range(1, len(sorted_dates)):
            if sorted_dates[i] - sorted_dates[i-1] == timedelta(days=1):
                current += 1
                longest = max(longest, current)
            else:
                current = 1
        
        return longest
    
    def update_habit_streaks(self, habit_id: str, user_id: str) -> Dict[str, Any]:
        """Recalculate and update streak for a habit"""
        
        streak_data = self.calculate_streak(user_id, habit_id)
        
        update_query = text("""
            UPDATE habit_habits
            SET current_streak = :current,
                longest_streak = GREATEST(longest_streak, :longest),
                updated_at = NOW()
            WHERE id = :habit_id AND user_id = :user_id
            RETURNING name, current_streak, longest_streak
        """)
        
        result = self.db.execute(update_query, {
            "current": streak_data["current_streak"],
            "longest": streak_data["longest_streak"],
            "habit_id": habit_id,
            "user_id": user_id
        })
        
        row = result.fetchone()
        self.db.commit()
        
        milestone = self._check_milestone(streak_data["current_streak"])
        
        return {
            "habitId": habit_id,
            "habitName": row[0] if row else None,
            "currentStreak": streak_data["current_streak"],
            "longestStreak": row[2] if row else streak_data["longest_streak"],
            "streakDays": streak_data["streak_days"],
            "milestone": milestone
        }
    
    def _check_milestone(self, streak: int) -> Optional[Dict[str, Any]]:
        """Check if streak has reached a milestone"""
        if streak in STREAK_MILESTONES:
            milestone_index = STREAK_MILESTONES.index(streak)
            next_milestone = STREAK_MILESTONES[milestone_index + 1] if milestone_index < len(STREAK_MILESTONES) - 1 else None
            
            return {
                "days": streak,
                "message": self._get_milestone_message(streak),
                "nextMilestone": next_milestone,
                "daysToNext": next_milestone - streak if next_milestone else None
            }
        return None
    
    def _get_milestone_message(self, days: int) -> str:
        """Get celebration message for milestone"""
        messages = {
            3: "3 days strong! You're building momentum!",
            7: "One week streak! You're forming a habit!",
            14: "Two weeks! This is becoming part of who you are!",
            21: "21 days - the habit is taking root!",
            30: "ONE MONTH! You've proven your commitment!",
            45: "45 days of dedication. Incredible!",
            60: "Two months! You're unstoppable!",
            75: "75 days - consistency is your superpower!",
            90: "90 DAYS! This habit is now part of your identity!",
            100: "TRIPLE DIGITS! You are a legend!",
            150: "150 days of excellence. Truly inspiring!",
            200: "200 days! You've mastered this habit!",
            300: "300 days - almost a year of dedication!",
            365: "ONE FULL YEAR! You've transformed your life!"
        }
        return messages.get(days, f"Amazing! {days} days of consistency!")
    
    def get_freeze_tokens(self, user_id: str) -> Dict[str, Any]:
        """Get user's streak freeze token status"""
        
        query = text("""
            SELECT 
                COALESCE(
                    (SELECT COUNT(*) FROM habit_streak_freezes 
                     WHERE user_id = :user_id AND used_at IS NULL AND expires_at > NOW()),
                    0
                ) as available,
                COALESCE(
                    (SELECT COUNT(*) FROM habit_streak_freezes 
                     WHERE user_id = :user_id AND used_at IS NOT NULL),
                    0
                ) as used
        """)
        
        try:
            result = self.db.execute(query, {"user_id": user_id}).fetchone()
            available = result[0] if result else 0
            used = result[1] if result else 0
        except Exception as e:
            logger.warning(f"Error fetching freeze tokens: {e}")
            self.db.rollback()
            available = 0
            used = 0
        
        return {
            "available": available,
            "used": used,
            "maxTokens": self.max_freeze_tokens,
            "canEarnMore": available < self.max_freeze_tokens
        }
    
    def use_freeze_token(self, user_id: str, habit_id: str) -> Dict[str, Any]:
        """Use a freeze token to protect streak for one day"""
        
        tokens = self.get_freeze_tokens(user_id)
        
        if tokens["available"] <= 0:
            return {
                "success": False,
                "error": "No freeze tokens available",
                "tokensRemaining": 0
            }
        
        try:
            use_query = text("""
                UPDATE habit_streak_freezes
                SET used_at = NOW(),
                    used_for_habit_id = :habit_id
                WHERE id = (
                    SELECT id FROM habit_streak_freezes
                    WHERE user_id = :user_id AND used_at IS NULL AND expires_at > NOW()
                    ORDER BY expires_at ASC
                    LIMIT 1
                )
                RETURNING id
            """)
            
            result = self.db.execute(use_query, {
                "user_id": user_id, "habit_id": habit_id
            })
            
            used_token = result.fetchone()
            
            if used_token:
                self.db.commit()
                
                HIPAAAuditLogger.log_access(
                    user_id=user_id,
                    user_role="patient",
                    action="streak_freeze_used",
                    resource_type="HabitStreakFreeze",
                    resource_id=used_token[0],
                    access_reason="streak_protection",
                    was_successful=True
                )
                
                return {
                    "success": True,
                    "message": "Streak freeze applied! Your streak is protected for today.",
                    "tokensRemaining": tokens["available"] - 1
                }
            
            return {"success": False, "error": "Failed to use freeze token"}
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error using freeze token: {e}")
            return {"success": False, "error": str(e)}
    
    def award_freeze_token(self, user_id: str, reason: str = "milestone") -> Dict[str, Any]:
        """Award a freeze token to user"""
        
        tokens = self.get_freeze_tokens(user_id)
        
        if tokens["available"] >= self.max_freeze_tokens:
            return {
                "success": False,
                "message": "Maximum freeze tokens reached",
                "tokensAvailable": tokens["available"]
            }
        
        try:
            insert_query = text("""
                INSERT INTO habit_streak_freezes 
                (id, user_id, earned_reason, expires_at)
                VALUES (gen_random_uuid(), :user_id, :reason, NOW() + INTERVAL '90 days')
                RETURNING id
            """)
            
            result = self.db.execute(insert_query, {
                "user_id": user_id, "reason": reason
            })
            
            token_id = result.fetchone()[0]
            self.db.commit()
            
            return {
                "success": True,
                "tokenId": token_id,
                "tokensAvailable": tokens["available"] + 1,
                "expiresIn": "90 days"
            }
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error awarding freeze token: {e}")
            return {"success": False, "error": str(e)}
    
    def can_recover_streak(self, user_id: str, habit_id: str) -> Dict[str, Any]:
        """Check if user can recover a broken streak within grace period"""
        
        query = text("""
            SELECT current_streak, longest_streak,
                   (SELECT MAX(completion_date) FROM habit_completions 
                    WHERE habit_id = :habit_id AND user_id = :user_id AND completed = true) as last_completion
            FROM habit_habits
            WHERE id = :habit_id AND user_id = :user_id
        """)
        
        result = self.db.execute(query, {
            "habit_id": habit_id, "user_id": user_id
        }).fetchone()
        
        if not result or not result[2]:
            return {"canRecover": False, "reason": "No completion history"}
        
        last_completion = result[2]
        hours_since = (datetime.utcnow() - last_completion).total_seconds() / 3600
        
        if hours_since <= self.grace_period_hours + 24:
            return {
                "canRecover": True,
                "previousStreak": result[0] or 0,
                "hoursSinceLastCompletion": round(hours_since, 1),
                "graceHoursRemaining": max(0, self.grace_period_hours + 24 - hours_since)
            }
        
        return {
            "canRecover": False,
            "reason": "Grace period expired",
            "hoursSinceLastCompletion": round(hours_since, 1)
        }
    
    def recover_streak(self, user_id: str, habit_id: str) -> Dict[str, Any]:
        """Attempt to recover a broken streak within grace period"""
        
        recovery_check = self.can_recover_streak(user_id, habit_id)
        
        if not recovery_check["canRecover"]:
            return {
                "success": False,
                "error": recovery_check.get("reason", "Cannot recover streak")
            }
        
        tokens = self.get_freeze_tokens(user_id)
        if tokens["available"] <= 0:
            return {
                "success": False,
                "error": "Streak recovery requires a freeze token"
            }
        
        use_result = self.use_freeze_token(user_id, habit_id)
        if not use_result["success"]:
            return use_result
        
        streak_data = self.calculate_streak(user_id, habit_id)
        
        update_query = text("""
            UPDATE habit_habits
            SET current_streak = :streak,
                updated_at = NOW()
            WHERE id = :habit_id AND user_id = :user_id
            RETURNING name, current_streak
        """)
        
        result = self.db.execute(update_query, {
            "streak": recovery_check["previousStreak"],
            "habit_id": habit_id,
            "user_id": user_id
        })
        
        row = result.fetchone()
        self.db.commit()
        
        logger.info(f"User {user_id} recovered streak on habit {habit_id}")
        
        return {
            "success": True,
            "message": f"Streak recovered! You're back to {recovery_check['previousStreak']} days!",
            "recoveredStreak": recovery_check["previousStreak"],
            "habitName": row[0] if row else None,
            "freezeTokensRemaining": use_result["tokensRemaining"]
        }
    
    def validate_all_user_streaks(self, user_id: str) -> Dict[str, Any]:
        """Validate and recalculate all streaks for a user"""
        
        habits_query = text("""
            SELECT id, name, current_streak, frequency
            FROM habit_habits
            WHERE user_id = :user_id AND is_active = true
        """)
        
        habits = self.db.execute(habits_query, {"user_id": user_id}).fetchall()
        
        results = {
            "validated": 0,
            "streaksUpdated": 0,
            "streaksBroken": 0,
            "details": []
        }
        
        for habit in habits:
            habit_id = habit[0]
            old_streak = habit[2] or 0
            
            streak_data = self.calculate_streak(user_id, habit_id)
            new_streak = streak_data["current_streak"]
            
            if new_streak != old_streak:
                update_query = text("""
                    UPDATE habit_habits
                    SET current_streak = :streak,
                        longest_streak = GREATEST(longest_streak, :streak),
                        updated_at = NOW()
                    WHERE id = :habit_id
                """)
                
                self.db.execute(update_query, {
                    "streak": new_streak, "habit_id": habit_id
                })
                
                results["streaksUpdated"] += 1
                
                if new_streak < old_streak:
                    results["streaksBroken"] += 1
                
                results["details"].append({
                    "habitId": habit_id,
                    "habitName": habit[1],
                    "oldStreak": old_streak,
                    "newStreak": new_streak,
                    "broken": new_streak < old_streak
                })
            
            results["validated"] += 1
        
        self.db.commit()
        
        return results


def nightly_streak_validation_job():
    """APScheduler job for nightly streak validation across all users"""
    from app.database import SessionLocal
    
    logger.info("Starting nightly streak validation job")
    
    db = SessionLocal()
    try:
        service = StreakCalculationService(db)
        
        users_query = text("""
            SELECT DISTINCT user_id FROM habit_habits WHERE is_active = true
        """)
        
        users = db.execute(users_query).fetchall()
        
        total_validated = 0
        total_broken = 0
        
        for user_row in users:
            user_id = user_row[0]
            results = service.validate_all_user_streaks(user_id)
            total_validated += results["validated"]
            total_broken += results["streaksBroken"]
        
        logger.info(f"Nightly streak validation complete: {total_validated} habits validated, {total_broken} streaks broken")
        
    except Exception as e:
        logger.error(f"Error in nightly streak validation: {e}")
    finally:
        db.close()
