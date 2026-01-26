"""
Gamification Service - Production-Grade Points, Badges, Levels System
=====================================================================

Complete gamification engine for habit tracking with:
- XP/Points system with configurable rewards
- Badge definitions and achievement triggers
- Leveling system with tier progression
- Achievement detection and award logic
- HIPAA-compliant audit logging
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.access_control import HIPAAAuditLogger, AccessScope, PHICategory

logger = logging.getLogger(__name__)


class BadgeCategory(str, Enum):
    STREAK = "streak"
    COMPLETION = "completion"
    CONSISTENCY = "consistency"
    MILESTONE = "milestone"
    QUIT = "quit"
    SOCIAL = "social"
    SPECIAL = "special"


BADGE_DEFINITIONS = {
    "first_step": {
        "id": "first_step",
        "name": "First Step",
        "description": "Complete your first habit",
        "category": BadgeCategory.MILESTONE,
        "icon": "footprints",
        "xp_reward": 10,
        "rarity": "common"
    },
    "week_warrior": {
        "id": "week_warrior",
        "name": "Week Warrior",
        "description": "Maintain a 7-day streak on any habit",
        "category": BadgeCategory.STREAK,
        "icon": "flame",
        "xp_reward": 50,
        "rarity": "common"
    },
    "two_week_titan": {
        "id": "two_week_titan",
        "name": "Two Week Titan",
        "description": "Maintain a 14-day streak on any habit",
        "category": BadgeCategory.STREAK,
        "icon": "fire",
        "xp_reward": 100,
        "rarity": "uncommon"
    },
    "month_master": {
        "id": "month_master",
        "name": "Month Master",
        "description": "Maintain a 30-day streak on any habit",
        "category": BadgeCategory.STREAK,
        "icon": "trophy",
        "xp_reward": 200,
        "rarity": "rare"
    },
    "sixty_day_legend": {
        "id": "sixty_day_legend",
        "name": "60-Day Legend",
        "description": "Maintain a 60-day streak - habit is forming!",
        "category": BadgeCategory.STREAK,
        "icon": "medal",
        "xp_reward": 400,
        "rarity": "epic"
    },
    "hundred_day_hero": {
        "id": "hundred_day_hero",
        "name": "100-Day Hero",
        "description": "Maintain a 100-day streak - true commitment!",
        "category": BadgeCategory.STREAK,
        "icon": "crown",
        "xp_reward": 1000,
        "rarity": "legendary"
    },
    "habit_collector": {
        "id": "habit_collector",
        "name": "Habit Collector",
        "description": "Create 5 different habits",
        "category": BadgeCategory.MILESTONE,
        "icon": "collection",
        "xp_reward": 30,
        "rarity": "common"
    },
    "habit_master": {
        "id": "habit_master",
        "name": "Habit Master",
        "description": "Create 10 different habits",
        "category": BadgeCategory.MILESTONE,
        "icon": "star",
        "xp_reward": 75,
        "rarity": "uncommon"
    },
    "century_club": {
        "id": "century_club",
        "name": "Century Club",
        "description": "Complete 100 total habit check-ins",
        "category": BadgeCategory.COMPLETION,
        "icon": "check-circle",
        "xp_reward": 100,
        "rarity": "uncommon"
    },
    "five_hundred_strong": {
        "id": "five_hundred_strong",
        "name": "500 Strong",
        "description": "Complete 500 total habit check-ins",
        "category": BadgeCategory.COMPLETION,
        "icon": "award",
        "xp_reward": 300,
        "rarity": "rare"
    },
    "thousand_completions": {
        "id": "thousand_completions",
        "name": "Thousand Completions",
        "description": "Complete 1000 total habit check-ins",
        "category": BadgeCategory.COMPLETION,
        "icon": "gem",
        "xp_reward": 750,
        "rarity": "epic"
    },
    "perfect_week": {
        "id": "perfect_week",
        "name": "Perfect Week",
        "description": "Complete all habits every day for a week",
        "category": BadgeCategory.CONSISTENCY,
        "icon": "calendar-check",
        "xp_reward": 75,
        "rarity": "uncommon"
    },
    "perfect_month": {
        "id": "perfect_month",
        "name": "Perfect Month",
        "description": "Complete all habits every day for a month",
        "category": BadgeCategory.CONSISTENCY,
        "icon": "calendar-star",
        "xp_reward": 500,
        "rarity": "epic"
    },
    "early_bird": {
        "id": "early_bird",
        "name": "Early Bird",
        "description": "Complete a habit before 7 AM",
        "category": BadgeCategory.SPECIAL,
        "icon": "sunrise",
        "xp_reward": 25,
        "rarity": "common"
    },
    "night_owl": {
        "id": "night_owl",
        "name": "Night Owl",
        "description": "Complete a habit after 10 PM",
        "category": BadgeCategory.SPECIAL,
        "icon": "moon",
        "xp_reward": 25,
        "rarity": "common"
    },
    "comeback_kid": {
        "id": "comeback_kid",
        "name": "Comeback Kid",
        "description": "Resume a habit after breaking a 7+ day streak",
        "category": BadgeCategory.SPECIAL,
        "icon": "refresh",
        "xp_reward": 50,
        "rarity": "uncommon"
    },
    "one_week_clean": {
        "id": "one_week_clean",
        "name": "One Week Clean",
        "description": "7 days free from a bad habit",
        "category": BadgeCategory.QUIT,
        "icon": "shield",
        "xp_reward": 75,
        "rarity": "uncommon"
    },
    "one_month_clean": {
        "id": "one_month_clean",
        "name": "One Month Clean",
        "description": "30 days free from a bad habit",
        "category": BadgeCategory.QUIT,
        "icon": "shield-check",
        "xp_reward": 200,
        "rarity": "rare"
    },
    "ninety_days_clean": {
        "id": "ninety_days_clean",
        "name": "90 Days Clean",
        "description": "90 days free - major milestone!",
        "category": BadgeCategory.QUIT,
        "icon": "star-shield",
        "xp_reward": 500,
        "rarity": "epic"
    },
    "one_year_clean": {
        "id": "one_year_clean",
        "name": "One Year Clean",
        "description": "365 days free - you've transformed!",
        "category": BadgeCategory.QUIT,
        "icon": "crown-shield",
        "xp_reward": 2000,
        "rarity": "legendary"
    },
    "encourager": {
        "id": "encourager",
        "name": "Encourager",
        "description": "Send 10 encouragement messages to buddies",
        "category": BadgeCategory.SOCIAL,
        "icon": "heart",
        "xp_reward": 50,
        "rarity": "uncommon"
    },
    "accountability_partner": {
        "id": "accountability_partner",
        "name": "Accountability Partner",
        "description": "Connect with 3 habit buddies",
        "category": BadgeCategory.SOCIAL,
        "icon": "users",
        "xp_reward": 40,
        "rarity": "uncommon"
    }
}

LEVEL_THRESHOLDS = [
    {"level": 1, "name": "Novice", "min_xp": 0, "max_xp": 99},
    {"level": 2, "name": "Beginner", "min_xp": 100, "max_xp": 249},
    {"level": 3, "name": "Learner", "min_xp": 250, "max_xp": 499},
    {"level": 4, "name": "Intermediate", "min_xp": 500, "max_xp": 999},
    {"level": 5, "name": "Skilled", "min_xp": 1000, "max_xp": 1999},
    {"level": 6, "name": "Advanced", "min_xp": 2000, "max_xp": 3499},
    {"level": 7, "name": "Expert", "min_xp": 3500, "max_xp": 5499},
    {"level": 8, "name": "Master", "min_xp": 5500, "max_xp": 7999},
    {"level": 9, "name": "Grandmaster", "min_xp": 8000, "max_xp": 11999},
    {"level": 10, "name": "Legend", "min_xp": 12000, "max_xp": float('inf')}
]

XP_REWARDS = {
    "habit_completion": 10,
    "streak_day": 5,
    "perfect_day": 25,
    "mood_log": 5,
    "journal_entry": 10,
    "cbt_session": 20,
    "quit_day_clean": 15
}


class GamificationService:
    """Production-grade gamification engine for habit tracking"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_gamification_state(self, user_id: str) -> Dict[str, Any]:
        """Get complete gamification state for user"""
        
        reward_query = text("""
            SELECT id, reward_type, current_level, growth_stage, total_points,
                   streak_bonus, completion_points, visual_state, unlocked_badges,
                   days_active, perfect_days
            FROM habit_rewards
            WHERE user_id = :user_id
        """)
        
        row = self.db.execute(reward_query, {"user_id": user_id}).fetchone()
        
        if not row:
            self._initialize_user_rewards(user_id)
            row = self.db.execute(reward_query, {"user_id": user_id}).fetchone()
        
        total_xp = row[4] or 0
        unlocked_badges = row[8] or []
        
        level_info = self._calculate_level(total_xp)
        
        all_badges = []
        for badge_id, badge_def in BADGE_DEFINITIONS.items():
            is_unlocked = badge_id in unlocked_badges
            all_badges.append({
                **badge_def,
                "unlocked": is_unlocked,
                "unlocked_at": None
            })
        
        return {
            "id": row[0],
            "userId": user_id,
            "level": level_info,
            "totalXp": total_xp,
            "streakBonus": row[5] or 0,
            "completionPoints": row[6] or 0,
            "daysActive": row[9] or 0,
            "perfectDays": row[10] or 0,
            "growthStage": row[3],
            "visualState": row[7],
            "badges": {
                "unlocked": [b for b in all_badges if b["unlocked"]],
                "locked": [b for b in all_badges if not b["unlocked"]],
                "totalUnlocked": len(unlocked_badges),
                "totalAvailable": len(BADGE_DEFINITIONS)
            },
            "nextMilestones": self._get_next_milestones(user_id, unlocked_badges)
        }
    
    def _initialize_user_rewards(self, user_id: str):
        """Initialize gamification record for new user"""
        insert_query = text("""
            INSERT INTO habit_rewards 
            (id, user_id, reward_type, current_level, growth_stage, total_points,
             streak_bonus, completion_points, visual_state, unlocked_badges, days_active, perfect_days)
            VALUES (gen_random_uuid(), :user_id, 'plant_growth', 1, 'seed', 0, 0, 0, 
                   '{"height": 1, "leaves": 0, "flowers": 0}'::jsonb, '[]'::jsonb, 0, 0)
            ON CONFLICT (user_id) DO NOTHING
        """)
        self.db.execute(insert_query, {"user_id": user_id})
        self.db.commit()
    
    def _calculate_level(self, total_xp: int) -> Dict[str, Any]:
        """Calculate user level from total XP"""
        current_level = LEVEL_THRESHOLDS[0]
        
        for level in LEVEL_THRESHOLDS:
            if level["min_xp"] <= total_xp <= level["max_xp"]:
                current_level = level
                break
            elif total_xp > level["max_xp"]:
                current_level = level
        
        xp_in_level = total_xp - current_level["min_xp"]
        xp_for_next = current_level["max_xp"] - current_level["min_xp"]
        progress = min(1.0, xp_in_level / xp_for_next) if xp_for_next != float('inf') else 1.0
        
        next_level = None
        for level in LEVEL_THRESHOLDS:
            if level["level"] == current_level["level"] + 1:
                next_level = level
                break
        
        return {
            "current": current_level["level"],
            "name": current_level["name"],
            "xpInLevel": xp_in_level,
            "xpForNextLevel": int(xp_for_next) if xp_for_next != float('inf') else None,
            "progress": round(progress, 2),
            "nextLevelName": next_level["name"] if next_level else None
        }
    
    def award_xp(self, user_id: str, xp_type: str, multiplier: float = 1.0) -> Tuple[int, Optional[Dict]]:
        """Award XP to user and check for level up"""
        
        base_xp = XP_REWARDS.get(xp_type, 0)
        xp_awarded = int(base_xp * multiplier)
        
        if xp_awarded <= 0:
            return 0, None
        
        old_state = self.get_user_gamification_state(user_id)
        old_level = old_state["level"]["current"]
        
        update_query = text("""
            UPDATE habit_rewards
            SET total_points = total_points + :xp,
                updated_at = NOW()
            WHERE user_id = :user_id
            RETURNING total_points
        """)
        
        result = self.db.execute(update_query, {"user_id": user_id, "xp": xp_awarded})
        new_total = result.fetchone()[0]
        self.db.commit()
        
        new_level_info = self._calculate_level(new_total)
        
        level_up = None
        if new_level_info["current"] > old_level:
            level_up = {
                "oldLevel": old_level,
                "newLevel": new_level_info["current"],
                "newLevelName": new_level_info["name"]
            }
            
            logger.info(f"User {user_id} leveled up: {old_level} -> {new_level_info['current']}")
        
        return xp_awarded, level_up
    
    def check_and_award_badges(self, user_id: str, event_type: str, event_data: Dict = None) -> List[Dict]:
        """Check badge conditions and award any earned badges"""
        
        event_data = event_data or {}
        awarded_badges = []
        
        current_badges = self._get_user_badges(user_id)
        
        badge_checks = []
        
        if event_type == "habit_completion":
            badge_checks.extend([
                ("first_step", self._check_first_completion),
                ("century_club", self._check_completion_milestone),
                ("five_hundred_strong", self._check_completion_milestone),
                ("thousand_completions", self._check_completion_milestone),
                ("early_bird", self._check_early_bird),
                ("night_owl", self._check_night_owl),
            ])
        
        elif event_type == "streak_update":
            badge_checks.extend([
                ("week_warrior", self._check_streak_milestone),
                ("two_week_titan", self._check_streak_milestone),
                ("month_master", self._check_streak_milestone),
                ("sixty_day_legend", self._check_streak_milestone),
                ("hundred_day_hero", self._check_streak_milestone),
                ("comeback_kid", self._check_comeback),
            ])
        
        elif event_type == "habit_created":
            badge_checks.extend([
                ("habit_collector", self._check_habit_count),
                ("habit_master", self._check_habit_count),
            ])
        
        elif event_type == "quit_day":
            badge_checks.extend([
                ("one_week_clean", self._check_quit_milestone),
                ("one_month_clean", self._check_quit_milestone),
                ("ninety_days_clean", self._check_quit_milestone),
                ("one_year_clean", self._check_quit_milestone),
            ])
        
        elif event_type == "perfect_day":
            badge_checks.extend([
                ("perfect_week", self._check_perfect_streak),
                ("perfect_month", self._check_perfect_streak),
            ])
        
        elif event_type == "social":
            badge_checks.extend([
                ("encourager", self._check_encouragements),
                ("accountability_partner", self._check_buddy_count),
            ])
        
        for badge_id, check_func in badge_checks:
            if badge_id not in current_badges:
                if check_func(user_id, badge_id, event_data):
                    badge_def = BADGE_DEFINITIONS[badge_id]
                    self._award_badge(user_id, badge_id)
                    self.award_xp(user_id, "badge_earned", badge_def["xp_reward"] / 10)
                    awarded_badges.append(badge_def)
                    
                    logger.info(f"User {user_id} earned badge: {badge_id}")
        
        return awarded_badges
    
    def _get_user_badges(self, user_id: str) -> List[str]:
        """Get list of badge IDs user has earned"""
        query = text("SELECT unlocked_badges FROM habit_rewards WHERE user_id = :user_id")
        result = self.db.execute(query, {"user_id": user_id}).fetchone()
        return result[0] if result and result[0] else []
    
    def _award_badge(self, user_id: str, badge_id: str):
        """Award badge to user"""
        update_query = text("""
            UPDATE habit_rewards
            SET unlocked_badges = unlocked_badges || :badge_json::jsonb,
                updated_at = NOW()
            WHERE user_id = :user_id
        """)
        
        import json
        self.db.execute(update_query, {
            "user_id": user_id,
            "badge_json": json.dumps([badge_id])
        })
        self.db.commit()
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=user_id,
            actor_role="patient",
            patient_id=user_id,
            action="badge_awarded",
            phi_categories=["behavioral_health"],
            resource_type="HabitReward",
            resource_id=badge_id,
            access_reason="gamification_achievement"
        )
    
    def _check_first_completion(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT COUNT(*) FROM habit_completions WHERE user_id = :user_id AND completed = true")
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        return count >= 1
    
    def _check_completion_milestone(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT COUNT(*) FROM habit_completions WHERE user_id = :user_id AND completed = true")
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        
        thresholds = {"century_club": 100, "five_hundred_strong": 500, "thousand_completions": 1000}
        return count >= thresholds.get(badge_id, float('inf'))
    
    def _check_streak_milestone(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT MAX(current_streak) FROM habit_habits WHERE user_id = :user_id AND is_active = true")
        max_streak = self.db.execute(query, {"user_id": user_id}).fetchone()[0] or 0
        
        thresholds = {
            "week_warrior": 7, "two_week_titan": 14, "month_master": 30,
            "sixty_day_legend": 60, "hundred_day_hero": 100
        }
        return max_streak >= thresholds.get(badge_id, float('inf'))
    
    def _check_early_bird(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("""
            SELECT COUNT(*) FROM habit_completions 
            WHERE user_id = :user_id AND completed = true 
            AND EXTRACT(HOUR FROM completion_date) < 7
        """)
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        return count >= 1
    
    def _check_night_owl(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("""
            SELECT COUNT(*) FROM habit_completions 
            WHERE user_id = :user_id AND completed = true 
            AND EXTRACT(HOUR FROM completion_date) >= 22
        """)
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        return count >= 1
    
    def _check_comeback(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("""
            SELECT COUNT(*) FROM habit_habits 
            WHERE user_id = :user_id AND longest_streak >= 7 AND current_streak >= 1 
            AND current_streak < longest_streak
        """)
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        return count >= 1
    
    def _check_habit_count(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT COUNT(*) FROM habit_habits WHERE user_id = :user_id")
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        
        thresholds = {"habit_collector": 5, "habit_master": 10}
        return count >= thresholds.get(badge_id, float('inf'))
    
    def _check_quit_milestone(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT MAX(days_clean) FROM habit_quit_plans WHERE user_id = :user_id AND status = 'active'")
        max_days = self.db.execute(query, {"user_id": user_id}).fetchone()[0] or 0
        
        thresholds = {"one_week_clean": 7, "one_month_clean": 30, "ninety_days_clean": 90, "one_year_clean": 365}
        return max_days >= thresholds.get(badge_id, float('inf'))
    
    def _check_perfect_streak(self, user_id: str, badge_id: str, data: Dict) -> bool:
        perfect_days = data.get("consecutive_perfect_days", 0)
        thresholds = {"perfect_week": 7, "perfect_month": 30}
        return perfect_days >= thresholds.get(badge_id, float('inf'))
    
    def _check_encouragements(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT COUNT(*) FROM habit_encouragements WHERE from_user_id = :user_id")
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        return count >= 10
    
    def _check_buddy_count(self, user_id: str, badge_id: str, data: Dict) -> bool:
        query = text("SELECT COUNT(*) FROM habit_buddies WHERE user_id = :user_id AND status = 'accepted'")
        count = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        return count >= 3
    
    def _get_next_milestones(self, user_id: str, current_badges: List[str]) -> List[Dict]:
        """Get upcoming milestones user is close to achieving"""
        milestones = []
        
        query = text("SELECT COUNT(*) FROM habit_completions WHERE user_id = :user_id AND completed = true")
        completions = self.db.execute(query, {"user_id": user_id}).fetchone()[0]
        
        completion_milestones = [
            (100, "century_club"), (500, "five_hundred_strong"), (1000, "thousand_completions")
        ]
        
        for threshold, badge_id in completion_milestones:
            if badge_id not in current_badges and completions < threshold:
                progress = completions / threshold
                if progress >= 0.5:
                    milestones.append({
                        "badge": BADGE_DEFINITIONS[badge_id],
                        "progress": round(progress, 2),
                        "remaining": threshold - completions
                    })
                break
        
        query = text("SELECT MAX(current_streak) FROM habit_habits WHERE user_id = :user_id AND is_active = true")
        max_streak = self.db.execute(query, {"user_id": user_id}).fetchone()[0] or 0
        
        streak_milestones = [
            (7, "week_warrior"), (14, "two_week_titan"), (30, "month_master"),
            (60, "sixty_day_legend"), (100, "hundred_day_hero")
        ]
        
        for threshold, badge_id in streak_milestones:
            if badge_id not in current_badges and max_streak < threshold:
                progress = max_streak / threshold
                if progress >= 0.3:
                    milestones.append({
                        "badge": BADGE_DEFINITIONS[badge_id],
                        "progress": round(progress, 2),
                        "remaining": threshold - max_streak
                    })
                break
        
        return milestones[:3]
    
    def on_habit_completed(self, user_id: str, habit_id: str) -> Dict[str, Any]:
        """Handle habit completion event - award XP and check badges"""
        
        xp_awarded, level_up = self.award_xp(user_id, "habit_completion")
        
        streak_query = text("SELECT current_streak FROM habit_habits WHERE id = :habit_id")
        streak = self.db.execute(streak_query, {"habit_id": habit_id}).fetchone()
        streak_days = streak[0] if streak else 0
        
        if streak_days > 1:
            streak_xp, _ = self.award_xp(user_id, "streak_day", min(streak_days / 10, 2.0))
            xp_awarded += streak_xp
        
        badges = self.check_and_award_badges(user_id, "habit_completion", {"habit_id": habit_id})
        badges.extend(self.check_and_award_badges(user_id, "streak_update", {"streak": streak_days}))
        
        return {
            "xpAwarded": xp_awarded,
            "levelUp": level_up,
            "badgesEarned": badges
        }
