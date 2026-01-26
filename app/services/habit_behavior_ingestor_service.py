"""
Habit Behavior Ingestor Service - Multi-Modal Data Integration
==============================================================

Feeds habit tracker data into the Behavior AI system for:
- Cross-signal correlation analysis
- Habit patterns as behavioral biomarkers
- Integration with risk scoring engine
- Deterioration trend detection from habit data
- HIPAA-compliant with audit logging
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.services.access_control import HIPAAAuditLogger, AccessScope, PHICategory

logger = logging.getLogger(__name__)


class HabitBehaviorIngestorService:
    """
    Ingests habit tracking data into the Behavior AI system.
    Maps habit events to behavioral signals for risk assessment.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def ingest_habit_completion(
        self,
        user_id: str,
        habit_id: str,
        completed: bool,
        mood: Optional[str] = None,
        difficulty: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ingest habit completion event as behavioral signal.
        Maps to behavior metrics for risk scoring.
        """
        
        habit_query = text("""
            SELECT name, category, current_streak, frequency
            FROM habit_habits
            WHERE id = :habit_id AND user_id = :user_id
        """)
        
        habit = self.db.execute(habit_query, {
            "habit_id": habit_id, "user_id": user_id
        }).fetchone()
        
        if not habit:
            return {"ingested": False, "error": "Habit not found"}
        
        signals = self._extract_behavioral_signals(
            user_id=user_id,
            habit_name=habit[0],
            category=habit[1],
            current_streak=habit[2] or 0,
            completed=completed,
            mood=mood,
            difficulty=difficulty,
            notes=notes
        )
        
        self._store_behavioral_signals(user_id, signals)
        
        return {
            "ingested": True,
            "signals": signals,
            "habitName": habit[0],
            "category": habit[1]
        }
    
    def _extract_behavioral_signals(
        self,
        user_id: str,
        habit_name: str,
        category: str,
        current_streak: int,
        completed: bool,
        mood: Optional[str],
        difficulty: Optional[int],
        notes: Optional[str]
    ) -> Dict[str, Any]:
        """Extract behavioral signals from habit completion data"""
        
        signals = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "habit_tracker",
            "habit_category": category,
            "habit_name": habit_name,
            "completion": {
                "completed": completed,
                "streak_day": current_streak if completed else 0,
                "streak_broken": not completed and current_streak > 0
            },
            "engagement": {
                "score": self._calculate_engagement_score(completed, current_streak, difficulty),
                "consistency_indicator": min(1.0, current_streak / 30) if completed else 0
            },
            "mood_context": None,
            "difficulty_context": None,
            "text_signals": None
        }
        
        if mood:
            mood_scores = {
                "great": 1.0, "good": 0.75, "okay": 0.5, 
                "neutral": 0.5, "bad": 0.25, "terrible": 0.1
            }
            signals["mood_context"] = {
                "label": mood,
                "normalized_score": mood_scores.get(mood.lower(), 0.5)
            }
        
        if difficulty:
            signals["difficulty_context"] = {
                "level": difficulty,
                "normalized": difficulty / 10.0,
                "struggle_indicator": difficulty >= 7
            }
        
        if notes:
            signals["text_signals"] = self._analyze_notes_for_signals(notes)
        
        return signals
    
    def _calculate_engagement_score(
        self,
        completed: bool,
        streak: int,
        difficulty: Optional[int]
    ) -> float:
        """Calculate engagement score from completion factors"""
        
        if not completed:
            return 0.2
        
        base_score = 0.5
        streak_bonus = min(0.3, streak * 0.01)
        difficulty_factor = 0
        if difficulty:
            difficulty_factor = 0.2 * (1 - (difficulty - 1) / 9)
        
        return min(1.0, base_score + streak_bonus + difficulty_factor)
    
    def _analyze_notes_for_signals(self, notes: str) -> Dict[str, Any]:
        """Extract behavioral signals from completion notes"""
        
        notes_lower = notes.lower()
        
        avoidance_phrases = [
            "didn't feel like", "too tired", "skipped", "forgot",
            "couldn't", "not in the mood", "gave up", "quit"
        ]
        
        positive_phrases = [
            "felt great", "easy", "loved it", "proud", "accomplished",
            "motivated", "energized", "happy"
        ]
        
        struggle_phrases = [
            "hard", "difficult", "struggled", "barely", "forced",
            "pushed through", "exhausted", "challenge"
        ]
        
        avoidance_count = sum(1 for p in avoidance_phrases if p in notes_lower)
        positive_count = sum(1 for p in positive_phrases if p in notes_lower)
        struggle_count = sum(1 for p in struggle_phrases if p in notes_lower)
        
        sentiment_score = 0.5
        if positive_count > 0:
            sentiment_score = min(1.0, 0.5 + positive_count * 0.15)
        if avoidance_count > 0:
            sentiment_score = max(0.0, sentiment_score - avoidance_count * 0.2)
        if struggle_count > 0:
            sentiment_score = max(0.2, sentiment_score - struggle_count * 0.1)
        
        return {
            "avoidance_detected": avoidance_count > 0,
            "positive_sentiment": positive_count > 0,
            "struggle_indicated": struggle_count > 0,
            "sentiment_score": sentiment_score,
            "word_count": len(notes.split())
        }
    
    def _store_behavioral_signals(self, user_id: str, signals: Dict[str, Any]):
        """Store behavioral signals in behavior metrics table"""
        
        try:
            import json
            
            today = datetime.utcnow().date()
            
            check_query = text("""
                SELECT id, habit_engagement_score, habit_completion_rate, habit_signals
                FROM behavior_metrics
                WHERE patient_id = :user_id AND DATE(date) = :today
            """)
            
            existing = self.db.execute(check_query, {
                "user_id": user_id, "today": today
            }).fetchone()
            
            if existing:
                current_signals = existing[3] or []
                if isinstance(current_signals, str):
                    current_signals = json.loads(current_signals)
                current_signals.append(signals)
                
                new_engagement = (float(existing[1] or 0) + signals["engagement"]["score"]) / 2
                
                update_query = text("""
                    UPDATE behavior_metrics
                    SET habit_engagement_score = :engagement,
                        habit_signals = :signals::jsonb,
                        updated_at = NOW()
                    WHERE id = :id
                """)
                
                self.db.execute(update_query, {
                    "engagement": new_engagement,
                    "signals": json.dumps(current_signals),
                    "id": existing[0]
                })
            else:
                insert_query = text("""
                    INSERT INTO behavior_metrics 
                    (id, patient_id, date, habit_engagement_score, habit_completion_rate, habit_signals)
                    VALUES (gen_random_uuid(), :user_id, :date, :engagement, :completion, :signals::jsonb)
                    ON CONFLICT DO NOTHING
                """)
                
                self.db.execute(insert_query, {
                    "user_id": user_id,
                    "date": datetime.utcnow(),
                    "engagement": signals["engagement"]["score"],
                    "completion": 1.0 if signals["completion"]["completed"] else 0.0,
                    "signals": json.dumps([signals])
                })
            
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"Failed to store behavioral signals: {e}")
            self.db.rollback()
    
    def aggregate_daily_habit_metrics(self, user_id: str) -> Dict[str, Any]:
        """Aggregate daily habit metrics for behavior AI analysis"""
        
        today = datetime.utcnow().date()
        
        query = text("""
            SELECT 
                COUNT(*) FILTER (WHERE completed = true) as completions,
                COUNT(*) FILTER (WHERE completed = false) as skips,
                COUNT(*) as total,
                AVG(CASE WHEN mood IN ('great', 'good') THEN 1.0 
                         WHEN mood IN ('okay', 'neutral') THEN 0.5 
                         ELSE 0.2 END) as avg_mood,
                AVG(difficulty_level) as avg_difficulty
            FROM habit_completions
            WHERE user_id = :user_id
            AND DATE(completion_date) = :today
        """)
        
        result = self.db.execute(query, {
            "user_id": user_id, "today": today
        }).fetchone()
        
        if not result or result[2] == 0:
            return {
                "date": today.isoformat(),
                "completions": 0,
                "skips": 0,
                "completionRate": 0.0,
                "avgMood": None,
                "avgDifficulty": None,
                "behaviorRiskContribution": 0.0
            }
        
        completions = result[0] or 0
        skips = result[1] or 0
        total = result[2]
        completion_rate = completions / total if total > 0 else 0
        
        risk_contribution = self._calculate_behavior_risk_contribution(
            completion_rate=completion_rate,
            avg_mood=float(result[3]) if result[3] else 0.5,
            avg_difficulty=float(result[4]) if result[4] else 5
        )
        
        return {
            "date": today.isoformat(),
            "completions": completions,
            "skips": skips,
            "completionRate": round(completion_rate, 2),
            "avgMood": round(float(result[3]), 2) if result[3] else None,
            "avgDifficulty": round(float(result[4]), 1) if result[4] else None,
            "behaviorRiskContribution": round(risk_contribution, 2)
        }
    
    def _calculate_behavior_risk_contribution(
        self,
        completion_rate: float,
        avg_mood: float,
        avg_difficulty: float
    ) -> float:
        """
        Calculate risk contribution from habit behavior.
        Higher values indicate higher risk (deterioration indicators).
        """
        
        completion_risk = (1 - completion_rate) * 40
        mood_risk = (1 - avg_mood) * 30
        difficulty_risk = (avg_difficulty / 10) * 30
        
        return min(100, completion_risk + mood_risk + difficulty_risk)
    
    def get_habit_trend_signals(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get habit behavior trends for deterioration detection"""
        
        query = text("""
            SELECT 
                DATE(completion_date) as day,
                COUNT(*) FILTER (WHERE completed = true) as completions,
                COUNT(*) as total,
                AVG(difficulty_level) as avg_difficulty
            FROM habit_completions
            WHERE user_id = :user_id
            AND completion_date >= NOW() - :days * INTERVAL '1 day'
            GROUP BY DATE(completion_date)
            ORDER BY day ASC
        """)
        
        results = self.db.execute(query, {
            "user_id": user_id, "days": days
        }).fetchall()
        
        if len(results) < 3:
            return {
                "trendAvailable": False,
                "reason": "Insufficient data",
                "daysAnalyzed": len(results)
            }
        
        completion_rates = [r[1] / r[2] if r[2] > 0 else 0 for r in results]
        
        trend_direction = "stable"
        if len(completion_rates) >= 3:
            recent_avg = sum(completion_rates[-3:]) / 3
            earlier_avg = sum(completion_rates[:3]) / 3
            
            if recent_avg < earlier_avg - 0.15:
                trend_direction = "declining"
            elif recent_avg > earlier_avg + 0.15:
                trend_direction = "improving"
        
        overall_rate = sum(r[1] for r in results) / sum(r[2] for r in results) if results else 0
        
        deterioration_signals = []
        if trend_direction == "declining":
            deterioration_signals.append("Declining habit completion rate")
        if overall_rate < 0.5:
            deterioration_signals.append("Low overall habit engagement")
        
        avg_difficulty_trend = [float(r[3]) if r[3] else 5 for r in results]
        if len(avg_difficulty_trend) >= 3:
            if avg_difficulty_trend[-1] > avg_difficulty_trend[0] + 2:
                deterioration_signals.append("Increasing difficulty reported")
        
        return {
            "trendAvailable": True,
            "daysAnalyzed": len(results),
            "overallCompletionRate": round(overall_rate, 2),
            "trendDirection": trend_direction,
            "dailyRates": [
                {"date": r[0].isoformat(), "rate": round(r[1]/r[2], 2) if r[2] > 0 else 0}
                for r in results
            ],
            "deteriorationSignals": deterioration_signals,
            "riskLevel": "high" if len(deterioration_signals) >= 2 else ("medium" if deterioration_signals else "low")
        }
    
    def feed_to_risk_scoring_engine(self, user_id: str) -> Dict[str, Any]:
        """
        Feed aggregated habit data to the behavior AI risk scoring engine.
        Returns risk contribution from habit behavior.
        """
        
        daily_metrics = self.aggregate_daily_habit_metrics(user_id)
        trend_signals = self.get_habit_trend_signals(user_id)
        
        base_risk = daily_metrics["behaviorRiskContribution"]
        trend_adjustment = 0
        
        if trend_signals.get("trendDirection") == "declining":
            trend_adjustment = 15
        elif trend_signals.get("trendDirection") == "improving":
            trend_adjustment = -10
        
        signal_adjustment = len(trend_signals.get("deteriorationSignals", [])) * 5
        
        final_risk = min(100, max(0, base_risk + trend_adjustment + signal_adjustment))
        
        try:
            update_query = text("""
                UPDATE behavior_risk_scores
                SET habit_behavior_risk = :risk,
                    habit_risk_factors = :factors::jsonb,
                    updated_at = NOW()
                WHERE patient_id = :user_id
                AND DATE(calculated_at) = CURRENT_DATE
            """)
            
            import json
            self.db.execute(update_query, {
                "risk": final_risk,
                "factors": json.dumps({
                    "daily_metrics": daily_metrics,
                    "trend": trend_signals.get("trendDirection"),
                    "signals": trend_signals.get("deteriorationSignals", [])
                }),
                "user_id": user_id
            })
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to update risk score with habit data: {e}")
        
        HIPAAAuditLogger.log_access(
            user_id=user_id,
            user_role="patient",
            action="habit_risk_feed",
            resource_type="BehaviorRiskScore",
            resource_id=user_id,
            access_reason="habit_behavior_ingestion",
            was_successful=True
        )
        
        return {
            "habitBehaviorRisk": round(final_risk, 1),
            "riskLevel": "high" if final_risk >= 60 else ("medium" if final_risk >= 30 else "low"),
            "dailyMetrics": daily_metrics,
            "trendSignals": trend_signals,
            "contributingFactors": trend_signals.get("deteriorationSignals", [])
        }
