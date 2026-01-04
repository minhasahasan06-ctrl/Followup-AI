"""
Patient Genius Features (E.12-E.14)
===================================
Advanced patient-facing AI capabilities.

E.12: Effort-aware daily check-in (minimal questions when stable)
E.13: Just-in-time micro-habits (template picks by engagement bucket)
E.14: "Explain in 1 sentence" safe trend explanations (no raw numbers)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PatientStability(str, Enum):
    """Patient stability levels for effort-aware check-ins"""
    STABLE = "stable"
    IMPROVING = "improving"
    MILD_CONCERN = "mild_concern"
    MODERATE_CONCERN = "moderate_concern"
    HIGH_CONCERN = "high_concern"


class EngagementBucket(str, Enum):
    """Patient engagement levels"""
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    AT_RISK = "at_risk"


class TrendDirection(str, Enum):
    """Direction of health trend"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    FLUCTUATING = "fluctuating"


@dataclass
class EffortAwareCheckin:
    """Effort-aware daily check-in configuration"""
    patient_id_hash: str
    stability_level: PatientStability
    question_count: int
    question_ids: List[str]
    estimated_time_seconds: int
    skip_allowed: bool
    next_full_checkin_date: Optional[str]
    reason: str


@dataclass
class MicroHabitSuggestion:
    """Just-in-time micro-habit suggestion"""
    habit_id: str
    habit_name: str
    habit_description: str
    trigger_context: str
    duration_minutes: int
    difficulty_level: str
    engagement_match: EngagementBucket
    personalization_reason: str


@dataclass
class SafeTrendExplanation:
    """Safe trend explanation without raw numbers"""
    metric_name: str
    trend_direction: TrendDirection
    explanation: str
    timeframe: str
    confidence: str
    action_suggestion: Optional[str]


QUESTION_BANK = {
    "Q_FEELING_GENERAL": {"text": "How are you feeling today?", "type": "scale", "time_seconds": 5},
    "Q_ENERGY_LEVEL": {"text": "What's your energy level?", "type": "scale", "time_seconds": 5},
    "Q_SLEEP_QUALITY": {"text": "How did you sleep last night?", "type": "scale", "time_seconds": 5},
    "Q_PAIN_LEVEL": {"text": "Are you experiencing any pain?", "type": "scale", "time_seconds": 5},
    "Q_APPETITE": {"text": "How is your appetite today?", "type": "scale", "time_seconds": 5},
    "Q_MOOD": {"text": "How would you describe your mood?", "type": "choice", "time_seconds": 8},
    "Q_MEDICATION_TAKEN": {"text": "Have you taken your medications?", "type": "boolean", "time_seconds": 3},
    "Q_NEW_SYMPTOMS": {"text": "Any new symptoms to report?", "type": "boolean", "time_seconds": 3},
    "Q_SYMPTOMS_DETAIL": {"text": "Please describe your symptoms", "type": "text", "time_seconds": 30},
    "Q_ACTIVITY_LEVEL": {"text": "How active were you today?", "type": "scale", "time_seconds": 5},
}

MICRO_HABITS = {
    "H_BREATHING_1MIN": {
        "name": "1-Minute Breathing",
        "description": "Take 6 deep breaths to calm your nervous system",
        "duration_minutes": 1,
        "difficulty": "easy",
        "contexts": ["morning", "stress", "before_bed"]
    },
    "H_HYDRATION_GLASS": {
        "name": "Drink a Glass of Water",
        "description": "Stay hydrated with a full glass of water",
        "duration_minutes": 1,
        "difficulty": "easy",
        "contexts": ["morning", "afternoon", "medication"]
    },
    "H_STRETCH_2MIN": {
        "name": "Quick Stretch",
        "description": "Gentle stretches for neck, shoulders, and back",
        "duration_minutes": 2,
        "difficulty": "easy",
        "contexts": ["morning", "afternoon", "after_sitting"]
    },
    "H_GRATITUDE_NOTE": {
        "name": "Gratitude Moment",
        "description": "Think of one thing you're grateful for today",
        "duration_minutes": 1,
        "difficulty": "easy",
        "contexts": ["morning", "evening"]
    },
    "H_WALK_5MIN": {
        "name": "5-Minute Walk",
        "description": "A short walk around your space",
        "duration_minutes": 5,
        "difficulty": "moderate",
        "contexts": ["afternoon", "energy_boost"]
    },
    "H_MEDICATION_CHECK": {
        "name": "Medication Check",
        "description": "Review and organize your medications for tomorrow",
        "duration_minutes": 3,
        "difficulty": "easy",
        "contexts": ["evening", "before_bed"]
    },
}

TREND_TEMPLATES = {
    "improving_energy": "Your energy levels have been trending upward over the past {timeframe}. Keep up the good work!",
    "stable_vitals": "Your {metric} readings have been consistent and within your typical range.",
    "declining_sleep": "Your sleep patterns suggest you may not be getting as much rest as usual. Consider your evening routine.",
    "improving_adherence": "Great job with your medication routine! Your consistency has improved recently.",
    "fluctuating_mood": "Your mood has shown some ups and downs. This is normal, but worth discussing if it continues.",
    "stable_overall": "Overall, your health indicators are holding steady. No major changes detected.",
    "declining_activity": "Your activity levels have been lower than usual. Even small movements can help.",
    "improving_symptoms": "The symptoms you've been tracking appear to be easing. Continue monitoring.",
}


class GeniusPatientService:
    """
    E.12-E.14: Patient Genius Features
    
    Provides personalized, effort-aware patient interactions.
    CRITICAL: Never expose raw PHI - only bucketed/templated responses.
    """
    
    STABLE_QUESTION_COUNT = 2
    MILD_CONCERN_QUESTION_COUNT = 4
    MODERATE_CONCERN_QUESTION_COUNT = 6
    HIGH_CONCERN_QUESTION_COUNT = 8
    
    def __init__(self):
        logger.info("GeniusPatientService initialized")
    
    def generate_effort_aware_checkin(
        self,
        patient_id_hash: str,
        stability_level: PatientStability,
        recent_responses: Optional[Dict[str, Any]] = None,
        consecutive_stable_days: int = 0
    ) -> EffortAwareCheckin:
        """
        E.12: Effort-aware daily check-in.
        
        Generates minimal questions when patient is stable, more thorough
        questions when there are concerns.
        
        Args:
            patient_id_hash: Hashed patient identifier
            stability_level: Current stability assessment
            recent_responses: Recent check-in responses (for pattern detection)
            consecutive_stable_days: Days of stable readings
            
        Returns:
            EffortAwareCheckin with personalized question set
        """
        if stability_level == PatientStability.STABLE:
            question_count = self.STABLE_QUESTION_COUNT
            question_ids = ["Q_FEELING_GENERAL", "Q_MEDICATION_TAKEN"]
            skip_allowed = consecutive_stable_days >= 3
            reason = "Stable health status - minimal check-in"
        
        elif stability_level == PatientStability.IMPROVING:
            question_count = self.STABLE_QUESTION_COUNT
            question_ids = ["Q_FEELING_GENERAL", "Q_ENERGY_LEVEL"]
            skip_allowed = consecutive_stable_days >= 5
            reason = "Improving trend - brief check-in to confirm progress"
        
        elif stability_level == PatientStability.MILD_CONCERN:
            question_count = self.MILD_CONCERN_QUESTION_COUNT
            question_ids = [
                "Q_FEELING_GENERAL", "Q_ENERGY_LEVEL",
                "Q_SLEEP_QUALITY", "Q_MEDICATION_TAKEN"
            ]
            skip_allowed = False
            reason = "Mild concern detected - expanded check-in"
        
        elif stability_level == PatientStability.MODERATE_CONCERN:
            question_count = self.MODERATE_CONCERN_QUESTION_COUNT
            question_ids = [
                "Q_FEELING_GENERAL", "Q_ENERGY_LEVEL", "Q_SLEEP_QUALITY",
                "Q_PAIN_LEVEL", "Q_NEW_SYMPTOMS", "Q_MEDICATION_TAKEN"
            ]
            skip_allowed = False
            reason = "Moderate concern - comprehensive check-in required"
        
        else:
            question_count = self.HIGH_CONCERN_QUESTION_COUNT
            question_ids = [
                "Q_FEELING_GENERAL", "Q_ENERGY_LEVEL", "Q_SLEEP_QUALITY",
                "Q_PAIN_LEVEL", "Q_APPETITE", "Q_MOOD",
                "Q_NEW_SYMPTOMS", "Q_MEDICATION_TAKEN"
            ]
            skip_allowed = False
            reason = "High concern - full assessment needed"
        
        estimated_time = sum(
            QUESTION_BANK.get(qid, {}).get("time_seconds", 10)
            for qid in question_ids
        )
        
        checkin = EffortAwareCheckin(
            patient_id_hash=patient_id_hash,
            stability_level=stability_level,
            question_count=question_count,
            question_ids=question_ids,
            estimated_time_seconds=estimated_time,
            skip_allowed=skip_allowed,
            next_full_checkin_date=None,
            reason=reason
        )
        
        logger.debug(f"Generated effort-aware checkin: {question_count} questions, {estimated_time}s estimated")
        return checkin
    
    def get_micro_habit_suggestions(
        self,
        patient_id_hash: str,
        engagement_bucket: EngagementBucket,
        current_context: str = "general",
        max_suggestions: int = 3
    ) -> List[MicroHabitSuggestion]:
        """
        E.13: Just-in-time micro-habits.
        
        Suggests personalized micro-habits based on engagement level and context.
        Lower engagement patients get easier, shorter habits.
        
        Args:
            patient_id_hash: Hashed patient identifier
            engagement_bucket: Current engagement level
            current_context: Context trigger (morning, evening, etc.)
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of MicroHabitSuggestion objects
        """
        suggestions = []
        
        if engagement_bucket == EngagementBucket.AT_RISK:
            max_duration = 1
            difficulty_filter = ["easy"]
            personalization = "Starting small to rebuild your wellness routine"
        elif engagement_bucket == EngagementBucket.LOW:
            max_duration = 2
            difficulty_filter = ["easy"]
            personalization = "Quick wins to boost your daily momentum"
        elif engagement_bucket == EngagementBucket.MODERATE:
            max_duration = 5
            difficulty_filter = ["easy", "moderate"]
            personalization = "Building on your consistent engagement"
        else:
            max_duration = 10
            difficulty_filter = ["easy", "moderate", "challenging"]
            personalization = "Expanding your wellness toolkit"
        
        for habit_id, habit_data in MICRO_HABITS.items():
            if habit_data["duration_minutes"] > max_duration:
                continue
            if habit_data["difficulty"] not in difficulty_filter:
                continue
            if current_context != "general" and current_context not in habit_data.get("contexts", []):
                continue
            
            suggestion = MicroHabitSuggestion(
                habit_id=habit_id,
                habit_name=habit_data["name"],
                habit_description=habit_data["description"],
                trigger_context=current_context,
                duration_minutes=habit_data["duration_minutes"],
                difficulty_level=habit_data["difficulty"],
                engagement_match=engagement_bucket,
                personalization_reason=personalization
            )
            suggestions.append(suggestion)
            
            if len(suggestions) >= max_suggestions:
                break
        
        logger.debug(f"Generated {len(suggestions)} micro-habit suggestions for engagement={engagement_bucket.value}")
        return suggestions
    
    def generate_safe_trend_explanation(
        self,
        metric_name: str,
        trend_direction: TrendDirection,
        timeframe_days: int = 7
    ) -> SafeTrendExplanation:
        """
        E.14: "Explain in 1 sentence" safe trend explanations.
        
        Generates patient-safe trend explanations WITHOUT raw numbers.
        Uses templated language to convey direction and significance.
        
        Args:
            metric_name: Name of the health metric
            trend_direction: Direction of the trend
            timeframe_days: Analysis timeframe in days
            
        Returns:
            SafeTrendExplanation with templated text
        """
        if timeframe_days <= 3:
            timeframe_text = "few days"
        elif timeframe_days <= 7:
            timeframe_text = "week"
        elif timeframe_days <= 14:
            timeframe_text = "two weeks"
        elif timeframe_days <= 30:
            timeframe_text = "month"
        else:
            timeframe_text = "recent period"
        
        metric_lower = metric_name.lower()
        
        if trend_direction == TrendDirection.IMPROVING:
            if "energy" in metric_lower:
                template_key = "improving_energy"
            elif "symptom" in metric_lower:
                template_key = "improving_symptoms"
            elif "adherence" in metric_lower or "medication" in metric_lower:
                template_key = "improving_adherence"
            else:
                template_key = "stable_overall"
            
            confidence = "high"
            action = "Continue what you're doing - it's working!"
        
        elif trend_direction == TrendDirection.STABLE:
            template_key = "stable_vitals"
            confidence = "high"
            action = None
        
        elif trend_direction == TrendDirection.DECLINING:
            if "sleep" in metric_lower:
                template_key = "declining_sleep"
            elif "activity" in metric_lower or "exercise" in metric_lower:
                template_key = "declining_activity"
            else:
                template_key = "stable_overall"
            
            confidence = "moderate"
            action = "Consider discussing this pattern with your care team."
        
        else:
            if "mood" in metric_lower:
                template_key = "fluctuating_mood"
            else:
                template_key = "stable_overall"
            
            confidence = "moderate"
            action = "Keep tracking - patterns become clearer over time."
        
        template = TREND_TEMPLATES.get(template_key, TREND_TEMPLATES["stable_overall"])
        explanation = template.format(
            metric=metric_name,
            timeframe=timeframe_text
        )
        
        trend_explanation = SafeTrendExplanation(
            metric_name=metric_name,
            trend_direction=trend_direction,
            explanation=explanation,
            timeframe=timeframe_text,
            confidence=confidence,
            action_suggestion=action
        )
        
        logger.debug(f"Generated safe trend explanation for {metric_name}: {trend_direction.value}")
        return trend_explanation
    
    def get_batch_trend_explanations(
        self,
        trends: List[Dict[str, Any]]
    ) -> List[SafeTrendExplanation]:
        """
        Generate multiple trend explanations at once.
        
        Args:
            trends: List of dicts with metric_name, trend_direction, timeframe_days
            
        Returns:
            List of SafeTrendExplanation objects
        """
        explanations = []
        for trend in trends:
            try:
                explanation = self.generate_safe_trend_explanation(
                    metric_name=trend.get("metric_name", "Health Metric"),
                    trend_direction=TrendDirection(trend.get("trend_direction", "stable")),
                    timeframe_days=trend.get("timeframe_days", 7)
                )
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error generating trend explanation: {e}")
        
        return explanations


_genius_patient_service: Optional[GeniusPatientService] = None


def get_genius_patient_service() -> GeniusPatientService:
    """Get or create singleton GeniusPatientService"""
    global _genius_patient_service
    if _genius_patient_service is None:
        _genius_patient_service = GeniusPatientService()
    return _genius_patient_service
