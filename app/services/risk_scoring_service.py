"""
Risk Scoring Engine for Health Deterioration Prediction

Aggregates deviations into a composite risk score (0-15 scale).
Used for wellness monitoring and change detection (NOT medical diagnosis).

Risk Score Classification:
- 0-2: Stable (no action needed)
- 3-5: Monitoring (increased awareness)
- 6-15: Urgent (discuss with healthcare provider)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
from pydantic import BaseModel

from app.models.health_baseline import BaselineDeviation
from app.models.user import User
from app.services.deviation_service import DeviationDetectionService


class RiskFactorContribution(BaseModel):
    """Individual risk factor contribution to total score"""
    metric_name: str
    z_score: float
    severity_level: str
    points: int
    description: str


class RiskScore(BaseModel):
    """Composite risk score with breakdown"""
    patient_id: str
    score: int  # 0-15 scale
    level: str  # stable, monitoring, urgent
    calculated_at: datetime
    
    # Risk factor breakdown
    factors: List[RiskFactorContribution]
    
    # Summary statistics
    total_deviations: int
    critical_deviations: int
    moderate_deviations: int
    
    # Recommendations
    recommendation: str
    action_items: List[str]


class RiskScoringService:
    """
    Service for calculating composite risk scores from health deviations.
    
    Scoring Rules (weighted by severity):
    - Respiratory rate deviation (z > 2): +3 points (critical health indicator)
    - Respiratory rate deviation (z < -1.5): +2 points (unusual decrease)
    - Pain facial deviation (z > 2): +3 points (worsening discomfort)
    - Pain self-reported deviation (z > 2): +2 points (patient-reported worsening)
    - Symptom severity deviation (z > 2): +2 points (increasing symptoms)
    - Activity impact increasing: +1 point (functional decline)
    
    Classification:
    - 0-2: Stable wellness pattern
    - 3-5: Monitoring needed (patterns changing)
    - 6-15: Urgent discussion with healthcare provider recommended
    """
    
    # Scoring weights by metric and severity
    SCORING_WEIGHTS = {
        "respiratory_rate": {
            "critical_high": 5,   # z > 3 (most critical health indicator)
            "moderate_high": 3,   # z > 2
            "moderate_low": 2,    # z < -1.5
            "critical_low": 4     # z < -2.5 (abnormally low)
        },
        "pain_facial": {
            "critical_high": 4,   # z > 3
            "moderate_high": 3,   # z > 2
            "moderate_low": 1,    # z < -1.5
            "critical_low": 2     # z < -2.5
        },
        "pain_self_reported": {
            "critical_high": 3,   # z > 3
            "moderate_high": 2,   # z > 2
            "moderate_low": 1,    # z < -1.5
            "critical_low": 2     # z < -2.5
        },
        "symptom_severity": {
            "critical_high": 3,   # z > 3
            "moderate_high": 2,   # z > 2
            "moderate_low": 1,    # z < -1.5
            "critical_low": 2     # z < -2.5
        },
        "activity_impact": {
            # Future enhancement: track activity/functional decline
            "critical_low": 3,    # z < -2.5 (severe functional decline)
            "moderate_low": 2     # z < -1.5 (functional decline)
        }
    }
    
    @staticmethod
    def calculate_patient_risk_score(
        db: Session,
        patient_id: str,
        lookback_hours: int = 24
    ) -> RiskScore:
        """
        Calculate composite risk score for a patient.
        
        Args:
            db: Database session
            patient_id: Patient identifier
            lookback_hours: Hours to look back for deviations (default: 24)
            
        Returns:
            RiskScore with breakdown and recommendations
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Get recent deviations
        recent_deviations = db.query(BaselineDeviation).filter(
            and_(
                BaselineDeviation.patient_id == patient_id,
                BaselineDeviation.measurement_date >= cutoff_time
            )
        ).order_by(desc(BaselineDeviation.measurement_date)).all()
        
        # Calculate risk score
        total_score = 0
        factors: List[RiskFactorContribution] = []
        critical_count = 0
        moderate_count = 0
        
        # Process each deviation
        for deviation in recent_deviations:
            points = RiskScoringService._score_deviation(deviation)
            
            if points > 0:
                factors.append(RiskFactorContribution(
                    metric_name=deviation.metric_name,
                    z_score=deviation.z_score,
                    severity_level=deviation.severity_level,
                    points=points,
                    description=RiskScoringService._get_factor_description(deviation)
                ))
                total_score += points
            
            # Count severity levels
            if deviation.severity_level == "critical":
                critical_count += 1
            elif deviation.severity_level == "moderate":
                moderate_count += 1
        
        # Cap score at 15 (max possible)
        final_score = min(total_score, 15)
        
        # Classify risk level
        risk_level = RiskScoringService._classify_risk_level(final_score)
        
        # Generate recommendations
        recommendation, action_items = RiskScoringService._generate_recommendations(
            final_score, risk_level, factors, critical_count
        )
        
        return RiskScore(
            patient_id=patient_id,
            score=final_score,
            level=risk_level,
            calculated_at=datetime.utcnow(),
            factors=factors,
            total_deviations=len(recent_deviations),
            critical_deviations=critical_count,
            moderate_deviations=moderate_count,
            recommendation=recommendation,
            action_items=action_items
        )
    
    @staticmethod
    def _score_deviation(deviation: BaselineDeviation) -> int:
        """
        Score a single deviation based on metric type and severity.
        
        Scoring categories:
        - critical_high: z > 3.0 (severe increase)
        - moderate_high: z > 2.0 (moderate increase)
        - critical_low: z < -2.5 (severe decrease)
        - moderate_low: z < -1.5 (moderate decrease)
        
        Returns points to add to total risk score (0-15 scale).
        """
        metric_weights = RiskScoringService.SCORING_WEIGHTS.get(deviation.metric_name)
        if not metric_weights:
            return 0
        
        z = deviation.z_score
        
        # Determine severity category and apply weights
        if z > 3.0:
            # Critical increase
            return metric_weights.get("critical_high", 0)
        elif z > 2.0:
            # Moderate increase
            return metric_weights.get("moderate_high", 0)
        elif z < -2.5:
            # Critical decrease (use critical_low, fallback to critical_high)
            return metric_weights.get("critical_low", metric_weights.get("critical_high", 0))
        elif z < -1.5:
            # Moderate decrease
            return metric_weights.get("moderate_low", 0)
        else:
            return 0
    
    @staticmethod
    def _classify_risk_level(score: int) -> str:
        """
        Classify risk level based on composite score.
        
        Returns: stable, monitoring, or urgent
        """
        if score <= 2:
            return "stable"
        elif score <= 5:
            return "monitoring"
        else:
            return "urgent"
    
    @staticmethod
    def _get_factor_description(deviation: BaselineDeviation) -> str:
        """
        Generate human-readable description of risk factor.
        Uses wellness monitoring language (NOT medical diagnosis).
        """
        metric_display = {
            "respiratory_rate": "breathing pattern",
            "pain_facial": "discomfort level",
            "pain_self_reported": "reported pain",
            "symptom_severity": "symptom intensity"
        }
        
        metric_name = metric_display.get(deviation.metric_name, deviation.metric_name)
        z = deviation.z_score
        trend = deviation.trend_direction or "stable"
        
        if z > 0:
            change = "increased"
        else:
            change = "decreased"
        
        if trend == "worsening":
            trend_desc = " and continuing to worsen"
        elif trend == "improving":
            trend_desc = " but showing signs of improvement"
        else:
            trend_desc = ""
        
        return f"Your {metric_name} has {change} significantly{trend_desc}"
    
    @staticmethod
    def _generate_recommendations(
        score: int,
        level: str,
        factors: List[RiskFactorContribution],
        critical_count: int
    ) -> Tuple[str, List[str]]:
        """
        Generate wellness recommendations based on risk score.
        
        Returns: (main_recommendation, action_items)
        """
        action_items = []
        
        if level == "stable":
            recommendation = (
                "Your health patterns are stable. Continue monitoring your wellness "
                "and maintaining healthy habits."
            )
            action_items = [
                "Keep tracking your daily measurements",
                "Maintain your current wellness routine",
                "Contact your healthcare provider for regular checkups"
            ]
        
        elif level == "monitoring":
            recommendation = (
                "Some of your health patterns are changing. Increased monitoring is recommended. "
                "These changes may be worth discussing with your healthcare provider."
            )
            action_items = [
                "Increase frequency of measurements if comfortable",
                "Note any new symptoms or changes in daily log",
                "Consider scheduling a checkup with your healthcare provider",
                "Review recent medication or lifestyle changes"
            ]
        
        else:  # urgent
            recommendation = (
                "Significant changes detected in your health patterns. "
                "We strongly recommend discussing these changes with your healthcare provider soon. "
                "Remember: This is a wellness monitoring tool, not a medical diagnosis."
            )
            action_items = [
                "Contact your healthcare provider to discuss these changes",
                "Continue taking measurements to track pattern evolution",
                "Review and share your baseline comparison with your doctor",
                "Note any additional symptoms or concerns"
            ]
            
            # Add specific concerns
            if critical_count > 0:
                action_items.insert(0, 
                    f"Priority: {critical_count} critical pattern change(s) detected"
                )
        
        # Add factor-specific action items
        respiratory_factors = [f for f in factors if "respiratory" in f.metric_name]
        pain_factors = [f for f in factors if "pain" in f.metric_name]
        
        if respiratory_factors and level in ["monitoring", "urgent"]:
            action_items.append(
                "Focus on breathing exercises and note any respiratory changes"
            )
        
        if pain_factors and level in ["monitoring", "urgent"]:
            action_items.append(
                "Track pain patterns and discuss pain management with your provider"
            )
        
        return recommendation, action_items
    
    @staticmethod
    def get_risk_score_history(
        db: Session,
        patient_id: str,
        days: int = 7
    ) -> List[Dict]:
        """
        Calculate historical risk scores (daily retrospective).
        
        Useful for trend visualization in deterioration dashboard.
        
        Args:
            db: Database session
            patient_id: Patient identifier
            days: Number of days of history to calculate
            
        Returns:
            List of {date, score, level} dictionaries
        """
        history = []
        
        for day_offset in range(days):
            # Calculate score for each day
            target_date = datetime.utcnow() - timedelta(days=day_offset)
            start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Get deviations for that day
            daily_deviations = db.query(BaselineDeviation).filter(
                and_(
                    BaselineDeviation.patient_id == patient_id,
                    BaselineDeviation.measurement_date >= start_of_day,
                    BaselineDeviation.measurement_date <= end_of_day
                )
            ).all()
            
            # Calculate score
            total_score = 0
            for deviation in daily_deviations:
                total_score += RiskScoringService._score_deviation(deviation)
            
            final_score = min(total_score, 15)
            risk_level = RiskScoringService._classify_risk_level(final_score)
            
            history.append({
                "date": start_of_day.date().isoformat(),
                "score": final_score,
                "level": risk_level,
                "deviation_count": len(daily_deviations)
            })
        
        # Return chronological order (oldest first)
        return list(reversed(history))
