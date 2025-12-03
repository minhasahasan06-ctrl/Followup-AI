"""
AI Health Alert Engine - Comprehensive trend detection, engagement monitoring, and clinician dashboard.

Features:
1. Trend Metrics: Z-scores, rolling slopes (3/7/14 day), volatility index, composite risk
2. Engagement Metrics: Adherence, check-ins, streaks, time-to-alert
3. Quality of Life: Wellness index, functional status, self-care consistency
4. Alert Generation: Trend, engagement, and QOL alerts with ML-based prioritization
5. Clinician Dashboard: Active alerts, acknowledge/dismiss workflow, audit trail

COMPLIANCE: All alerts are observational pattern alerts - NOT medical diagnoses.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from decimal import Decimal
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, desc, func

from app.database import get_db

# Health Section Analytics integration
try:
    from app.services.health_section_analytics import (
        HealthSectionAnalyticsEngine,
        HealthSection,
        TrendDirection,
        RiskLevel,
    )
    HEALTH_SECTION_ANALYTICS_AVAILABLE = True
except ImportError:
    HEALTH_SECTION_ANALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai-health-alerts", tags=["AI Health Alerts"])

COMPLIANCE_DISCLAIMER = "This is an observational pattern alert. Not a diagnosis or medical opinion."


class TrendMetricCreate(BaseModel):
    patient_id: str
    metric_name: str
    metric_category: str
    raw_value: float
    recorded_at: Optional[datetime] = None


class TrendMetricResponse(BaseModel):
    id: str
    patient_id: str
    metric_name: str
    metric_category: str
    raw_value: float
    baseline_14d_mean: Optional[float] = None
    baseline_14d_std: Optional[float] = None
    z_score: Optional[float] = None
    z_score_severity: Optional[str] = None
    slope_3d: Optional[float] = None
    slope_7d: Optional[float] = None
    slope_14d: Optional[float] = None
    slope_direction: Optional[str] = None
    volatility_index: Optional[float] = None
    volatility_level: Optional[str] = None
    composite_trend_score: Optional[float] = None
    recorded_at: datetime
    computed_at: datetime


class EngagementMetricResponse(BaseModel):
    id: str
    patient_id: str
    period_start: datetime
    period_end: datetime
    adherence_score: float
    checkins_completed: int
    checkins_expected: int
    captures_completed: int
    surveys_completed: int
    engagement_score: float
    engagement_trend: str
    engagement_drop_14d: Optional[float] = None
    current_streak: int
    longest_streak: int
    alerts_generated: int
    alerts_dismissed: int
    computed_at: datetime


class QolMetricResponse(BaseModel):
    id: str
    patient_id: str
    wellness_index: float
    wellness_components: Optional[Dict[str, float]] = None
    wellness_trend: str
    functional_status: float
    functional_components: Optional[Dict[str, Any]] = None
    selfcare_score: float
    selfcare_components: Optional[Dict[str, float]] = None
    stability_score: float
    behavior_patterns: Optional[Dict[str, float]] = None
    recorded_at: datetime
    computed_at: datetime


class HealthAlertResponse(BaseModel):
    id: str
    patient_id: str
    alert_type: str
    alert_category: str
    severity: str
    priority: int
    escalation_probability: Optional[float] = None
    title: str
    message: str
    disclaimer: str
    contributing_metrics: Optional[List[Dict[str, Any]]] = None
    trigger_rule: Optional[str] = None
    status: str
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    dismissed_by: Optional[str] = None
    dismissed_at: Optional[datetime] = None
    dismiss_reason: Optional[str] = None
    clinician_notes: Optional[str] = None
    created_at: datetime


class AlertUpdateRequest(BaseModel):
    status: str
    dismiss_reason: Optional[str] = None
    clinician_notes: Optional[str] = None


class AlertRuleConfig(BaseModel):
    z_score_threshold: float = 2.5
    slope_threshold: float = -0.5
    volatility_threshold: float = 2.0
    composite_score_threshold: float = 60.0
    engagement_drop_threshold: float = 30.0
    wellness_drop_threshold: float = 20.0
    missed_checkins_threshold: int = 3
    adherence_minimum: float = 60.0


class DashboardSummary(BaseModel):
    total_active_alerts: int
    critical_alerts: int
    high_alerts: int
    moderate_alerts: int
    low_alerts: int
    alerts_by_type: Dict[str, int]
    avg_escalation_probability: float
    patients_with_alerts: int
    recent_alerts: List[HealthAlertResponse]


class TrendComputationService:
    """Service for computing trend metrics: Z-scores, slopes, volatility"""
    
    @staticmethod
    def compute_z_score(current_value: float, values_14d: List[float]) -> tuple:
        """
        Compute Z-score: z = (current - mean_14d) / std_14d
        Returns: (z_score, mean, std, severity)
        """
        if len(values_14d) < 3:
            return None, None, None, "insufficient_data"
        
        mean_14d = np.mean(values_14d)
        std_14d = np.std(values_14d)
        
        if std_14d == 0 or std_14d < 0.001:
            return 0.0, mean_14d, std_14d, "stable"
        
        z_score = (current_value - mean_14d) / std_14d
        
        abs_z = abs(z_score)
        if abs_z >= 3.0:
            severity = "critical"
        elif abs_z >= 2.5:
            severity = "high"
        elif abs_z >= 2.0:
            severity = "elevated"
        else:
            severity = "normal"
        
        return round(z_score, 3), round(mean_14d, 4), round(std_14d, 4), severity
    
    @staticmethod
    def compute_rolling_slope(values: List[float], days: int) -> Optional[float]:
        """
        Compute rolling slope using linear regression
        Returns: slope (rate of change per day)
        """
        if len(values) < 2:
            return None
        
        recent_values = values[-days:] if len(values) >= days else values
        if len(recent_values) < 2:
            return None
        
        X = np.arange(len(recent_values)).reshape(-1, 1)
        y = np.array(recent_values)
        
        try:
            model = LinearRegression()
            model.fit(X, y)
            return round(float(model.coef_[0]), 5)
        except Exception:
            return None
    
    @staticmethod
    def get_slope_direction(slope_7d: Optional[float], slope_14d: Optional[float]) -> str:
        """Determine overall slope direction"""
        if slope_7d is None and slope_14d is None:
            return "unknown"
        
        slope = slope_7d if slope_7d is not None else (slope_14d if slope_14d is not None else 0.0)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    @staticmethod
    def compute_volatility(values_14d: List[float]) -> tuple:
        """
        Compute volatility index = std(14-day values)
        Returns: (volatility_index, volatility_level)
        """
        if len(values_14d) < 3:
            return None, "insufficient_data"
        
        volatility = np.std(values_14d)
        
        if volatility >= 3.0:
            level = "extreme"
        elif volatility >= 2.0:
            level = "high"
        elif volatility >= 1.0:
            level = "moderate"
        else:
            level = "stable"
        
        return round(volatility, 4), level
    
    @staticmethod
    def compute_composite_trend_score(
        z_score: Optional[float],
        volatility: Optional[float],
        slope_7d: Optional[float],
        engagement_drop: Optional[float] = None
    ) -> float:
        """
        Compute composite trend risk index (0-100)
        Weighted sum of: z-score magnitude, volatility, slope, engagement drop
        """
        score = 0.0
        weights_used = 0.0
        
        if z_score is not None:
            z_contribution = min(abs(z_score) * 15, 40)
            score += z_contribution * 0.35
            weights_used += 0.35
        
        if volatility is not None:
            vol_contribution = min(volatility * 10, 30)
            score += vol_contribution * 0.25
            weights_used += 0.25
        
        if slope_7d is not None:
            slope_contribution = max(-slope_7d * 20, 0)
            slope_contribution = min(slope_contribution, 20)
            score += slope_contribution * 0.25
            weights_used += 0.25
        
        if engagement_drop is not None and engagement_drop > 0:
            engagement_contribution = min(engagement_drop, 20)
            score += engagement_contribution * 0.15
            weights_used += 0.15
        
        if weights_used > 0:
            score = score / weights_used * 1.0
        
        return min(round(score, 2), 100.0)


class EngagementComputationService:
    """Service for computing engagement metrics"""
    
    @staticmethod
    def compute_adherence_score(completed: int, expected: int) -> float:
        """adherence = (completed_actions / expected_actions) * 100"""
        if expected == 0:
            return 100.0
        return round((completed / expected) * 100, 2)
    
    @staticmethod
    def compute_engagement_score(
        checkins: int,
        captures: int,
        surveys: int,
        streak: int,
        max_streak: int = 30
    ) -> float:
        """
        Composite engagement score based on multiple factors
        """
        checkin_score = min(checkins * 10, 40)
        capture_score = min(captures * 15, 30)
        survey_score = min(surveys * 10, 20)
        streak_score = min(streak / max_streak * 10, 10)
        
        return round(checkin_score + capture_score + survey_score + streak_score, 2)
    
    @staticmethod
    def compute_engagement_trend(
        current_score: float,
        previous_scores: List[float]
    ) -> tuple:
        """
        Determine engagement trend and drop percentage
        Returns: (trend, drop_percentage)
        """
        if not previous_scores:
            return "stable", 0.0
        
        avg_previous = np.mean(previous_scores)
        if avg_previous == 0:
            return "stable", 0.0
        
        change = ((current_score - avg_previous) / avg_previous) * 100
        
        if change > 10:
            trend = "improving"
        elif change < -10:
            trend = "declining"
        else:
            trend = "stable"
        
        drop = max(-change, 0)
        return trend, round(drop, 2)


class QolComputationService:
    """Service for computing quality of life metrics"""
    
    @staticmethod
    def compute_wellness_index(
        mood_score: float,
        energy_score: float,
        mobility_trend: float,
        adherence: float
    ) -> tuple:
        """
        Daily Wellness Index (0-100)
        Combine: mood, energy, mobility trend, adherence
        """
        mood_contribution = mood_score * 0.30
        energy_contribution = energy_score * 0.25
        mobility_contribution = mobility_trend * 0.25
        adherence_contribution = (adherence / 100) * 100 * 0.20
        
        wellness = mood_contribution + energy_contribution + mobility_contribution + adherence_contribution
        
        components = {
            "moodScore": round(mood_contribution, 2),
            "energyScore": round(energy_contribution, 2),
            "mobilityTrend": round(mobility_contribution, 2),
            "adherenceContribution": round(adherence_contribution, 2)
        }
        
        return min(round(wellness, 2), 100.0), components
    
    @staticmethod
    def compute_functional_status(
        activity_level: float,
        gait_speed: Optional[float],
        engagement: float
    ) -> tuple:
        """
        Functional Status Proxy (0-100)
        weighted(activity_level, gait_speed, engagement)
        """
        activity_contribution = activity_level * 0.40
        engagement_contribution = (engagement / 100) * 100 * 0.30
        
        if gait_speed is not None:
            gait_contribution = gait_speed * 0.30
        else:
            gait_contribution = 50 * 0.30
        
        functional = activity_contribution + engagement_contribution + gait_contribution
        
        components = {
            "activityLevel": round(activity_contribution, 2),
            "gaitSpeed": round(gait_contribution, 2) if gait_speed else None,
            "engagementFactor": round(engagement_contribution, 2)
        }
        
        return min(round(functional, 2), 100.0), components
    
    @staticmethod
    def compute_selfcare_score(
        medication_adherence: float,
        hydration_logs: int,
        checkin_streak: int
    ) -> tuple:
        """
        Self-care Consistency Score (0-100)
        """
        med_contribution = medication_adherence * 0.50
        hydration_contribution = min(hydration_logs * 5, 25)
        streak_contribution = min(checkin_streak * 2.5, 25)
        
        selfcare = med_contribution + hydration_contribution + streak_contribution
        
        components = {
            "medicationAdherence": round(med_contribution, 2),
            "hydrationLogs": round(hydration_contribution, 2),
            "checkinStreak": round(streak_contribution, 2)
        }
        
        return min(round(selfcare, 2), 100.0), components
    
    @staticmethod
    def compute_stability_score(
        volatility: float,
        negative_slopes: float,
        missed_checkins: int,
        penalty_per_missed: float = 5.0
    ) -> float:
        """
        Daily Stability Score = 100 - composite_volatility - negative_slopes - missed_checkins * penalty
        """
        volatility_penalty = min(volatility * 10, 30)
        slope_penalty = min(abs(negative_slopes) * 10, 30)
        checkin_penalty = min(missed_checkins * penalty_per_missed, 30)
        
        stability = 100 - volatility_penalty - slope_penalty - checkin_penalty
        return max(round(stability, 2), 0.0)
    
    @staticmethod
    def compute_behavior_patterns(
        activity_data: Dict[str, float],
        mood_data: Dict[str, float],
        checkin_data: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Organ-System Behavior Scores (statistical patterns, NOT medical)
        """
        respiratory_pattern = min(
            activity_data.get("pacing", 50) * 0.4 +
            activity_data.get("breath_rate", 50) * 0.6,
            100.0
        )
        
        fluid_pattern = min(
            activity_data.get("weight_trend", 50) * 0.5 +
            activity_data.get("activity_trend", 50) * 0.5,
            100.0
        )
        
        mood_neuro_pattern = min(
            mood_data.get("mood", 50) * 0.4 +
            mood_data.get("consistency", 50) * 0.3 +
            mood_data.get("stability", 50) * 0.3,
            100.0
        )
        
        behavioral_stability = min(
            checkin_data.get("consistency", 50) * 0.4 +
            checkin_data.get("interactions", 50) * 0.3 +
            checkin_data.get("checkins", 50) * 0.3,
            100.0
        )
        
        return {
            "respiratoryLikePattern": round(respiratory_pattern, 2),
            "fluidLikePattern": round(fluid_pattern, 2),
            "moodNeuroPattern": round(mood_neuro_pattern, 2),
            "behavioralStabilityPattern": round(behavioral_stability, 2)
        }


class AlertGenerationService:
    """Service for generating and managing health alerts"""
    
    def __init__(self, config: Optional[AlertRuleConfig] = None):
        self.config = config if config is not None else AlertRuleConfig()
    
    def check_trend_alerts(
        self,
        z_score: Optional[float],
        slope_7d: Optional[float],
        volatility: Optional[float],
        composite_score: Optional[float],
        baseline_composite: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Check for trend-based alerts
        Triggers:
        1. z-score >= 2.5
        2. 7-day slope strongly negative
        3. volatility index > threshold
        4. composite trend score crosses baseline by >20%
        """
        alerts = []
        
        if z_score is not None and abs(z_score) >= self.config.z_score_threshold:
            severity = "critical" if abs(z_score) >= 3.0 else "high"
            alerts.append({
                "alert_type": "trend",
                "alert_category": "zscore_deviation",
                "severity": severity,
                "priority": 9 if severity == "critical" else 7,
                "title": "Significant Pattern Deviation Detected",
                "message": f"A tracked metric shows deviation (z-score: {z_score:.2f}) from your established baseline. This observational pattern warrants review.",
                "trigger_rule": "z_score_threshold",
                "trigger_threshold": self.config.z_score_threshold,
                "trigger_value": z_score
            })
        
        if slope_7d is not None and slope_7d <= self.config.slope_threshold:
            alerts.append({
                "alert_type": "trend",
                "alert_category": "slope_negative",
                "severity": "high" if slope_7d <= -1.0 else "moderate",
                "priority": 7 if slope_7d <= -1.0 else 5,
                "title": "Declining Trend Observed",
                "message": f"A 7-day declining trend (slope: {slope_7d:.3f}) has been observed. This pattern indicates a change from your baseline.",
                "trigger_rule": "slope_threshold",
                "trigger_threshold": self.config.slope_threshold,
                "trigger_value": slope_7d
            })
        
        if volatility is not None and volatility >= self.config.volatility_threshold:
            alerts.append({
                "alert_type": "trend",
                "alert_category": "volatility_high",
                "severity": "moderate",
                "priority": 5,
                "title": "Increased Variability Detected",
                "message": f"Higher than usual variability (index: {volatility:.2f}) has been detected in your tracked metrics.",
                "trigger_rule": "volatility_threshold",
                "trigger_threshold": self.config.volatility_threshold,
                "trigger_value": volatility
            })
        
        if composite_score is not None and baseline_composite is not None:
            if baseline_composite > 0:
                change_pct = ((composite_score - baseline_composite) / baseline_composite) * 100
                if change_pct >= 20:
                    alerts.append({
                        "alert_type": "trend",
                        "alert_category": "composite_change",
                        "severity": "high" if change_pct >= 30 else "moderate",
                        "priority": 7 if change_pct >= 30 else 5,
                        "title": "Overall Trend Score Changed",
                        "message": f"Your composite trend score has changed by {change_pct:.1f}% from baseline.",
                        "trigger_rule": "composite_baseline_change",
                        "trigger_threshold": 20.0,
                        "trigger_value": change_pct
                    })
        
        return alerts
    
    def check_engagement_alerts(
        self,
        missed_checkins_48h: int,
        adherence: float,
        engagement_drop: float
    ) -> List[Dict[str, Any]]:
        """
        Check for engagement-based alerts
        Triggers:
        1. missed 3 check-ins in 48 hours
        2. adherence < 60%
        3. engagement drop > 30% vs 14-day baseline
        """
        alerts = []
        
        if missed_checkins_48h >= self.config.missed_checkins_threshold:
            alerts.append({
                "alert_type": "engagement",
                "alert_category": "missed_checkins",
                "severity": "moderate",
                "priority": 6,
                "title": "Check-ins Missed",
                "message": f"{missed_checkins_48h} check-ins have been missed in the past 48 hours. Regular check-ins help track your wellness patterns.",
                "trigger_rule": "missed_checkins_threshold",
                "trigger_threshold": self.config.missed_checkins_threshold,
                "trigger_value": missed_checkins_48h
            })
        
        if adherence < self.config.adherence_minimum:
            alerts.append({
                "alert_type": "engagement",
                "alert_category": "low_adherence",
                "severity": "moderate",
                "priority": 5,
                "title": "Low Adherence Observed",
                "message": f"Your current adherence score ({adherence:.1f}%) is below the recommended level. Consistent engagement helps maintain accurate tracking.",
                "trigger_rule": "adherence_minimum",
                "trigger_threshold": self.config.adherence_minimum,
                "trigger_value": adherence
            })
        
        if engagement_drop >= self.config.engagement_drop_threshold:
            alerts.append({
                "alert_type": "engagement",
                "alert_category": "engagement_decline",
                "severity": "moderate",
                "priority": 5,
                "title": "Engagement Decline Detected",
                "message": f"Your engagement has dropped by {engagement_drop:.1f}% compared to your 14-day baseline.",
                "trigger_rule": "engagement_drop_threshold",
                "trigger_threshold": self.config.engagement_drop_threshold,
                "trigger_value": engagement_drop
            })
        
        return alerts
    
    def check_qol_alerts(
        self,
        wellness_index: float,
        previous_wellness: Optional[float],
        functional_status: float,
        previous_functional: Optional[float],
        selfcare_score: float
    ) -> List[Dict[str, Any]]:
        """
        Check for quality-of-life alerts
        Triggers:
        1. wellness index drops by >20 points
        2. functional status drops >30% in 3 days
        3. self-care consistency drops below threshold
        """
        alerts = []
        
        if previous_wellness is not None:
            wellness_drop = previous_wellness - wellness_index
            if wellness_drop >= self.config.wellness_drop_threshold:
                alerts.append({
                    "alert_type": "qol",
                    "alert_category": "wellness_drop",
                    "severity": "high" if wellness_drop >= 30 else "moderate",
                    "priority": 7 if wellness_drop >= 30 else 5,
                    "title": "Wellness Index Decline",
                    "message": f"Your wellness index has dropped by {wellness_drop:.1f} points. This reflects changes in mood, energy, and activity patterns.",
                    "trigger_rule": "wellness_drop_threshold",
                    "trigger_threshold": self.config.wellness_drop_threshold,
                    "trigger_value": wellness_drop
                })
        
        if previous_functional is not None and previous_functional > 0:
            functional_drop_pct = ((previous_functional - functional_status) / previous_functional) * 100
            if functional_drop_pct >= 30:
                alerts.append({
                    "alert_type": "qol",
                    "alert_category": "functional_decline",
                    "severity": "high",
                    "priority": 7,
                    "title": "Functional Status Change",
                    "message": f"Your functional status proxy has declined by {functional_drop_pct:.1f}% over the recent period.",
                    "trigger_rule": "functional_drop_threshold",
                    "trigger_threshold": 30.0,
                    "trigger_value": functional_drop_pct
                })
        
        if selfcare_score < 40:
            alerts.append({
                "alert_type": "qol",
                "alert_category": "selfcare_low",
                "severity": "moderate",
                "priority": 4,
                "title": "Self-care Attention Needed",
                "message": f"Your self-care consistency score ({selfcare_score:.1f}) indicates room for improvement in medication adherence and daily routines.",
                "trigger_rule": "selfcare_threshold",
                "trigger_threshold": 40.0,
                "trigger_value": selfcare_score
            })
        
        return alerts


@router.get("/trend-metrics/{patient_id}", response_model=List[TrendMetricResponse])
async def get_trend_metrics(
    patient_id: str,
    metric_name: Optional[str] = None,
    days: int = Query(default=14, le=90),
    db: Session = Depends(get_db)
):
    """Get trend metrics for a patient with optional filtering"""
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT id, patient_id, metric_name, metric_category, raw_value,
                   baseline_14d_mean, baseline_14d_std, z_score, z_score_severity,
                   slope_3d, slope_7d, slope_14d, slope_direction,
                   volatility_index, volatility_level, composite_trend_score,
                   recorded_at, computed_at
            FROM ai_trend_metrics
            WHERE patient_id = :patient_id
            AND recorded_at >= :cutoff
            """ + (" AND metric_name = :metric_name" if metric_name else "") + """
            ORDER BY recorded_at DESC
            LIMIT 500
        """)
        
        params = {"patient_id": patient_id, "cutoff": cutoff}
        if metric_name:
            params["metric_name"] = metric_name
        
        result = db.execute(query, params)
        rows = result.fetchall()
        
        return [
            TrendMetricResponse(
                id=str(row[0]),
                patient_id=row[1],
                metric_name=row[2],
                metric_category=row[3],
                raw_value=float(row[4]) if row[4] else 0,
                baseline_14d_mean=float(row[5]) if row[5] else None,
                baseline_14d_std=float(row[6]) if row[6] else None,
                z_score=float(row[7]) if row[7] else None,
                z_score_severity=row[8],
                slope_3d=float(row[9]) if row[9] else None,
                slope_7d=float(row[10]) if row[10] else None,
                slope_14d=float(row[11]) if row[11] else None,
                slope_direction=row[12],
                volatility_index=float(row[13]) if row[13] else None,
                volatility_level=row[14],
                composite_trend_score=float(row[15]) if row[15] else None,
                recorded_at=row[16],
                computed_at=row[17]
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching trend metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trend-metrics/compute/{patient_id}")
async def compute_trend_metrics(
    patient_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Compute and store trend metrics for all tracked data sources"""
    try:
        metrics_computed = []
        trend_service = TrendComputationService()
        cutoff_14d = datetime.utcnow() - timedelta(days=14)
        
        checkins_query = text("""
            SELECT pain_level, fatigue_level, breathlessness_level, sleep_quality, 
                   mobility_score, mood, created_at
            FROM symptom_checkins
            WHERE user_id = :patient_id
            AND created_at >= :cutoff
            ORDER BY created_at ASC
        """)
        checkins = db.execute(checkins_query, {"patient_id": patient_id, "cutoff": cutoff_14d}).fetchall()
        
        if checkins:
            metric_mappings = [
                ("pain_level", "symptom", 0),
                ("fatigue_level", "symptom", 1),
                ("breathlessness_level", "symptom", 2),
                ("sleep_quality", "symptom", 3),
                ("mobility_score", "symptom", 4),
                ("mood", "behavioral", 5)
            ]
            
            for metric_name, category, idx in metric_mappings:
                values = [float(row[idx]) for row in checkins if row[idx] is not None]
                if len(values) >= 2:
                    current_value = values[-1]
                    z_score, mean, std, severity = trend_service.compute_z_score(current_value, values)
                    slope_3d = trend_service.compute_rolling_slope(values, 3)
                    slope_7d = trend_service.compute_rolling_slope(values, 7)
                    slope_14d = trend_service.compute_rolling_slope(values, 14)
                    slope_direction = trend_service.get_slope_direction(slope_7d, slope_14d)
                    volatility, vol_level = trend_service.compute_volatility(values)
                    composite = trend_service.compute_composite_trend_score(z_score, volatility, slope_7d)
                    
                    insert_query = text("""
                        INSERT INTO ai_trend_metrics (
                            id, patient_id, metric_name, metric_category, raw_value,
                            baseline_14d_mean, baseline_14d_std, z_score, z_score_severity,
                            slope_3d, slope_7d, slope_14d, slope_direction,
                            volatility_index, volatility_level, composite_trend_score,
                            recorded_at, computed_at
                        ) VALUES (
                            gen_random_uuid(), :patient_id, :metric_name, :category, :raw_value,
                            :mean, :std, :z_score, :severity,
                            :slope_3d, :slope_7d, :slope_14d, :slope_direction,
                            :volatility, :vol_level, :composite,
                            :recorded_at, NOW()
                        )
                    """)
                    
                    db.execute(insert_query, {
                        "patient_id": patient_id,
                        "metric_name": metric_name,
                        "category": category,
                        "raw_value": current_value,
                        "mean": mean,
                        "std": std,
                        "z_score": z_score,
                        "severity": severity,
                        "slope_3d": slope_3d,
                        "slope_7d": slope_7d,
                        "slope_14d": slope_14d,
                        "slope_direction": slope_direction,
                        "volatility": volatility,
                        "vol_level": vol_level,
                        "composite": composite,
                        "recorded_at": checkins[-1][6]
                    })
                    
                    metrics_computed.append({
                        "metric_name": metric_name,
                        "z_score": z_score,
                        "slope_7d": slope_7d,
                        "volatility": volatility,
                        "composite_score": composite
                    })
        
        db.commit()
        
        return {
            "success": True,
            "patient_id": patient_id,
            "metrics_computed": len(metrics_computed),
            "details": metrics_computed
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error computing trend metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/engagement-metrics/{patient_id}", response_model=List[EngagementMetricResponse])
async def get_engagement_metrics(
    patient_id: str,
    days: int = Query(default=30, le=90),
    db: Session = Depends(get_db)
):
    """Get engagement metrics for a patient"""
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT id, patient_id, period_start, period_end, adherence_score,
                   checkins_completed, checkins_expected, captures_completed, surveys_completed,
                   engagement_score, engagement_trend, engagement_drop_14d,
                   current_streak, longest_streak, alerts_generated, alerts_dismissed, computed_at
            FROM ai_engagement_metrics
            WHERE patient_id = :patient_id
            AND period_end >= :cutoff
            ORDER BY period_end DESC
        """)
        
        result = db.execute(query, {"patient_id": patient_id, "cutoff": cutoff})
        rows = result.fetchall()
        
        return [
            EngagementMetricResponse(
                id=str(row[0]),
                patient_id=row[1],
                period_start=row[2],
                period_end=row[3],
                adherence_score=float(row[4]) if row[4] else 0,
                checkins_completed=row[5] or 0,
                checkins_expected=row[6] or 0,
                captures_completed=row[7] or 0,
                surveys_completed=row[8] or 0,
                engagement_score=float(row[9]) if row[9] else 0,
                engagement_trend=row[10] or "stable",
                engagement_drop_14d=float(row[11]) if row[11] else None,
                current_streak=row[12] or 0,
                longest_streak=row[13] or 0,
                alerts_generated=row[14] or 0,
                alerts_dismissed=row[15] or 0,
                computed_at=row[16]
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching engagement metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/engagement-metrics/compute/{patient_id}")
async def compute_engagement_metrics(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Compute engagement metrics for a patient"""
    try:
        eng_service = EngagementComputationService()
        now = datetime.utcnow()
        period_start = now - timedelta(days=7)
        
        checkins_query = text("""
            SELECT COUNT(*) FROM symptom_checkins
            WHERE user_id = :patient_id AND created_at >= :start
        """)
        checkins_completed = db.execute(checkins_query, {"patient_id": patient_id, "start": period_start}).scalar() or 0
        
        checkins_expected = 7
        
        captures_query = text("""
            SELECT COUNT(*) FROM video_exam_sessions
            WHERE patient_id = :patient_id AND started_at >= :start
        """)
        captures_completed = db.execute(captures_query, {"patient_id": patient_id, "start": period_start}).scalar() or 0
        
        surveys_query = text("""
            SELECT COUNT(*) FROM mental_health_responses
            WHERE patient_id = :patient_id AND completed_at >= :start
        """)
        surveys_completed = db.execute(surveys_query, {"patient_id": patient_id, "start": period_start}).scalar() or 0
        
        streak_query = text("""
            SELECT DISTINCT DATE(created_at) as day
            FROM symptom_checkins
            WHERE user_id = :patient_id
            ORDER BY day DESC
        """)
        streak_days = [row[0] for row in db.execute(streak_query, {"patient_id": patient_id}).fetchall()]
        
        current_streak = 0
        if streak_days:
            today = datetime.utcnow().date()
            for i, day in enumerate(streak_days):
                expected_day = today - timedelta(days=i)
                if day == expected_day:
                    current_streak += 1
                else:
                    break
        
        adherence_score = eng_service.compute_adherence_score(
            checkins_completed + captures_completed + surveys_completed,
            checkins_expected + 2 + 1
        )
        
        engagement_score = eng_service.compute_engagement_score(
            checkins_completed, captures_completed, surveys_completed, current_streak
        )
        
        prev_scores_query = text("""
            SELECT engagement_score FROM ai_engagement_metrics
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC LIMIT 4
        """)
        prev_scores = [float(row[0]) for row in db.execute(prev_scores_query, {"patient_id": patient_id}).fetchall() if row[0]]
        
        engagement_trend, engagement_drop = eng_service.compute_engagement_trend(engagement_score, prev_scores)
        
        alerts_query = text("""
            SELECT 
                COUNT(*) FILTER (WHERE status != 'dismissed') as generated,
                COUNT(*) FILTER (WHERE status = 'dismissed') as dismissed
            FROM ai_health_alerts
            WHERE patient_id = :patient_id AND created_at >= :start
        """)
        alerts_result = db.execute(alerts_query, {"patient_id": patient_id, "start": period_start}).fetchone()
        alerts_generated = alerts_result[0] if alerts_result else 0
        alerts_dismissed = alerts_result[1] if alerts_result else 0
        
        insert_query = text("""
            INSERT INTO ai_engagement_metrics (
                id, patient_id, period_start, period_end, adherence_score,
                checkins_completed, checkins_expected, captures_completed, surveys_completed,
                engagement_score, engagement_trend, engagement_drop_14d,
                current_streak, longest_streak, alerts_generated, alerts_dismissed, computed_at
            ) VALUES (
                gen_random_uuid(), :patient_id, :period_start, :period_end, :adherence_score,
                :checkins_completed, :checkins_expected, :captures_completed, :surveys_completed,
                :engagement_score, :engagement_trend, :engagement_drop,
                :current_streak, :longest_streak, :alerts_generated, :alerts_dismissed, NOW()
            )
        """)
        
        db.execute(insert_query, {
            "patient_id": patient_id,
            "period_start": period_start,
            "period_end": now,
            "adherence_score": adherence_score,
            "checkins_completed": checkins_completed,
            "checkins_expected": checkins_expected,
            "captures_completed": captures_completed,
            "surveys_completed": surveys_completed,
            "engagement_score": engagement_score,
            "engagement_trend": engagement_trend,
            "engagement_drop": engagement_drop,
            "current_streak": current_streak,
            "longest_streak": max(current_streak, 0),
            "alerts_generated": alerts_generated,
            "alerts_dismissed": alerts_dismissed
        })
        
        db.commit()
        
        return {
            "success": True,
            "patient_id": patient_id,
            "adherence_score": adherence_score,
            "engagement_score": engagement_score,
            "engagement_trend": engagement_trend,
            "current_streak": current_streak
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error computing engagement metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/qol-metrics/{patient_id}", response_model=List[QolMetricResponse])
async def get_qol_metrics(
    patient_id: str,
    days: int = Query(default=30, le=90),
    db: Session = Depends(get_db)
):
    """Get quality of life metrics for a patient"""
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT id, patient_id, wellness_index, wellness_components, wellness_trend,
                   functional_status, functional_components, selfcare_score, selfcare_components,
                   stability_score, behavior_patterns, recorded_at, computed_at
            FROM ai_qol_metrics
            WHERE patient_id = :patient_id
            AND recorded_at >= :cutoff
            ORDER BY recorded_at DESC
        """)
        
        result = db.execute(query, {"patient_id": patient_id, "cutoff": cutoff})
        rows = result.fetchall()
        
        return [
            QolMetricResponse(
                id=str(row[0]),
                patient_id=row[1],
                wellness_index=float(row[2]) if row[2] else 0,
                wellness_components=row[3],
                wellness_trend=row[4] or "stable",
                functional_status=float(row[5]) if row[5] else 0,
                functional_components=row[6],
                selfcare_score=float(row[7]) if row[7] else 0,
                selfcare_components=row[8],
                stability_score=float(row[9]) if row[9] else 0,
                behavior_patterns=row[10],
                recorded_at=row[11],
                computed_at=row[12]
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching QoL metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/qol-metrics/compute/{patient_id}")
async def compute_qol_metrics(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Compute quality of life metrics for a patient"""
    try:
        qol_service = QolComputationService()
        now = datetime.utcnow()
        
        latest_checkin_query = text("""
            SELECT mood, fatigue_level, mobility_score, created_at
            FROM symptom_checkins
            WHERE user_id = :patient_id
            ORDER BY created_at DESC LIMIT 1
        """)
        latest_checkin = db.execute(latest_checkin_query, {"patient_id": patient_id}).fetchone()
        
        mood_score = float(latest_checkin[0]) if latest_checkin and latest_checkin[0] else 50
        energy_score = 100 - float(latest_checkin[1]) * 10 if latest_checkin and latest_checkin[1] else 50
        mobility_score = float(latest_checkin[2]) * 10 if latest_checkin and latest_checkin[2] else 50
        
        engagement_query = text("""
            SELECT adherence_score, engagement_score, current_streak
            FROM ai_engagement_metrics
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC LIMIT 1
        """)
        engagement_data = db.execute(engagement_query, {"patient_id": patient_id}).fetchone()
        
        adherence = float(engagement_data[0]) if engagement_data and engagement_data[0] else 50
        engagement = float(engagement_data[1]) if engagement_data and engagement_data[1] else 50
        streak = engagement_data[2] if engagement_data else 0
        
        wellness_index, wellness_components = qol_service.compute_wellness_index(
            mood_score, energy_score, mobility_score, adherence
        )
        
        functional_status, functional_components = qol_service.compute_functional_status(
            mobility_score, None, engagement
        )
        
        selfcare_score, selfcare_components = qol_service.compute_selfcare_score(
            adherence, 3, streak
        )
        
        trend_query = text("""
            SELECT AVG(volatility_index), AVG(slope_7d)
            FROM ai_trend_metrics
            WHERE patient_id = :patient_id
            AND computed_at >= NOW() - INTERVAL '7 days'
        """)
        trend_data = db.execute(trend_query, {"patient_id": patient_id}).fetchone()
        avg_volatility = float(trend_data[0]) if trend_data and trend_data[0] else 0
        avg_slope = float(trend_data[1]) if trend_data and trend_data[1] else 0
        
        stability_score = qol_service.compute_stability_score(
            avg_volatility, avg_slope, max(0, 7 - streak)
        )
        
        behavior_patterns = qol_service.compute_behavior_patterns(
            {"pacing": mobility_score, "breath_rate": 50, "weight_trend": 50, "activity_trend": mobility_score},
            {"mood": mood_score, "consistency": min(streak * 10, 100), "stability": stability_score},
            {"consistency": int(min(streak * 10, 100)), "interactions": int(engagement), "checkins": int(min(streak * 10, 100))}
        )
        
        prev_qol_query = text("""
            SELECT wellness_index FROM ai_qol_metrics
            WHERE patient_id = :patient_id
            ORDER BY recorded_at DESC LIMIT 1
        """)
        prev_qol = db.execute(prev_qol_query, {"patient_id": patient_id}).fetchone()
        prev_wellness = float(prev_qol[0]) if prev_qol and prev_qol[0] else None
        
        if prev_wellness:
            change = wellness_index - prev_wellness
            if change > 5:
                wellness_trend = "improving"
            elif change < -5:
                wellness_trend = "declining"
            else:
                wellness_trend = "stable"
        else:
            wellness_trend = "stable"
        
        import json
        insert_query = text("""
            INSERT INTO ai_qol_metrics (
                id, patient_id, wellness_index, wellness_components, wellness_trend,
                functional_status, functional_components, selfcare_score, selfcare_components,
                stability_score, behavior_patterns, recorded_at, computed_at
            ) VALUES (
                gen_random_uuid(), :patient_id, :wellness_index, :wellness_components::jsonb, :wellness_trend,
                :functional_status, :functional_components::jsonb, :selfcare_score, :selfcare_components::jsonb,
                :stability_score, :behavior_patterns::jsonb, :recorded_at, NOW()
            )
        """)
        
        db.execute(insert_query, {
            "patient_id": patient_id,
            "wellness_index": wellness_index,
            "wellness_components": json.dumps(wellness_components),
            "wellness_trend": wellness_trend,
            "functional_status": functional_status,
            "functional_components": json.dumps(functional_components),
            "selfcare_score": selfcare_score,
            "selfcare_components": json.dumps(selfcare_components),
            "stability_score": stability_score,
            "behavior_patterns": json.dumps(behavior_patterns),
            "recorded_at": now
        })
        
        db.commit()
        
        return {
            "success": True,
            "patient_id": patient_id,
            "wellness_index": wellness_index,
            "functional_status": functional_status,
            "selfcare_score": selfcare_score,
            "stability_score": stability_score
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error computing QoL metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts/{patient_id}", response_model=List[HealthAlertResponse])
async def get_health_alerts(
    patient_id: str,
    status: Optional[str] = None,
    alert_type: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db)
):
    """Get health alerts for a patient with optional filtering"""
    try:
        filters = ["patient_id = :patient_id"]
        params = {"patient_id": patient_id, "limit": limit}
        
        if status:
            filters.append("status = :status")
            params["status"] = status
        if alert_type:
            filters.append("alert_type = :alert_type")
            params["alert_type"] = alert_type
        if severity:
            filters.append("severity = :severity")
            params["severity"] = severity
        
        query = text(f"""
            SELECT id, patient_id, alert_type, alert_category, severity, priority,
                   escalation_probability, title, message, disclaimer,
                   contributing_metrics, trigger_rule, status,
                   acknowledged_by, acknowledged_at, dismissed_by, dismissed_at,
                   dismiss_reason, clinician_notes, created_at
            FROM ai_health_alerts
            WHERE {' AND '.join(filters)}
            ORDER BY priority DESC, created_at DESC
            LIMIT :limit
        """)
        
        result = db.execute(query, params)
        rows = result.fetchall()
        
        return [
            HealthAlertResponse(
                id=str(row[0]),
                patient_id=row[1],
                alert_type=row[2],
                alert_category=row[3],
                severity=row[4],
                priority=row[5],
                escalation_probability=float(row[6]) if row[6] else None,
                title=row[7],
                message=row[8],
                disclaimer=row[9],
                contributing_metrics=row[10],
                trigger_rule=row[11],
                status=row[12],
                acknowledged_by=row[13],
                acknowledged_at=row[14],
                dismissed_by=row[15],
                dismissed_at=row[16],
                dismiss_reason=row[17],
                clinician_notes=row[18],
                created_at=row[19]
            )
            for row in rows
        ]
    except Exception as e:
        logger.error(f"Error fetching health alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/generate/{patient_id}")
async def generate_alerts(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Generate alerts based on current metrics"""
    try:
        alert_service = AlertGenerationService()
        alerts_created = []
        
        trend_query = text("""
            SELECT metric_name, z_score, slope_7d, volatility_index, composite_trend_score
            FROM ai_trend_metrics
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC LIMIT 10
        """)
        trend_metrics = db.execute(trend_query, {"patient_id": patient_id}).fetchall()
        
        for metric in trend_metrics:
            z_score = float(metric[1]) if metric[1] else None
            slope_7d = float(metric[2]) if metric[2] else None
            volatility = float(metric[3]) if metric[3] else None
            composite = float(metric[4]) if metric[4] else None
            
            trend_alerts = alert_service.check_trend_alerts(
                z_score, slope_7d, volatility, composite
            )
            
            for alert_data in trend_alerts:
                alert_data["contributing_metrics"] = [{
                    "metricName": metric[0],
                    "value": float(metric[4]) if metric[4] else 0,
                    "zScore": z_score,
                    "slope": slope_7d,
                    "threshold": alert_data.get("trigger_threshold", 0),
                    "contribution": "primary"
                }]
                alerts_created.append(alert_data)
        
        engagement_query = text("""
            SELECT adherence_score, engagement_drop_14d, checkins_expected - checkins_completed as missed
            FROM ai_engagement_metrics
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC LIMIT 1
        """)
        engagement = db.execute(engagement_query, {"patient_id": patient_id}).fetchone()
        
        if engagement:
            adherence = float(engagement[0]) if engagement[0] else 100
            engagement_drop = float(engagement[1]) if engagement[1] else 0
            missed = engagement[2] if engagement[2] else 0
            
            engagement_alerts = alert_service.check_engagement_alerts(
                missed, adherence, engagement_drop
            )
            alerts_created.extend(engagement_alerts)
        
        qol_query = text("""
            SELECT wellness_index, functional_status, selfcare_score
            FROM ai_qol_metrics
            WHERE patient_id = :patient_id
            ORDER BY recorded_at DESC LIMIT 2
        """)
        qol_metrics = db.execute(qol_query, {"patient_id": patient_id}).fetchall()
        
        if qol_metrics:
            current_wellness = float(qol_metrics[0][0]) if qol_metrics[0][0] else 50
            current_functional = float(qol_metrics[0][1]) if qol_metrics[0][1] else 50
            selfcare = float(qol_metrics[0][2]) if qol_metrics[0][2] else 50
            
            prev_wellness = float(qol_metrics[1][0]) if len(qol_metrics) > 1 and qol_metrics[1][0] else None
            prev_functional = float(qol_metrics[1][1]) if len(qol_metrics) > 1 and qol_metrics[1][1] else None
            
            qol_alerts = alert_service.check_qol_alerts(
                current_wellness, prev_wellness,
                current_functional, prev_functional,
                selfcare
            )
            alerts_created.extend(qol_alerts)
        
        import json
        for alert_data in alerts_created:
            existing_query = text("""
                SELECT COUNT(*) FROM ai_health_alerts
                WHERE patient_id = :patient_id
                AND alert_category = :category
                AND status IN ('new', 'acknowledged')
                AND created_at >= NOW() - INTERVAL '24 hours'
            """)
            existing = db.execute(existing_query, {
                "patient_id": patient_id,
                "category": alert_data["alert_category"]
            }).scalar()
            
            if existing == 0:
                insert_query = text("""
                    INSERT INTO ai_health_alerts (
                        id, patient_id, alert_type, alert_category, severity, priority,
                        escalation_probability, title, message, disclaimer,
                        contributing_metrics, trigger_rule, trigger_threshold, trigger_value,
                        status, created_at, updated_at
                    ) VALUES (
                        gen_random_uuid(), :patient_id, :alert_type, :alert_category, :severity, :priority,
                        :escalation_prob, :title, :message, :disclaimer,
                        :contributing_metrics::jsonb, :trigger_rule, :trigger_threshold, :trigger_value,
                        'new', NOW(), NOW()
                    )
                """)
                
                db.execute(insert_query, {
                    "patient_id": patient_id,
                    "alert_type": alert_data["alert_type"],
                    "alert_category": alert_data["alert_category"],
                    "severity": alert_data["severity"],
                    "priority": alert_data["priority"],
                    "escalation_prob": alert_data["priority"] / 10.0,
                    "title": alert_data["title"],
                    "message": alert_data["message"],
                    "disclaimer": COMPLIANCE_DISCLAIMER,
                    "contributing_metrics": json.dumps(alert_data.get("contributing_metrics", [])),
                    "trigger_rule": alert_data.get("trigger_rule"),
                    "trigger_threshold": alert_data.get("trigger_threshold"),
                    "trigger_value": alert_data.get("trigger_value")
                })
        
        db.commit()
        
        return {
            "success": True,
            "patient_id": patient_id,
            "alerts_generated": len(alerts_created),
            "alerts": alerts_created
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error generating alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/alerts/{alert_id}")
async def update_alert(
    alert_id: str,
    update: AlertUpdateRequest,
    clinician_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update alert status (acknowledge, dismiss, add notes)"""
    try:
        if update.status == "acknowledged":
            query = text("""
                UPDATE ai_health_alerts
                SET status = :status, acknowledged_by = :clinician_id, acknowledged_at = NOW(),
                    clinician_notes = COALESCE(:notes, clinician_notes), updated_at = NOW()
                WHERE id = :alert_id
            """)
        elif update.status == "dismissed":
            query = text("""
                UPDATE ai_health_alerts
                SET status = :status, dismissed_by = :clinician_id, dismissed_at = NOW(),
                    dismiss_reason = :dismiss_reason, clinician_notes = COALESCE(:notes, clinician_notes),
                    updated_at = NOW()
                WHERE id = :alert_id
            """)
        else:
            query = text("""
                UPDATE ai_health_alerts
                SET status = :status, clinician_notes = COALESCE(:notes, clinician_notes),
                    updated_at = NOW()
                WHERE id = :alert_id
            """)
        
        db.execute(query, {
            "alert_id": alert_id,
            "status": update.status,
            "clinician_id": clinician_id,
            "dismiss_reason": update.dismiss_reason,
            "notes": update.clinician_notes
        })
        
        db.commit()
        
        return {"success": True, "alert_id": alert_id, "new_status": update.status}
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def get_dashboard_summary(
    clinician_id: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get clinician dashboard summary"""
    try:
        summary_query = text("""
            SELECT 
                COUNT(*) FILTER (WHERE status IN ('new', 'acknowledged')) as total_active,
                COUNT(*) FILTER (WHERE severity = 'critical' AND status IN ('new', 'acknowledged')) as critical,
                COUNT(*) FILTER (WHERE severity = 'high' AND status IN ('new', 'acknowledged')) as high,
                COUNT(*) FILTER (WHERE severity = 'moderate' AND status IN ('new', 'acknowledged')) as moderate,
                COUNT(*) FILTER (WHERE severity = 'low' AND status IN ('new', 'acknowledged')) as low,
                AVG(escalation_probability) FILTER (WHERE status IN ('new', 'acknowledged')) as avg_escalation,
                COUNT(DISTINCT patient_id) FILTER (WHERE status IN ('new', 'acknowledged')) as patients_with_alerts
            FROM ai_health_alerts
            WHERE created_at >= NOW() - INTERVAL '7 days'
        """)
        
        summary = db.execute(summary_query).fetchone()
        
        type_query = text("""
            SELECT alert_type, COUNT(*)
            FROM ai_health_alerts
            WHERE status IN ('new', 'acknowledged')
            AND created_at >= NOW() - INTERVAL '7 days'
            GROUP BY alert_type
        """)
        
        type_results = db.execute(type_query).fetchall()
        alerts_by_type = {row[0]: row[1] for row in type_results}
        
        recent_query = text("""
            SELECT id, patient_id, alert_type, alert_category, severity, priority,
                   escalation_probability, title, message, disclaimer,
                   contributing_metrics, trigger_rule, status,
                   acknowledged_by, acknowledged_at, dismissed_by, dismissed_at,
                   dismiss_reason, clinician_notes, created_at
            FROM ai_health_alerts
            WHERE status IN ('new', 'acknowledged')
            ORDER BY priority DESC, created_at DESC
            LIMIT 20
        """)
        
        recent_results = db.execute(recent_query).fetchall()
        recent_alerts = [
            HealthAlertResponse(
                id=str(row[0]),
                patient_id=row[1],
                alert_type=row[2],
                alert_category=row[3],
                severity=row[4],
                priority=row[5],
                escalation_probability=float(row[6]) if row[6] else None,
                title=row[7],
                message=row[8],
                disclaimer=row[9],
                contributing_metrics=row[10],
                trigger_rule=row[11],
                status=row[12],
                acknowledged_by=row[13],
                acknowledged_at=row[14],
                dismissed_by=row[15],
                dismissed_at=row[16],
                dismiss_reason=row[17],
                clinician_notes=row[18],
                created_at=row[19]
            )
            for row in recent_results
        ]
        
        if summary is None:
            return DashboardSummary(
                total_active_alerts=0,
                critical_alerts=0,
                high_alerts=0,
                moderate_alerts=0,
                low_alerts=0,
                alerts_by_type=alerts_by_type,
                avg_escalation_probability=0,
                patients_with_alerts=0,
                recent_alerts=recent_alerts
            )
        
        return DashboardSummary(
            total_active_alerts=summary[0] or 0,
            critical_alerts=summary[1] or 0,
            high_alerts=summary[2] or 0,
            moderate_alerts=summary[3] or 0,
            low_alerts=summary[4] or 0,
            alerts_by_type=alerts_by_type,
            avg_escalation_probability=float(summary[5]) if summary[5] else 0,
            patients_with_alerts=summary[6] or 0,
            recent_alerts=recent_alerts
        )
        
    except Exception as e:
        logger.error(f"Error fetching dashboard summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute-all/{patient_id}")
async def compute_all_metrics(
    patient_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Compute all metrics for a patient:
    1. Trend metrics (z-scores, slopes, volatility)
    2. Engagement metrics (adherence, streaks)
    3. QoL metrics (wellness, functional status)
    4. Generate alerts based on computed metrics
    """
    try:
        trend_result = await compute_trend_metrics(patient_id, background_tasks, db)
        
        engagement_result = await compute_engagement_metrics(patient_id, db)
        
        qol_result = await compute_qol_metrics(patient_id, db)
        
        alerts_result = await generate_alerts(patient_id, db)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "trend_metrics": trend_result,
            "engagement_metrics": engagement_result,
            "qol_metrics": qol_result,
            "alerts": alerts_result
        }
        
    except Exception as e:
        logger.error(f"Error computing all metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/patient-overview/{patient_id}")
async def get_patient_overview(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Get comprehensive overview for a single patient"""
    try:
        latest_trend_query = text("""
            SELECT metric_name, raw_value, z_score, z_score_severity, 
                   slope_7d, volatility_index, composite_trend_score, recorded_at
            FROM ai_trend_metrics
            WHERE patient_id = :patient_id
            AND computed_at = (
                SELECT MAX(computed_at) FROM ai_trend_metrics WHERE patient_id = :patient_id
            )
        """)
        trend_metrics = db.execute(latest_trend_query, {"patient_id": patient_id}).fetchall()
        
        engagement_query = text("""
            SELECT adherence_score, engagement_score, engagement_trend,
                   current_streak, checkins_completed, computed_at
            FROM ai_engagement_metrics
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC LIMIT 1
        """)
        engagement = db.execute(engagement_query, {"patient_id": patient_id}).fetchone()
        
        qol_query = text("""
            SELECT wellness_index, wellness_trend, functional_status,
                   selfcare_score, stability_score, behavior_patterns, recorded_at
            FROM ai_qol_metrics
            WHERE patient_id = :patient_id
            ORDER BY recorded_at DESC LIMIT 1
        """)
        qol = db.execute(qol_query, {"patient_id": patient_id}).fetchone()
        
        alerts_query = text("""
            SELECT id, alert_type, alert_category, severity, priority, title, status, created_at
            FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND status IN ('new', 'acknowledged')
            ORDER BY priority DESC, created_at DESC
            LIMIT 10
        """)
        active_alerts = db.execute(alerts_query, {"patient_id": patient_id}).fetchall()
        
        return {
            "patient_id": patient_id,
            "trend_metrics": [
                {
                    "metric_name": row[0],
                    "raw_value": float(row[1]) if row[1] else None,
                    "z_score": float(row[2]) if row[2] else None,
                    "z_score_severity": row[3],
                    "slope_7d": float(row[4]) if row[4] else None,
                    "volatility": float(row[5]) if row[5] else None,
                    "composite_score": float(row[6]) if row[6] else None,
                    "recorded_at": row[7].isoformat() if row[7] else None
                }
                for row in trend_metrics
            ],
            "engagement": {
                "adherence_score": float(engagement[0]) if engagement and engagement[0] else None,
                "engagement_score": float(engagement[1]) if engagement and engagement[1] else None,
                "trend": engagement[2] if engagement else None,
                "current_streak": engagement[3] if engagement else 0,
                "checkins_completed": engagement[4] if engagement else 0,
                "computed_at": engagement[5].isoformat() if engagement and engagement[5] else None
            } if engagement else None,
            "quality_of_life": {
                "wellness_index": float(qol[0]) if qol and qol[0] else None,
                "wellness_trend": qol[1] if qol else None,
                "functional_status": float(qol[2]) if qol and qol[2] else None,
                "selfcare_score": float(qol[3]) if qol and qol[3] else None,
                "stability_score": float(qol[4]) if qol and qol[4] else None,
                "behavior_patterns": qol[5] if qol else None,
                "recorded_at": qol[6].isoformat() if qol and qol[6] else None
            } if qol else None,
            "active_alerts": [
                {
                    "id": str(row[0]),
                    "alert_type": row[1],
                    "alert_category": row[2],
                    "severity": row[3],
                    "priority": row[4],
                    "title": row[5],
                    "status": row[6],
                    "created_at": row[7].isoformat() if row[7] else None
                }
                for row in active_alerts
            ],
            "total_active_alerts": len(active_alerts)
        }
        
    except Exception as e:
        logger.error(f"Error fetching patient overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# ENHANCED ALERT ENGINE ENDPOINTS - V2
# Includes: Metrics Ingest, Organ Scoring, DPI, Rule-Based Alerts, Notifications
# =============================================================================

try:
    from app.services.alert_engine import (
        MetricsIngestService,
        OrganScoringService,
        DPIComputationService,
        RuleBasedAlertEngine,
        NotificationService,
        EscalationService,
        MLRankingService,
        AlertConfigService
    )
    ALERT_ENGINE_V2_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Alert Engine V2 services not available: {e}")
    ALERT_ENGINE_V2_AVAILABLE = False


class MetricIngestPayload(BaseModel):
    """Payload for metric ingestion"""
    patient_id: str
    metric_name: str
    metric_value: float
    unit: str = ""
    timestamp: Optional[datetime] = None
    confidence: float = 1.0
    source: str = "app"
    capture_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BatchMetricIngestPayload(BaseModel):
    """Payload for batch metric ingestion"""
    metrics: List[MetricIngestPayload]


class OrganScoresResponse(BaseModel):
    """Response for organ scores"""
    patient_id: str
    respiratory: Dict[str, Any]
    cardio_fluid: Dict[str, Any]
    hepatic: Dict[str, Any]
    mobility: Dict[str, Any]
    cognitive: Dict[str, Any]
    total_metrics: int
    computed_at: str


class DPIResponse(BaseModel):
    """Response for DPI"""
    patient_id: str
    dpi_raw: float
    dpi_normalized: float
    dpi_bucket: str
    bucket_color: str
    bucket_description: str
    components: List[Dict[str, Any]]
    previous_dpi: Optional[float] = None
    previous_bucket: Optional[str] = None
    bucket_changed: bool = False
    dpi_delta_24h: Optional[float] = None
    jump_detected: bool = False
    computed_at: str


class AlertConfigUpdate(BaseModel):
    """Payload for updating alert config"""
    baseline_window_days: Optional[int] = None
    z_yellow_threshold: Optional[float] = None
    z_red_threshold: Optional[float] = None
    dpi_green_max: Optional[float] = None
    dpi_yellow_max: Optional[float] = None
    dpi_orange_max: Optional[float] = None
    suppression_window_hours: Optional[int] = None
    max_alerts_per_patient_per_day: Optional[int] = None
    escalation_timeout_critical_hours: Optional[float] = None
    escalation_timeout_high_hours: Optional[float] = None
    sms_enabled: Optional[bool] = None
    email_enabled: Optional[bool] = None
    ml_ranking_enabled: Optional[bool] = None


class EscalationRequest(BaseModel):
    """Request for manual escalation"""
    escalate_to: str
    reason: str


@router.post("/v2/metrics/ingest")
async def ingest_metric(
    payload: MetricIngestPayload,
    db: Session = Depends(get_db)
):
    """
    Ingest a single health metric for real-time processing.
    
    The metric will be validated, stored, and queued for Alert Engine processing.
    Quality flags will be applied based on confidence and capture age.
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        from app.services.alert_engine.metrics_ingest import MetricIngestRequest
        
        service = MetricsIngestService(db)
        await service.initialize()
        
        request = MetricIngestRequest(
            patient_id=payload.patient_id,
            timestamp=payload.timestamp or datetime.utcnow(),
            metric_name=payload.metric_name,
            metric_value=payload.metric_value,
            unit=payload.unit,
            confidence=payload.confidence,
            source=payload.source,
            capture_id=payload.capture_id,
            metadata=payload.metadata
        )
        
        result = await service.ingest_metric(request)
        
        return {
            "success": result.success,
            "metric_id": result.metric_id,
            "processed_at": result.processed_at.isoformat(),
            "queued_for_processing": result.queued_for_processing,
            "quality_flags": result.quality_flags,
            "message": result.message
        }
        
    except Exception as e:
        logger.error(f"Error ingesting metric: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/metrics/ingest/batch")
async def ingest_metrics_batch(
    payload: BatchMetricIngestPayload,
    db: Session = Depends(get_db)
):
    """Ingest multiple metrics in a single batch request."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        from app.services.alert_engine.metrics_ingest import MetricIngestRequest
        
        service = MetricsIngestService(db)
        await service.initialize()
        
        requests = [
            MetricIngestRequest(
                patient_id=m.patient_id,
                timestamp=m.timestamp or datetime.utcnow(),
                metric_name=m.metric_name,
                metric_value=m.metric_value,
                unit=m.unit,
                confidence=m.confidence,
                source=m.source,
                capture_id=m.capture_id,
                metadata=m.metadata
            )
            for m in payload.metrics
        ]
        
        result = await service.ingest_batch(requests)
        return result
        
    except Exception as e:
        logger.error(f"Error ingesting batch metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/metrics/recent/{patient_id}")
async def get_recent_metrics(
    patient_id: str,
    metric_name: Optional[str] = None,
    hours: int = Query(default=24, le=168),
    db: Session = Depends(get_db)
):
    """Get recently ingested metrics for a patient."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        service = MetricsIngestService(db)
        await service.initialize()
        
        metrics = await service.get_recent_metrics(patient_id, metric_name, hours)
        return {"patient_id": patient_id, "hours": hours, "metrics": metrics}
        
    except Exception as e:
        logger.error(f"Error fetching recent metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/organ-scores/{patient_id}")
async def get_organ_scores(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current organ-level health scores for a patient.
    
    Returns scores for 5 organ groups:
    - Respiratory
    - Cardio/Fluid
    - Hepatic/Hematologic
    - Mobility
    - Cognitive/Behavioral
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        service = OrganScoringService(db)
        
        # Get latest from history
        history = await service.get_organ_scores_history(patient_id, days=1)
        
        if history:
            latest = history[0]
            return OrganScoresResponse(
                patient_id=patient_id,
                respiratory=latest["respiratory"],
                cardio_fluid=latest["cardio_fluid"],
                hepatic=latest["hepatic"],
                mobility=latest["mobility"],
                cognitive=latest["cognitive"],
                total_metrics=latest["total_metrics"],
                computed_at=latest["computed_at"]
            )
        
        # Compute fresh scores
        result = await service.compute_from_recent_data(patient_id)
        await service.store_organ_scores(result)
        
        return OrganScoresResponse(
            patient_id=patient_id,
            respiratory={
                "score": result.organ_scores.get("respiratory", {}).normalized_score if result.organ_scores.get("respiratory") else 50,
                "severity": result.organ_scores.get("respiratory", {}).severity if result.organ_scores.get("respiratory") else "normal"
            },
            cardio_fluid={
                "score": result.organ_scores.get("cardio_fluid", {}).normalized_score if result.organ_scores.get("cardio_fluid") else 50,
                "severity": result.organ_scores.get("cardio_fluid", {}).severity if result.organ_scores.get("cardio_fluid") else "normal"
            },
            hepatic={
                "score": result.organ_scores.get("hepatic_hematologic", {}).normalized_score if result.organ_scores.get("hepatic_hematologic") else 50,
                "severity": result.organ_scores.get("hepatic_hematologic", {}).severity if result.organ_scores.get("hepatic_hematologic") else "normal"
            },
            mobility={
                "score": result.organ_scores.get("mobility", {}).normalized_score if result.organ_scores.get("mobility") else 50,
                "severity": result.organ_scores.get("mobility", {}).severity if result.organ_scores.get("mobility") else "normal"
            },
            cognitive={
                "score": result.organ_scores.get("cognitive_behavioral", {}).normalized_score if result.organ_scores.get("cognitive_behavioral") else 50,
                "severity": result.organ_scores.get("cognitive_behavioral", {}).severity if result.organ_scores.get("cognitive_behavioral") else "normal"
            },
            total_metrics=result.total_metrics,
            computed_at=result.computed_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching organ scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/organ-scores/compute/{patient_id}")
async def compute_organ_scores(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Compute and store organ scores for a patient."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        service = OrganScoringService(db)
        result = await service.compute_from_recent_data(patient_id)
        await service.store_organ_scores(result)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "organ_scores": {
                name: {
                    "score": score.normalized_score,
                    "severity": score.severity,
                    "num_metrics": score.num_metrics
                }
                for name, score in result.organ_scores.items()
            },
            "total_metrics": result.total_metrics,
            "computed_at": result.computed_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error computing organ scores: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/organ-scores/history/{patient_id}")
async def get_organ_scores_history(
    patient_id: str,
    days: int = Query(default=30, le=90),
    db: Session = Depends(get_db)
):
    """Get historical organ scores for a patient."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        service = OrganScoringService(db)
        history = await service.get_organ_scores_history(patient_id, days)
        return {"patient_id": patient_id, "days": days, "history": history}
        
    except Exception as e:
        logger.error(f"Error fetching organ scores history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/dpi/{patient_id}")
async def get_dpi(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current Composite Deterioration Index (DPI) for a patient.
    
    DPI Color Buckets:
    - Green (< 25): Stable
    - Yellow (25-50): Elevated
    - Orange (50-75): Concerning
    - Red (>= 75): Critical
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        dpi_service = DPIComputationService(db)
        current_dpi = await dpi_service.get_current_dpi(patient_id)
        
        if current_dpi:
            return DPIResponse(
                patient_id=patient_id,
                dpi_raw=current_dpi["dpi_raw"],
                dpi_normalized=current_dpi["dpi_normalized"],
                dpi_bucket=current_dpi["dpi_bucket"],
                bucket_color=dpi_service.get_bucket_color(current_dpi["dpi_bucket"]),
                bucket_description=dpi_service.get_bucket_description(current_dpi["dpi_bucket"]),
                components=current_dpi["components"] or [],
                previous_dpi=current_dpi.get("previous_dpi"),
                previous_bucket=current_dpi.get("previous_bucket"),
                bucket_changed=current_dpi.get("bucket_changed", False),
                dpi_delta_24h=current_dpi.get("dpi_delta_24h"),
                jump_detected=current_dpi.get("jump_detected", False),
                computed_at=current_dpi["computed_at"]
            )
        
        # No DPI found - compute fresh
        organ_service = OrganScoringService(db)
        organ_result = await organ_service.compute_from_recent_data(patient_id)
        dpi_result = dpi_service.compute_dpi(patient_id, organ_result)
        await dpi_service.store_dpi(dpi_result)
        
        return DPIResponse(
            patient_id=patient_id,
            dpi_raw=dpi_result.dpi_raw,
            dpi_normalized=dpi_result.dpi_normalized,
            dpi_bucket=dpi_result.dpi_bucket,
            bucket_color=dpi_service.get_bucket_color(dpi_result.dpi_bucket),
            bucket_description=dpi_service.get_bucket_description(dpi_result.dpi_bucket),
            components=[
                {
                    "organ_name": c.organ_name,
                    "organ_score": c.organ_score,
                    "weight": c.weight,
                    "contribution": c.contribution,
                    "percentage": c.percentage
                }
                for c in dpi_result.components
            ],
            previous_dpi=dpi_result.previous_dpi,
            previous_bucket=dpi_result.previous_bucket,
            bucket_changed=dpi_result.bucket_changed,
            dpi_delta_24h=dpi_result.dpi_delta_24h,
            jump_detected=dpi_result.jump_detected,
            computed_at=dpi_result.computed_at.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error fetching DPI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/dpi/compute/{patient_id}")
async def compute_dpi(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """Compute and store DPI for a patient."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        organ_service = OrganScoringService(db)
        dpi_service = DPIComputationService(db)
        
        # Compute organ scores first
        organ_result = await organ_service.compute_from_recent_data(patient_id)
        await organ_service.store_organ_scores(organ_result)
        
        # Then compute DPI
        dpi_result = dpi_service.compute_dpi(patient_id, organ_result)
        await dpi_service.store_dpi(dpi_result)
        
        return {
            "success": True,
            "patient_id": patient_id,
            "dpi_normalized": dpi_result.dpi_normalized,
            "dpi_bucket": dpi_result.dpi_bucket,
            "bucket_color": dpi_service.get_bucket_color(dpi_result.dpi_bucket),
            "bucket_changed": dpi_result.bucket_changed,
            "jump_detected": dpi_result.jump_detected,
            "computed_at": dpi_result.computed_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error computing DPI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/dpi/history/{patient_id}")
async def get_dpi_history(
    patient_id: str,
    days: int = Query(default=30, le=90),
    db: Session = Depends(get_db)
):
    """Get historical DPI values for a patient."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        service = DPIComputationService(db)
        history = await service.get_dpi_history(patient_id, days)
        return {"patient_id": patient_id, "days": days, "history": history}
        
    except Exception as e:
        logger.error(f"Error fetching DPI history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/alerts/generate/{patient_id}")
async def generate_v2_alerts(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Generate alerts using the V2 rule-based engine.
    
    Evaluates 7 clinical rules:
    1. Risk Jump (Green → Yellow)
    2. Persistent Yellow (>48h)
    3. Any Red
    4. Respiratory Spike
    5. Daily Check-in Deviation
    6. Composite Sudden Increase
    7. Multi-Signal Corroboration
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        # Compute organ scores and DPI first
        organ_service = OrganScoringService(db)
        dpi_service = DPIComputationService(db)
        rule_engine = RuleBasedAlertEngine(db)
        
        organ_result = await organ_service.compute_from_recent_data(patient_id)
        await organ_service.store_organ_scores(organ_result)
        
        dpi_result = dpi_service.compute_dpi(patient_id, organ_result)
        await dpi_service.store_dpi(dpi_result)
        
        # Get metric z-scores
        metric_z_scores = {}
        for group_name, organ_score in organ_result.organ_scores.items():
            for metric in organ_score.contributing_metrics:
                metric_z_scores[metric.metric_name] = metric.z_score
        
        # Evaluate all rules
        triggered_alerts = await rule_engine.evaluate_all_rules(
            patient_id=patient_id,
            dpi_result=dpi_result,
            organ_result=organ_result,
            metric_z_scores=metric_z_scores
        )
        
        # Create alert records
        created_alerts = []
        for trigger in triggered_alerts:
            record = await rule_engine.create_alert_record(patient_id, trigger)
            created_alerts.append({
                "id": record.id,
                "rule": record.trigger_rule,
                "severity": record.severity,
                "priority": record.priority,
                "title": record.title,
                "corroborated": record.corroborated
            })
        
        return {
            "success": True,
            "patient_id": patient_id,
            "dpi": {
                "score": dpi_result.dpi_normalized,
                "bucket": dpi_result.dpi_bucket,
                "bucket_changed": dpi_result.bucket_changed
            },
            "alerts_generated": len(created_alerts),
            "alerts": created_alerts
        }
        
    except Exception as e:
        logger.error(f"Error generating V2 alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/alerts/{alert_id}/escalate")
async def escalate_alert(
    alert_id: str,
    request: EscalationRequest,
    clinician_id: str = Query(...),
    db: Session = Depends(get_db)
):
    """Manually escalate an alert to another clinician."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        escalation_service = EscalationService(db)
        
        success = await escalation_service.manual_escalate(
            alert_id=alert_id,
            escalated_by=clinician_id,
            escalate_to=request.escalate_to,
            reason=request.reason
        )
        
        if success:
            return {"success": True, "alert_id": alert_id, "escalated_to": request.escalate_to}
        else:
            raise HTTPException(status_code=400, detail="Failed to escalate alert")
        
    except Exception as e:
        logger.error(f"Error escalating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/alerts/{alert_id}/escalation-history")
async def get_alert_escalation_history(
    alert_id: str,
    db: Session = Depends(get_db)
):
    """Get escalation history for an alert."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        escalation_service = EscalationService(db)
        history = await escalation_service.get_escalation_history(alert_id)
        return {"alert_id": alert_id, "history": history}
        
    except Exception as e:
        logger.error(f"Error fetching escalation history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/notifications/unread/{user_id}")
async def get_unread_notifications(
    user_id: str,
    limit: int = Query(default=50, le=100),
    db: Session = Depends(get_db)
):
    """Get unread dashboard notifications for a clinician."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        notification_service = NotificationService(db)
        notifications = await notification_service.get_unread_notifications(user_id, limit)
        return {"user_id": user_id, "unread_count": len(notifications), "notifications": notifications}
        
    except Exception as e:
        logger.error(f"Error fetching notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: str,
    user_id: str = Query(...),
    db: Session = Depends(get_db)
):
    """Mark a notification as read."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        notification_service = NotificationService(db)
        success = await notification_service.mark_notification_read(notification_id, user_id)
        return {"success": success, "notification_id": notification_id}
        
    except Exception as e:
        logger.error(f"Error marking notification read: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/notifications/stats")
async def get_notification_stats(
    days: int = Query(default=7, le=30),
    db: Session = Depends(get_db)
):
    """Get notification delivery statistics."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        notification_service = NotificationService(db)
        stats = await notification_service.get_delivery_stats(days)
        return {"days": days, "stats": stats}
        
    except Exception as e:
        logger.error(f"Error fetching notification stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/alerts/ranked/{patient_id}")
async def get_ranked_alerts(
    patient_id: str,
    limit: int = Query(default=20, le=50),
    db: Session = Depends(get_db)
):
    """Get ML-ranked alerts for a patient."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        # Get active alerts
        query = text("""
            SELECT id, patient_id, alert_type, alert_category, severity, priority,
                   title, message, trigger_rule, contributing_metrics,
                   dpi_at_trigger, organ_scores, corroborated, status, created_at
            FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND status NOT IN ('dismissed', 'closed')
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        
        results = db.execute(query, {"patient_id": patient_id, "limit": limit}).fetchall()
        
        alerts = [
            {
                "id": str(row[0]),
                "patient_id": row[1],
                "alert_type": row[2],
                "alert_category": row[3],
                "severity": row[4],
                "priority": row[5],
                "title": row[6],
                "message": row[7],
                "trigger_rule": row[8],
                "trigger_metrics": row[9] or [],
                "dpi_at_trigger": float(row[10]) if row[10] else None,
                "organ_scores": row[11],
                "corroborated": row[12] or False,
                "status": row[13],
                "created_at": row[14].isoformat() if row[14] else None
            }
            for row in results
        ]
        
        # Apply ML ranking
        ml_service = MLRankingService(db)
        await ml_service.initialize()
        
        ranked_results = await ml_service.rank_alerts(alerts)
        
        # Combine alerts with rankings
        ranked_alerts = []
        for result in ranked_results:
            alert = next((a for a in alerts if a["id"] == result.alert_id), None)
            if alert:
                alert["ml_priority_score"] = result.priority_score
                alert["ml_confidence"] = result.confidence
                alert["ml_explanation"] = result.explanation
                ranked_alerts.append(alert)
        
        return {
            "patient_id": patient_id,
            "total_alerts": len(ranked_alerts),
            "alerts": ranked_alerts
        }
        
    except Exception as e:
        logger.error(f"Error fetching ranked alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/config")
async def get_alert_config():
    """Get current Alert Engine configuration."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        config_service = AlertConfigService()
        return config_service.config.to_dict()
        
    except Exception as e:
        logger.error(f"Error fetching config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/v2/config")
async def update_alert_config(
    updates: AlertConfigUpdate
):
    """Update Alert Engine configuration."""
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        config_service = AlertConfigService()
        update_dict = {k: v for k, v in updates.dict().items() if v is not None}
        
        if update_dict:
            config_service.update_config(update_dict)
        
        return {"success": True, "updated_fields": list(update_dict.keys())}
        
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/compute-all/{patient_id}")
async def compute_all_v2(
    patient_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Complete V2 Alert Engine pipeline:
    1. Compute organ scores
    2. Compute DPI
    3. Evaluate all 7 alert rules
    4. Apply ML ranking
    5. Return comprehensive result
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(status_code=503, detail="Alert Engine V2 not available")
    
    try:
        # Initialize services
        organ_service = OrganScoringService(db)
        dpi_service = DPIComputationService(db)
        rule_engine = RuleBasedAlertEngine(db)
        ml_service = MLRankingService(db)
        await ml_service.initialize()
        
        # Step 1: Compute organ scores
        organ_result = await organ_service.compute_from_recent_data(patient_id)
        await organ_service.store_organ_scores(organ_result)
        
        # Step 2: Compute DPI
        dpi_result = dpi_service.compute_dpi(patient_id, organ_result)
        await dpi_service.store_dpi(dpi_result)
        
        # Step 3: Get metric z-scores
        metric_z_scores = {}
        for group_name, organ_score in organ_result.organ_scores.items():
            for metric in organ_score.contributing_metrics:
                metric_z_scores[metric.metric_name] = metric.z_score
        
        # Step 4: Evaluate rules and create alerts
        triggered_alerts = await rule_engine.evaluate_all_rules(
            patient_id=patient_id,
            dpi_result=dpi_result,
            organ_result=organ_result,
            metric_z_scores=metric_z_scores
        )
        
        created_alerts = []
        for trigger in triggered_alerts:
            record = await rule_engine.create_alert_record(patient_id, trigger)
            created_alerts.append({
                "id": record.id,
                "rule": record.trigger_rule,
                "severity": record.severity,
                "priority": record.priority,
                "title": record.title
            })
        
        # Step 5: Apply ML ranking to existing alerts
        query = text("""
            SELECT id, patient_id, alert_type, severity, priority, trigger_rule,
                   dpi_at_trigger, organ_scores, corroborated, contributing_metrics
            FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND status NOT IN ('dismissed', 'closed')
            ORDER BY created_at DESC
            LIMIT 20
        """)
        
        alert_rows = db.execute(query, {"patient_id": patient_id}).fetchall()
        alerts_for_ranking = [
            {
                "id": str(row[0]),
                "patient_id": row[1],
                "alert_type": row[2],
                "severity": row[3],
                "priority": row[4],
                "trigger_rule": row[5],
                "dpi_at_trigger": float(row[6]) if row[6] else None,
                "organ_scores": row[7],
                "corroborated": row[8] or False,
                "trigger_metrics": row[9] or []
            }
            for row in alert_rows
        ]
        
        ranked = await ml_service.rank_alerts(alerts_for_ranking) if alerts_for_ranking else []
        
        return {
            "success": True,
            "patient_id": patient_id,
            "organ_scores": {
                name: {
                    "score": score.normalized_score,
                    "severity": score.severity,
                    "num_metrics": score.num_metrics
                }
                for name, score in organ_result.organ_scores.items()
            },
            "dpi": {
                "score": dpi_result.dpi_normalized,
                "bucket": dpi_result.dpi_bucket,
                "bucket_color": dpi_service.get_bucket_color(dpi_result.dpi_bucket),
                "bucket_changed": dpi_result.bucket_changed,
                "jump_detected": dpi_result.jump_detected
            },
            "alerts_generated": len(created_alerts),
            "new_alerts": created_alerts,
            "total_active_alerts": len(ranked),
            "top_ranked_alerts": [
                {
                    "id": r.alert_id,
                    "priority_score": r.priority_score,
                    "confidence": r.confidence
                }
                for r in ranked[:5]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in V2 compute-all: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/patient-overview/{patient_id}")
async def get_v2_patient_overview(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get comprehensive V2 patient overview including:
    - Current organ scores
    - Current DPI with bucket
    - Active alerts (ML-ranked)
    - Historical trends
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        # Fall back to V1 overview
        return await get_patient_overview(patient_id, db)
    
    try:
        organ_service = OrganScoringService(db)
        dpi_service = DPIComputationService(db)
        ml_service = MLRankingService(db)
        await ml_service.initialize()
        
        # Get organ scores
        organ_history = await organ_service.get_organ_scores_history(patient_id, days=1)
        current_organs = organ_history[0] if organ_history else None
        
        # Get DPI
        current_dpi = await dpi_service.get_current_dpi(patient_id)
        
        # Get DPI history for trend
        dpi_history = await dpi_service.get_dpi_history(patient_id, days=7)
        
        # Get active alerts
        alerts_query = text("""
            SELECT id, alert_type, alert_category, severity, priority, title, message,
                   trigger_rule, dpi_at_trigger, organ_scores, corroborated,
                   status, created_at
            FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND status NOT IN ('dismissed', 'closed')
            ORDER BY priority DESC, created_at DESC
            LIMIT 20
        """)
        
        alert_rows = db.execute(alerts_query, {"patient_id": patient_id}).fetchall()
        
        alerts = [
            {
                "id": str(row[0]),
                "alert_type": row[1],
                "alert_category": row[2],
                "severity": row[3],
                "priority": row[4],
                "title": row[5],
                "message": row[6],
                "trigger_rule": row[7],
                "dpi_at_trigger": float(row[8]) if row[8] else None,
                "organ_scores": row[9],
                "corroborated": row[10] or False,
                "status": row[11],
                "created_at": row[12].isoformat() if row[12] else None
            }
            for row in alert_rows
        ]
        
        # Apply ML ranking
        ranked = await ml_service.rank_alerts(alerts) if alerts else []
        
        for i, result in enumerate(ranked):
            for alert in alerts:
                if alert["id"] == result.alert_id:
                    alert["ml_priority_score"] = result.priority_score
                    alert["ml_rank"] = i + 1
        
        # Sort alerts by ML rank
        alerts.sort(key=lambda x: x.get("ml_priority_score", 0), reverse=True)
        
        return {
            "patient_id": patient_id,
            "organ_scores": current_organs,
            "dpi": {
                "current": current_dpi,
                "bucket_color": dpi_service.get_bucket_color(current_dpi["dpi_bucket"]) if current_dpi else "#6b7280",
                "bucket_description": dpi_service.get_bucket_description(current_dpi["dpi_bucket"]) if current_dpi else "No data",
                "trend": [
                    {"date": h["computed_at"], "score": h["dpi_normalized"], "bucket": h["dpi_bucket"]}
                    for h in dpi_history[:7]
                ]
            } if current_dpi else None,
            "active_alerts": alerts,
            "total_active_alerts": len(alerts),
            "alert_summary": {
                "critical": len([a for a in alerts if a["severity"] == "critical"]),
                "high": len([a for a in alerts if a["severity"] == "high"]),
                "moderate": len([a for a in alerts if a["severity"] == "moderate"]),
                "low": len([a for a in alerts if a["severity"] == "low"])
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching V2 patient overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ML PREDICTION ENDPOINTS (V2)
# ============================================================================

class PredictionRequest(BaseModel):
    patient_id: str
    horizon_hours: Optional[List[int]] = Field(default=[3, 6, 12, 24], description="Prediction horizons in hours")
    include_feature_importance: bool = Field(default=True)


class PredictionHorizonResponse(BaseModel):
    horizon: str
    deterioration_probability: float
    confidence: float
    risk_level: str
    feature_importance: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    patient_id: str
    ensemble_score: float
    ensemble_confidence: float
    statistical_weight: float
    ml_weight: float
    trend_direction: str
    risk_trajectory: str
    predictions: Dict[str, PredictionHorizonResponse]
    computed_at: datetime
    model_version: str
    disclaimer: str


@router.post("/v2/predictions/compute")
async def compute_predictions(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Compute ML deterioration predictions for a patient across multiple time horizons.
    Uses LSTM-based deep learning models for time-series vital sign analysis.
    """
    try:
        from app.services.alert_engine.prediction_service import PredictionService
        
        prediction_service = PredictionService(db)
        await prediction_service.initialize()
        
        # Get vital signs data for the patient
        vital_query = text("""
            SELECT recorded_at, 
                   heart_rate, respiratory_rate, blood_pressure_systolic, blood_pressure_diastolic,
                   temperature, spo2, weight
            FROM health_metrics
            WHERE user_id = :patient_id
            AND recorded_at >= NOW() - INTERVAL '7 days'
            ORDER BY recorded_at ASC
        """)
        
        vital_rows = db.execute(vital_query, {"patient_id": request.patient_id}).fetchall()
        
        if not vital_rows:
            return {
                "patient_id": request.patient_id,
                "status": "insufficient_data",
                "message": "Insufficient vital signs data for prediction. Need at least 7 days of data.",
                "disclaimer": COMPLIANCE_DISCLAIMER
            }
        
        # Convert to vital signs dict
        vital_signs_history = {
            "timestamps": [],
            "heart_rate": [],
            "respiratory_rate": [],
            "bp_systolic": [],
            "bp_diastolic": [],
            "temperature": [],
            "spo2": [],
            "weight": []
        }
        
        for row in vital_rows:
            vital_signs_history["timestamps"].append(row[0])
            vital_signs_history["heart_rate"].append(float(row[1]) if row[1] else None)
            vital_signs_history["respiratory_rate"].append(float(row[2]) if row[2] else None)
            vital_signs_history["bp_systolic"].append(float(row[3]) if row[3] else None)
            vital_signs_history["bp_diastolic"].append(float(row[4]) if row[4] else None)
            vital_signs_history["temperature"].append(float(row[5]) if row[5] else None)
            vital_signs_history["spo2"].append(float(row[6]) if row[6] else None)
            vital_signs_history["weight"].append(float(row[7]) if row[7] else None)
        
        # Compute predictions
        result = await prediction_service.predict_deterioration(
            patient_id=request.patient_id,
            vital_signs_history=vital_signs_history,
            horizon_hours=request.horizon_hours
        )
        
        # Store prediction in background
        background_tasks.add_task(prediction_service.store_prediction, result)
        
        # Format response
        predictions_response = {}
        for horizon, pred in result.predictions.items():
            predictions_response[horizon] = {
                "horizon": pred.horizon,
                "deterioration_probability": pred.deterioration_probability,
                "confidence": pred.confidence,
                "risk_level": pred.risk_level,
                "feature_importance": pred.feature_importance if request.include_feature_importance else None
            }
        
        return {
            "patient_id": result.patient_id,
            "ensemble_score": result.ensemble_score,
            "ensemble_confidence": result.ensemble_confidence,
            "statistical_weight": result.statistical_weight,
            "ml_weight": result.ml_weight,
            "trend_direction": result.trend_direction,
            "risk_trajectory": result.risk_trajectory,
            "predictions": predictions_response,
            "computed_at": result.computed_at.isoformat(),
            "model_version": result.model_version,
            "disclaimer": COMPLIANCE_DISCLAIMER
        }
        
    except Exception as e:
        logger.error(f"Error computing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/predictions/{patient_id}")
async def get_latest_prediction(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Get the most recent ML prediction for a patient.
    """
    try:
        from app.services.alert_engine.prediction_service import PredictionService
        
        prediction_service = PredictionService(db)
        result = await prediction_service.get_latest_prediction(patient_id)
        
        if not result:
            return {
                "patient_id": patient_id,
                "status": "no_predictions",
                "message": "No predictions available for this patient.",
                "disclaimer": COMPLIANCE_DISCLAIMER
            }
        
        return {
            **result,
            "disclaimer": COMPLIANCE_DISCLAIMER
        }
        
    except Exception as e:
        logger.error(f"Error fetching prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v2/predictions/{patient_id}/history")
async def get_prediction_history(
    patient_id: str,
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
):
    """
    Get ML prediction history for a patient.
    """
    try:
        from app.services.alert_engine.prediction_service import PredictionService
        
        prediction_service = PredictionService(db)
        history = await prediction_service.get_prediction_history(patient_id, days=days)
        
        return {
            "patient_id": patient_id,
            "days": days,
            "predictions": history,
            "total": len(history),
            "disclaimer": COMPLIANCE_DISCLAIMER
        }
        
    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v2/compute-all-with-ml/{patient_id}")
async def compute_all_with_ml_predictions(
    patient_id: str,
    db: Session = Depends(get_db)
):
    """
    Full V2 computation pipeline with ML predictions integrated:
    1. Compute organ scores
    2. Get/compute ML predictions
    3. Compute ensemble DPI (statistical + ML)
    4. Evaluate all 10 rules (7 statistical + 3 ML-based)
    5. Store alerts and apply ML ranking
    """
    if not ALERT_ENGINE_V2_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Alert Engine V2 not available. Check dependencies."
        )
    
    try:
        from app.services.alert_engine.prediction_service import PredictionService
        
        organ_service = OrganScoringService(db)
        dpi_service = DPIComputationService(db)
        rule_engine = RuleBasedAlertEngine(db)
        ml_service = MLRankingService(db)
        prediction_service = PredictionService(db)
        
        await ml_service.initialize()
        await prediction_service.initialize()
        
        # Step 1: Compute organ scores
        organ_result = await organ_service.compute_from_recent_data(patient_id)
        await organ_service.store_organ_scores(organ_result)
        
        # Step 2: Get/compute ML predictions
        ml_prediction = await prediction_service.get_latest_prediction(patient_id)
        
        # If no recent prediction, compute one
        if not ml_prediction or _prediction_is_stale(ml_prediction):
            # Fetch vital signs
            vital_query = text("""
                SELECT recorded_at, heart_rate, respiratory_rate, 
                       blood_pressure_systolic, blood_pressure_diastolic,
                       temperature, spo2
                FROM health_metrics
                WHERE user_id = :patient_id
                AND recorded_at >= NOW() - INTERVAL '7 days'
                ORDER BY recorded_at ASC
            """)
            
            vital_rows = db.execute(vital_query, {"patient_id": patient_id}).fetchall()
            
            if vital_rows and len(vital_rows) >= 10:
                vital_signs_history = {
                    "timestamps": [row[0] for row in vital_rows],
                    "heart_rate": [float(row[1]) if row[1] else None for row in vital_rows],
                    "respiratory_rate": [float(row[2]) if row[2] else None for row in vital_rows],
                    "bp_systolic": [float(row[3]) if row[3] else None for row in vital_rows],
                    "bp_diastolic": [float(row[4]) if row[4] else None for row in vital_rows],
                    "temperature": [float(row[5]) if row[5] else None for row in vital_rows],
                    "spo2": [float(row[6]) if row[6] else None for row in vital_rows]
                }
                
                try:
                    pred_result = await prediction_service.predict_deterioration(
                        patient_id=patient_id,
                        vital_signs_history=vital_signs_history
                    )
                    
                    ml_prediction = {
                        "ensemble_score": pred_result.ensemble_score,
                        "ensemble_confidence": pred_result.ensemble_confidence,
                        "predictions": {
                            horizon: {
                                "deterioration_probability": pred.deterioration_probability,
                                "confidence": pred.confidence,
                                "risk_level": pred.risk_level
                            }
                            for horizon, pred in pred_result.predictions.items()
                        },
                        "trend_direction": pred_result.trend_direction,
                        "risk_trajectory": pred_result.risk_trajectory,
                        "model_version": pred_result.model_version
                    }
                    
                    await prediction_service.store_prediction(pred_result)
                except Exception as pred_error:
                    logger.warning(f"ML prediction failed: {pred_error}")
                    ml_prediction = None
        
        # Step 3: Compute ensemble DPI
        dpi_result = await dpi_service.compute_ensemble_dpi(
            patient_id, organ_result, ml_prediction
        )
        await dpi_service.store_dpi(dpi_result)
        
        # Step 4: Get metric z-scores
        metric_z_scores = {}
        for group_name, organ_score in organ_result.organ_scores.items():
            for metric in organ_score.contributing_metrics:
                metric_z_scores[metric.metric_name] = metric.z_score
        
        # Step 5: Evaluate all 10 rules (including ML-based)
        triggered_alerts = await rule_engine.evaluate_all_rules(
            patient_id=patient_id,
            dpi_result=dpi_result,
            organ_result=organ_result,
            metric_z_scores=metric_z_scores,
            ml_prediction=ml_prediction
        )
        
        created_alerts = []
        for trigger in triggered_alerts:
            record = await rule_engine.create_alert_record(patient_id, trigger)
            created_alerts.append({
                "id": record.id,
                "rule": record.trigger_rule,
                "severity": record.severity,
                "priority": record.priority,
                "title": record.title,
                "is_ml_based": record.trigger_rule.startswith("ml_")
            })
        
        # Step 6: Apply ML ranking to existing alerts
        query = text("""
            SELECT id, patient_id, alert_type, severity, priority, trigger_rule,
                   dpi_at_trigger, organ_scores, corroborated, contributing_metrics
            FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND status NOT IN ('dismissed', 'closed')
            ORDER BY created_at DESC
            LIMIT 20
        """)
        
        alert_rows = db.execute(query, {"patient_id": patient_id}).fetchall()
        alerts_for_ranking = [
            {
                "id": str(row[0]),
                "patient_id": row[1],
                "alert_type": row[2],
                "severity": row[3],
                "priority": row[4],
                "trigger_rule": row[5],
                "dpi_at_trigger": float(row[6]) if row[6] else None,
                "organ_scores": row[7],
                "corroborated": row[8] or False,
                "trigger_metrics": row[9] or []
            }
            for row in alert_rows
        ]
        
        ranked = await ml_service.rank_alerts(alerts_for_ranking) if alerts_for_ranking else []
        
        return {
            "success": True,
            "patient_id": patient_id,
            "organ_scores": {
                name: {
                    "score": score.normalized_score,
                    "severity": score.severity,
                    "num_metrics": score.num_metrics
                }
                for name, score in organ_result.organ_scores.items()
            },
            "dpi": {
                "score": dpi_result.dpi_normalized,
                "bucket": dpi_result.dpi_bucket,
                "bucket_color": dpi_service.get_bucket_color(dpi_result.dpi_bucket),
                "bucket_changed": dpi_result.bucket_changed,
                "jump_detected": dpi_result.jump_detected,
                "statistical_dpi": dpi_result.statistical_dpi,
                "ml_dpi": dpi_result.ml_dpi,
                "ensemble_weights": {
                    "statistical": dpi_result.ensemble_weight_statistical,
                    "ml": dpi_result.ensemble_weight_ml
                },
                "ensemble_confidence": dpi_result.ensemble_confidence
            },
            "ml_prediction": {
                "available": ml_prediction is not None,
                "ensemble_score": ml_prediction.get("ensemble_score") if ml_prediction else None,
                "confidence": ml_prediction.get("ensemble_confidence") if ml_prediction else None,
                "trend": ml_prediction.get("trend_direction") if ml_prediction else None,
                "trajectory": ml_prediction.get("risk_trajectory") if ml_prediction else None
            },
            "alerts_generated": len(created_alerts),
            "new_alerts": created_alerts,
            "ml_based_alerts": len([a for a in created_alerts if a.get("is_ml_based")]),
            "total_active_alerts": len(ranked),
            "top_ranked_alerts": [
                {
                    "id": r.alert_id,
                    "priority_score": r.priority_score,
                    "confidence": r.confidence
                }
                for r in ranked[:5]
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in V2 compute-all-with-ml: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _prediction_is_stale(prediction: Dict[str, Any], max_age_hours: int = 6) -> bool:
    """Check if a prediction is too old to use"""
    computed_at = prediction.get("computed_at")
    if not computed_at:
        return True
    
    if isinstance(computed_at, str):
        computed_at = datetime.fromisoformat(computed_at.replace("Z", "+00:00"))
    
    age = datetime.utcnow() - computed_at.replace(tzinfo=None)
    return age.total_seconds() > max_age_hours * 3600


# ============================================
# DEVICE DATA → HEALTH ALERTS PIPELINE
# ============================================

class SectionAlertResponse(BaseModel):
    """Response model for section-based health alerts"""
    section: str
    deterioration_index: float
    risk_score: float
    risk_level: str
    trend: str
    stability_score: float
    alert_triggered: bool
    alert_reason: Optional[str] = None
    data_coverage: float
    anomalies_detected: int
    recommendations: List[str] = []


class DeviceAlertPipelineResponse(BaseModel):
    """Response model for device data → alerts pipeline"""
    patient_id: str
    generated_at: str
    overall_risk_score: float
    overall_trend: str
    sections: List[SectionAlertResponse]
    critical_alerts: List[Dict[str, Any]]
    alerts_generated: int
    new_alerts: List[Dict[str, Any]]


@router.post("/device-data-pipeline/{patient_id}")
async def run_device_data_alert_pipeline(
    patient_id: str,
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
) -> DeviceAlertPipelineResponse:
    """
    Run the complete device data → health alerts pipeline.
    
    This endpoint:
    1. Fetches device readings from wearables/medical devices
    2. Analyzes each health section (cardiovascular, respiratory, etc.)
    3. Computes deterioration indices, risk scores, and trends
    4. Generates alerts based on threshold violations
    5. Creates alert records for clinician review
    
    Args:
        patient_id: Patient's user ID
        days: Number of days to analyze (default 7)
    
    Returns:
        Comprehensive pipeline results with section analytics and generated alerts
    """
    if not HEALTH_SECTION_ANALYTICS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Health Section Analytics engine not available"
        )
    
    try:
        # Initialize analytics engine with async session
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        import asyncio
        
        database_url = os.environ.get("DATABASE_URL", "")
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            async_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        else:
            async_url = database_url
        
        async_engine = create_async_engine(async_url, echo=False)
        AsyncSessionLocal = sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async def analyze_and_generate_alerts():
            async with AsyncSessionLocal() as async_session:
                analytics_engine = HealthSectionAnalyticsEngine(async_session)
                profile = await analytics_engine.analyze_patient(patient_id, days=days)
                return profile
        
        # Run async analysis
        profile = asyncio.get_event_loop().run_until_complete(analyze_and_generate_alerts())
        
        # Convert section results to response format
        section_responses = []
        alerts_to_create = []
        
        for section, analytics in profile.sections.items():
            section_data = SectionAlertResponse(
                section=section.value,
                deterioration_index=analytics.deterioration_index,
                risk_score=analytics.risk_score,
                risk_level=analytics.risk_level.value,
                trend=analytics.trend.value,
                stability_score=analytics.stability_score,
                alert_triggered=analytics.alert_triggered,
                alert_reason=analytics.alert_reason,
                data_coverage=analytics.data_coverage,
                anomalies_detected=analytics.anomalies_detected,
                recommendations=analytics.recommendations,
            )
            section_responses.append(section_data)
            
            # If alert triggered, prepare alert record
            if analytics.alert_triggered:
                severity = "critical" if analytics.risk_score >= 13 else \
                          "high" if analytics.risk_score >= 10 else \
                          "moderate" if analytics.risk_score >= 5 else "low"
                priority = int(min(10, max(1, analytics.risk_score)))
                
                alerts_to_create.append({
                    "patient_id": patient_id,
                    "alert_type": "device_analytics",
                    "alert_category": f"section_{section.value}",
                    "severity": severity,
                    "priority": priority,
                    "title": f"{section.value.replace('_', ' ').title()} Alert",
                    "message": analytics.alert_reason or f"Health section {section.value} requires attention",
                    "disclaimer": COMPLIANCE_DISCLAIMER,
                    "contributing_metrics": [
                        {"metric": f.get("factor"), "value": f.get("value"), "impact": f.get("impact")}
                        for f in analytics.risk_factors
                    ],
                    "trigger_rule": "device_analytics_threshold",
                    "trigger_threshold": 10.0,
                    "trigger_value": analytics.deterioration_index,
                })
        
        # Create alert records in database
        created_alerts = []
        for alert_data in alerts_to_create:
            try:
                import uuid
                alert_id = str(uuid.uuid4())
                
                db.execute(
                    text("""
                        INSERT INTO ai_health_alerts (
                            id, patient_id, alert_type, alert_category, severity, priority,
                            title, message, disclaimer, contributing_metrics,
                            trigger_rule, status, created_at
                        ) VALUES (
                            :id, :patient_id, :alert_type, :alert_category, :severity, :priority,
                            :title, :message, :disclaimer, :contributing_metrics,
                            :trigger_rule, 'active', NOW()
                        )
                    """),
                    {
                        "id": alert_id,
                        "patient_id": alert_data["patient_id"],
                        "alert_type": alert_data["alert_type"],
                        "alert_category": alert_data["alert_category"],
                        "severity": alert_data["severity"],
                        "priority": alert_data["priority"],
                        "title": alert_data["title"],
                        "message": alert_data["message"],
                        "disclaimer": alert_data["disclaimer"],
                        "contributing_metrics": None,  # JSON would need proper handling
                        "trigger_rule": alert_data["trigger_rule"],
                    }
                )
                db.commit()
                
                created_alerts.append({
                    "id": alert_id,
                    "section": alert_data["alert_category"].replace("section_", ""),
                    "severity": alert_data["severity"],
                    "priority": alert_data["priority"],
                    "title": alert_data["title"],
                })
            except Exception as e:
                logger.warning(f"Could not create alert record: {e}")
                db.rollback()
        
        # Log HIPAA audit event
        try:
            db.execute(
                text("""
                    INSERT INTO audit_logs (
                        id, user_id, action, resource_type, resource_id,
                        details, timestamp
                    ) VALUES (
                        gen_random_uuid(), :user_id, 'device_data_pipeline_run',
                        'health_alerts', :patient_id,
                        :details, NOW()
                    )
                """),
                {
                    "user_id": patient_id,
                    "patient_id": patient_id,
                    "details": f'{{"sections_analyzed": {len(section_responses)}, "alerts_generated": {len(created_alerts)}}}',
                }
            )
            db.commit()
        except Exception as e:
            logger.warning(f"Could not log audit event: {e}")
        
        return DeviceAlertPipelineResponse(
            patient_id=patient_id,
            generated_at=profile.generated_at,
            overall_risk_score=profile.overall_risk_score,
            overall_trend=profile.overall_trend.value,
            sections=section_responses,
            critical_alerts=profile.critical_alerts,
            alerts_generated=len(created_alerts),
            new_alerts=created_alerts,
        )
        
    except Exception as e:
        logger.error(f"Error in device data alert pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/device-analytics/{patient_id}")
async def get_device_analytics(
    patient_id: str,
    section: Optional[str] = Query(default=None, description="Specific section to analyze"),
    days: int = Query(default=7, ge=1, le=30),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get device-based health analytics for a patient without generating alerts.
    
    Use this endpoint for read-only analytics viewing.
    For alert generation, use /device-data-pipeline/{patient_id}
    
    Args:
        patient_id: Patient's user ID
        section: Optional specific section (cardiovascular, respiratory, etc.)
        days: Number of days to analyze
    
    Returns:
        Health section analytics with risk scores, trends, and predictions
    """
    if not HEALTH_SECTION_ANALYTICS_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Health Section Analytics engine not available"
        )
    
    try:
        from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
        from sqlalchemy.orm import sessionmaker
        import asyncio
        
        database_url = os.environ.get("DATABASE_URL", "")
        if database_url.startswith("postgresql://"):
            async_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif database_url.startswith("postgres://"):
            async_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)
        else:
            async_url = database_url
        
        async_engine = create_async_engine(async_url, echo=False)
        AsyncSessionLocal = sessionmaker(
            async_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async def run_analysis():
            async with AsyncSessionLocal() as async_session:
                analytics_engine = HealthSectionAnalyticsEngine(async_session)
                
                if section:
                    try:
                        health_section = HealthSection(section)
                        result = await analytics_engine.analyze_section(
                            patient_id, health_section, days=days
                        )
                        return {"single_section": result}
                    except ValueError:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid section: {section}. Valid options: {[s.value for s in HealthSection]}"
                        )
                else:
                    profile = await analytics_engine.analyze_patient(patient_id, days=days)
                    return {"profile": profile}
        
        result = asyncio.get_event_loop().run_until_complete(run_analysis())
        
        if "single_section" in result:
            analytics = result["single_section"]
            return {
                "patient_id": patient_id,
                "section": analytics.section.value,
                "analysis": {
                    "deterioration_index": analytics.deterioration_index,
                    "risk_score": analytics.risk_score,
                    "risk_level": analytics.risk_level.value,
                    "trend": analytics.trend.value,
                    "trend_slope": analytics.trend_slope,
                    "trend_confidence": analytics.trend_confidence,
                    "stability_score": analytics.stability_score,
                    "data_coverage": analytics.data_coverage,
                    "data_points": analytics.data_points,
                    "anomalies_detected": analytics.anomalies_detected,
                    "anomaly_details": analytics.anomaly_details,
                    "risk_factors": analytics.risk_factors,
                    "predictions": analytics.predictions,
                    "recommendations": analytics.recommendations,
                    "alert_triggered": analytics.alert_triggered,
                    "alert_reason": analytics.alert_reason,
                    "timestamp": analytics.timestamp,
                }
            }
        else:
            profile = result["profile"]
            return {
                "patient_id": profile.patient_id,
                "generated_at": profile.generated_at,
                "overall_risk_score": profile.overall_risk_score,
                "overall_trend": profile.overall_trend.value,
                "critical_alerts": profile.critical_alerts,
                "recommendations": profile.recommendations,
                "sections": {
                    section.value: {
                        "deterioration_index": analytics.deterioration_index,
                        "risk_score": analytics.risk_score,
                        "risk_level": analytics.risk_level.value,
                        "trend": analytics.trend.value,
                        "stability_score": analytics.stability_score,
                        "data_coverage": analytics.data_coverage,
                        "anomalies_detected": analytics.anomalies_detected,
                        "alert_triggered": analytics.alert_triggered,
                    }
                    for section, analytics in profile.sections.items()
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in device analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========================================================================
# CORRELATION INSIGHTS ENDPOINTS
# ========================================================================

class CorrelationPair(BaseModel):
    """Individual correlation between two metrics"""
    metric_a: str
    metric_a_label: str
    metric_b: str
    metric_b_label: str
    correlation_coefficient: float
    p_value: float
    is_significant: bool
    strength: str  # strong, moderate, weak, none
    direction: str  # positive, negative
    sample_size: int
    category: str


class CorrelationCategory(BaseModel):
    """Category of correlations"""
    category_name: str
    category_label: str
    category_description: str
    correlations: List[CorrelationPair]
    total_correlations: int
    significant_correlations: int


class CorrelationInsightsResponse(BaseModel):
    """Comprehensive correlation insights for a patient"""
    patient_id: str
    generated_at: datetime
    categories: List[CorrelationCategory]
    summary: Dict[str, Any]
    recommendations: List[str]


def _calculate_correlation_strength(r: float, p_value: float) -> str:
    """Determine correlation strength based on coefficient"""
    if p_value > 0.05:
        return "none"
    abs_r = abs(r)
    if abs_r >= 0.7:
        return "strong"
    elif abs_r >= 0.4:
        return "moderate"
    elif abs_r >= 0.2:
        return "weak"
    return "none"


def _compute_pearson_correlation(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Compute Pearson correlation with statistical significance"""
    if len(x) < 5 or len(y) < 5 or len(x) != len(y):
        return {
            "r": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "sample_size": min(len(x), len(y))
        }
    
    try:
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)
        
        # Remove NaN values
        mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
        x_clean = x_arr[mask]
        y_clean = y_arr[mask]
        
        if len(x_clean) < 5:
            return {
                "r": 0.0,
                "p_value": 1.0,
                "is_significant": False,
                "sample_size": len(x_clean)
            }
        
        r, p_value = stats.pearsonr(x_clean, y_clean)
        
        return {
            "r": float(r) if not np.isnan(r) else 0.0,
            "p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "is_significant": p_value < 0.05 if not np.isnan(p_value) else False,
            "sample_size": len(x_clean)
        }
    except Exception as e:
        logger.warning(f"Error computing correlation: {e}")
        return {
            "r": 0.0,
            "p_value": 1.0,
            "is_significant": False,
            "sample_size": 0
        }


@router.get("/correlation-insights/{patient_id}")
async def get_correlation_insights(
    patient_id: str,
    days: int = Query(30, ge=7, le=90, description="Days of data to analyze"),
    db: Session = Depends(get_db)
) -> CorrelationInsightsResponse:
    """
    Get comprehensive correlation insights for a patient.
    
    Analyzes correlations across:
    - Symptom-Medication: Temporal patterns between medications and symptoms
    - Activity-Sleep: Relationships between physical activity and sleep quality
    - Environmental-Health: Environmental factors affecting health metrics
    - Device Metrics: Cross-metric correlations from connected devices
    
    Returns correlation coefficients with statistical significance.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        categories = []
        total_significant = 0
        total_correlations = 0
        
        # ========= SYMPTOM-MEDICATION CORRELATIONS =========
        med_symptom_correlations = []
        
        # Query medication-symptom correlation data from stored records
        med_query = text("""
            SELECT 
                sec.medication_name,
                sec.symptom_name,
                sec.confidence_score,
                sec.correlation_strength,
                sec.time_to_onset_hours
            FROM side_effect_correlations sec
            JOIN medication_timeline mt ON mt.id = sec.medication_timeline_id
            WHERE mt.patient_id = :patient_id
            AND sec.analysis_date >= :cutoff
            ORDER BY sec.confidence_score DESC
            LIMIT 20
        """)
        
        try:
            med_results = db.execute(med_query, {"patient_id": patient_id, "cutoff": cutoff_date}).fetchall()
            
            for row in med_results:
                strength = row[3] if row[3] else "unknown"
                confidence = float(row[2]) if row[2] else 0.5
                
                med_symptom_correlations.append(CorrelationPair(
                    metric_a=f"med_{row[0].lower().replace(' ', '_')}" if row[0] else "unknown_med",
                    metric_a_label=row[0] or "Unknown Medication",
                    metric_b=f"symptom_{row[1].lower().replace(' ', '_')}" if row[1] else "unknown_symptom",
                    metric_b_label=row[1] or "Unknown Symptom",
                    correlation_coefficient=confidence,
                    p_value=0.05 if confidence > 0.6 else 0.1,
                    is_significant=confidence > 0.6,
                    strength=strength.lower() if strength else "unknown",
                    direction="positive",
                    sample_size=int(row[4]) if row[4] else 1,
                    category="symptom_medication"
                ))
        except Exception as e:
            logger.warning(f"Could not fetch medication correlations: {e}")
        
        if med_symptom_correlations:
            sig_count = sum(1 for c in med_symptom_correlations if c.is_significant)
            categories.append(CorrelationCategory(
                category_name="symptom_medication",
                category_label="Symptom-Medication Correlations",
                category_description="Temporal patterns between medication changes and symptom onset",
                correlations=med_symptom_correlations,
                total_correlations=len(med_symptom_correlations),
                significant_correlations=sig_count
            ))
            total_correlations += len(med_symptom_correlations)
            total_significant += sig_count
        
        # ========= ACTIVITY-SLEEP CORRELATIONS =========
        activity_sleep_correlations = []
        
        # Query activity and sleep data
        sleep_activity_query = text("""
            SELECT 
                dr.recorded_at::date as record_date,
                dr.heart_rate,
                dr.steps,
                dr.sleep_score,
                dr.respiratory_rate,
                dr.spo2
            FROM device_readings dr
            WHERE dr.patient_id = :patient_id
            AND dr.recorded_at >= :cutoff
            ORDER BY dr.recorded_at
        """)
        
        try:
            device_results = db.execute(sleep_activity_query, {"patient_id": patient_id, "cutoff": cutoff_date}).fetchall()
            
            if device_results and len(device_results) >= 5:
                # Extract data for correlation
                steps_data = [float(r[2]) for r in device_results if r[2] is not None]
                sleep_data = [float(r[3]) for r in device_results if r[3] is not None]
                hr_data = [float(r[1]) for r in device_results if r[1] is not None]
                rr_data = [float(r[4]) for r in device_results if r[4] is not None]
                spo2_data = [float(r[5]) for r in device_results if r[5] is not None]
                
                # Steps vs Sleep Quality
                if len(steps_data) >= 5 and len(sleep_data) >= 5:
                    min_len = min(len(steps_data), len(sleep_data))
                    corr = _compute_pearson_correlation(steps_data[:min_len], sleep_data[:min_len])
                    
                    activity_sleep_correlations.append(CorrelationPair(
                        metric_a="daily_steps",
                        metric_a_label="Daily Steps",
                        metric_b="sleep_score",
                        metric_b_label="Sleep Quality Score",
                        correlation_coefficient=corr["r"],
                        p_value=corr["p_value"],
                        is_significant=corr["is_significant"],
                        strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                        direction="positive" if corr["r"] > 0 else "negative",
                        sample_size=corr["sample_size"],
                        category="activity_sleep"
                    ))
                
                # Heart Rate vs Sleep Quality
                if len(hr_data) >= 5 and len(sleep_data) >= 5:
                    min_len = min(len(hr_data), len(sleep_data))
                    corr = _compute_pearson_correlation(hr_data[:min_len], sleep_data[:min_len])
                    
                    activity_sleep_correlations.append(CorrelationPair(
                        metric_a="resting_heart_rate",
                        metric_a_label="Resting Heart Rate",
                        metric_b="sleep_score",
                        metric_b_label="Sleep Quality Score",
                        correlation_coefficient=corr["r"],
                        p_value=corr["p_value"],
                        is_significant=corr["is_significant"],
                        strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                        direction="positive" if corr["r"] > 0 else "negative",
                        sample_size=corr["sample_size"],
                        category="activity_sleep"
                    ))
                
                # Steps vs Respiratory Rate
                if len(steps_data) >= 5 and len(rr_data) >= 5:
                    min_len = min(len(steps_data), len(rr_data))
                    corr = _compute_pearson_correlation(steps_data[:min_len], rr_data[:min_len])
                    
                    activity_sleep_correlations.append(CorrelationPair(
                        metric_a="daily_steps",
                        metric_a_label="Daily Steps",
                        metric_b="respiratory_rate",
                        metric_b_label="Respiratory Rate",
                        correlation_coefficient=corr["r"],
                        p_value=corr["p_value"],
                        is_significant=corr["is_significant"],
                        strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                        direction="positive" if corr["r"] > 0 else "negative",
                        sample_size=corr["sample_size"],
                        category="activity_sleep"
                    ))
        except Exception as e:
            logger.warning(f"Could not compute activity-sleep correlations: {e}")
        
        if activity_sleep_correlations:
            sig_count = sum(1 for c in activity_sleep_correlations if c.is_significant)
            categories.append(CorrelationCategory(
                category_name="activity_sleep",
                category_label="Activity-Sleep Correlations",
                category_description="Relationships between physical activity patterns and sleep quality",
                correlations=activity_sleep_correlations,
                total_correlations=len(activity_sleep_correlations),
                significant_correlations=sig_count
            ))
            total_correlations += len(activity_sleep_correlations)
            total_significant += sig_count
        
        # ========= ENVIRONMENTAL-HEALTH CORRELATIONS =========
        env_health_correlations = []
        
        # Query environmental risk data if available
        env_query = text("""
            SELECT 
                er.recorded_at::date,
                er.aqi_value,
                er.pollen_level,
                er.temperature,
                er.humidity
            FROM environmental_risks er
            WHERE er.patient_id = :patient_id
            AND er.recorded_at >= :cutoff
            ORDER BY er.recorded_at
        """)
        
        try:
            env_results = db.execute(env_query, {"patient_id": patient_id, "cutoff": cutoff_date}).fetchall()
            
            # Query symptom severity data for same period
            symptom_severity_query = text("""
                SELECT 
                    DATE(reported_at) as report_date,
                    AVG(severity) as avg_severity
                FROM symptom_logs
                WHERE patient_id = :patient_id
                AND reported_at >= :cutoff
                GROUP BY DATE(reported_at)
                ORDER BY report_date
            """)
            
            symptom_results = db.execute(symptom_severity_query, {"patient_id": patient_id, "cutoff": cutoff_date}).fetchall()
            
            if env_results and symptom_results and len(env_results) >= 5 and len(symptom_results) >= 5:
                # Match dates and compute correlations
                env_by_date = {r[0]: r for r in env_results}
                symptom_by_date = {r[0]: r[1] for r in symptom_results}
                
                common_dates = set(env_by_date.keys()) & set(symptom_by_date.keys())
                
                if len(common_dates) >= 5:
                    sorted_dates = sorted(common_dates)
                    aqi_values = [float(env_by_date[d][1]) for d in sorted_dates if env_by_date[d][1] is not None]
                    severity_values = [float(symptom_by_date[d]) for d in sorted_dates]
                    
                    if len(aqi_values) >= 5 and len(severity_values) >= 5:
                        min_len = min(len(aqi_values), len(severity_values))
                        corr = _compute_pearson_correlation(aqi_values[:min_len], severity_values[:min_len])
                        
                        env_health_correlations.append(CorrelationPair(
                            metric_a="air_quality_index",
                            metric_a_label="Air Quality Index",
                            metric_b="symptom_severity",
                            metric_b_label="Average Symptom Severity",
                            correlation_coefficient=corr["r"],
                            p_value=corr["p_value"],
                            is_significant=corr["is_significant"],
                            strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                            direction="positive" if corr["r"] > 0 else "negative",
                            sample_size=corr["sample_size"],
                            category="environmental_health"
                        ))
        except Exception as e:
            logger.warning(f"Could not compute environmental correlations: {e}")
        
        if env_health_correlations:
            sig_count = sum(1 for c in env_health_correlations if c.is_significant)
            categories.append(CorrelationCategory(
                category_name="environmental_health",
                category_label="Environmental-Health Correlations",
                category_description="Impact of environmental factors on health symptoms",
                correlations=env_health_correlations,
                total_correlations=len(env_health_correlations),
                significant_correlations=sig_count
            ))
            total_correlations += len(env_health_correlations)
            total_significant += sig_count
        
        # ========= DEVICE METRIC CORRELATIONS =========
        device_correlations = []
        
        try:
            if device_results and len(device_results) >= 5:
                # Heart Rate vs SpO2
                if len(hr_data) >= 5 and len(spo2_data) >= 5:
                    min_len = min(len(hr_data), len(spo2_data))
                    corr = _compute_pearson_correlation(hr_data[:min_len], spo2_data[:min_len])
                    
                    device_correlations.append(CorrelationPair(
                        metric_a="heart_rate",
                        metric_a_label="Heart Rate",
                        metric_b="spo2",
                        metric_b_label="Blood Oxygen (SpO2)",
                        correlation_coefficient=corr["r"],
                        p_value=corr["p_value"],
                        is_significant=corr["is_significant"],
                        strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                        direction="positive" if corr["r"] > 0 else "negative",
                        sample_size=corr["sample_size"],
                        category="device_metrics"
                    ))
                
                # Heart Rate vs Respiratory Rate
                if len(hr_data) >= 5 and len(rr_data) >= 5:
                    min_len = min(len(hr_data), len(rr_data))
                    corr = _compute_pearson_correlation(hr_data[:min_len], rr_data[:min_len])
                    
                    device_correlations.append(CorrelationPair(
                        metric_a="heart_rate",
                        metric_a_label="Heart Rate",
                        metric_b="respiratory_rate",
                        metric_b_label="Respiratory Rate",
                        correlation_coefficient=corr["r"],
                        p_value=corr["p_value"],
                        is_significant=corr["is_significant"],
                        strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                        direction="positive" if corr["r"] > 0 else "negative",
                        sample_size=corr["sample_size"],
                        category="device_metrics"
                    ))
                
                # SpO2 vs Respiratory Rate
                if len(spo2_data) >= 5 and len(rr_data) >= 5:
                    min_len = min(len(spo2_data), len(rr_data))
                    corr = _compute_pearson_correlation(spo2_data[:min_len], rr_data[:min_len])
                    
                    device_correlations.append(CorrelationPair(
                        metric_a="spo2",
                        metric_a_label="Blood Oxygen (SpO2)",
                        metric_b="respiratory_rate",
                        metric_b_label="Respiratory Rate",
                        correlation_coefficient=corr["r"],
                        p_value=corr["p_value"],
                        is_significant=corr["is_significant"],
                        strength=_calculate_correlation_strength(corr["r"], corr["p_value"]),
                        direction="positive" if corr["r"] > 0 else "negative",
                        sample_size=corr["sample_size"],
                        category="device_metrics"
                    ))
        except Exception as e:
            logger.warning(f"Could not compute device metric correlations: {e}")
        
        if device_correlations:
            sig_count = sum(1 for c in device_correlations if c.is_significant)
            categories.append(CorrelationCategory(
                category_name="device_metrics",
                category_label="Device Metric Correlations",
                category_description="Cross-metric correlations from connected medical devices",
                correlations=device_correlations,
                total_correlations=len(device_correlations),
                significant_correlations=sig_count
            ))
            total_correlations += len(device_correlations)
            total_significant += sig_count
        
        # Generate recommendations based on significant correlations
        recommendations = []
        
        for category in categories:
            for corr in category.correlations:
                if corr.is_significant and corr.strength in ["strong", "moderate"]:
                    if corr.category == "symptom_medication":
                        recommendations.append(
                            f"Monitor {corr.metric_b_label} closely when taking {corr.metric_a_label} - "
                            f"a {corr.strength} correlation was detected."
                        )
                    elif corr.category == "activity_sleep":
                        direction_text = "increases" if corr.direction == "positive" else "decreases"
                        recommendations.append(
                            f"Your {corr.metric_a_label} {direction_text} with {corr.metric_b_label}. "
                            f"Consider tracking both metrics together."
                        )
                    elif corr.category == "environmental_health":
                        if "aqi" in corr.metric_a.lower() or "air" in corr.metric_a.lower():
                            recommendations.append(
                                "Air quality significantly impacts your symptoms. "
                                "Consider checking air quality before outdoor activities."
                            )
                    elif corr.category == "device_metrics":
                        recommendations.append(
                            f"{corr.metric_a_label} and {corr.metric_b_label} show a {corr.strength} correlation. "
                            "This relationship may be clinically relevant."
                        )
        
        # Limit recommendations
        recommendations = recommendations[:5]
        
        if not recommendations:
            recommendations = [
                "Continue collecting health data to improve correlation analysis.",
                "Consistent daily tracking helps identify meaningful patterns."
            ]
        
        return CorrelationInsightsResponse(
            patient_id=patient_id,
            generated_at=datetime.utcnow(),
            categories=categories,
            summary={
                "total_correlations": total_correlations,
                "significant_correlations": total_significant,
                "categories_analyzed": len(categories),
                "analysis_period_days": days
            },
            recommendations=recommendations
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing correlation insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))
