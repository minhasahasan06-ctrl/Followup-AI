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
