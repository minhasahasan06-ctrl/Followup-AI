"""
Rule-Based Alert Engine - 7 clinical alert rules with deduplication and suppression.

Core Rules:
1. Risk Jump: DPI bucket Green → Yellow within 24h
2. Persistent Yellow: Yellow bucket for >48h continuously
3. Any Red: DPI enters Red OR any organ_score >= red_threshold
4. Respiratory Spike: RR z>=3 OR RR absolute >= 30 bpm with confidence>0.8
5. Daily Check-in Deviation: Self-reported pain/fatigue z>=2 from baseline
6. Composite Sudden Increase: DPI increases >X points in 24h
7. Multi-Signal Corroboration: 2+ independent signals cross Yellow simultaneously

Includes deduplication via suppression keys, rate-limiting, and acknowledgment snooze.
"""

import os
import logging
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

from .config_service import AlertConfigService
from .dpi_computation import DPIResult
from .organ_scoring import OrganScoringResult, OrganScore


class AlertRule(Enum):
    """Alert rule identifiers"""
    RISK_JUMP = "risk_jump"
    PERSISTENT_YELLOW = "persistent_yellow"
    ANY_RED = "any_red"
    RESPIRATORY_SPIKE = "respiratory_spike"
    CHECKIN_DEVIATION = "checkin_deviation"
    COMPOSITE_JUMP = "composite_jump"
    MULTI_SIGNAL_CORROBORATION = "multi_signal_corroboration"
    # ML-based triggers (V2)
    ML_HIGH_RISK_PREDICTION = "ml_high_risk_prediction"
    ML_TRAJECTORY_DETERIORATING = "ml_trajectory_deteriorating"
    ML_CONFIDENCE_SPIKE = "ml_confidence_spike"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


@dataclass
class AlertTrigger:
    """Individual alert trigger from a rule"""
    rule: AlertRule
    severity: AlertSeverity
    priority: int  # 1-10, higher = more urgent
    title: str
    message: str
    trigger_metrics: List[Dict[str, Any]]
    dpi_at_trigger: Optional[float]
    organ_scores: Optional[Dict[str, float]]
    suppression_key: str
    corroborated: bool = False
    confidence: float = 1.0


@dataclass
class AlertRecord:
    """Full alert record for storage"""
    id: str
    patient_id: str
    alert_type: str
    alert_category: str
    severity: str
    priority: int
    title: str
    message: str
    disclaimer: str
    trigger_rule: str
    trigger_metrics: List[Dict[str, Any]]
    dpi_at_trigger: Optional[float]
    organ_scores: Optional[Dict[str, float]]
    suppression_key: str
    corroborated: bool
    status: str  # new, sent, acknowledged, escalated, closed
    escalation_probability: Optional[float] = None
    ml_priority_score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class RuleBasedAlertEngine:
    """Engine for generating alerts based on clinical rules"""
    
    COMPLIANCE_DISCLAIMER = "This is an observational pattern alert. Not a diagnosis or medical opinion. Requires clinician review."
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
    
    async def evaluate_all_rules(
        self,
        patient_id: str,
        dpi_result: DPIResult,
        organ_result: OrganScoringResult,
        metric_z_scores: Dict[str, float] = None,
        ml_prediction: Optional[Dict[str, Any]] = None
    ) -> List[AlertTrigger]:
        """
        Evaluate all 10 alert rules (7 statistical + 3 ML-based) and return triggered alerts.
        Applies deduplication and rate limiting.
        """
        config = self.config_service.config
        triggered_alerts = []
        
        # Get current patient data for context
        checkin_data = await self._get_recent_checkin_data(patient_id)
        
        # === STATISTICAL RULES (1-7) ===
        
        # Rule 1: Risk Jump (Green → Yellow)
        alert = self._check_risk_jump(patient_id, dpi_result)
        if alert:
            triggered_alerts.append(alert)
        
        # Rule 2: Persistent Yellow (>48h)
        alert = await self._check_persistent_yellow(patient_id, dpi_result)
        if alert:
            triggered_alerts.append(alert)
        
        # Rule 3: Any Red
        alerts = self._check_any_red(patient_id, dpi_result, organ_result)
        triggered_alerts.extend(alerts)
        
        # Rule 4: Respiratory Spike
        alert = await self._check_respiratory_spike(patient_id, metric_z_scores or {})
        if alert:
            triggered_alerts.append(alert)
        
        # Rule 5: Daily Check-in Deviation
        alert = await self._check_checkin_deviation(patient_id, checkin_data)
        if alert:
            triggered_alerts.append(alert)
        
        # Rule 6: Composite Sudden Increase
        alert = self._check_composite_jump(patient_id, dpi_result)
        if alert:
            triggered_alerts.append(alert)
        
        # Rule 7: Multi-Signal Corroboration
        alert = self._check_multi_signal_corroboration(patient_id, organ_result, metric_z_scores or {})
        if alert:
            triggered_alerts.append(alert)
        
        # === ML-BASED RULES (8-10) ===
        
        if ml_prediction:
            # Rule 8: ML High Risk Prediction
            alert = self._check_ml_high_risk_prediction(patient_id, ml_prediction, dpi_result)
            if alert:
                triggered_alerts.append(alert)
            
            # Rule 9: ML Trajectory Deteriorating
            alert = self._check_ml_trajectory_deteriorating(patient_id, ml_prediction, dpi_result)
            if alert:
                triggered_alerts.append(alert)
            
            # Rule 10: ML Confidence Spike
            alert = self._check_ml_confidence_spike(patient_id, ml_prediction, dpi_result)
            if alert:
                triggered_alerts.append(alert)
        
        # Apply deduplication and rate limiting
        filtered_alerts = await self._apply_suppression_rules(patient_id, triggered_alerts)
        
        return filtered_alerts
    
    def _check_risk_jump(
        self,
        patient_id: str,
        dpi_result: DPIResult
    ) -> Optional[AlertTrigger]:
        """Rule 1: DPI bucket moves Green → Yellow within 24h"""
        if not dpi_result.bucket_changed:
            return None
        
        if dpi_result.previous_bucket == "green" and dpi_result.dpi_bucket in ["yellow", "orange", "red"]:
            suppression_key = self._generate_suppression_key(
                patient_id, "risk_jump", datetime.utcnow()
            )
            
            severity = AlertSeverity.HIGH if dpi_result.dpi_bucket in ["orange", "red"] else AlertSeverity.MODERATE
            
            return AlertTrigger(
                rule=AlertRule.RISK_JUMP,
                severity=severity,
                priority=7 if severity == AlertSeverity.HIGH else 5,
                title="Deterioration Pattern Change Observed",
                message=f"Health status indicator moved from stable (green) to elevated ({dpi_result.dpi_bucket}). "
                        f"DPI changed from {dpi_result.previous_dpi:.1f} to {dpi_result.dpi_normalized:.1f}. "
                        f"Top contributing factors: {self._format_top_contributors(dpi_result.components[:3])}",
                trigger_metrics=[{
                    "name": "DPI",
                    "old_bucket": dpi_result.previous_bucket,
                    "new_bucket": dpi_result.dpi_bucket,
                    "old_value": dpi_result.previous_dpi,
                    "new_value": dpi_result.dpi_normalized
                }],
                dpi_at_trigger=dpi_result.dpi_normalized,
                organ_scores={c.organ_name: c.organ_score for c in dpi_result.components},
                suppression_key=suppression_key
            )
        return None
    
    async def _check_persistent_yellow(
        self,
        patient_id: str,
        dpi_result: DPIResult
    ) -> Optional[AlertTrigger]:
        """Rule 2: Patient in Yellow bucket for >48h continuously"""
        if dpi_result.dpi_bucket not in ["yellow", "orange"]:
            return None
        
        # Check how long patient has been in elevated state
        query = text("""
            SELECT computed_at, dpi_bucket
            FROM dpi_history
            WHERE patient_id = :patient_id
            AND computed_at >= NOW() - INTERVAL '48 hours'
            ORDER BY computed_at ASC
        """)
        
        try:
            results = self.db.execute(query, {"patient_id": patient_id}).fetchall()
            
            if not results:
                return None
            
            # Check if all readings in last 48h are yellow or worse
            all_elevated = all(row[1] in ["yellow", "orange", "red"] for row in results)
            earliest_elevated = results[0][0] if all_elevated else None
            
            if all_elevated and earliest_elevated:
                duration_hours = (datetime.utcnow() - earliest_elevated).total_seconds() / 3600
                
                if duration_hours >= 48:
                    suppression_key = self._generate_suppression_key(
                        patient_id, "persistent_yellow", datetime.utcnow()
                    )
                    
                    return AlertTrigger(
                        rule=AlertRule.PERSISTENT_YELLOW,
                        severity=AlertSeverity.HIGH,
                        priority=7,
                        title="Sustained Elevated Pattern Detected",
                        message=f"Health indicators have remained elevated for over {int(duration_hours)} hours. "
                                f"Current status: {dpi_result.dpi_bucket}. This persistent pattern warrants clinical review.",
                        trigger_metrics=[{
                            "name": "DPI_duration",
                            "bucket": dpi_result.dpi_bucket,
                            "duration_hours": duration_hours,
                            "current_dpi": dpi_result.dpi_normalized
                        }],
                        dpi_at_trigger=dpi_result.dpi_normalized,
                        organ_scores={c.organ_name: c.organ_score for c in dpi_result.components},
                        suppression_key=suppression_key
                    )
        except Exception as e:
            logger.warning(f"Error checking persistent yellow: {e}")
        
        return None
    
    def _check_any_red(
        self,
        patient_id: str,
        dpi_result: DPIResult,
        organ_result: OrganScoringResult
    ) -> List[AlertTrigger]:
        """Rule 3: DPI enters Red OR any organ score >= red threshold"""
        alerts = []
        config = self.config_service.config
        
        # Check if overall DPI is red
        if dpi_result.dpi_bucket == "red":
            suppression_key = self._generate_suppression_key(
                patient_id, "any_red_dpi", datetime.utcnow()
            )
            
            alerts.append(AlertTrigger(
                rule=AlertRule.ANY_RED,
                severity=AlertSeverity.CRITICAL,
                priority=9,
                title="Critical Pattern Alert",
                message=f"Overall health indicator has reached critical level (DPI: {dpi_result.dpi_normalized:.1f}). "
                        f"This requires immediate clinical review. "
                        f"Primary contributors: {self._format_top_contributors(dpi_result.components[:3])}",
                trigger_metrics=[{
                    "name": "DPI",
                    "value": dpi_result.dpi_normalized,
                    "bucket": "red"
                }],
                dpi_at_trigger=dpi_result.dpi_normalized,
                organ_scores={c.organ_name: c.organ_score for c in dpi_result.components},
                suppression_key=suppression_key
            ))
        
        # Check individual organ scores
        for group_name, organ_score in organ_result.organ_scores.items():
            group_config = config.organ_groups.get(group_name)
            if group_config and organ_score.normalized_score >= group_config.red_threshold:
                suppression_key = self._generate_suppression_key(
                    patient_id, f"any_red_{group_name}", datetime.utcnow()
                )
                
                alerts.append(AlertTrigger(
                    rule=AlertRule.ANY_RED,
                    severity=AlertSeverity.CRITICAL,
                    priority=9,
                    title=f"Critical {organ_score.organ_name} Pattern Alert",
                    message=f"{organ_score.organ_name} indicators have reached critical level "
                            f"(score: {organ_score.normalized_score:.1f}). "
                            f"Based on {organ_score.num_metrics} tracked metrics. "
                            f"Immediate clinical review recommended.",
                    trigger_metrics=[{
                        "name": f"{group_name}_score",
                        "value": organ_score.normalized_score,
                        "severity": "red",
                        "num_metrics": organ_score.num_metrics
                    }],
                    dpi_at_trigger=dpi_result.dpi_normalized,
                    organ_scores={c.organ_name: c.organ_score for c in dpi_result.components},
                    suppression_key=suppression_key
                ))
        
        return alerts
    
    async def _check_respiratory_spike(
        self,
        patient_id: str,
        metric_z_scores: Dict[str, float]
    ) -> Optional[AlertTrigger]:
        """Rule 4: RR z>=3 OR RR absolute >= 30 bpm with confidence>0.8"""
        config = self.config_service.config
        
        # Check z-score based respiratory metrics
        respiratory_metrics = ["respiratory_rate", "rr", "breath_rate", "audio_rr"]
        
        for metric in respiratory_metrics:
            if metric in metric_z_scores:
                z_score = metric_z_scores[metric]
                if abs(z_score) >= config.z_red_threshold:
                    suppression_key = self._generate_suppression_key(
                        patient_id, "respiratory_spike", datetime.utcnow()
                    )
                    
                    return AlertTrigger(
                        rule=AlertRule.RESPIRATORY_SPIKE,
                        severity=AlertSeverity.CRITICAL,
                        priority=9,
                        title="Respiratory Pattern Alert",
                        message=f"Significant deviation detected in respiratory metrics (z-score: {z_score:.2f}). "
                                f"This pattern indicates a notable change from baseline and requires review.",
                        trigger_metrics=[{
                            "name": metric,
                            "z_score": z_score,
                            "threshold": config.z_red_threshold
                        }],
                        dpi_at_trigger=None,
                        organ_scores=None,
                        suppression_key=suppression_key
                    )
        
        # Check absolute respiratory rate
        try:
            query = text("""
                SELECT metric_value, confidence
                FROM metric_ingest_log
                WHERE patient_id = :patient_id
                AND metric_name IN ('respiratory_rate', 'rr', 'breath_rate')
                AND timestamp >= NOW() - INTERVAL '1 hour'
                ORDER BY timestamp DESC
                LIMIT 1
            """)
            
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            
            if result:
                rr_value = float(result[0]) if result[0] else 0
                confidence = float(result[1]) if result[1] else 0
                
                if rr_value >= config.respiratory_rate_absolute_max and confidence >= config.respiratory_confidence_min:
                    suppression_key = self._generate_suppression_key(
                        patient_id, "respiratory_spike_absolute", datetime.utcnow()
                    )
                    
                    return AlertTrigger(
                        rule=AlertRule.RESPIRATORY_SPIKE,
                        severity=AlertSeverity.CRITICAL,
                        priority=9,
                        title="Respiratory Rate Alert",
                        message=f"Respiratory rate ({rr_value:.0f} bpm) exceeds threshold with high confidence ({confidence:.0%}). "
                                f"This elevated reading requires clinical review.",
                        trigger_metrics=[{
                            "name": "respiratory_rate",
                            "value": rr_value,
                            "confidence": confidence,
                            "threshold": config.respiratory_rate_absolute_max
                        }],
                        dpi_at_trigger=None,
                        organ_scores=None,
                        suppression_key=suppression_key
                    )
        except Exception as e:
            logger.warning(f"Error checking absolute respiratory rate: {e}")
        
        return None
    
    async def _check_checkin_deviation(
        self,
        patient_id: str,
        checkin_data: Dict[str, Any]
    ) -> Optional[AlertTrigger]:
        """Rule 5: Self-reported pain/fatigue deviates from baseline by z>=2"""
        config = self.config_service.config
        
        if not checkin_data:
            return None
        
        deviating_metrics = []
        
        for metric_name in ["pain_level", "fatigue_level", "breathlessness_level"]:
            if metric_name in checkin_data:
                z_score = checkin_data.get(f"{metric_name}_z", 0)
                if abs(z_score) >= config.z_yellow_threshold:
                    deviating_metrics.append({
                        "name": metric_name,
                        "value": checkin_data[metric_name],
                        "z_score": z_score
                    })
        
        if deviating_metrics:
            suppression_key = self._generate_suppression_key(
                patient_id, "checkin_deviation", datetime.utcnow()
            )
            
            max_z = max(abs(m["z_score"]) for m in deviating_metrics)
            severity = AlertSeverity.HIGH if max_z >= config.z_red_threshold else AlertSeverity.MODERATE
            
            metric_names = ", ".join(m["name"].replace("_", " ") for m in deviating_metrics)
            
            return AlertTrigger(
                rule=AlertRule.CHECKIN_DEVIATION,
                severity=severity,
                priority=7 if severity == AlertSeverity.HIGH else 5,
                title="Self-Reported Symptom Change",
                message=f"Patient-reported values for {metric_names} show significant deviation from baseline patterns. "
                        f"This self-reported change warrants clinician attention.",
                trigger_metrics=deviating_metrics,
                dpi_at_trigger=None,
                organ_scores=None,
                suppression_key=suppression_key
            )
        
        return None
    
    def _check_composite_jump(
        self,
        patient_id: str,
        dpi_result: DPIResult
    ) -> Optional[AlertTrigger]:
        """Rule 6: DPI increases >X points in 24h"""
        config = self.config_service.config
        
        if dpi_result.jump_detected and dpi_result.dpi_delta_24h:
            if dpi_result.dpi_delta_24h >= config.dpi_jump_threshold_24h:
                suppression_key = self._generate_suppression_key(
                    patient_id, "composite_jump", datetime.utcnow()
                )
                
                return AlertTrigger(
                    rule=AlertRule.COMPOSITE_JUMP,
                    severity=AlertSeverity.HIGH,
                    priority=7,
                    title="Rapid Deterioration Pattern",
                    message=f"Overall health indicator increased by {dpi_result.dpi_delta_24h:.1f} points in the past 24 hours "
                            f"(current: {dpi_result.dpi_normalized:.1f}). This rapid change pattern requires review.",
                    trigger_metrics=[{
                        "name": "DPI_delta_24h",
                        "delta": dpi_result.dpi_delta_24h,
                        "current": dpi_result.dpi_normalized,
                        "threshold": config.dpi_jump_threshold_24h
                    }],
                    dpi_at_trigger=dpi_result.dpi_normalized,
                    organ_scores={c.organ_name: c.organ_score for c in dpi_result.components},
                    suppression_key=suppression_key
                )
        
        return None
    
    def _check_multi_signal_corroboration(
        self,
        patient_id: str,
        organ_result: OrganScoringResult,
        metric_z_scores: Dict[str, float]
    ) -> Optional[AlertTrigger]:
        """Rule 7: 2+ independent signals cross Yellow simultaneously"""
        config = self.config_service.config
        
        elevated_signals = []
        
        # Check organ-level elevations
        for group_name, organ_score in organ_result.organ_scores.items():
            group_config = config.organ_groups.get(group_name)
            if group_config and organ_score.normalized_score >= group_config.yellow_threshold:
                elevated_signals.append({
                    "type": "organ",
                    "name": organ_score.organ_name,
                    "score": organ_score.normalized_score
                })
        
        # Check metric-level elevations
        for metric_name, z_score in metric_z_scores.items():
            if abs(z_score) >= config.z_yellow_threshold:
                elevated_signals.append({
                    "type": "metric",
                    "name": metric_name,
                    "z_score": z_score
                })
        
        # Need at least min_corroborating_signals
        if len(elevated_signals) >= config.min_corroborating_signals:
            suppression_key = self._generate_suppression_key(
                patient_id, "multi_signal", datetime.utcnow()
            )
            
            signal_names = ", ".join(s["name"] for s in elevated_signals[:5])
            
            return AlertTrigger(
                rule=AlertRule.MULTI_SIGNAL_CORROBORATION,
                severity=AlertSeverity.HIGH,
                priority=8,
                title="Multiple Elevated Indicators",
                message=f"{len(elevated_signals)} independent health indicators are showing elevated patterns: {signal_names}. "
                        f"This corroborated pattern provides stronger evidence for clinical review.",
                trigger_metrics=elevated_signals,
                dpi_at_trigger=None,
                organ_scores={s.organ_name: s.normalized_score for s in organ_result.organ_scores.values()},
                suppression_key=suppression_key,
                corroborated=True
            )
        
        return None
    
    # === ML-BASED RULE METHODS (V2) ===
    
    def _check_ml_high_risk_prediction(
        self,
        patient_id: str,
        ml_prediction: Dict[str, Any],
        dpi_result: DPIResult
    ) -> Optional[AlertTrigger]:
        """
        Rule 8: ML model predicts high deterioration risk (>=0.7) 
        with sufficient confidence (>=0.6)
        """
        HIGH_RISK_THRESHOLD = 0.7
        MIN_CONFIDENCE = 0.6
        
        ensemble_score = ml_prediction.get("ensemble_score", 0)
        confidence = ml_prediction.get("ensemble_confidence", 0)
        
        if ensemble_score >= HIGH_RISK_THRESHOLD and confidence >= MIN_CONFIDENCE:
            suppression_key = self._generate_suppression_key(
                patient_id, "ml_high_risk", datetime.utcnow()
            )
            
            # Get horizon-specific risks
            horizons = ml_prediction.get("predictions", {})
            horizon_alerts = []
            for horizon, pred in horizons.items():
                if isinstance(pred, dict):
                    prob = pred.get("deterioration_probability", 0)
                    if prob >= HIGH_RISK_THRESHOLD:
                        horizon_alerts.append(f"{horizon}: {prob:.0%}")
            
            horizon_summary = ", ".join(horizon_alerts) if horizon_alerts else "multiple horizons"
            
            return AlertTrigger(
                rule=AlertRule.ML_HIGH_RISK_PREDICTION,
                severity=AlertSeverity.CRITICAL if ensemble_score >= 0.85 else AlertSeverity.HIGH,
                priority=9 if ensemble_score >= 0.85 else 8,
                title="Predictive Risk Alert",
                message=f"AI analysis indicates elevated deterioration risk ({ensemble_score:.0%}) across {horizon_summary}. "
                        f"Confidence level: {confidence:.0%}. This predictive signal warrants proactive clinical review.",
                trigger_metrics=[{
                    "name": "ml_ensemble_score",
                    "value": ensemble_score,
                    "confidence": confidence,
                    "threshold": HIGH_RISK_THRESHOLD,
                    "horizons": horizons
                }],
                dpi_at_trigger=dpi_result.dpi_normalized if dpi_result else None,
                organ_scores={c.organ_name: c.organ_score for c in dpi_result.components} if dpi_result else None,
                suppression_key=suppression_key,
                confidence=confidence
            )
        
        return None
    
    def _check_ml_trajectory_deteriorating(
        self,
        patient_id: str,
        ml_prediction: Dict[str, Any],
        dpi_result: DPIResult
    ) -> Optional[AlertTrigger]:
        """
        Rule 9: ML predicts worsening trajectory across time horizons
        (short-term risk increasing to long-term risk)
        """
        MIN_TRAJECTORY_INCREASE = 0.2  # 20% increase from short to long term
        MIN_CONFIDENCE = 0.5
        
        confidence = ml_prediction.get("ensemble_confidence", 0)
        if confidence < MIN_CONFIDENCE:
            return None
        
        trend_direction = ml_prediction.get("trend_direction", "stable")
        risk_trajectory = ml_prediction.get("risk_trajectory", "low")
        
        # Check if trajectory shows deterioration pattern
        if trend_direction == "deteriorating" or risk_trajectory in ["high", "critical"]:
            # Analyze horizon progression
            horizons = ml_prediction.get("predictions", {})
            if len(horizons) >= 2:
                sorted_horizons = sorted(
                    [(k, v) for k, v in horizons.items() if isinstance(v, dict)],
                    key=lambda x: int(x[0].replace("h", "").replace("_hours", ""))
                )
                
                if len(sorted_horizons) >= 2:
                    short_term_prob = sorted_horizons[0][1].get("deterioration_probability", 0)
                    long_term_prob = sorted_horizons[-1][1].get("deterioration_probability", 0)
                    
                    trajectory_increase = long_term_prob - short_term_prob
                    
                    if trajectory_increase >= MIN_TRAJECTORY_INCREASE:
                        suppression_key = self._generate_suppression_key(
                            patient_id, "ml_trajectory", datetime.utcnow()
                        )
                        
                        return AlertTrigger(
                            rule=AlertRule.ML_TRAJECTORY_DETERIORATING,
                            severity=AlertSeverity.HIGH,
                            priority=7,
                            title="Deteriorating Trajectory Predicted",
                            message=f"AI analysis indicates worsening trajectory over time. "
                                    f"Short-term risk: {short_term_prob:.0%} → Long-term risk: {long_term_prob:.0%}. "
                                    f"This escalating pattern suggests attention may be needed.",
                            trigger_metrics=[{
                                "name": "ml_trajectory",
                                "short_term": short_term_prob,
                                "long_term": long_term_prob,
                                "increase": trajectory_increase,
                                "trend": trend_direction
                            }],
                            dpi_at_trigger=dpi_result.dpi_normalized if dpi_result else None,
                            organ_scores={c.organ_name: c.organ_score for c in dpi_result.components} if dpi_result else None,
                            suppression_key=suppression_key,
                            confidence=confidence
                        )
        
        return None
    
    def _check_ml_confidence_spike(
        self,
        patient_id: str,
        ml_prediction: Dict[str, Any],
        dpi_result: DPIResult
    ) -> Optional[AlertTrigger]:
        """
        Rule 10: ML confidence suddenly increases while risk is moderate-high
        (model becoming more certain of deterioration)
        """
        CONFIDENCE_SPIKE_THRESHOLD = 0.8  # Very high confidence
        MIN_RISK_FOR_SPIKE = 0.5  # At least moderate risk
        
        ensemble_score = ml_prediction.get("ensemble_score", 0)
        confidence = ml_prediction.get("ensemble_confidence", 0)
        
        # High confidence about moderate+ risk is concerning
        if confidence >= CONFIDENCE_SPIKE_THRESHOLD and ensemble_score >= MIN_RISK_FOR_SPIKE:
            # This rule is less severe - it's about certainty increasing
            suppression_key = self._generate_suppression_key(
                patient_id, "ml_confidence_spike", datetime.utcnow()
            )
            
            return AlertTrigger(
                rule=AlertRule.ML_CONFIDENCE_SPIKE,
                severity=AlertSeverity.MODERATE,
                priority=6,
                title="Increased Prediction Certainty",
                message=f"AI analysis shows high certainty ({confidence:.0%}) about elevated risk ({ensemble_score:.0%}). "
                        f"The model has strong confidence in this assessment based on current data patterns.",
                trigger_metrics=[{
                    "name": "ml_confidence",
                    "confidence": confidence,
                    "ensemble_score": ensemble_score,
                    "confidence_threshold": CONFIDENCE_SPIKE_THRESHOLD
                }],
                dpi_at_trigger=dpi_result.dpi_normalized if dpi_result else None,
                organ_scores={c.organ_name: c.organ_score for c in dpi_result.components} if dpi_result else None,
                suppression_key=suppression_key,
                confidence=confidence
            )
        
        return None
    
    async def _get_recent_checkin_data(self, patient_id: str) -> Dict[str, Any]:
        """Get recent checkin data with z-scores"""
        try:
            query = text("""
                SELECT pain_level, fatigue_level, breathlessness_level, mood, created_at
                FROM symptom_checkins
                WHERE user_id = :patient_id
                ORDER BY created_at DESC
                LIMIT 1
            """)
            
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            
            if result:
                # Get baselines and compute z-scores
                data = {
                    "pain_level": float(result[0]) if result[0] else None,
                    "fatigue_level": float(result[1]) if result[1] else None,
                    "breathlessness_level": float(result[2]) if result[2] else None,
                    "mood": float(result[3]) if result[3] else None,
                    "created_at": result[4]
                }
                
                # Get z-scores from trend metrics
                z_query = text("""
                    SELECT metric_name, z_score
                    FROM ai_trend_metrics
                    WHERE patient_id = :patient_id
                    AND metric_name IN ('pain_level', 'fatigue_level', 'breathlessness_level')
                    AND computed_at >= NOW() - INTERVAL '24 hours'
                    ORDER BY computed_at DESC
                """)
                
                z_results = self.db.execute(z_query, {"patient_id": patient_id}).fetchall()
                for row in z_results:
                    data[f"{row[0]}_z"] = float(row[1]) if row[1] else 0
                
                return data
            
            return {}
        except Exception as e:
            logger.warning(f"Error getting checkin data: {e}")
            return {}
    
    async def _apply_suppression_rules(
        self,
        patient_id: str,
        alerts: List[AlertTrigger]
    ) -> List[AlertTrigger]:
        """Apply deduplication, rate-limiting, and suppression rules"""
        config = self.config_service.config
        filtered = []
        
        for alert in alerts:
            # Check for existing open alert with same suppression key
            if await self._has_open_alert(patient_id, alert.suppression_key):
                logger.info(f"Suppressing duplicate alert: {alert.suppression_key}")
                continue
            
            # Check rate limiting
            daily_count = await self._get_daily_alert_count(patient_id)
            if daily_count >= config.max_alerts_per_patient_per_day:
                # Only allow critical alerts when rate limited
                if alert.severity != AlertSeverity.CRITICAL:
                    logger.info(f"Rate limiting alert: {alert.rule.value}")
                    continue
            
            # Check acknowledgment snooze
            if await self._is_snoozed(patient_id, alert.rule.value):
                logger.info(f"Snoozed alert: {alert.rule.value}")
                continue
            
            # Check corroboration requirement for high severity
            if config.corroboration_required_for_high_severity:
                if alert.severity == AlertSeverity.HIGH and not alert.corroborated:
                    # Downgrade to moderate if not corroborated
                    alert = AlertTrigger(
                        rule=alert.rule,
                        severity=AlertSeverity.MODERATE,
                        priority=alert.priority - 1,
                        title=alert.title,
                        message=alert.message,
                        trigger_metrics=alert.trigger_metrics,
                        dpi_at_trigger=alert.dpi_at_trigger,
                        organ_scores=alert.organ_scores,
                        suppression_key=alert.suppression_key,
                        corroborated=alert.corroborated
                    )
            
            filtered.append(alert)
        
        return filtered
    
    async def _has_open_alert(self, patient_id: str, suppression_key: str) -> bool:
        """Check if there's an open alert with the same suppression key"""
        config = self.config_service.config
        window_start = datetime.utcnow() - timedelta(hours=config.suppression_window_hours)
        
        query = text("""
            SELECT COUNT(*) FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND suppression_key = :suppression_key
            AND status NOT IN ('dismissed', 'closed')
            AND created_at >= :window_start
        """)
        
        try:
            count = self.db.execute(query, {
                "patient_id": patient_id,
                "suppression_key": suppression_key,
                "window_start": window_start
            }).scalar() or 0
            
            return count > 0
        except Exception as e:
            logger.warning(f"Error checking open alerts: {e}")
            return False
    
    async def _get_daily_alert_count(self, patient_id: str) -> int:
        """Get count of alerts created today for a patient"""
        query = text("""
            SELECT COUNT(*) FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND DATE(created_at) = DATE(NOW())
        """)
        
        try:
            return self.db.execute(query, {"patient_id": patient_id}).scalar() or 0
        except Exception as e:
            logger.warning(f"Error getting daily alert count: {e}")
            return 0
    
    async def _is_snoozed(self, patient_id: str, rule_type: str) -> bool:
        """Check if alert type is snoozed after recent acknowledgment"""
        config = self.config_service.config
        snooze_cutoff = datetime.utcnow() - timedelta(minutes=config.ack_snooze_minutes)
        
        query = text("""
            SELECT COUNT(*) FROM ai_health_alerts
            WHERE patient_id = :patient_id
            AND trigger_rule = :rule_type
            AND status = 'acknowledged'
            AND acknowledged_at >= :snooze_cutoff
        """)
        
        try:
            count = self.db.execute(query, {
                "patient_id": patient_id,
                "rule_type": rule_type,
                "snooze_cutoff": snooze_cutoff
            }).scalar() or 0
            
            return count > 0
        except Exception as e:
            logger.warning(f"Error checking snooze status: {e}")
            return False
    
    def _generate_suppression_key(
        self,
        patient_id: str,
        alert_type: str,
        timestamp: datetime
    ) -> str:
        """Generate suppression key for deduplication"""
        config = self.config_service.config
        # Round timestamp to suppression window
        window_start = timestamp.replace(
            hour=(timestamp.hour // config.suppression_window_hours) * config.suppression_window_hours,
            minute=0, second=0, microsecond=0
        )
        
        key_string = f"{patient_id}:{alert_type}:{window_start.isoformat()}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _format_top_contributors(self, components) -> str:
        """Format top contributing organs for alert message"""
        if not components:
            return "No specific contributors identified"
        
        parts = []
        for c in components[:3]:
            parts.append(f"{c.organ_name} ({c.percentage:.0f}%)")
        
        return ", ".join(parts)
    
    async def create_alert_record(
        self,
        patient_id: str,
        trigger: AlertTrigger
    ) -> AlertRecord:
        """Create and store an alert record from a trigger"""
        alert_id = str(uuid.uuid4())
        
        record = AlertRecord(
            id=alert_id,
            patient_id=patient_id,
            alert_type=trigger.rule.value,
            alert_category=trigger.rule.name,
            severity=trigger.severity.value,
            priority=trigger.priority,
            title=trigger.title,
            message=trigger.message,
            disclaimer=self.COMPLIANCE_DISCLAIMER,
            trigger_rule=trigger.rule.value,
            trigger_metrics=trigger.trigger_metrics,
            dpi_at_trigger=trigger.dpi_at_trigger,
            organ_scores=trigger.organ_scores,
            suppression_key=trigger.suppression_key,
            corroborated=trigger.corroborated,
            status="new"
        )
        
        # Store in database
        try:
            insert_query = text("""
                INSERT INTO ai_health_alerts (
                    id, patient_id, alert_type, alert_category, severity, priority,
                    title, message, disclaimer, trigger_rule, contributing_metrics,
                    dpi_at_trigger, organ_scores, suppression_key, corroborated,
                    status, created_at
                ) VALUES (
                    :id, :patient_id, :alert_type, :alert_category, :severity, :priority,
                    :title, :message, :disclaimer, :trigger_rule, :contributing_metrics::jsonb,
                    :dpi_at_trigger, :organ_scores::jsonb, :suppression_key, :corroborated,
                    :status, NOW()
                )
            """)
            
            self.db.execute(insert_query, {
                "id": record.id,
                "patient_id": record.patient_id,
                "alert_type": record.alert_type,
                "alert_category": record.alert_category,
                "severity": record.severity,
                "priority": record.priority,
                "title": record.title,
                "message": record.message,
                "disclaimer": record.disclaimer,
                "trigger_rule": record.trigger_rule,
                "contributing_metrics": json.dumps(record.trigger_metrics),
                "dpi_at_trigger": record.dpi_at_trigger,
                "organ_scores": json.dumps(record.organ_scores) if record.organ_scores else None,
                "suppression_key": record.suppression_key,
                "corroborated": record.corroborated,
                "status": record.status
            })
            self.db.commit()
            
        except Exception as e:
            logger.error(f"Error creating alert record: {e}")
            self.db.rollback()
            raise
        
        return record
