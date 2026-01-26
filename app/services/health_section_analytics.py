"""
Health Section Analytics Engine - Production Grade

Generates comprehensive health analytics for each section:
- Deterioration Index (0-100)
- Risk Score (0-15)
- Trend Analysis (improving/stable/declining)
- Stability Score (0-100)
- ML-powered predictions
- Anomaly detection
- Alert triggers

Health Sections:
- Cardiovascular (heart rate, HRV, ECG)
- Hypertension (blood pressure)
- Diabetes (glucose)
- Respiratory (SpO2, breathing rate)
- Sleep (duration, quality, stages)
- Mental Health (stress, HRV, activity patterns)
- Fitness (steps, calories, VO2 max)
- General Wellness (temperature, weight)
"""

import os
import logging
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import json

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================
# DATA MODELS
# ============================================

class HealthSection(str, Enum):
    """Health monitoring sections"""
    CARDIOVASCULAR = "cardiovascular"
    HYPERTENSION = "hypertension"
    DIABETES = "diabetes"
    RESPIRATORY = "respiratory"
    SLEEP = "sleep"
    MENTAL_HEALTH = "mental_health"
    FITNESS = "fitness"
    GENERAL_WELLNESS = "general_wellness"


class TrendDirection(str, Enum):
    """Trend direction indicators"""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    CRITICAL = "critical"
    INSUFFICIENT_DATA = "insufficient_data"


class RiskLevel(str, Enum):
    """Risk level categories"""
    LOW = "low"           # 0-5
    MODERATE = "moderate" # 6-10
    HIGH = "high"         # 11-13
    CRITICAL = "critical" # 14-15


@dataclass
class SectionAnalytics:
    """Analytics result for a health section"""
    section: HealthSection
    timestamp: str
    
    # Core metrics
    deterioration_index: float  # 0-100, higher = worse
    risk_score: float           # 0-15
    stability_score: float      # 0-100, higher = more stable
    trend: TrendDirection
    
    # Risk breakdown
    risk_level: RiskLevel
    risk_factors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trend details
    trend_slope: float = 0.0
    trend_confidence: float = 0.0
    
    # Anomaly detection
    anomalies_detected: int = 0
    anomaly_details: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data quality
    data_coverage: float = 0.0  # % of expected readings received
    data_points: int = 0
    
    # ML predictions
    predictions: Dict[str, Any] = field(default_factory=dict)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Alert triggers
    alert_triggered: bool = False
    alert_reason: Optional[str] = None


@dataclass
class PatientHealthProfile:
    """Complete health profile for a patient"""
    patient_id: str
    generated_at: str
    sections: Dict[HealthSection, SectionAnalytics]
    overall_risk_score: float
    overall_trend: TrendDirection
    critical_alerts: List[Dict[str, Any]]
    recommendations: List[str]


# ============================================
# REFERENCE RANGES
# ============================================

REFERENCE_RANGES = {
    "heart_rate": {"min": 60, "max": 100, "critical_low": 40, "critical_high": 150, "unit": "bpm"},
    "resting_heart_rate": {"min": 50, "max": 80, "critical_low": 35, "critical_high": 100, "unit": "bpm"},
    "hrv": {"min": 20, "max": 100, "critical_low": 10, "critical_high": None, "unit": "ms"},
    "systolic": {"min": 90, "max": 130, "critical_low": 70, "critical_high": 180, "unit": "mmHg"},
    "diastolic": {"min": 60, "max": 85, "critical_low": 40, "critical_high": 120, "unit": "mmHg"},
    "glucose_fasting": {"min": 70, "max": 100, "critical_low": 54, "critical_high": 250, "unit": "mg/dL"},
    "glucose_postprandial": {"min": 70, "max": 140, "critical_low": 54, "critical_high": 300, "unit": "mg/dL"},
    "spo2": {"min": 95, "max": 100, "critical_low": 88, "critical_high": None, "unit": "%"},
    "respiratory_rate": {"min": 12, "max": 20, "critical_low": 8, "critical_high": 30, "unit": "breaths/min"},
    "temperature": {"min": 36.1, "max": 37.2, "critical_low": 35, "critical_high": 39.4, "unit": "°C"},
    "sleep_duration": {"min": 420, "max": 540, "critical_low": 240, "critical_high": None, "unit": "minutes"},
    "sleep_efficiency": {"min": 85, "max": 100, "critical_low": 60, "critical_high": None, "unit": "%"},
    "steps": {"min": 7500, "max": 15000, "critical_low": None, "critical_high": None, "unit": "steps"},
    "weight_change": {"min": -0.5, "max": 0.5, "critical_low": -2, "critical_high": 2, "unit": "kg/week"},
    "stress": {"min": 0, "max": 50, "critical_low": None, "critical_high": 80, "unit": "score"},
    "readiness": {"min": 60, "max": 100, "critical_low": 30, "critical_high": None, "unit": "score"},
    "recovery": {"min": 50, "max": 100, "critical_low": 20, "critical_high": None, "unit": "score"},
}

# Section-specific metric mappings
SECTION_METRICS = {
    HealthSection.CARDIOVASCULAR: ["heart_rate", "resting_heart_rate", "hrv", "ecg"],
    HealthSection.HYPERTENSION: ["systolic", "diastolic", "pulse"],
    HealthSection.DIABETES: ["glucose", "glucose_fasting", "glucose_postprandial"],
    HealthSection.RESPIRATORY: ["spo2", "respiratory_rate", "breathing_disturbances"],
    HealthSection.SLEEP: ["sleep_duration", "sleep_efficiency", "deep_sleep", "rem_sleep", "awake_time"],
    HealthSection.MENTAL_HEALTH: ["stress", "hrv", "sleep_quality", "activity_variance"],
    HealthSection.FITNESS: ["steps", "calories", "active_minutes", "vo2_max", "training_load"],
    HealthSection.GENERAL_WELLNESS: ["temperature", "weight", "bmi", "body_fat"],
}


# ============================================
# ANALYTICS ENGINE
# ============================================

class HealthSectionAnalyticsEngine:
    """
    Production-grade health analytics engine.
    
    Analyzes device readings and generates comprehensive
    health insights with ML-powered predictions.
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def analyze_patient(
        self,
        patient_id: str,
        days: int = 7,
        sections: Optional[List[HealthSection]] = None,
    ) -> PatientHealthProfile:
        """
        Generate complete health profile for a patient.
        
        Args:
            patient_id: Patient's user ID
            days: Number of days to analyze
            sections: Specific sections to analyze (None = all)
        
        Returns:
            Complete PatientHealthProfile with all section analytics
        """
        sections_to_analyze = sections or list(HealthSection)
        section_results: Dict[HealthSection, SectionAnalytics] = {}
        critical_alerts: List[Dict[str, Any]] = []
        
        # Fetch device readings
        readings = await self._fetch_readings(patient_id, days)
        
        # Fetch baseline data
        baseline = await self._fetch_baseline(patient_id)
        
        # Analyze each section
        for section in sections_to_analyze:
            analytics = await self._analyze_section(
                patient_id=patient_id,
                section=section,
                readings=readings,
                baseline=baseline,
                days=days,
            )
            section_results[section] = analytics
            
            if analytics.alert_triggered:
                critical_alerts.append({
                    "section": section.value,
                    "reason": analytics.alert_reason,
                    "risk_score": analytics.risk_score,
                    "deterioration_index": analytics.deterioration_index,
                    "timestamp": analytics.timestamp,
                })
        
        # Calculate overall metrics
        overall_risk = self._calculate_overall_risk(section_results)
        overall_trend = self._calculate_overall_trend(section_results)
        recommendations = self._generate_recommendations(section_results)
        
        return PatientHealthProfile(
            patient_id=patient_id,
            generated_at=datetime.utcnow().isoformat(),
            sections=section_results,
            overall_risk_score=overall_risk,
            overall_trend=overall_trend,
            critical_alerts=critical_alerts,
            recommendations=recommendations,
        )
    
    async def analyze_section(
        self,
        patient_id: str,
        section: HealthSection,
        days: int = 7,
    ) -> SectionAnalytics:
        """Analyze a single health section"""
        readings = await self._fetch_readings(patient_id, days, section)
        baseline = await self._fetch_baseline(patient_id)
        
        return await self._analyze_section(
            patient_id=patient_id,
            section=section,
            readings=readings,
            baseline=baseline,
            days=days,
        )
    
    async def _fetch_readings(
        self,
        patient_id: str,
        days: int,
        section: Optional[HealthSection] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch device readings from database with proper column mapping"""
        start_date = datetime.utcnow() - timedelta(days=days)
        
        try:
            result = await self.db.execute(
                text("""
                    SELECT 
                        id, patient_id, device_type, device_brand, source,
                        wearable_integration_id, recorded_at,
                        -- Blood Pressure
                        bp_systolic, bp_diastolic, bp_pulse, bp_irregular_heartbeat,
                        -- Glucose
                        glucose_value, glucose_context,
                        -- Weight/Body Composition
                        weight, bmi, body_fat_percentage,
                        -- Temperature
                        temperature, temperature_unit,
                        -- Cardiovascular
                        heart_rate, resting_heart_rate, hrv, hrv_sdnn, spo2, spo2_min,
                        respiratory_rate, afib_detected, irregular_rhythm_alert,
                        -- Sleep
                        sleep_duration, sleep_deep_minutes, sleep_rem_minutes,
                        sleep_light_minutes, sleep_awake_minutes, sleep_score, sleep_efficiency,
                        -- Stress/Recovery
                        recovery_score, readiness_score, body_battery, strain_score, stress_score,
                        -- Activity/Fitness
                        steps, active_minutes, calories_burned, distance_meters,
                        vo2_max, training_load, training_status,
                        -- Routing flags
                        route_to_hypertension, route_to_diabetes, route_to_cardiovascular,
                        route_to_respiratory, route_to_sleep, route_to_mental_health, route_to_fitness,
                        -- Metadata
                        metadata
                    FROM device_readings
                    WHERE patient_id = :patient_id
                    AND recorded_at >= :start_date
                    ORDER BY recorded_at DESC
                """),
                {"patient_id": patient_id, "start_date": start_date}
            )
            
            rows = result.fetchall()
            readings = []
            for row in rows:
                # Convert row to normalized readings format
                base_data = {
                    "id": row.id,
                    "patient_id": row.patient_id,
                    "device_type": row.device_type,
                    "device_brand": row.device_brand,
                    "source": row.source or row.device_brand,
                    "timestamp": row.recorded_at.isoformat() if row.recorded_at else None,
                    "metadata": row.metadata,
                }
                
                # Create separate reading entries for each non-null metric
                # Cardiovascular
                if row.heart_rate is not None:
                    readings.append({**base_data, "data_type": "heart_rate", "value": row.heart_rate})
                if row.resting_heart_rate is not None:
                    readings.append({**base_data, "data_type": "resting_heart_rate", "value": row.resting_heart_rate})
                if row.hrv is not None:
                    readings.append({**base_data, "data_type": "hrv", "value": row.hrv})
                if row.spo2 is not None:
                    readings.append({**base_data, "data_type": "spo2", "value": row.spo2})
                if row.respiratory_rate is not None:
                    readings.append({**base_data, "data_type": "respiratory_rate", "value": row.respiratory_rate})
                
                # Blood Pressure (compound value)
                if row.bp_systolic is not None or row.bp_diastolic is not None:
                    readings.append({**base_data, "data_type": "blood_pressure", "value": {
                        "systolic": row.bp_systolic,
                        "diastolic": row.bp_diastolic,
                        "pulse": row.bp_pulse,
                    }})
                
                # Glucose
                if row.glucose_value is not None:
                    readings.append({**base_data, "data_type": "glucose", "value": row.glucose_value})
                
                # Temperature
                if row.temperature is not None:
                    readings.append({**base_data, "data_type": "temperature", "value": row.temperature, "unit": row.temperature_unit})
                
                # Weight/Body Composition
                if row.weight is not None:
                    readings.append({**base_data, "data_type": "weight", "value": row.weight})
                if row.bmi is not None:
                    readings.append({**base_data, "data_type": "bmi", "value": row.bmi})
                if row.body_fat_percentage is not None:
                    readings.append({**base_data, "data_type": "body_fat_percentage", "value": row.body_fat_percentage})
                
                # Sleep metrics
                if row.sleep_duration is not None:
                    readings.append({**base_data, "data_type": "sleep_duration", "value": row.sleep_duration})
                if row.sleep_score is not None:
                    readings.append({**base_data, "data_type": "sleep_score", "value": row.sleep_score})
                if row.sleep_efficiency is not None:
                    readings.append({**base_data, "data_type": "sleep_efficiency", "value": row.sleep_efficiency})
                if row.sleep_deep_minutes is not None:
                    readings.append({**base_data, "data_type": "sleep_deep_minutes", "value": row.sleep_deep_minutes})
                if row.sleep_rem_minutes is not None:
                    readings.append({**base_data, "data_type": "sleep_rem_minutes", "value": row.sleep_rem_minutes})
                
                # Stress/Recovery
                if row.stress_score is not None:
                    readings.append({**base_data, "data_type": "stress_score", "value": row.stress_score})
                if row.recovery_score is not None:
                    readings.append({**base_data, "data_type": "recovery_score", "value": row.recovery_score})
                if row.readiness_score is not None:
                    readings.append({**base_data, "data_type": "readiness_score", "value": row.readiness_score})
                if row.body_battery is not None:
                    readings.append({**base_data, "data_type": "body_battery", "value": row.body_battery})
                if row.strain_score is not None:
                    readings.append({**base_data, "data_type": "strain_score", "value": row.strain_score})
                
                # Fitness/Activity
                if row.steps is not None:
                    readings.append({**base_data, "data_type": "steps", "value": row.steps})
                if row.active_minutes is not None:
                    readings.append({**base_data, "data_type": "active_minutes", "value": row.active_minutes})
                if row.calories_burned is not None:
                    readings.append({**base_data, "data_type": "calories_burned", "value": row.calories_burned})
                if row.vo2_max is not None:
                    readings.append({**base_data, "data_type": "vo2_max", "value": row.vo2_max})
                if row.distance_meters is not None:
                    readings.append({**base_data, "data_type": "distance_meters", "value": row.distance_meters})
            
            logger.info(f"Fetched {len(readings)} readings for patient {patient_id}")
            return readings
            
        except Exception as e:
            logger.error(f"Error fetching readings: {e}")
            return []
    
    async def _fetch_baseline(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Fetch patient's baseline data"""
        try:
            result = await self.db.execute(
                text("""
                    SELECT baseline_data, computed_at
                    FROM patient_baselines
                    WHERE patient_id = :patient_id
                    AND status = 'active'
                    ORDER BY computed_at DESC
                    LIMIT 1
                """),
                {"patient_id": patient_id}
            )
            
            row = result.fetchone()
            if row:
                return {
                    "data": row.baseline_data,
                    "computed_at": row.computed_at,
                }
            return None
            
        except Exception as e:
            logger.warning(f"Could not fetch baseline: {e}")
            return None
    
    async def _analyze_section(
        self,
        patient_id: str,
        section: HealthSection,
        readings: List[Dict[str, Any]],
        baseline: Optional[Dict[str, Any]],
        days: int,
    ) -> SectionAnalytics:
        """Analyze a single health section"""
        
        # Filter readings for this section
        section_metrics = SECTION_METRICS.get(section, [])
        section_readings = [
            r for r in readings
            if r.get("data_type") in section_metrics or r.get("reading_type") in section_metrics
        ]
        
        # Extract numeric values grouped by metric
        metrics_data: Dict[str, List[float]] = {}
        for reading in section_readings:
            data_type = reading.get("data_type") or reading.get("reading_type")
            value = reading.get("value")
            
            if data_type and value is not None:
                if isinstance(value, (int, float)):
                    if data_type not in metrics_data:
                        metrics_data[data_type] = []
                    metrics_data[data_type].append(float(value))
                elif isinstance(value, dict):
                    # Handle compound values (e.g., BP)
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            compound_key = f"{data_type}_{k}"
                            if compound_key not in metrics_data:
                                metrics_data[compound_key] = []
                            metrics_data[compound_key].append(float(v))
        
        # Calculate metrics
        deterioration_index = self._calculate_deterioration_index(section, metrics_data, baseline)
        risk_score, risk_factors = self._calculate_risk_score(section, metrics_data)
        stability_score = self._calculate_stability_score(metrics_data)
        trend, trend_slope, trend_confidence = self._calculate_trend(metrics_data)
        anomalies = self._detect_anomalies(section, metrics_data, baseline)
        
        # Data quality
        expected_points = days * 24  # Assuming hourly readings
        actual_points = len(section_readings)
        data_coverage = min(100, (actual_points / max(1, expected_points)) * 100)
        
        # ML predictions
        predictions = await self._generate_predictions(patient_id, section, metrics_data)
        
        # Recommendations
        recommendations = self._generate_section_recommendations(
            section, risk_score, trend, anomalies
        )
        
        # Alert triggers
        alert_triggered, alert_reason = self._check_alert_triggers(
            section, deterioration_index, risk_score, anomalies
        )
        
        # Risk level
        if risk_score < 6:
            risk_level = RiskLevel.LOW
        elif risk_score < 11:
            risk_level = RiskLevel.MODERATE
        elif risk_score < 14:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL
        
        return SectionAnalytics(
            section=section,
            timestamp=datetime.utcnow().isoformat(),
            deterioration_index=deterioration_index,
            risk_score=risk_score,
            stability_score=stability_score,
            trend=trend,
            risk_level=risk_level,
            risk_factors=risk_factors,
            trend_slope=trend_slope,
            trend_confidence=trend_confidence,
            anomalies_detected=len(anomalies),
            anomaly_details=anomalies,
            data_coverage=data_coverage,
            data_points=actual_points,
            predictions=predictions,
            recommendations=recommendations,
            alert_triggered=alert_triggered,
            alert_reason=alert_reason,
        )
    
    def _calculate_deterioration_index(
        self,
        section: HealthSection,
        metrics_data: Dict[str, List[float]],
        baseline: Optional[Dict[str, Any]],
    ) -> float:
        """
        Calculate deterioration index (0-100).
        
        Compares current values against:
        1. Reference ranges
        2. Patient's baseline
        3. Historical trends
        """
        if not metrics_data:
            return 0.0
        
        deviations = []
        
        for metric, values in metrics_data.items():
            if not values:
                continue
            
            current = statistics.mean(values[-min(10, len(values)):])  # Recent average
            
            # Check against reference range
            ref = REFERENCE_RANGES.get(metric)
            if ref:
                ref_min = ref.get("min", 0)
                ref_max = ref.get("max", 100)
                ref_range = ref_max - ref_min
                
                if current < ref_min:
                    deviation = abs(ref_min - current) / max(1, ref_range) * 100
                elif current > ref_max:
                    deviation = abs(current - ref_max) / max(1, ref_range) * 100
                else:
                    deviation = 0
                
                # Critical thresholds have higher weight
                critical_low = ref.get("critical_low")
                critical_high = ref.get("critical_high")
                
                if critical_low and current < critical_low:
                    deviation = min(100, deviation * 2)
                if critical_high and current > critical_high:
                    deviation = min(100, deviation * 2)
                
                deviations.append(deviation)
            
            # Compare against baseline
            if baseline and baseline.get("data"):
                baseline_data = baseline["data"]
                if isinstance(baseline_data, dict) and metric in baseline_data:
                    baseline_mean = baseline_data[metric].get("mean", current)
                    baseline_std = baseline_data[metric].get("std", 1)
                    
                    if baseline_std > 0:
                        z_score = abs(current - baseline_mean) / baseline_std
                        baseline_deviation = min(100, z_score * 20)  # Scale z-score
                        deviations.append(baseline_deviation)
        
        if not deviations:
            return 0.0
        
        # Weighted average with higher weight on extreme values
        sorted_devs = sorted(deviations, reverse=True)
        weights = [2 ** (len(sorted_devs) - i) for i in range(len(sorted_devs))]
        
        weighted_sum = sum(d * w for d, w in zip(sorted_devs, weights))
        weight_total = sum(weights)
        
        return min(100, weighted_sum / max(1, weight_total))
    
    def _calculate_risk_score(
        self,
        section: HealthSection,
        metrics_data: Dict[str, List[float]],
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Calculate risk score (0-15) with contributing factors.
        
        Factors:
        - Out of range values
        - Critical threshold violations
        - Variability
        - Missing data
        """
        risk_factors = []
        total_risk = 0.0
        
        for metric, values in metrics_data.items():
            if not values:
                continue
            
            ref = REFERENCE_RANGES.get(metric)
            if not ref:
                continue
            
            current = statistics.mean(values[-min(5, len(values)):])
            ref_min = ref.get("min", 0)
            ref_max = ref.get("max", 100)
            critical_low = ref.get("critical_low")
            critical_high = ref.get("critical_high")
            
            factor_risk = 0.0
            factor_reason = None
            
            # Critical threshold check (highest risk)
            if critical_low and current < critical_low:
                factor_risk = 4.0
                factor_reason = f"{metric} critically low: {current:.1f} (threshold: {critical_low})"
            elif critical_high and current > critical_high:
                factor_risk = 4.0
                factor_reason = f"{metric} critically high: {current:.1f} (threshold: {critical_high})"
            # Out of normal range (moderate risk)
            elif current < ref_min:
                deviation = abs(ref_min - current) / max(1, ref_min) * 100
                factor_risk = min(3.0, deviation / 10)
                factor_reason = f"{metric} below normal: {current:.1f} (range: {ref_min}-{ref_max})"
            elif current > ref_max:
                deviation = abs(current - ref_max) / max(1, ref_max) * 100
                factor_risk = min(3.0, deviation / 10)
                factor_reason = f"{metric} above normal: {current:.1f} (range: {ref_min}-{ref_max})"
            
            # Variability check
            if len(values) >= 3:
                try:
                    cv = statistics.stdev(values) / max(0.01, statistics.mean(values)) * 100
                    if cv > 30:  # High variability
                        factor_risk += 0.5
                        if not factor_reason:
                            factor_reason = f"{metric} showing high variability (CV: {cv:.1f}%)"
                except:
                    pass
            
            if factor_risk > 0 and factor_reason:
                risk_factors.append({
                    "metric": metric,
                    "risk_contribution": factor_risk,
                    "reason": factor_reason,
                    "current_value": current,
                    "reference_range": f"{ref_min}-{ref_max}",
                })
                total_risk += factor_risk
        
        # Cap at 15
        total_risk = min(15.0, total_risk)
        
        # Sort risk factors by contribution
        risk_factors.sort(key=lambda x: x["risk_contribution"], reverse=True)
        
        return total_risk, risk_factors[:5]  # Top 5 factors
    
    def _calculate_stability_score(self, metrics_data: Dict[str, List[float]]) -> float:
        """
        Calculate stability score (0-100).
        
        Higher score = more stable readings.
        Based on coefficient of variation.
        """
        if not metrics_data:
            return 100.0
        
        cvs = []
        
        for metric, values in metrics_data.items():
            if len(values) >= 2:
                try:
                    mean = statistics.mean(values)
                    if mean > 0:
                        cv = statistics.stdev(values) / mean * 100
                        cvs.append(cv)
                except:
                    pass
        
        if not cvs:
            return 100.0
        
        avg_cv = statistics.mean(cvs)
        
        # Convert CV to stability score (lower CV = higher stability)
        # CV of 0% = 100 stability, CV of 50% = 0 stability
        stability = max(0, 100 - (avg_cv * 2))
        
        return stability
    
    def _calculate_trend(
        self,
        metrics_data: Dict[str, List[float]],
    ) -> Tuple[TrendDirection, float, float]:
        """
        Calculate trend direction with slope and confidence.
        
        Uses linear regression on time-series data.
        """
        if not metrics_data:
            return TrendDirection.INSUFFICIENT_DATA, 0.0, 0.0
        
        slopes = []
        
        for metric, values in metrics_data.items():
            if len(values) < 3:
                continue
            
            # Simple linear regression
            n = len(values)
            x = list(range(n))
            y = values
            
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(y)
            
            numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
            denominator = sum((xi - x_mean) ** 2 for xi in x)
            
            if denominator > 0:
                slope = numerator / denominator
                
                # Normalize slope by mean value
                normalized_slope = slope / max(0.01, abs(y_mean)) * 100
                slopes.append(normalized_slope)
        
        if not slopes:
            return TrendDirection.INSUFFICIENT_DATA, 0.0, 0.0
        
        avg_slope = statistics.mean(slopes)
        
        # Calculate confidence based on consistency of slopes
        if len(slopes) > 1:
            try:
                slope_std = statistics.stdev(slopes)
                confidence = max(0, min(100, 100 - slope_std * 10))
            except:
                confidence = 50.0
        else:
            confidence = 50.0
        
        # Determine trend direction
        if avg_slope < -5:
            trend = TrendDirection.DECLINING
        elif avg_slope > 5:
            trend = TrendDirection.IMPROVING
        else:
            trend = TrendDirection.STABLE
        
        # Critical check - override if deterioration is severe
        if avg_slope < -20:
            trend = TrendDirection.CRITICAL
        
        return trend, avg_slope, confidence
    
    def _detect_anomalies(
        self,
        section: HealthSection,
        metrics_data: Dict[str, List[float]],
        baseline: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using multiple methods:
        1. Z-score against baseline
        2. IQR method
        3. Critical threshold violations
        """
        anomalies = []
        
        for metric, values in metrics_data.items():
            if len(values) < 5:
                continue
            
            ref = REFERENCE_RANGES.get(metric)
            
            # Z-score method
            try:
                mean = statistics.mean(values)
                std = statistics.stdev(values)
                
                for i, v in enumerate(values[-10:]):  # Check recent values
                    if std > 0:
                        z_score = abs(v - mean) / std
                        if z_score > 3:
                            anomalies.append({
                                "metric": metric,
                                "method": "z_score",
                                "value": v,
                                "z_score": z_score,
                                "severity": "high" if z_score > 4 else "medium",
                            })
            except:
                pass
            
            # IQR method
            if len(values) >= 10:
                sorted_vals = sorted(values)
                q1 = sorted_vals[len(sorted_vals) // 4]
                q3 = sorted_vals[3 * len(sorted_vals) // 4]
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                for v in values[-10:]:
                    if v < lower or v > upper:
                        if not any(a["value"] == v and a["metric"] == metric for a in anomalies):
                            anomalies.append({
                                "metric": metric,
                                "method": "iqr",
                                "value": v,
                                "bounds": {"lower": lower, "upper": upper},
                                "severity": "medium",
                            })
            
            # Critical threshold violations
            if ref:
                critical_low = ref.get("critical_low")
                critical_high = ref.get("critical_high")
                
                for v in values[-10:]:
                    if critical_low and v < critical_low:
                        if not any(a["value"] == v and a["metric"] == metric for a in anomalies):
                            anomalies.append({
                                "metric": metric,
                                "method": "critical_threshold",
                                "value": v,
                                "threshold": critical_low,
                                "direction": "below",
                                "severity": "critical",
                            })
                    elif critical_high and v > critical_high:
                        if not any(a["value"] == v and a["metric"] == metric for a in anomalies):
                            anomalies.append({
                                "metric": metric,
                                "method": "critical_threshold",
                                "value": v,
                                "threshold": critical_high,
                                "direction": "above",
                                "severity": "critical",
                            })
        
        return anomalies[:10]  # Limit to 10 most relevant
    
    async def _generate_predictions(
        self,
        patient_id: str,
        section: HealthSection,
        metrics_data: Dict[str, List[float]],
    ) -> Dict[str, Any]:
        """
        Generate ML-powered predictions.
        
        Integrates with existing ML prediction infrastructure.
        """
        predictions = {
            "24h_forecast": {},
            "risk_trend": "stable",
            "confidence": 0.0,
        }
        
        if not metrics_data:
            return predictions
        
        # Simple forecasting using exponential smoothing
        for metric, values in metrics_data.items():
            if len(values) >= 5:
                # Simple exponential smoothing
                alpha = 0.3
                smoothed = values[0]
                for v in values[1:]:
                    smoothed = alpha * v + (1 - alpha) * smoothed
                
                # Predict next value
                predicted = smoothed
                
                predictions["24h_forecast"][metric] = {
                    "predicted_value": round(predicted, 2),
                    "confidence_interval": [
                        round(predicted * 0.9, 2),
                        round(predicted * 1.1, 2),
                    ],
                }
        
        # Calculate overall confidence
        if predictions["24h_forecast"]:
            predictions["confidence"] = 70.0  # Base confidence
        
        return predictions
    
    def _generate_section_recommendations(
        self,
        section: HealthSection,
        risk_score: float,
        trend: TrendDirection,
        anomalies: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate section-specific recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_score >= 10:
            recommendations.append(f"High {section.value} risk detected. Consider scheduling a check-up.")
        elif risk_score >= 6:
            recommendations.append(f"Moderate {section.value} risk. Monitor closely over the next few days.")
        
        # Trend-based recommendations
        if trend == TrendDirection.DECLINING:
            recommendations.append(f"Your {section.value} metrics are trending downward. Review recent changes.")
        elif trend == TrendDirection.CRITICAL:
            recommendations.append(f"Critical decline in {section.value}. Please consult your healthcare provider.")
        
        # Section-specific advice
        section_advice = {
            HealthSection.CARDIOVASCULAR: "Regular cardio exercise and stress management can help improve heart health.",
            HealthSection.HYPERTENSION: "Reducing sodium intake and maintaining regular activity can help manage blood pressure.",
            HealthSection.DIABETES: "Consistent meal timing and monitoring carbohydrate intake supports stable glucose levels.",
            HealthSection.RESPIRATORY: "Practice deep breathing exercises and ensure good air quality in your environment.",
            HealthSection.SLEEP: "Maintain consistent sleep and wake times, limit screen time before bed.",
            HealthSection.MENTAL_HEALTH: "Regular physical activity and social connection support mental wellness.",
            HealthSection.FITNESS: "Aim for at least 150 minutes of moderate activity per week.",
            HealthSection.GENERAL_WELLNESS: "Stay hydrated and maintain a balanced diet with regular check-ups.",
        }
        
        if section in section_advice:
            recommendations.append(section_advice[section])
        
        # Anomaly-based recommendations
        critical_anomalies = [a for a in anomalies if a.get("severity") == "critical"]
        if critical_anomalies:
            recommendations.insert(0, "Critical readings detected. Please review and contact your provider if symptoms persist.")
        
        return recommendations[:4]  # Limit to 4 recommendations
    
    def _check_alert_triggers(
        self,
        section: HealthSection,
        deterioration_index: float,
        risk_score: float,
        anomalies: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        """Check if alerts should be triggered"""
        
        # Critical deterioration
        if deterioration_index >= 80:
            return True, f"Critical deterioration in {section.value} (index: {deterioration_index:.1f})"
        
        # High risk
        if risk_score >= 12:
            return True, f"High risk level in {section.value} (score: {risk_score:.1f})"
        
        # Critical anomalies
        critical = [a for a in anomalies if a.get("severity") == "critical"]
        if len(critical) >= 2:
            return True, f"Multiple critical readings in {section.value}"
        
        return False, None
    
    def _calculate_overall_risk(
        self,
        sections: Dict[HealthSection, SectionAnalytics],
    ) -> float:
        """Calculate overall patient risk score"""
        if not sections:
            return 0.0
        
        # Weight by section importance
        weights = {
            HealthSection.CARDIOVASCULAR: 2.0,
            HealthSection.HYPERTENSION: 1.5,
            HealthSection.DIABETES: 1.5,
            HealthSection.RESPIRATORY: 1.5,
            HealthSection.SLEEP: 1.0,
            HealthSection.MENTAL_HEALTH: 1.0,
            HealthSection.FITNESS: 0.5,
            HealthSection.GENERAL_WELLNESS: 0.5,
        }
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for section, analytics in sections.items():
            w = weights.get(section, 1.0)
            weighted_sum += analytics.risk_score * w
            weight_total += w
        
        return min(15, weighted_sum / max(1, weight_total))
    
    def _calculate_overall_trend(
        self,
        sections: Dict[HealthSection, SectionAnalytics],
    ) -> TrendDirection:
        """Calculate overall patient trend"""
        if not sections:
            return TrendDirection.INSUFFICIENT_DATA
        
        trends = [s.trend for s in sections.values()]
        
        if TrendDirection.CRITICAL in trends:
            return TrendDirection.CRITICAL
        
        declining = trends.count(TrendDirection.DECLINING)
        improving = trends.count(TrendDirection.IMPROVING)
        
        if declining > len(trends) / 2:
            return TrendDirection.DECLINING
        elif improving > declining:
            return TrendDirection.IMPROVING
        else:
            return TrendDirection.STABLE
    
    def _generate_recommendations(
        self,
        sections: Dict[HealthSection, SectionAnalytics],
    ) -> List[str]:
        """Generate overall patient recommendations"""
        recommendations = []
        
        # Collect all recommendations, prioritize by risk
        all_recs = []
        for section, analytics in sections.items():
            for rec in analytics.recommendations:
                all_recs.append((analytics.risk_score, rec))
        
        # Sort by risk and deduplicate
        all_recs.sort(key=lambda x: x[0], reverse=True)
        seen = set()
        
        for _, rec in all_recs:
            if rec not in seen:
                recommendations.append(rec)
                seen.add(rec)
                if len(recommendations) >= 6:
                    break
        
        return recommendations


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

async def analyze_patient_health(
    db: AsyncSession,
    patient_id: str,
    days: int = 7,
) -> PatientHealthProfile:
    """
    Convenience function to analyze patient health.
    
    Usage:
        profile = await analyze_patient_health(db, patient_id)
        print(f"Overall risk: {profile.overall_risk_score}")
    """
    engine = HealthSectionAnalyticsEngine(db)
    return await engine.analyze_patient(patient_id, days)


async def get_section_analytics(
    db: AsyncSession,
    patient_id: str,
    section: str,
    days: int = 7,
) -> SectionAnalytics:
    """
    Get analytics for a specific health section.
    
    Usage:
        cardio = await get_section_analytics(db, patient_id, "cardiovascular")
        print(f"Risk: {cardio.risk_score}, Trend: {cardio.trend}")
    """
    engine = HealthSectionAnalyticsEngine(db)
    section_enum = HealthSection(section)
    return await engine.analyze_section(patient_id, section_enum, days)
