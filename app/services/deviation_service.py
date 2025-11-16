"""
Deviation Detection Service for Health Change Monitoring

Calculates z-scores and detects health pattern changes.
Used for wellness monitoring (NOT medical diagnosis).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import statistics

from app.models.health_baseline import HealthBaseline, BaselineDeviation
from app.models.pain_tracking import PainMeasurement, PainQuestionnaire
from app.models.symptom_journal import SymptomMeasurement, BodyArea
from app.models.medication_side_effects import SymptomLog
from app.services.baseline_service import BaselineCalculationService


class DeviationDetectionService:
    """
    Service for detecting health pattern deviations from baseline.
    Supports wellness monitoring and change detection.
    """
    
    # Z-score thresholds for flagging anomalies
    Z_SCORE_HIGH_THRESHOLD = 2.0  # Above baseline
    Z_SCORE_LOW_THRESHOLD = -1.5  # Below baseline
    
    @staticmethod
    def calculate_z_score(value: float, baseline_mean: float, baseline_std: float) -> float:
        """
        Calculate z-score (standard deviations from mean).
        
        Z-score = (value - mean) / std
        
        If std is 0 (single sample baseline), return 0 for same value, else large number.
        """
        if baseline_std == 0.0:
            # Single-sample baseline
            if value == baseline_mean:
                return 0.0
            else:
                # Significant deviation from single-sample baseline
                return 10.0 if value > baseline_mean else -10.0
        
        return (value - baseline_mean) / baseline_std
    
    @staticmethod
    def calculate_percent_change(value: float, baseline_mean: float) -> float:
        """Calculate percentage change from baseline mean"""
        if baseline_mean == 0:
            return 0.0
        return ((value - baseline_mean) / baseline_mean) * 100.0
    
    @staticmethod
    def calculate_trend_slope(values: List[Tuple[datetime, float]]) -> Optional[float]:
        """
        Calculate linear regression slope for trend detection.
        
        Positive slope = increasing trend (worsening for pain/symptoms)
        Negative slope = decreasing trend (improving for pain/symptoms)
        
        Returns slope in units per day.
        """
        if len(values) < 2:
            return None
        
        # Convert to days since first measurement
        first_date = values[0][0]
        x_values = [(dt - first_date).total_seconds() / 86400.0 for dt, _ in values]
        y_values = [val for _, val in values]
        
        # Simple linear regression
        n = len(x_values)
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(y_values)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    @staticmethod
    def classify_deviation(z_score: float) -> Tuple[str, str]:
        """
        Classify deviation severity based on z-score.
        
        Thresholds:
        - Above: z > 2.0 triggers alert, z > 3.0 is critical
        - Below: z < -1.5 triggers alert, z < -2.5 is critical
        
        Returns: (deviation_type, severity_level)
        """
        if z_score > DeviationDetectionService.Z_SCORE_HIGH_THRESHOLD:
            # Above threshold (z > 2.0)
            if z_score > 3.0:
                return ("above_threshold", "critical")
            else:
                return ("above_threshold", "moderate")
        elif z_score < DeviationDetectionService.Z_SCORE_LOW_THRESHOLD:
            # Below threshold (z < -1.5)
            if z_score < -2.5:
                return ("below_threshold", "critical")
            else:
                return ("below_threshold", "moderate")
        else:
            # Within normal range (-1.5 <= z <= 2.0)
            return ("normal", "normal")
    
    @staticmethod
    def detect_pain_facial_deviation(
        db: Session,
        patient_id: str,
        measurement_id: int
    ) -> Optional[BaselineDeviation]:
        """
        Detect deviation for new pain facial measurement.
        Compares to current baseline and creates deviation record if anomaly detected.
        
        SECURITY: Enforces patient ownership - only processes patient's own measurements.
        """
        # Get current baseline
        baseline = BaselineCalculationService.get_current_baseline(db, patient_id)
        if not baseline or baseline.pain_facial_mean is None:
            return None
        
        # Get measurement - SECURITY: Verify ownership
        measurement = db.query(PainMeasurement).filter(
            and_(
                PainMeasurement.id == measurement_id,
                PainMeasurement.patient_id == patient_id  # CRITICAL: Ownership check
            )
        ).first()
        
        if not measurement or measurement.facial_stress_score is None:
            return None
        
        value = float(measurement.facial_stress_score)
        
        # Calculate z-score
        z_score = DeviationDetectionService.calculate_z_score(
            value, baseline.pain_facial_mean, baseline.pain_facial_std
        )
        
        percent_change = DeviationDetectionService.calculate_percent_change(
            value, baseline.pain_facial_mean
        )
        
        # Get trend (last 3 and 7 days)
        trend_3day = DeviationDetectionService._get_pain_facial_trend(db, patient_id, 3)
        trend_7day = DeviationDetectionService._get_pain_facial_trend(db, patient_id, 7)
        
        # Classify deviation
        deviation_type, severity_level = DeviationDetectionService.classify_deviation(z_score)
        
        # Determine trend direction
        trend_direction = "stable"
        if trend_3day and abs(trend_3day) > 1.0:  # >1 point change per day
            trend_direction = "worsening" if trend_3day > 0 else "improving"
        
        # Create deviation record
        deviation = BaselineDeviation(
            patient_id=patient_id,
            baseline_id=baseline.id,
            metric_name="pain_facial",
            measurement_value=value,
            measurement_date=measurement.created_at,
            z_score=z_score,
            percent_change=percent_change,
            baseline_mean=baseline.pain_facial_mean,
            baseline_std=baseline.pain_facial_std,
            trend_3day_slope=trend_3day,
            trend_7day_slope=trend_7day,
            trend_direction=trend_direction,
            deviation_type=deviation_type,
            severity_level=severity_level,
            source_measurement_id=measurement_id,
            source_table="pain_measurements"
        )
        
        # Generate alert if threshold exceeded
        if deviation_type != "normal":
            deviation.alert_triggered = True
            deviation.alert_message = DeviationDetectionService._generate_alert_message(
                "discomfort level", z_score, percent_change, trend_direction
            )
        
        db.add(deviation)
        db.commit()
        db.refresh(deviation)
        
        return deviation
    
    @staticmethod
    def detect_pain_self_reported_deviation(
        db: Session,
        patient_id: str,
        questionnaire_id: int
    ) -> Optional[BaselineDeviation]:
        """
        Detect deviation for self-reported pain score.
        
        SECURITY: Enforces patient ownership.
        """
        baseline = BaselineCalculationService.get_current_baseline(db, patient_id)
        if not baseline or baseline.pain_self_reported_mean is None:
            return None
        
        # SECURITY: Verify ownership
        questionnaire = db.query(PainQuestionnaire).filter(
            and_(
                PainQuestionnaire.id == questionnaire_id,
                PainQuestionnaire.patient_id == patient_id  # CRITICAL: Ownership check
            )
        ).first()
        
        if not questionnaire or questionnaire.pain_level_self_reported is None:
            return None
        
        value = float(questionnaire.pain_level_self_reported)
        
        z_score = DeviationDetectionService.calculate_z_score(
            value, baseline.pain_self_reported_mean, baseline.pain_self_reported_std
        )
        
        percent_change = DeviationDetectionService.calculate_percent_change(
            value, baseline.pain_self_reported_mean
        )
        
        trend_3day = DeviationDetectionService._get_pain_self_reported_trend(db, patient_id, 3)
        trend_7day = DeviationDetectionService._get_pain_self_reported_trend(db, patient_id, 7)
        
        deviation_type, severity_level = DeviationDetectionService.classify_deviation(z_score)
        
        trend_direction = "stable"
        if trend_3day and abs(trend_3day) > 0.5:  # >0.5 point change per day on 0-10 scale
            trend_direction = "worsening" if trend_3day > 0 else "improving"
        
        deviation = BaselineDeviation(
            patient_id=patient_id,
            baseline_id=baseline.id,
            metric_name="pain_self_reported",
            measurement_value=value,
            measurement_date=questionnaire.created_at,
            z_score=z_score,
            percent_change=percent_change,
            baseline_mean=baseline.pain_self_reported_mean,
            baseline_std=baseline.pain_self_reported_std,
            trend_3day_slope=trend_3day,
            trend_7day_slope=trend_7day,
            trend_direction=trend_direction,
            deviation_type=deviation_type,
            severity_level=severity_level,
            source_measurement_id=questionnaire_id,
            source_table="pain_questionnaires"
        )
        
        if deviation_type != "normal":
            deviation.alert_triggered = True
            deviation.alert_message = DeviationDetectionService._generate_alert_message(
                "pain level", z_score, percent_change, trend_direction
            )
        
        db.add(deviation)
        db.commit()
        db.refresh(deviation)
        
        return deviation
    
    @staticmethod
    def detect_respiratory_rate_deviation(
        db: Session,
        patient_id: str,
        measurement_id: int
    ) -> Optional[BaselineDeviation]:
        """
        Detect deviation for respiratory rate measurement.
        
        SECURITY: Enforces patient ownership.
        """
        baseline = BaselineCalculationService.get_current_baseline(db, patient_id)
        if not baseline or baseline.respiratory_rate_mean is None:
            return None
        
        # SECURITY: Verify ownership
        measurement = db.query(SymptomMeasurement).filter(
            and_(
                SymptomMeasurement.id == measurement_id,
                SymptomMeasurement.patient_id == patient_id  # CRITICAL: Ownership check
            )
        ).first()
        
        if not measurement or measurement.respiratory_rate_bpm is None:
            return None
        
        value = float(measurement.respiratory_rate_bpm)
        
        z_score = DeviationDetectionService.calculate_z_score(
            value, baseline.respiratory_rate_mean, baseline.respiratory_rate_std
        )
        
        percent_change = DeviationDetectionService.calculate_percent_change(
            value, baseline.respiratory_rate_mean
        )
        
        trend_3day = DeviationDetectionService._get_respiratory_rate_trend(db, patient_id, 3)
        trend_7day = DeviationDetectionService._get_respiratory_rate_trend(db, patient_id, 7)
        
        deviation_type, severity_level = DeviationDetectionService.classify_deviation(z_score)
        
        trend_direction = "stable"
        if trend_3day and abs(trend_3day) > 1.0:  # >1 bpm change per day
            trend_direction = "worsening" if trend_3day > 0 else "improving"
        
        deviation = BaselineDeviation(
            patient_id=patient_id,
            baseline_id=baseline.id,
            metric_name="respiratory_rate",
            measurement_value=value,
            measurement_date=measurement.created_at,
            z_score=z_score,
            percent_change=percent_change,
            baseline_mean=baseline.respiratory_rate_mean,
            baseline_std=baseline.respiratory_rate_std,
            trend_3day_slope=trend_3day,
            trend_7day_slope=trend_7day,
            trend_direction=trend_direction,
            deviation_type=deviation_type,
            severity_level=severity_level,
            source_measurement_id=measurement_id,
            source_table="symptom_measurements"
        )
        
        if deviation_type != "normal":
            deviation.alert_triggered = True
            deviation.alert_message = DeviationDetectionService._generate_alert_message(
                "respiratory rate", z_score, percent_change, trend_direction
            )
        
        db.add(deviation)
        db.commit()
        db.refresh(deviation)
        
        return deviation
    
    @staticmethod
    def detect_symptom_severity_deviation(
        db: Session,
        patient_id: str,
        symptom_log_id: int
    ) -> Optional[BaselineDeviation]:
        """
        Detect deviation for symptom severity.
        
        SECURITY: Enforces patient ownership.
        """
        baseline = BaselineCalculationService.get_current_baseline(db, patient_id)
        if not baseline or baseline.symptom_severity_mean is None:
            return None
        
        # SECURITY: Verify ownership
        symptom_log = db.query(SymptomLog).filter(
            and_(
                SymptomLog.id == symptom_log_id,
                SymptomLog.patient_id == patient_id  # CRITICAL: Ownership check
            )
        ).first()
        
        if not symptom_log or symptom_log.severity is None:
            return None
        
        value = float(symptom_log.severity)
        
        z_score = DeviationDetectionService.calculate_z_score(
            value, baseline.symptom_severity_mean, baseline.symptom_severity_std
        )
        
        percent_change = DeviationDetectionService.calculate_percent_change(
            value, baseline.symptom_severity_mean
        )
        
        trend_3day = DeviationDetectionService._get_symptom_severity_trend(db, patient_id, 3)
        trend_7day = DeviationDetectionService._get_symptom_severity_trend(db, patient_id, 7)
        
        deviation_type, severity_level = DeviationDetectionService.classify_deviation(z_score)
        
        trend_direction = "stable"
        if trend_3day and abs(trend_3day) > 0.5:
            trend_direction = "worsening" if trend_3day > 0 else "improving"
        
        deviation = BaselineDeviation(
            patient_id=patient_id,
            baseline_id=baseline.id,
            metric_name="symptom_severity",
            measurement_value=value,
            measurement_date=symptom_log.logged_at,
            z_score=z_score,
            percent_change=percent_change,
            baseline_mean=baseline.symptom_severity_mean,
            baseline_std=baseline.symptom_severity_std,
            trend_3day_slope=trend_3day,
            trend_7day_slope=trend_7day,
            trend_direction=trend_direction,
            deviation_type=deviation_type,
            severity_level=severity_level,
            source_measurement_id=symptom_log_id,
            source_table="symptom_logs"
        )
        
        if deviation_type != "normal":
            deviation.alert_triggered = True
            deviation.alert_message = DeviationDetectionService._generate_alert_message(
                "symptom severity", z_score, percent_change, trend_direction
            )
        
        db.add(deviation)
        db.commit()
        db.refresh(deviation)
        
        return deviation
    
    # Helper methods for trend calculation
    
    @staticmethod
    def _get_pain_facial_trend(db: Session, patient_id: str, days: int) -> Optional[float]:
        """Get pain facial stress score trend over specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        measurements = db.query(
            PainMeasurement.created_at,
            PainMeasurement.facial_stress_score
        ).filter(
            and_(
                PainMeasurement.patient_id == patient_id,
                PainMeasurement.created_at >= cutoff_date,
                PainMeasurement.facial_stress_score.isnot(None)
            )
        ).order_by(PainMeasurement.created_at).all()
        
        if len(measurements) < 2:
            return None
        
        values = [(m[0], float(m[1])) for m in measurements]
        return DeviationDetectionService.calculate_trend_slope(values)
    
    @staticmethod
    def _get_pain_self_reported_trend(db: Session, patient_id: str, days: int) -> Optional[float]:
        """Get self-reported pain trend over specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        questionnaires = db.query(
            PainQuestionnaire.created_at,
            PainQuestionnaire.pain_level_self_reported
        ).filter(
            and_(
                PainQuestionnaire.patient_id == patient_id,
                PainQuestionnaire.created_at >= cutoff_date,
                PainQuestionnaire.pain_level_self_reported.isnot(None)
            )
        ).order_by(PainQuestionnaire.created_at).all()
        
        if len(questionnaires) < 2:
            return None
        
        values = [(q[0], float(q[1])) for q in questionnaires]
        return DeviationDetectionService.calculate_trend_slope(values)
    
    @staticmethod
    def _get_respiratory_rate_trend(db: Session, patient_id: str, days: int) -> Optional[float]:
        """Get respiratory rate trend over specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        measurements = db.query(
            SymptomMeasurement.created_at,
            SymptomMeasurement.respiratory_rate_bpm
        ).filter(
            and_(
                SymptomMeasurement.patient_id == patient_id,
                SymptomMeasurement.body_area == BodyArea.CHEST,
                SymptomMeasurement.created_at >= cutoff_date,
                SymptomMeasurement.respiratory_rate_bpm.isnot(None)
            )
        ).order_by(SymptomMeasurement.created_at).all()
        
        if len(measurements) < 2:
            return None
        
        values = [(m[0], float(m[1])) for m in measurements]
        return DeviationDetectionService.calculate_trend_slope(values)
    
    @staticmethod
    def _get_symptom_severity_trend(db: Session, patient_id: str, days: int) -> Optional[float]:
        """Get symptom severity trend over specified days"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        symptoms = db.query(
            SymptomLog.logged_at,
            SymptomLog.severity
        ).filter(
            and_(
                SymptomLog.patient_id == patient_id,
                SymptomLog.logged_at >= cutoff_date,
                SymptomLog.severity.isnot(None)
            )
        ).order_by(SymptomLog.logged_at).all()
        
        if len(symptoms) < 2:
            return None
        
        values = [(s[0], float(s[1])) for s in symptoms]
        return DeviationDetectionService.calculate_trend_slope(values)
    
    @staticmethod
    def _generate_alert_message(
        metric_name: str,
        z_score: float,
        percent_change: float,
        trend_direction: str
    ) -> str:
        """
        Generate wellness alert message (regulatory-compliant language).
        NOT medical diagnosis - wellness monitoring only.
        """
        if z_score > 0:
            change_desc = f"increased by {abs(percent_change):.1f}%"
        else:
            change_desc = f"decreased by {abs(percent_change):.1f}%"
        
        trend_desc = ""
        if trend_direction == "worsening":
            trend_desc = " and shows a worsening trend"
        elif trend_direction == "improving":
            trend_desc = " and shows an improving trend"
        
        return (
            f"Your {metric_name} has {change_desc} from your personal baseline{trend_desc}. "
            f"This change may be worth discussing with your healthcare provider."
        )
    
    @staticmethod
    def get_patient_deviations(
        db: Session,
        patient_id: str,
        days: Optional[int] = 7,
        alert_only: bool = False
    ) -> List[BaselineDeviation]:
        """
        Get deviation records for a patient.
        
        Args:
            db: Database session
            patient_id: Patient identifier
            days: Number of days to look back (default: 7)
            alert_only: Only return deviations that triggered alerts
            
        Returns:
            List of BaselineDeviation records
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days) if days else None
        
        query = db.query(BaselineDeviation).filter(
            BaselineDeviation.patient_id == patient_id
        )
        
        if cutoff_date:
            query = query.filter(BaselineDeviation.measurement_date >= cutoff_date)
        
        if alert_only:
            query = query.filter(BaselineDeviation.alert_triggered == True)
        
        return query.order_by(desc(BaselineDeviation.measurement_date)).all()
