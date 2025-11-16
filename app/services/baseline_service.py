"""
Baseline Calculation Service for Health Deterioration Detection

Calculates 7-day rolling baselines for patient health metrics.
Used for wellness monitoring and change detection (NOT medical diagnosis).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from sqlalchemy import func, and_
from sqlalchemy.orm import Session
import statistics

from app.models.health_baseline import HealthBaseline, BaselineDeviation
from app.models.pain_tracking import PainMeasurement, PainQuestionnaire
from app.models.symptom_journal import SymptomMeasurement, BodyArea
from app.models.medication_side_effects import SymptomLog


class BaselineCalculationService:
    """
    Service for calculating patient health baselines from historical data.
    Supports wellness monitoring and change detection.
    """
    
    @staticmethod
    def calculate_patient_baseline(
        db: Session,
        patient_id: str,
        end_date: Optional[datetime] = None,
        window_days: int = 7
    ) -> Optional[HealthBaseline]:
        """
        Calculate 7-day rolling baseline for a patient's health metrics.
        
        Args:
            db: Database session
            patient_id: Patient identifier
            end_date: End date for baseline window (defaults to now)
            window_days: Number of days to include in baseline (default: 7)
            
        Returns:
            HealthBaseline object if sufficient data exists, None otherwise
        """
        if end_date is None:
            end_date = datetime.utcnow()
        
        start_date = end_date - timedelta(days=window_days)
        
        # Fetch pain data
        pain_facial_values = BaselineCalculationService._get_pain_facial_data(
            db, patient_id, start_date, end_date
        )
        pain_self_reported_values = BaselineCalculationService._get_pain_self_reported_data(
            db, patient_id, start_date, end_date
        )
        
        # Fetch respiratory rate data
        respiratory_rate_values = BaselineCalculationService._get_respiratory_rate_data(
            db, patient_id, start_date, end_date
        )
        
        # Fetch symptom severity data
        symptom_severity_values = BaselineCalculationService._get_symptom_severity_data(
            db, patient_id, start_date, end_date
        )
        
        # Fetch activity impact data
        activity_impact_rate = BaselineCalculationService._get_activity_impact_rate(
            db, patient_id, start_date, end_date
        )
        
        # Calculate total data points
        total_data_points = (
            len(pain_facial_values) +
            len(pain_self_reported_values) +
            len(respiratory_rate_values) +
            len(symptom_severity_values)
        )
        
        # Require at least 3 data points to create baseline
        if total_data_points < 3:
            return None
        
        # Calculate baseline statistics
        baseline = HealthBaseline(
            patient_id=patient_id,
            baseline_start_date=start_date,
            baseline_end_date=end_date,
            data_points_count=total_data_points,
            is_current=False  # Will be set to True after saving
        )
        
        # Pain facial metrics
        if len(pain_facial_values) >= 1:
            baseline.pain_facial_mean = statistics.mean(pain_facial_values)
            baseline.pain_facial_std = statistics.stdev(pain_facial_values) if len(pain_facial_values) > 1 else 0.0
            baseline.pain_facial_min = min(pain_facial_values)
            baseline.pain_facial_max = max(pain_facial_values)
        
        # Pain self-reported metrics
        if len(pain_self_reported_values) >= 1:
            baseline.pain_self_reported_mean = statistics.mean(pain_self_reported_values)
            baseline.pain_self_reported_std = statistics.stdev(pain_self_reported_values) if len(pain_self_reported_values) > 1 else 0.0
            baseline.pain_self_reported_min = min(pain_self_reported_values)
            baseline.pain_self_reported_max = max(pain_self_reported_values)
        
        # Respiratory rate metrics
        if len(respiratory_rate_values) >= 1:
            baseline.respiratory_rate_mean = statistics.mean(respiratory_rate_values)
            baseline.respiratory_rate_std = statistics.stdev(respiratory_rate_values) if len(respiratory_rate_values) > 1 else 0.0
            baseline.respiratory_rate_min = min(respiratory_rate_values)
            baseline.respiratory_rate_max = max(respiratory_rate_values)
        
        # Symptom severity metrics
        if len(symptom_severity_values) >= 1:
            baseline.symptom_severity_mean = statistics.mean(symptom_severity_values)
            baseline.symptom_severity_std = statistics.stdev(symptom_severity_values) if len(symptom_severity_values) > 1 else 0.0
            baseline.symptom_severity_min = min(symptom_severity_values)
            baseline.symptom_severity_max = max(symptom_severity_values)
        
        # Activity impact
        baseline.activity_impact_rate = activity_impact_rate
        
        # Assess baseline quality
        baseline.baseline_quality = BaselineCalculationService._assess_baseline_quality(
            total_data_points, window_days
        )
        
        # Store raw daily values for debugging
        baseline.raw_daily_values = {
            "pain_facial": pain_facial_values,
            "pain_self_reported": pain_self_reported_values,
            "respiratory_rate": respiratory_rate_values,
            "symptom_severity": symptom_severity_values,
            "activity_impact_rate": activity_impact_rate
        }
        
        # Mark previous baselines as not current
        db.query(HealthBaseline).filter(
            HealthBaseline.patient_id == patient_id,
            HealthBaseline.is_current == True
        ).update({"is_current": False})
        
        # Mark this baseline as current
        baseline.is_current = True
        
        db.add(baseline)
        db.commit()
        db.refresh(baseline)
        
        return baseline
    
    @staticmethod
    def _get_pain_facial_data(
        db: Session,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Fetch facial stress scores from pain measurements"""
        measurements = db.query(PainMeasurement.facial_stress_score).filter(
            and_(
                PainMeasurement.patient_id == patient_id,
                PainMeasurement.created_at >= start_date,
                PainMeasurement.created_at <= end_date,
                PainMeasurement.facial_stress_score.isnot(None)
            )
        ).all()
        
        return [float(m[0]) for m in measurements if m[0] is not None]
    
    @staticmethod
    def _get_pain_self_reported_data(
        db: Session,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Fetch self-reported pain levels from questionnaires"""
        questionnaires = db.query(PainQuestionnaire.pain_level_self_reported).filter(
            and_(
                PainQuestionnaire.patient_id == patient_id,
                PainQuestionnaire.created_at >= start_date,
                PainQuestionnaire.created_at <= end_date,
                PainQuestionnaire.pain_level_self_reported.isnot(None)
            )
        ).all()
        
        return [float(q[0]) for q in questionnaires if q[0] is not None]
    
    @staticmethod
    def _get_respiratory_rate_data(
        db: Session,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Fetch respiratory rate from symptom measurements (chest area only)"""
        measurements = db.query(SymptomMeasurement.respiratory_rate_bpm).filter(
            and_(
                SymptomMeasurement.patient_id == patient_id,
                SymptomMeasurement.body_area == BodyArea.CHEST,
                SymptomMeasurement.created_at >= start_date,
                SymptomMeasurement.created_at <= end_date,
                SymptomMeasurement.respiratory_rate_bpm.isnot(None)
            )
        ).all()
        
        return [float(m[0]) for m in measurements if m[0] is not None]
    
    @staticmethod
    def _get_symptom_severity_data(
        db: Session,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[float]:
        """Fetch symptom severity from symptom logs"""
        symptoms = db.query(SymptomLog.severity).filter(
            and_(
                SymptomLog.patient_id == patient_id,
                SymptomLog.logged_at >= start_date,
                SymptomLog.logged_at <= end_date,
                SymptomLog.severity.isnot(None)
            )
        ).all()
        
        return [float(s[0]) for s in symptoms if s[0] is not None]
    
    @staticmethod
    def _get_activity_impact_rate(
        db: Session,
        patient_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> float:
        """
        Calculate percentage of days where daily activities were affected.
        Returns value between 0 and 1.
        """
        total_responses = db.query(func.count(PainQuestionnaire.id)).filter(
            and_(
                PainQuestionnaire.patient_id == patient_id,
                PainQuestionnaire.created_at >= start_date,
                PainQuestionnaire.created_at <= end_date
            )
        ).scalar()
        
        if total_responses == 0:
            return 0.0
        
        affected_responses = db.query(func.count(PainQuestionnaire.id)).filter(
            and_(
                PainQuestionnaire.patient_id == patient_id,
                PainQuestionnaire.created_at >= start_date,
                PainQuestionnaire.created_at <= end_date,
                PainQuestionnaire.affects_daily_activities == True
            )
        ).scalar()
        
        return float(affected_responses) / float(total_responses) if total_responses > 0 else 0.0
    
    @staticmethod
    def _assess_baseline_quality(data_points: int, window_days: int) -> str:
        """
        Assess the quality of baseline based on data density.
        
        Ideal: 1+ data point per day
        Good: 0.7+ data points per day
        Fair: 0.4+ data points per day
        Poor: < 0.4 data points per day
        """
        density = data_points / window_days
        
        if density >= 1.0:
            return "excellent"
        elif density >= 0.7:
            return "good"
        elif density >= 0.4:
            return "fair"
        else:
            return "poor"
    
    @staticmethod
    def get_current_baseline(db: Session, patient_id: str) -> Optional[HealthBaseline]:
        """Get the most recent current baseline for a patient"""
        return db.query(HealthBaseline).filter(
            HealthBaseline.patient_id == patient_id,
            HealthBaseline.is_current == True
        ).order_by(HealthBaseline.created_at.desc()).first()
    
    @staticmethod
    def calculate_all_patient_baselines(
        db: Session,
        patient_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Calculate baselines for all patients or specified list.
        Useful for daily batch updates.
        
        Returns summary statistics of baseline calculations.
        """
        from app.models.user import User
        
        if patient_ids is None:
            # Get all patient IDs
            patients = db.query(User.id).filter(User.role == "patient").all()
            patient_ids = [p[0] for p in patients]
        
        results = {
            "total_patients": len(patient_ids),
            "baselines_created": 0,
            "baselines_skipped": 0,
            "errors": []
        }
        
        for patient_id in patient_ids:
            try:
                baseline = BaselineCalculationService.calculate_patient_baseline(
                    db, patient_id
                )
                if baseline:
                    results["baselines_created"] += 1
                else:
                    results["baselines_skipped"] += 1
            except Exception as e:
                results["errors"].append({
                    "patient_id": patient_id,
                    "error": str(e)
                })
        
        return results
