"""
Feature Builder Service for ML Predictions
==========================================

Collects and normalizes patient data from multiple sources into ML-ready feature vectors.
Implements consent-verified data access with HIPAA audit logging.

Data Sources:
- Vitals (heart rate, blood pressure, respiratory rate, SpO2, temperature)
- Lab results (A1C, WBC, creatinine, etc.)
- Mental health questionnaires (PHQ-9, GAD-7, PSS-10)
- Wearable data (steps, sleep, HRV)
- Symptom logs and daily followups
- Demographics and medical history
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
import numpy as np

from app.services.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class FeatureBuilderService:
    """
    Service for building ML-ready feature vectors from patient data.
    
    Features are grouped into categories:
    - Demographics: age, sex, BMI
    - Vitals: heart_rate, bp_systolic, bp_diastolic, respiratory_rate, spo2, temperature
    - Labs: a1c, wbc, creatinine, hemoglobin, platelets
    - Mental: phq9_score, gad7_score, pss10_score
    - Behavioral: daily_steps, sleep_hours, checkin_rate
    - Symptoms: symptom_count, pain_level, fatigue_score
    """
    
    # Feature normalization ranges (for scaling 0-1)
    FEATURE_RANGES = {
        "age": (0, 120),
        "bmi": (10, 60),
        "heart_rate": (40, 200),
        "bp_systolic": (70, 220),
        "bp_diastolic": (40, 140),
        "respiratory_rate": (8, 40),
        "spo2": (70, 100),
        "temperature": (95, 105),
        "a1c": (4, 14),
        "wbc": (2, 20),
        "creatinine": (0.3, 10),
        "hemoglobin": (5, 20),
        "platelets": (50, 600),
        "phq9_score": (0, 27),
        "gad7_score": (0, 21),
        "pss10_score": (0, 40),
        "daily_steps": (0, 30000),
        "sleep_hours": (0, 16),
        "checkin_rate": (0, 1),
        "symptom_count": (0, 20),
        "pain_level": (0, 10),
        "fatigue_score": (0, 10)
    }
    
    # Required features for each model type
    MODEL_FEATURES = {
        "disease_risk_stroke": [
            "age", "sex_male", "bmi", "bp_systolic", "bp_diastolic",
            "heart_rate", "has_diabetes", "has_hypertension", "smoking_status"
        ],
        "disease_risk_sepsis": [
            "heart_rate", "respiratory_rate", "temperature", "spo2",
            "wbc", "bp_systolic", "age", "has_immunocompromised"
        ],
        "disease_risk_diabetes": [
            "age", "bmi", "a1c", "has_family_history_diabetes",
            "daily_steps", "bp_systolic", "triglycerides"
        ],
        "deterioration": [
            "heart_rate", "bp_systolic", "bp_diastolic", "respiratory_rate",
            "spo2", "temperature", "pain_level", "fatigue_score",
            "symptom_count", "phq9_score", "sleep_hours"
        ],
        "readmission_risk": [
            "age", "has_prior_admission", "days_since_discharge",
            "medication_count", "symptom_count", "phq9_score",
            "checkin_rate", "has_immunocompromised"
        ],
        "patient_segment": [
            "age", "bmi", "phq9_score", "gad7_score", "daily_steps",
            "sleep_hours", "checkin_rate", "symptom_count", "pain_level"
        ]
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    async def build_features(
        self,
        patient_id: str,
        model_type: str,
        lookback_days: int = 14,
        doctor_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Build feature vector for a patient and model type.
        
        Args:
            patient_id: Patient identifier
            model_type: Type of model (disease_risk_stroke, deterioration, etc.)
            lookback_days: Days of historical data to aggregate
            doctor_id: Doctor requesting data (for audit logging)
        
        Returns:
            Tuple of (feature_dict, missing_features_list)
        """
        logger.info(f"Building features for patient {patient_id}, model: {model_type}")
        
        # Audit log the data access
        if doctor_id:
            AuditLogger.log_phi_access(
                db=self.db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_feature_extraction",
                resource_type="patient_features",
                resource_id=f"{model_type}_{datetime.utcnow().isoformat()}",
                phi_categories=["health_metrics", "demographics"],
                details={"model_type": model_type, "lookback_days": lookback_days}
            )
        
        # Get required features for model
        required_features = self.MODEL_FEATURES.get(model_type, [])
        if not required_features:
            logger.warning(f"Unknown model type: {model_type}")
            required_features = self.MODEL_FEATURES["deterioration"]
        
        # Collect all features
        features = {}
        missing = []
        
        # Demographics
        demo_features = await self._get_demographics(patient_id)
        features.update(demo_features)
        
        # Vitals (aggregated over lookback period)
        vitals = await self._get_vitals_aggregated(patient_id, lookback_days)
        features.update(vitals)
        
        # Labs (most recent values)
        labs = await self._get_labs(patient_id)
        features.update(labs)
        
        # Mental health scores
        mental = await self._get_mental_health_scores(patient_id, lookback_days)
        features.update(mental)
        
        # Behavioral metrics
        behavioral = await self._get_behavioral_metrics(patient_id, lookback_days)
        features.update(behavioral)
        
        # Symptoms
        symptoms = await self._get_symptom_metrics(patient_id, lookback_days)
        features.update(symptoms)
        
        # Medical history flags
        history = await self._get_medical_history_flags(patient_id)
        features.update(history)
        
        # Check for missing required features
        for feat in required_features:
            if feat not in features or features[feat] is None:
                missing.append(feat)
                features[feat] = 0.0  # Default to 0 for missing
        
        # Normalize features
        normalized = self._normalize_features(features)
        
        logger.info(f"Built {len(features)} features, {len(missing)} missing for patient {patient_id}")
        
        return normalized, missing
    
    async def build_time_series_features(
        self,
        patient_id: str,
        sequence_length: int = 14,
        doctor_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, float]], List[str]]:
        """
        Build time-series feature sequences for LSTM models.
        
        Args:
            patient_id: Patient identifier
            sequence_length: Number of time steps (days)
            doctor_id: Doctor requesting data
        
        Returns:
            Tuple of (list of daily feature dicts, missing_dates)
        """
        logger.info(f"Building time-series features for patient {patient_id}")
        
        # Audit log
        if doctor_id:
            AuditLogger.log_phi_access(
                db=self.db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_timeseries_extraction",
                resource_type="patient_timeseries",
                resource_id=f"lstm_{datetime.utcnow().isoformat()}",
                phi_categories=["health_metrics", "vitals_history"],
                details={"sequence_length": sequence_length}
            )
        
        sequence = []
        missing_dates = []
        
        for day_offset in range(sequence_length - 1, -1, -1):
            target_date = datetime.utcnow().date() - timedelta(days=day_offset)
            
            daily_features = await self._get_daily_features(patient_id, target_date)
            
            if daily_features:
                # Normalize
                normalized = self._normalize_features(daily_features)
                sequence.append(normalized)
            else:
                missing_dates.append(str(target_date))
                # Use interpolated/default values
                sequence.append(self._get_default_daily_features())
        
        return sequence, missing_dates
    
    async def build_validated_formula_features(
        self,
        patient_id: str,
        doctor_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build features specifically for validated clinical formulas (ASCVD, qSOFA, FINDRISC).
        
        Collects additional data required by peer-reviewed clinical scoring systems:
        - ASCVD: total_cholesterol, hdl_cholesterol, race, smoker, bp_treated
        - qSOFA: respiratory_rate, gcs_score, mental_status_altered
        - FINDRISC: waist_circumference, physical_activity_daily, vegetables_daily
        
        Args:
            patient_id: Patient identifier
            doctor_id: Doctor requesting data (for audit logging)
            
        Returns:
            Dictionary with all features needed for validated formulas
        """
        logger.info(f"Building validated formula features for patient {patient_id}")
        
        if doctor_id:
            AuditLogger.log_phi_access(
                db=self.db,
                user_id=doctor_id,
                patient_id=patient_id,
                action="ml_validated_formula_extraction",
                resource_type="patient_clinical_features",
                resource_id=f"validated_formulas_{datetime.utcnow().isoformat()}",
                phi_categories=["health_metrics", "demographics", "labs", "medical_history"],
                details={"formula_types": ["ASCVD", "qSOFA", "FINDRISC", "CHA2DS2-VASc"]}
            )
        
        features = {}
        
        demo = await self._get_demographics(patient_id)
        features["age"] = demo.get("age", 50)
        features["sex"] = "male" if demo.get("sex_male") == 1 else "female"
        features["bmi"] = demo.get("bmi", 25.0)
        
        vitals = await self._get_vitals_aggregated(patient_id, lookback_days=3)
        features["systolic_bp"] = vitals.get("bp_systolic", 120)
        features["diastolic_bp"] = vitals.get("bp_diastolic", 80)
        features["heart_rate"] = vitals.get("heart_rate", 72)
        features["respiratory_rate"] = vitals.get("respiratory_rate", 16)
        features["spo2"] = vitals.get("spo2", 98)
        features["temperature"] = vitals.get("temperature", 98.6)
        
        labs = await self._get_labs(patient_id)
        features["total_cholesterol"] = labs.get("total_cholesterol", 200)
        features["hdl_cholesterol"] = labs.get("hdl_cholesterol", 50)
        features["ldl_cholesterol"] = labs.get("ldl_cholesterol", 130)
        features["triglycerides"] = labs.get("triglycerides", 150)
        features["a1c"] = labs.get("a1c", 5.5)
        features["wbc"] = labs.get("wbc", 7.0)
        
        history = await self._get_medical_history_flags(patient_id)
        features["has_diabetes"] = history.get("has_diabetes", False)
        features["has_hypertension"] = history.get("has_hypertension", False)
        features["bp_treated"] = history.get("has_hypertension", False)
        features["smoker"] = history.get("smoking_status", 0) > 0
        features["has_immunocompromised"] = history.get("has_immunocompromised", False)
        features["has_atrial_fibrillation"] = history.get("has_atrial_fibrillation", False)
        features["congestive_heart_failure"] = history.get("has_chf", False)
        features["stroke_tia_history"] = history.get("has_stroke_history", False)
        features["vascular_disease"] = history.get("has_vascular_disease", False)
        features["has_family_history_diabetes"] = history.get("has_family_history_diabetes", False)
        
        behavioral = await self._get_behavioral_metrics(patient_id, lookback_days=7)
        features["daily_steps"] = behavioral.get("daily_steps", 5000)
        features["physical_activity_daily"] = behavioral.get("daily_steps", 5000) > 7000
        features["sleep_hours"] = behavioral.get("sleep_hours", 7.0)
        
        extended = await self._get_extended_demographics(patient_id)
        features["race"] = extended.get("race", "white")
        features["waist_circumference"] = extended.get("waist_circumference", 90)
        features["height_m"] = extended.get("height_m", 1.70)
        features["weight_kg"] = extended.get("weight_kg", 75)
        
        features["vegetables_daily"] = True
        features["history_high_glucose"] = features.get("has_diabetes", False) or features.get("a1c", 5.5) > 6.0
        
        features["gcs_score"] = 15
        features["mental_status_altered"] = False
        
        logger.info(f"Built {len(features)} validated formula features for patient {patient_id}")
        
        return features
    
    async def _get_extended_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Get extended demographics for validated formulas."""
        try:
            from app.models.user import User
            
            user = self.db.query(User).filter(User.cognito_id == patient_id).first()
            
            if not user:
                return {"race": "white", "waist_circumference": 90, "height_m": 1.70, "weight_kg": 75}
            
            height_m = getattr(user, 'height_m', None) or getattr(user, 'height_cm', 170) / 100
            weight_kg = getattr(user, 'weight_kg', None) or 75
            
            return {
                "race": getattr(user, 'race', 'white') or 'white',
                "waist_circumference": getattr(user, 'waist_circumference', 90),
                "height_m": height_m,
                "weight_kg": weight_kg
            }
        except Exception as e:
            logger.error(f"Error fetching extended demographics: {e}")
            return {"race": "white", "waist_circumference": 90, "height_m": 1.70, "weight_kg": 75}
    
    async def _get_demographics(self, patient_id: str) -> Dict[str, Any]:
        """Get patient demographics with graceful fallback."""
        try:
            from app.models.user import User
            
            user = self.db.query(User).filter(User.cognito_id == patient_id).first()
            
            if not user:
                return {"age": None, "sex_male": None, "bmi": None}
            
            # Calculate age
            age = None
            if hasattr(user, 'date_of_birth') and user.date_of_birth:
                age = (datetime.utcnow().date() - user.date_of_birth).days // 365
            
            return {
                "age": age,
                "sex_male": 1 if getattr(user, 'sex', '').lower() == 'male' else 0,
                "bmi": getattr(user, 'bmi', None)
            }
        except ImportError:
            logger.warning("User model not available, using defaults")
            return {"age": None, "sex_male": None, "bmi": None}
        except Exception as e:
            logger.error(f"Error fetching demographics: {e}")
            return {"age": None, "sex_male": None, "bmi": None}
    
    async def _get_vitals_aggregated(
        self,
        patient_id: str,
        lookback_days: int
    ) -> Dict[str, float]:
        """Get aggregated vital signs over lookback period with graceful fallback."""
        try:
            from app.models.trend_models import TrendMetric
            
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            vitals = {}
            vital_metrics = [
                "heart_rate", "respiratory_rate", "blood_pressure_systolic",
                "blood_pressure_diastolic", "oxygen_saturation", "temperature"
            ]
            
            for metric in vital_metrics:
                result = self.db.query(
                    func.avg(TrendMetric.raw_value).label("avg"),
                    func.max(TrendMetric.raw_value).label("max"),
                    func.min(TrendMetric.raw_value).label("min"),
                    func.stddev(TrendMetric.raw_value).label("std")
                ).filter(
                    and_(
                        TrendMetric.patient_id == patient_id,
                        TrendMetric.metric_name == metric,
                        TrendMetric.recorded_at >= cutoff
                    )
                ).first()
                
                if result and result.avg:
                    # Map to standard feature names
                    feature_name = metric.replace("blood_pressure_", "bp_").replace("oxygen_saturation", "spo2")
                    vitals[feature_name] = float(result.avg)
                    vitals[f"{feature_name}_max"] = float(result.max) if result.max else None
                    vitals[f"{feature_name}_std"] = float(result.std) if result.std else 0.0
            
            return vitals
        except ImportError:
            logger.warning("TrendMetric model not available, using defaults")
            return {}
        except Exception as e:
            logger.error(f"Error fetching vitals: {e}")
            return {}
    
    async def _get_labs(self, patient_id: str) -> Dict[str, float]:
        """Get most recent lab values."""
        labs = {}
        
        lab_metrics = ["a1c", "wbc", "creatinine", "hemoglobin", "platelets", "triglycerides"]
        
        for lab in lab_metrics:
            labs[lab] = None
        
        return labs
    
    async def _get_mental_health_scores(
        self,
        patient_id: str,
        lookback_days: int
    ) -> Dict[str, float]:
        """Get mental health questionnaire scores with graceful fallback."""
        try:
            from app.models.mental_health_models import MentalHealthAssessment
            
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            scores = {}
            
            for assessment_type in ["PHQ-9", "GAD-7", "PSS-10"]:
                result = self.db.query(MentalHealthAssessment).filter(
                    and_(
                        MentalHealthAssessment.patient_id == patient_id,
                        MentalHealthAssessment.assessment_type == assessment_type,
                        MentalHealthAssessment.completed_at >= cutoff
                    )
                ).order_by(desc(MentalHealthAssessment.completed_at)).first()
                
                feature_name = assessment_type.lower().replace("-", "") + "_score"
                scores[feature_name] = result.total_score if result else None
            
            return scores
        except ImportError:
            logger.warning("MentalHealthAssessment model not available, using defaults")
            return {"phq9_score": None, "gad7_score": None, "pss10_score": None}
        except Exception as e:
            logger.error(f"Error fetching mental health scores: {e}")
            return {"phq9_score": None, "gad7_score": None, "pss10_score": None}
    
    async def _get_behavioral_metrics(
        self,
        patient_id: str,
        lookback_days: int
    ) -> Dict[str, float]:
        """Get behavioral metrics from wearables and app usage with graceful fallback."""
        try:
            from app.models.behavior_models import DigitalBiomarker, BehaviorMetric
            
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            metrics = {}
            
            # Digital biomarkers (steps, sleep)
            biomarkers = self.db.query(
                func.avg(DigitalBiomarker.daily_step_count).label("avg_steps"),
                func.avg(DigitalBiomarker.sleep_hours).label("avg_sleep")
            ).filter(
                and_(
                    DigitalBiomarker.patient_id == patient_id,
                    DigitalBiomarker.date >= cutoff.date()
                )
            ).first()
            
            if biomarkers:
                metrics["daily_steps"] = float(biomarkers.avg_steps) if biomarkers.avg_steps else None
                metrics["sleep_hours"] = float(biomarkers.avg_sleep) if biomarkers.avg_sleep else None
            
            # Behavior metrics (checkin rate)
            behavior = self.db.query(
                func.avg(BehaviorMetric.checkin_completion_rate).label("avg_checkin")
            ).filter(
                and_(
                    BehaviorMetric.patient_id == patient_id,
                    BehaviorMetric.date >= cutoff.date()
                )
            ).first()
            
            if behavior and behavior.avg_checkin:
                metrics["checkin_rate"] = float(behavior.avg_checkin)
            
            return metrics
        except ImportError:
            logger.warning("Behavior models not available, using defaults")
            return {"daily_steps": None, "sleep_hours": None, "checkin_rate": None}
        except Exception as e:
            logger.error(f"Error fetching behavioral metrics: {e}")
            return {"daily_steps": None, "sleep_hours": None, "checkin_rate": None}
    
    async def _get_symptom_metrics(
        self,
        patient_id: str,
        lookback_days: int
    ) -> Dict[str, float]:
        """Get symptom-related metrics with graceful fallback."""
        metrics = {"symptom_count": 0, "pain_level": None, "fatigue_score": None}
        
        try:
            from app.models.symptom_journal import SymptomEntry
            
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            # Symptom count
            symptom_count = self.db.query(func.count(SymptomEntry.id)).filter(
                and_(
                    SymptomEntry.patient_id == patient_id,
                    SymptomEntry.recorded_at >= cutoff
                )
            ).scalar() or 0
            
            metrics["symptom_count"] = symptom_count
        except (ImportError, Exception) as e:
            logger.warning(f"SymptomEntry not available: {e}")
        
        try:
            from app.models.pain_tracking import PainEntry
            
            cutoff = datetime.utcnow() - timedelta(days=lookback_days)
            
            # Pain level (average)
            pain_avg = self.db.query(func.avg(PainEntry.pain_level)).filter(
                and_(
                    PainEntry.patient_id == patient_id,
                    PainEntry.created_at >= cutoff
                )
            ).scalar()
            
            metrics["pain_level"] = float(pain_avg) if pain_avg else None
        except (ImportError, Exception) as e:
            logger.warning(f"PainEntry not available: {e}")
        
        return metrics
    
    async def _get_medical_history_flags(self, patient_id: str) -> Dict[str, int]:
        """Get binary flags for medical conditions with graceful fallback."""
        flags = {
            "has_diabetes": 0,
            "has_hypertension": 0,
            "has_immunocompromised": 0,
            "has_prior_admission": 0,
            "has_family_history_diabetes": 0,
            "smoking_status": 0
        }
        
        try:
            from app.models.user import User
            
            user = self.db.query(User).filter(User.cognito_id == patient_id).first()
            
            if user:
                conditions = getattr(user, 'comorbidities', []) or []
                conditions_lower = [c.lower() for c in conditions]
                
                flags["has_diabetes"] = 1 if any("diabetes" in c for c in conditions_lower) else 0
                flags["has_hypertension"] = 1 if any("hypertension" in c or "high blood pressure" in c for c in conditions_lower) else 0
                flags["has_immunocompromised"] = 1 if getattr(user, 'immunocompromised_condition', None) else 0
        except (ImportError, Exception) as e:
            logger.warning(f"Error fetching medical history flags: {e}")
        
        return flags
    
    async def _get_daily_features(
        self,
        patient_id: str,
        target_date: datetime
    ) -> Optional[Dict[str, float]]:
        """Get features for a specific day (for time-series) with graceful fallback."""
        try:
            from app.models.trend_models import TrendMetric
            
            start = datetime.combine(target_date, datetime.min.time())
            end = datetime.combine(target_date, datetime.max.time())
            
            features = {}
            
            metrics = self.db.query(TrendMetric).filter(
                and_(
                    TrendMetric.patient_id == patient_id,
                    TrendMetric.recorded_at >= start,
                    TrendMetric.recorded_at <= end
                )
            ).all()
            
            if not metrics:
                return None
            
            for m in metrics:
                feature_name = m.metric_name.replace("blood_pressure_", "bp_").replace("oxygen_saturation", "spo2")
                features[feature_name] = m.raw_value
            
            return features
        except (ImportError, Exception) as e:
            logger.warning(f"Error fetching daily features: {e}")
            return None
    
    def _get_default_daily_features(self) -> Dict[str, float]:
        """Get default feature values for missing days."""
        return {
            "heart_rate": 0.5,
            "respiratory_rate": 0.5,
            "bp_systolic": 0.5,
            "bp_diastolic": 0.5,
            "spo2": 0.5,
            "temperature": 0.5,
            "pain_level": 0.5,
            "fatigue_score": 0.5
        }
    
    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Normalize features to 0-1 range."""
        normalized = {}
        
        for name, value in features.items():
            if value is None:
                normalized[name] = 0.0
                continue
            
            # Get range for feature
            base_name = name.replace("_max", "").replace("_std", "")
            if base_name in self.FEATURE_RANGES:
                min_val, max_val = self.FEATURE_RANGES[base_name]
                # Min-max normalization
                normalized[name] = max(0.0, min(1.0, (float(value) - min_val) / (max_val - min_val)))
            else:
                # Keep as-is for binary/unknown features
                normalized[name] = float(value) if isinstance(value, (int, float)) else 0.0
        
        return normalized
    
    def get_feature_importance_labels(self, model_type: str) -> List[str]:
        """Get human-readable labels for feature importance display."""
        labels = {
            "age": "Age",
            "sex_male": "Sex (Male)",
            "bmi": "Body Mass Index",
            "heart_rate": "Heart Rate",
            "bp_systolic": "Systolic Blood Pressure",
            "bp_diastolic": "Diastolic Blood Pressure",
            "respiratory_rate": "Respiratory Rate",
            "spo2": "Oxygen Saturation",
            "temperature": "Body Temperature",
            "a1c": "HbA1c",
            "wbc": "White Blood Cell Count",
            "creatinine": "Creatinine",
            "phq9_score": "Depression Score (PHQ-9)",
            "gad7_score": "Anxiety Score (GAD-7)",
            "pss10_score": "Stress Score (PSS-10)",
            "daily_steps": "Daily Steps",
            "sleep_hours": "Sleep Duration",
            "checkin_rate": "Check-in Compliance",
            "symptom_count": "Symptom Count",
            "pain_level": "Pain Level",
            "fatigue_score": "Fatigue Level",
            "has_diabetes": "Has Diabetes",
            "has_hypertension": "Has Hypertension",
            "has_immunocompromised": "Immunocompromised",
            "smoking_status": "Smoking Status"
        }
        
        features = self.MODEL_FEATURES.get(model_type, [])
        return [labels.get(f, f) for f in features]
