"""
Feature Aggregation Pipeline for Followup Autopilot

Computes 30-day rolling metrics from raw signals:
- Pain/Fatigue/Mood averages
- Steps, HR, Sleep aggregates
- Environmental risk scores
- Medication adherence
- Mental health scores
- Video/Audio AI scores
- Risk & Exposures features

All features are normalized and z-scored for ML model input.
"""

import os
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from decimal import Decimal
import statistics

logger = logging.getLogger(__name__)


def safe_mean(values: List[float], default: float = 0.0) -> float:
    """Compute mean with fallback for empty lists"""
    filtered = [v for v in values if v is not None]
    return statistics.mean(filtered) if filtered else default


def safe_stdev(values: List[float], default: float = 0.0) -> float:
    """Compute standard deviation with fallback"""
    filtered = [v for v in values if v is not None]
    return statistics.stdev(filtered) if len(filtered) > 1 else default


def normalize_value(value: float, min_val: float, max_val: float) -> float:
    """Normalize value to 0-1 range"""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


class FeatureBuilder:
    """
    Daily feature aggregation pipeline.
    
    Responsibilities:
    1. Aggregate raw signals into daily features
    2. Compute rolling metrics (7-day, 14-day, 30-day)
    3. Normalize features for ML model input
    4. Handle missing data with sensible defaults
    """

    FEATURE_COLUMNS = [
        "avg_pain", "avg_fatigue", "avg_mood", "checkins_count",
        "steps", "resting_hr", "sleep_hours", "weight",
        "env_risk_score", "pollen_index", "aqi", "temp_c",
        "med_adherence_7d", "mh_score", "video_resp_risk",
        "audio_emotion_score", "pain_severity_score", "engagement_rate_14d",
        "infectious_exposure_score", "immunization_status",
        "occupational_risk_score"
    ]

    NORMALIZATION_RANGES = {
        "avg_pain": (0, 10),
        "avg_fatigue": (0, 10),
        "avg_mood": (0, 10),
        "checkins_count": (0, 20),
        "steps": (0, 20000),
        "resting_hr": (40, 120),
        "sleep_hours": (0, 12),
        "weight": (30, 200),
        "env_risk_score": (0, 100),
        "pollen_index": (0, 12),
        "aqi": (0, 500),
        "temp_c": (-20, 50),
        "med_adherence_7d": (0, 1),
        "mh_score": (0, 1),
        "video_resp_risk": (0, 1),
        "audio_emotion_score": (0, 1),
        "pain_severity_score": (0, 1),
        "engagement_rate_14d": (0, 1),
        "infectious_exposure_score": (0, 1),
        "immunization_status": (0, 1),
        "occupational_risk_score": (0, 1),
    }

    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)

    def build_daily_features(
        self,
        patient_id: str,
        target_date: date
    ) -> Dict[str, Any]:
        """
        Build aggregated features for a single day.
        
        Args:
            patient_id: Patient identifier
            target_date: Date to aggregate features for
            
        Returns:
            Dictionary of feature values
        """
        signals = self._get_signals_for_date(patient_id, target_date)
        
        features = {
            "patient_id": patient_id,
            "date": target_date,
            "checkins_count": len(signals),
        }

        features.update(self._aggregate_device_signals(signals))
        features.update(self._aggregate_symptom_signals(signals))
        features.update(self._aggregate_pain_signals(signals))
        features.update(self._aggregate_mental_signals(signals))
        features.update(self._aggregate_environment_signals(signals))
        features.update(self._aggregate_video_signals(signals))
        features.update(self._aggregate_audio_signals(signals))
        features.update(self._aggregate_medication_signals(patient_id, target_date))
        features.update(self._aggregate_exposure_signals(signals))

        features["engagement_rate_14d"] = self._compute_engagement_rate(
            patient_id, target_date, days=14
        )

        return features

    def build_features_range(
        self,
        patient_id: str,
        start_date: date,
        end_date: date
    ) -> List[Dict[str, Any]]:
        """Build features for a date range"""
        features_list = []
        current = start_date
        
        while current <= end_date:
            features = self.build_daily_features(patient_id, current)
            features_list.append(features)
            current += timedelta(days=1)
            
        return features_list

    def get_feature_sequence(
        self,
        patient_id: str,
        end_date: date,
        sequence_length: int = 30
    ) -> List[List[float]]:
        """
        Get feature sequence for ML model input.
        
        Returns list of 30 daily feature vectors, padded with zeros if needed.
        """
        start_date = end_date - timedelta(days=sequence_length - 1)
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotDailyFeatures
            rows = self.db.query(AutopilotDailyFeatures).filter(
                AutopilotDailyFeatures.patient_id == patient_id,
                AutopilotDailyFeatures.date >= start_date,
                AutopilotDailyFeatures.date <= end_date
            ).order_by(AutopilotDailyFeatures.date).all()
            
            features_by_date = {row.date: row for row in rows}
        else:
            features_by_date = {}

        sequence = []
        current = start_date
        
        while current <= end_date:
            if current in features_by_date:
                row = features_by_date[current]
                vector = self._row_to_normalized_vector(row)
            else:
                vector = [0.0] * len(self.FEATURE_COLUMNS)
            sequence.append(vector)
            current += timedelta(days=1)

        while len(sequence) < sequence_length:
            sequence.insert(0, [0.0] * len(self.FEATURE_COLUMNS))
            
        return sequence[-sequence_length:]

    def save_daily_features(
        self,
        patient_id: str,
        target_date: date,
        features: Dict[str, Any]
    ) -> bool:
        """Save or update daily features in database"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotDailyFeatures
                
                existing = self.db.query(AutopilotDailyFeatures).filter(
                    AutopilotDailyFeatures.patient_id == patient_id,
                    AutopilotDailyFeatures.date == target_date
                ).first()
                
                if existing:
                    for key, value in features.items():
                        if hasattr(existing, key) and key not in ("patient_id", "date", "id"):
                            setattr(existing, key, value)
                else:
                    record = AutopilotDailyFeatures(
                        patient_id=patient_id,
                        date=target_date,
                        **{k: v for k, v in features.items() 
                           if k not in ("patient_id", "date") and hasattr(AutopilotDailyFeatures, k)}
                    )
                    self.db.add(record)
                    
                self.db.commit()
                return True
            else:
                return self._save_features_raw(patient_id, target_date, features)
                
        except Exception as e:
            self.logger.error(f"Failed to save daily features: {e}")
            if self.db:
                self.db.rollback()
            return False

    def _get_signals_for_date(
        self,
        patient_id: str,
        target_date: date
    ) -> List[Dict[str, Any]]:
        """Fetch all signals for a patient on a specific date"""
        start_dt = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_dt = datetime.combine(target_date, datetime.max.time()).replace(tzinfo=timezone.utc)
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotPatientSignal
            rows = self.db.query(AutopilotPatientSignal).filter(
                AutopilotPatientSignal.patient_id == patient_id,
                AutopilotPatientSignal.signal_time >= start_dt,
                AutopilotPatientSignal.signal_time <= end_dt
            ).all()
            
            return [
                {
                    "category": row.category,
                    "source": row.source,
                    "raw_payload": row.raw_payload or {},
                    "ml_score": row.ml_score,
                    "signal_time": row.signal_time
                }
                for row in rows
            ]
        
        return self._get_signals_raw(patient_id, start_dt, end_dt)

    def _aggregate_device_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate device/wearable signals"""
        device_signals = [s for s in signals if s["category"] == "device"]
        
        steps_list = []
        hr_list = []
        sleep_list = []
        weight_list = []
        
        for sig in device_signals:
            payload = sig.get("raw_payload", {})
            metrics = payload.get("metrics", payload)
            
            if "steps" in metrics:
                steps_list.append(float(metrics["steps"]))
            if "resting_hr" in metrics or "heart_rate" in metrics:
                hr_list.append(float(metrics.get("resting_hr", metrics.get("heart_rate", 0))))
            if "sleep_hours" in metrics or "sleep_duration" in metrics:
                sleep_list.append(float(metrics.get("sleep_hours", metrics.get("sleep_duration", 0))))
            if "weight" in metrics:
                weight_list.append(float(metrics["weight"]))
        
        return {
            "steps": int(sum(steps_list)) if steps_list else 0,
            "resting_hr": safe_mean(hr_list),
            "sleep_hours": safe_mean(sleep_list),
            "weight": safe_mean(weight_list) if weight_list else None,
        }

    def _aggregate_symptom_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate symptom check-in signals"""
        symptom_signals = [s for s in signals if s["category"] == "symptom"]
        
        fatigue_list = []
        mood_list = []
        
        for sig in symptom_signals:
            payload = sig.get("raw_payload", {})
            symptoms = payload.get("symptoms", [])
            
            for symptom in symptoms:
                if isinstance(symptom, dict):
                    if symptom.get("type") == "fatigue":
                        fatigue_list.append(float(symptom.get("severity", 5)))
                    if symptom.get("type") == "mood":
                        mood_list.append(float(symptom.get("value", 5)))
        
        return {
            "avg_fatigue": safe_mean(fatigue_list, 5.0),
            "avg_mood": safe_mean(mood_list, 5.0),
        }

    def _aggregate_pain_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate pain tracking signals"""
        pain_signals = [s for s in signals if s["category"] == "pain"]
        
        pain_scores = []
        for sig in pain_signals:
            if sig.get("ml_score") is not None:
                pain_scores.append(float(sig["ml_score"]) * 10)
        
        avg_pain = safe_mean(pain_scores, 0.0)
        severity = avg_pain / 10.0 if avg_pain > 0 else 0.0
        
        return {
            "avg_pain": avg_pain,
            "pain_severity_score": severity,
        }

    def _aggregate_mental_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate mental health signals"""
        mental_signals = [s for s in signals if s["category"] == "mental"]
        
        mh_scores = []
        for sig in mental_signals:
            if sig.get("ml_score") is not None:
                mh_scores.append(float(sig["ml_score"]))
        
        return {
            "mh_score": safe_mean(mh_scores, 0.0),
        }

    def _aggregate_environment_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate environmental risk signals"""
        env_signals = [s for s in signals if s["category"] == "environment"]
        
        env_risk_list = []
        pollen_list = []
        aqi_list = []
        temp_list = []
        
        for sig in env_signals:
            payload = sig.get("raw_payload", {})
            
            if sig.get("ml_score") is not None:
                env_risk_list.append(float(sig["ml_score"]) * 100)
            if "pollen_index" in payload:
                pollen_list.append(float(payload["pollen_index"]))
            if "aqi" in payload:
                aqi_list.append(float(payload["aqi"]))
            if "temp_c" in payload:
                temp_list.append(float(payload["temp_c"]))
        
        return {
            "env_risk_score": safe_mean(env_risk_list, 0.0),
            "pollen_index": safe_mean(pollen_list, 0.0),
            "aqi": safe_mean(aqi_list, 0.0),
            "temp_c": safe_mean(temp_list) if temp_list else None,
        }

    def _aggregate_video_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate Video AI signals"""
        video_signals = [s for s in signals if s["category"] == "video"]
        
        resp_risk_list = []
        for sig in video_signals:
            if sig.get("ml_score") is not None:
                resp_risk_list.append(float(sig["ml_score"]))
        
        return {
            "video_resp_risk": safe_mean(resp_risk_list, 0.0),
        }

    def _aggregate_audio_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate Audio AI signals"""
        audio_signals = [s for s in signals if s["category"] == "audio"]
        
        emotion_list = []
        for sig in audio_signals:
            if sig.get("ml_score") is not None:
                emotion_list.append(float(sig["ml_score"]))
        
        return {
            "audio_emotion_score": safe_mean(emotion_list, 0.5),
        }

    def _aggregate_medication_signals(
        self,
        patient_id: str,
        target_date: date
    ) -> Dict[str, float]:
        """Aggregate medication adherence over 7 days"""
        start_date = target_date - timedelta(days=6)
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotPatientSignal
            from sqlalchemy import func
            
            meds_signals = self.db.query(AutopilotPatientSignal).filter(
                AutopilotPatientSignal.patient_id == patient_id,
                AutopilotPatientSignal.category == "meds",
                func.date(AutopilotPatientSignal.signal_time) >= start_date,
                func.date(AutopilotPatientSignal.signal_time) <= target_date
            ).all()
            
            adherence_scores = [
                float(sig.ml_score) for sig in meds_signals 
                if sig.ml_score is not None
            ]
        else:
            adherence_scores = []
        
        return {
            "med_adherence_7d": safe_mean(adherence_scores, 1.0),
        }

    def _aggregate_exposure_signals(self, signals: List[Dict]) -> Dict[str, float]:
        """Aggregate Risk & Exposures signals"""
        exposure_signals = [s for s in signals if s["category"] == "exposure"]
        
        infectious_scores = []
        immunization_scores = []
        occupational_scores = []
        genetic_flags = []
        
        for sig in exposure_signals:
            payload = sig.get("raw_payload", {})
            exposure_type = payload.get("exposure_type", "")
            score = sig.get("ml_score", 0) or 0
            
            if exposure_type == "infectious":
                infectious_scores.append(float(score))
            elif exposure_type == "immunization":
                immunization_scores.append(float(score))
            elif exposure_type == "occupational":
                occupational_scores.append(float(score))
            elif exposure_type == "genetic":
                genetic_flags.extend(payload.get("details", {}).get("flags", []))
        
        return {
            "infectious_exposure_score": safe_mean(infectious_scores, 0.0),
            "immunization_status": safe_mean(immunization_scores, 1.0),
            "occupational_risk_score": safe_mean(occupational_scores, 0.0),
            "genetic_risk_flags": genetic_flags,
        }

    def _compute_engagement_rate(
        self,
        patient_id: str,
        target_date: date,
        days: int = 14
    ) -> float:
        """Compute fraction of last N days with at least one signal"""
        start_date = target_date - timedelta(days=days - 1)
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotPatientSignal
            from sqlalchemy import func
            
            active_days = self.db.query(
                func.date(AutopilotPatientSignal.signal_time)
            ).filter(
                AutopilotPatientSignal.patient_id == patient_id,
                func.date(AutopilotPatientSignal.signal_time) >= start_date,
                func.date(AutopilotPatientSignal.signal_time) <= target_date
            ).distinct().count()
            
            return active_days / days
        
        return 0.5

    def _row_to_normalized_vector(self, row) -> List[float]:
        """Convert a database row to normalized feature vector"""
        vector = []
        
        for col in self.FEATURE_COLUMNS:
            value = getattr(row, col, 0) or 0
            if isinstance(value, Decimal):
                value = float(value)
            
            if col in self.NORMALIZATION_RANGES:
                min_val, max_val = self.NORMALIZATION_RANGES[col]
                value = normalize_value(float(value), min_val, max_val)
            
            vector.append(float(value))
            
        return vector

    def _get_signals_raw(
        self,
        patient_id: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[Dict[str, Any]]:
        """Direct database query when ORM not available"""
        import psycopg2
        from psycopg2.extras import RealDictCursor
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            return []
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT category, source, raw_payload, ml_score, signal_time
                    FROM autopilot_patient_signals
                    WHERE patient_id = %s AND signal_time >= %s AND signal_time <= %s
                """, (patient_id, start_dt, end_dt))
                return [dict(row) for row in cur.fetchall()]

    def _save_features_raw(
        self,
        patient_id: str,
        target_date: date,
        features: Dict[str, Any]
    ) -> bool:
        """Direct database save when ORM not available"""
        import psycopg2
        from psycopg2.extras import Json
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            return False
            
        columns = ["patient_id", "date"]
        values = [patient_id, target_date]
        
        for col in self.FEATURE_COLUMNS:
            if col in features:
                columns.append(col)
                val = features[col]
                if isinstance(val, list):
                    values.append(Json(val))
                else:
                    values.append(val)
        
        placeholders = ", ".join(["%s"] * len(values))
        col_names = ", ".join(columns)
        
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    INSERT INTO autopilot_daily_features ({col_names})
                    VALUES ({placeholders})
                    ON CONFLICT (patient_id, date) DO UPDATE SET
                    {", ".join(f"{c} = EXCLUDED.{c}" for c in columns[2:])}
                """, values)
            conn.commit()
            
        return True
