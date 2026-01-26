"""
ML Training Pipeline Service
Production-grade ML training infrastructure with:
- Consent-filtered patient data extraction
- Data preprocessing and feature engineering
- Multi-source data integration (patient data + public datasets)
- HIPAA-compliant audit logging
- Model training orchestration
- ONNX/joblib model export
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class ConsentedDataTypes:
    """Data types a patient has consented to contribute"""
    vitals: bool = False
    symptoms: bool = False
    medications: bool = False
    mental_health: bool = False
    behavioral_data: bool = False
    wearable_data: bool = False
    lab_results: bool = False
    imaging_data: bool = False
    daily_followup: bool = False
    habits: bool = False
    wellness: bool = False
    connected_apps: bool = False
    wearable_devices: bool = False
    wearable_heart: bool = False
    wearable_activity: bool = False
    wearable_sleep: bool = False
    wearable_oxygen: bool = False
    wearable_stress: bool = False
    environmental_risk: bool = False
    medical_history: bool = False
    current_conditions: bool = False
    device_readings_bp: bool = False
    device_readings_glucose: bool = False
    device_readings_scale: bool = False
    device_readings_thermometer: bool = False
    device_readings_stethoscope: bool = False
    device_readings_smartwatch: bool = False


@dataclass
class PatientDataContribution:
    """Represents anonymized patient data contribution for training"""
    patient_id_hash: str
    consent_id: str
    data_types: ConsentedDataTypes
    anonymization_level: str
    record_counts: Dict[str, int] = field(default_factory=dict)
    feature_vectors: Dict[str, np.ndarray] = field(default_factory=dict)
    date_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class TrainingDataset:
    """Aggregated training dataset"""
    features: np.ndarray
    labels: np.ndarray
    feature_names: List[str]
    patient_contributions: List[PatientDataContribution]
    public_dataset_sources: List[str] = field(default_factory=list)
    total_records: int = 0
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingJobConfig:
    """Configuration for a training job"""
    job_id: str
    model_name: str
    model_type: str
    target_version: str
    data_sources: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    validation_split: float = 0.2
    early_stopping: bool = True
    max_epochs: int = 100
    batch_size: int = 32


class DataAnonymizer:
    """HIPAA-compliant data anonymization utilities"""
    
    @staticmethod
    def hash_patient_id(patient_id: str, salt: Optional[str] = None) -> str:
        """Create irreversible hash of patient ID"""
        if salt is None:
            salt = os.environ.get("PATIENT_HASH_SALT", "default-training-salt")
        combined = f"{patient_id}:{salt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    @staticmethod
    def apply_differential_privacy(
        values: np.ndarray,
        epsilon: float = 1.0,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """Add Laplacian noise for differential privacy"""
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, values.shape)
        return values + noise
    
    @staticmethod
    def k_anonymize_age(age: Optional[int], k: int = 5) -> Optional[str]:
        """Generalize age into k-anonymous buckets"""
        if age is None:
            return None
        lower = (age // k) * k
        upper = lower + k - 1
        return f"{lower}-{upper}"
    
    @staticmethod
    def suppress_quasi_identifiers(data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or generalize quasi-identifiers with null-safe handling"""
        if not data:
            return {}
        
        suppressed = data.copy()
        
        quasi_identifiers = [
            'zip_code', 'birth_date', 'admission_date',
            'city', 'state', 'occupation', 'employer'
        ]
        
        for qi in quasi_identifiers:
            if qi in suppressed:
                del suppressed[qi]
        
        if 'age' in suppressed and suppressed.get('age') is not None:
            age_range = DataAnonymizer.k_anonymize_age(suppressed['age'])
            if age_range:
                suppressed['age_range'] = age_range
            del suppressed['age']
        elif 'age' in suppressed:
            del suppressed['age']
        
        return suppressed


class ConsentVerificationService:
    """Verify and filter patient data based on consent"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def get_consented_patients(
        self,
        required_data_types: List[str]
    ) -> List[Dict[str, Any]]:
        """Get list of patients who have consented to contribute specified data types"""
        
        query = text("""
            SELECT 
                id,
                patient_id,
                consent_enabled,
                data_types,
                anonymization_level,
                consent_signed_at
            FROM ml_training_consent
            WHERE consent_enabled = true
            AND consent_withdrawn_at IS NULL
            AND consent_signed_at IS NOT NULL
        """)
        
        result = await self.db.execute(query)
        rows = result.fetchall()
        
        consented = []
        
        def snake_to_camel(s: str) -> str:
            """Convert snake_case to camelCase"""
            parts = s.split('_')
            return parts[0] + ''.join(p.capitalize() for p in parts[1:])
        
        for row in rows:
            data_types = row.data_types or {}
            
            has_required = all(
                data_types.get(dt, False) or
                data_types.get(snake_to_camel(dt), False)
                for dt in required_data_types
            )
            
            if has_required:
                consented.append({
                    'consent_id': row.id,
                    'patient_id': row.patient_id,
                    'data_types': data_types,
                    'anonymization_level': row.anonymization_level,
                    'consent_signed_at': row.consent_signed_at
                })
        
        return consented
    
    async def verify_patient_consent(
        self,
        patient_id: str,
        data_type: str
    ) -> bool:
        """Verify if a specific patient has consented to a data type"""
        
        def snake_to_camel(s: str) -> str:
            """Convert snake_case to camelCase"""
            parts = s.split('_')
            return parts[0] + ''.join(p.capitalize() for p in parts[1:])
        
        query = text("""
            SELECT data_types
            FROM ml_training_consent
            WHERE patient_id = :patient_id
            AND consent_enabled = true
            AND consent_withdrawn_at IS NULL
        """)
        
        result = await self.db.execute(query, {"patient_id": patient_id})
        row = result.fetchone()
        
        if not row:
            return False
        
        data_types = row.data_types or {}
        return data_types.get(data_type, False) or data_types.get(snake_to_camel(data_type), False)


class PatientDataExtractor:
    """Extract and preprocess patient data for training"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.anonymizer = DataAnonymizer()
    
    async def extract_vitals_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract vital signs features for a patient"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                heart_rate,
                blood_pressure_systolic,
                blood_pressure_diastolic,
                temperature,
                spo2,
                respiratory_rate,
                recorded_at
            FROM health_metrics
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        result = await self.db.execute(query, {
            "patient_id": patient_id,
            "start_date": start_date,
            "end_date": end_date
        })
        rows = result.fetchall()
        
        if not rows:
            return {"features": None, "count": 0}
        
        features = {
            'heart_rate': [],
            'bp_systolic': [],
            'bp_diastolic': [],
            'temperature': [],
            'spo2': [],
            'respiratory_rate': []
        }
        
        for row in rows:
            if row.heart_rate:
                features['heart_rate'].append(float(row.heart_rate))
            if row.blood_pressure_systolic:
                features['bp_systolic'].append(float(row.blood_pressure_systolic))
            if row.blood_pressure_diastolic:
                features['bp_diastolic'].append(float(row.blood_pressure_diastolic))
            if row.temperature:
                features['temperature'].append(float(row.temperature))
            if row.spo2:
                features['spo2'].append(float(row.spo2))
            if row.respiratory_rate:
                features['respiratory_rate'].append(float(row.respiratory_rate))
        
        aggregated = {}
        for name, values in features.items():
            if values:
                arr = np.array(values)
                aggregated[f'{name}_mean'] = float(np.mean(arr))
                aggregated[f'{name}_std'] = float(np.std(arr))
                aggregated[f'{name}_min'] = float(np.min(arr))
                aggregated[f'{name}_max'] = float(np.max(arr))
                aggregated[f'{name}_trend'] = float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        
        return {"features": aggregated, "count": len(rows)}
    
    async def extract_symptom_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract symptom check-in features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                fatigue_level,
                pain_level,
                mood_level,
                appetite_level,
                sleep_quality,
                symptoms,
                created_at
            FROM daily_followups
            WHERE patient_id = :patient_id
            AND created_at BETWEEN :start_date AND :end_date
            ORDER BY created_at
        """)
        
        result = await self.db.execute(query, {
            "patient_id": patient_id,
            "start_date": start_date,
            "end_date": end_date
        })
        rows = result.fetchall()
        
        if not rows:
            return {"features": None, "count": 0}
        
        features = {
            'fatigue': [],
            'pain': [],
            'mood': [],
            'appetite': [],
            'sleep': []
        }
        symptom_counts = {}
        
        for row in rows:
            if row.fatigue_level is not None:
                features['fatigue'].append(float(row.fatigue_level))
            if row.pain_level is not None:
                features['pain'].append(float(row.pain_level))
            if row.mood_level is not None:
                features['mood'].append(float(row.mood_level))
            if row.appetite_level is not None:
                features['appetite'].append(float(row.appetite_level))
            if row.sleep_quality is not None:
                features['sleep'].append(float(row.sleep_quality))
            
            if row.symptoms:
                for symptom in row.symptoms:
                    symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
        
        aggregated = {}
        for name, values in features.items():
            if values:
                arr = np.array(values)
                aggregated[f'{name}_mean'] = float(np.mean(arr))
                aggregated[f'{name}_std'] = float(np.std(arr))
                aggregated[f'{name}_trend'] = float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        
        aggregated['unique_symptom_count'] = len(symptom_counts)
        aggregated['total_symptom_mentions'] = sum(symptom_counts.values())
        
        return {"features": aggregated, "count": len(rows)}
    
    async def extract_mental_health_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract mental health questionnaire features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                questionnaire_type,
                total_score,
                responses,
                completed_at
            FROM mental_health_questionnaires
            WHERE patient_id = :patient_id
            AND completed_at BETWEEN :start_date AND :end_date
            ORDER BY completed_at
        """)
        
        result = await self.db.execute(query, {
            "patient_id": patient_id,
            "start_date": start_date,
            "end_date": end_date
        })
        rows = result.fetchall()
        
        if not rows:
            return {"features": None, "count": 0}
        
        questionnaire_scores = {
            'phq9': [],
            'gad7': [],
            'pss10': []
        }
        
        for row in rows:
            q_type = row.questionnaire_type.lower() if row.questionnaire_type else ''
            score = row.total_score
            
            if 'phq' in q_type and score is not None:
                questionnaire_scores['phq9'].append(float(score))
            elif 'gad' in q_type and score is not None:
                questionnaire_scores['gad7'].append(float(score))
            elif 'pss' in q_type and score is not None:
                questionnaire_scores['pss10'].append(float(score))
        
        aggregated = {}
        for name, scores in questionnaire_scores.items():
            if scores:
                arr = np.array(scores)
                aggregated[f'{name}_latest'] = float(arr[-1])
                aggregated[f'{name}_mean'] = float(np.mean(arr))
                aggregated[f'{name}_trend'] = float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        
        return {"features": aggregated, "count": len(rows)}
    
    async def extract_medication_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract medication adherence features"""
        
        query = text("""
            SELECT 
                COUNT(*) as medication_count,
                AVG(CASE WHEN is_active THEN 1 ELSE 0 END) as active_rate
            FROM medications
            WHERE patient_id = :patient_id
        """)
        
        result = await self.db.execute(query, {"patient_id": patient_id})
        row = result.fetchone()
        
        features = {
            'total_medications': int(row.medication_count) if row else 0,
            'active_medication_rate': float(row.active_rate) if row and row.active_rate else 0.0
        }
        
        return {"features": features, "count": 1}
    
    async def extract_wearable_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract wearable device data features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                steps,
                active_minutes,
                sleep_hours,
                sleep_quality,
                calories_burned,
                recorded_at
            FROM wearable_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        features = {
            'steps': [],
            'active_minutes': [],
            'sleep_hours': [],
            'sleep_quality': [],
            'calories': []
        }
        
        for row in rows:
            if row.steps:
                features['steps'].append(float(row.steps))
            if row.active_minutes:
                features['active_minutes'].append(float(row.active_minutes))
            if row.sleep_hours:
                features['sleep_hours'].append(float(row.sleep_hours))
            if row.sleep_quality:
                features['sleep_quality'].append(float(row.sleep_quality))
            if row.calories_burned:
                features['calories'].append(float(row.calories_burned))
        
        aggregated = {}
        for name, values in features.items():
            if values:
                arr = np.array(values)
                aggregated[f'{name}_mean'] = float(np.mean(arr))
                aggregated[f'{name}_std'] = float(np.std(arr))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_daily_followup_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract comprehensive daily followup features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                fatigue_level,
                pain_level,
                mood_level,
                appetite_level,
                sleep_quality,
                overall_wellness,
                symptoms,
                notes,
                created_at
            FROM daily_followups
            WHERE patient_id = :patient_id
            AND created_at BETWEEN :start_date AND :end_date
            ORDER BY created_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        features = {
            'fatigue': [],
            'pain': [],
            'mood': [],
            'appetite': [],
            'sleep': [],
            'wellness': []
        }
        
        for row in rows:
            if row.fatigue_level is not None:
                features['fatigue'].append(float(row.fatigue_level))
            if row.pain_level is not None:
                features['pain'].append(float(row.pain_level))
            if row.mood_level is not None:
                features['mood'].append(float(row.mood_level))
            if row.appetite_level is not None:
                features['appetite'].append(float(row.appetite_level))
            if row.sleep_quality is not None:
                features['sleep'].append(float(row.sleep_quality))
            if hasattr(row, 'overall_wellness') and row.overall_wellness is not None:
                features['wellness'].append(float(row.overall_wellness))
        
        aggregated = {
            'followup_count': len(rows),
            'followup_consistency': len(rows) / max((end_date - start_date).days, 1)
        }
        
        for name, values in features.items():
            if values:
                arr = np.array(values)
                aggregated[f'daily_{name}_mean'] = float(np.mean(arr))
                aggregated[f'daily_{name}_std'] = float(np.std(arr))
                aggregated[f'daily_{name}_trend'] = float(np.polyfit(range(len(arr)), arr, 1)[0]) if len(arr) > 1 else 0.0
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_habit_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract habit tracker features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                h.id,
                h.name,
                h.category,
                h.frequency,
                h.target_value,
                h.current_streak,
                h.longest_streak,
                h.total_completions,
                h.created_at
            FROM habits h
            WHERE h.patient_id = :patient_id
            AND h.created_at <= :end_date
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        total_habits = len(rows)
        current_streaks = []
        longest_streaks = []
        total_completions = []
        categories = {}
        
        for row in rows:
            if hasattr(row, 'current_streak') and row.current_streak:
                current_streaks.append(int(row.current_streak))
            if hasattr(row, 'longest_streak') and row.longest_streak:
                longest_streaks.append(int(row.longest_streak))
            if hasattr(row, 'total_completions') and row.total_completions:
                total_completions.append(int(row.total_completions))
            if hasattr(row, 'category') and row.category:
                categories[row.category] = categories.get(row.category, 0) + 1
        
        aggregated = {
            'total_habits': total_habits,
            'habit_categories_count': len(categories),
            'avg_current_streak': float(np.mean(current_streaks)) if current_streaks else 0.0,
            'avg_longest_streak': float(np.mean(longest_streaks)) if longest_streaks else 0.0,
            'total_habit_completions': sum(total_completions),
            'avg_completions_per_habit': float(np.mean(total_completions)) if total_completions else 0.0
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wellness_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract wellness activity features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                activity_type,
                duration_minutes,
                intensity,
                calories_burned,
                notes,
                created_at
            FROM wellness_activities
            WHERE patient_id = :patient_id
            AND created_at BETWEEN :start_date AND :end_date
            ORDER BY created_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        durations = []
        intensities = []
        calories = []
        activity_types = {}
        
        for row in rows:
            if hasattr(row, 'duration_minutes') and row.duration_minutes:
                durations.append(float(row.duration_minutes))
            if hasattr(row, 'intensity') and row.intensity:
                intensities.append(float(row.intensity))
            if hasattr(row, 'calories_burned') and row.calories_burned:
                calories.append(float(row.calories_burned))
            if hasattr(row, 'activity_type') and row.activity_type:
                activity_types[row.activity_type] = activity_types.get(row.activity_type, 0) + 1
        
        aggregated = {
            'wellness_activity_count': len(rows),
            'wellness_activity_types': len(activity_types),
            'avg_duration_minutes': float(np.mean(durations)) if durations else 0.0,
            'total_duration_minutes': float(sum(durations)) if durations else 0.0,
            'avg_intensity': float(np.mean(intensities)) if intensities else 0.0,
            'total_calories': float(sum(calories)) if calories else 0.0
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_connected_apps_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract features from connected health apps"""
        
        query = text("""
            SELECT 
                app_name,
                app_type,
                last_sync_at,
                data_types_synced,
                is_active
            FROM connected_apps
            WHERE patient_id = :patient_id
        """)
        
        try:
            result = await self.db.execute(query, {"patient_id": patient_id})
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        active_apps = sum(1 for row in rows if hasattr(row, 'is_active') and row.is_active)
        app_types = set()
        
        for row in rows:
            if hasattr(row, 'app_type') and row.app_type:
                app_types.add(row.app_type)
        
        aggregated = {
            'total_connected_apps': len(rows),
            'active_connected_apps': active_apps,
            'app_types_count': len(app_types)
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wearable_device_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract wearable device metadata features"""
        
        query = text("""
            SELECT 
                device_type,
                device_name,
                last_sync_at,
                is_active
            FROM wearable_devices
            WHERE patient_id = :patient_id
        """)
        
        try:
            result = await self.db.execute(query, {"patient_id": patient_id})
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        device_types = set()
        active_devices = 0
        
        for row in rows:
            if hasattr(row, 'device_type') and row.device_type:
                device_types.add(row.device_type)
            if hasattr(row, 'is_active') and row.is_active:
                active_devices += 1
        
        aggregated = {
            'total_wearable_devices': len(rows),
            'active_wearable_devices': active_devices,
            'wearable_device_types': len(device_types)
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wearable_heart_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract heart and cardiovascular features from wearables"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                heart_rate,
                heart_rate_variability,
                resting_heart_rate,
                recorded_at
            FROM wearable_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            AND (heart_rate IS NOT NULL OR heart_rate_variability IS NOT NULL)
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        heart_rates = []
        hrvs = []
        resting_hrs = []
        
        for row in rows:
            if hasattr(row, 'heart_rate') and row.heart_rate:
                heart_rates.append(float(row.heart_rate))
            if hasattr(row, 'heart_rate_variability') and row.heart_rate_variability:
                hrvs.append(float(row.heart_rate_variability))
            if hasattr(row, 'resting_heart_rate') and row.resting_heart_rate:
                resting_hrs.append(float(row.resting_heart_rate))
        
        aggregated = {}
        if heart_rates:
            aggregated['wearable_hr_mean'] = float(np.mean(heart_rates))
            aggregated['wearable_hr_std'] = float(np.std(heart_rates))
            aggregated['wearable_hr_min'] = float(np.min(heart_rates))
            aggregated['wearable_hr_max'] = float(np.max(heart_rates))
        if hrvs:
            aggregated['wearable_hrv_mean'] = float(np.mean(hrvs))
            aggregated['wearable_hrv_std'] = float(np.std(hrvs))
        if resting_hrs:
            aggregated['wearable_rhr_mean'] = float(np.mean(resting_hrs))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wearable_activity_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract activity and movement features from wearables"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                steps,
                distance,
                active_minutes,
                calories_burned,
                floors_climbed,
                recorded_at
            FROM wearable_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        steps = []
        distances = []
        active_mins = []
        calories = []
        
        for row in rows:
            if hasattr(row, 'steps') and row.steps:
                steps.append(float(row.steps))
            if hasattr(row, 'distance') and row.distance:
                distances.append(float(row.distance))
            if hasattr(row, 'active_minutes') and row.active_minutes:
                active_mins.append(float(row.active_minutes))
            if hasattr(row, 'calories_burned') and row.calories_burned:
                calories.append(float(row.calories_burned))
        
        aggregated = {}
        if steps:
            aggregated['wearable_steps_mean'] = float(np.mean(steps))
            aggregated['wearable_steps_total'] = float(sum(steps))
        if distances:
            aggregated['wearable_distance_mean'] = float(np.mean(distances))
        if active_mins:
            aggregated['wearable_active_mins_mean'] = float(np.mean(active_mins))
        if calories:
            aggregated['wearable_calories_mean'] = float(np.mean(calories))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wearable_sleep_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract sleep data features from wearables"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                sleep_hours,
                sleep_quality,
                deep_sleep_hours,
                rem_sleep_hours,
                light_sleep_hours,
                awake_time_minutes,
                recorded_at
            FROM wearable_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            AND sleep_hours IS NOT NULL
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        sleep_hours = []
        sleep_quality = []
        deep_sleep = []
        rem_sleep = []
        
        for row in rows:
            if hasattr(row, 'sleep_hours') and row.sleep_hours:
                sleep_hours.append(float(row.sleep_hours))
            if hasattr(row, 'sleep_quality') and row.sleep_quality:
                sleep_quality.append(float(row.sleep_quality))
            if hasattr(row, 'deep_sleep_hours') and row.deep_sleep_hours:
                deep_sleep.append(float(row.deep_sleep_hours))
            if hasattr(row, 'rem_sleep_hours') and row.rem_sleep_hours:
                rem_sleep.append(float(row.rem_sleep_hours))
        
        aggregated = {}
        if sleep_hours:
            aggregated['wearable_sleep_hours_mean'] = float(np.mean(sleep_hours))
            aggregated['wearable_sleep_hours_std'] = float(np.std(sleep_hours))
        if sleep_quality:
            aggregated['wearable_sleep_quality_mean'] = float(np.mean(sleep_quality))
        if deep_sleep:
            aggregated['wearable_deep_sleep_mean'] = float(np.mean(deep_sleep))
        if rem_sleep:
            aggregated['wearable_rem_sleep_mean'] = float(np.mean(rem_sleep))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wearable_oxygen_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract blood oxygen and respiratory features from wearables"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                spo2,
                respiratory_rate,
                recorded_at
            FROM wearable_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            AND (spo2 IS NOT NULL OR respiratory_rate IS NOT NULL)
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        spo2_values = []
        resp_rates = []
        
        for row in rows:
            if hasattr(row, 'spo2') and row.spo2:
                spo2_values.append(float(row.spo2))
            if hasattr(row, 'respiratory_rate') and row.respiratory_rate:
                resp_rates.append(float(row.respiratory_rate))
        
        aggregated = {}
        if spo2_values:
            aggregated['wearable_spo2_mean'] = float(np.mean(spo2_values))
            aggregated['wearable_spo2_min'] = float(np.min(spo2_values))
        if resp_rates:
            aggregated['wearable_resp_rate_mean'] = float(np.mean(resp_rates))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_wearable_stress_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract stress and recovery features from wearables"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                stress_level,
                recovery_score,
                body_battery,
                readiness_score,
                recorded_at
            FROM wearable_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        stress = []
        recovery = []
        battery = []
        readiness = []
        
        for row in rows:
            if hasattr(row, 'stress_level') and row.stress_level:
                stress.append(float(row.stress_level))
            if hasattr(row, 'recovery_score') and row.recovery_score:
                recovery.append(float(row.recovery_score))
            if hasattr(row, 'body_battery') and row.body_battery:
                battery.append(float(row.body_battery))
            if hasattr(row, 'readiness_score') and row.readiness_score:
                readiness.append(float(row.readiness_score))
        
        aggregated = {}
        if stress:
            aggregated['wearable_stress_mean'] = float(np.mean(stress))
            aggregated['wearable_stress_max'] = float(np.max(stress))
        if recovery:
            aggregated['wearable_recovery_mean'] = float(np.mean(recovery))
        if battery:
            aggregated['wearable_body_battery_mean'] = float(np.mean(battery))
        if readiness:
            aggregated['wearable_readiness_mean'] = float(np.mean(readiness))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_environmental_risk_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract environmental risk features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                air_quality_index,
                pollen_count,
                uv_index,
                temperature,
                humidity,
                allergen_alerts,
                recorded_at
            FROM environmental_data
            WHERE patient_id = :patient_id
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        aqi = []
        pollen = []
        uv = []
        
        for row in rows:
            if hasattr(row, 'air_quality_index') and row.air_quality_index:
                aqi.append(float(row.air_quality_index))
            if hasattr(row, 'pollen_count') and row.pollen_count:
                pollen.append(float(row.pollen_count))
            if hasattr(row, 'uv_index') and row.uv_index:
                uv.append(float(row.uv_index))
        
        aggregated = {
            'environmental_data_points': len(rows)
        }
        if aqi:
            aggregated['env_aqi_mean'] = float(np.mean(aqi))
            aggregated['env_aqi_max'] = float(np.max(aqi))
        if pollen:
            aggregated['env_pollen_mean'] = float(np.mean(pollen))
        if uv:
            aggregated['env_uv_mean'] = float(np.mean(uv))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_medical_history_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract medical history features"""
        
        query = text("""
            SELECT 
                condition_name,
                condition_type,
                diagnosis_date,
                resolved_date,
                severity,
                is_chronic
            FROM medical_history
            WHERE patient_id = :patient_id
        """)
        
        try:
            result = await self.db.execute(query, {"patient_id": patient_id})
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        total_conditions = len(rows)
        chronic_conditions = sum(1 for row in rows if hasattr(row, 'is_chronic') and row.is_chronic)
        resolved_conditions = sum(1 for row in rows if hasattr(row, 'resolved_date') and row.resolved_date)
        condition_types = set()
        
        for row in rows:
            if hasattr(row, 'condition_type') and row.condition_type:
                condition_types.add(row.condition_type)
        
        aggregated = {
            'total_historical_conditions': total_conditions,
            'chronic_condition_count': chronic_conditions,
            'resolved_condition_count': resolved_conditions,
            'condition_type_diversity': len(condition_types)
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_current_conditions_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract current conditions and active diagnoses features"""
        
        query = text("""
            SELECT 
                condition_name,
                condition_type,
                diagnosis_date,
                severity,
                treatment_status,
                is_primary
            FROM current_conditions
            WHERE patient_id = :patient_id
            AND is_active = true
        """)
        
        try:
            result = await self.db.execute(query, {"patient_id": patient_id})
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        active_count = len(rows)
        primary_conditions = sum(1 for row in rows if hasattr(row, 'is_primary') and row.is_primary)
        severities = []
        
        for row in rows:
            if hasattr(row, 'severity') and row.severity:
                severity_map = {'mild': 1, 'moderate': 2, 'severe': 3, 'critical': 4}
                sev_value = severity_map.get(str(row.severity).lower(), 2)
                severities.append(sev_value)
        
        aggregated = {
            'active_condition_count': active_count,
            'primary_condition_count': primary_conditions,
            'avg_condition_severity': float(np.mean(severities)) if severities else 0.0,
            'max_condition_severity': float(np.max(severities)) if severities else 0.0
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_lab_results_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract lab results features (anonymized)"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                test_name,
                test_category,
                result_value,
                result_unit,
                reference_range_low,
                reference_range_high,
                is_abnormal,
                test_date
            FROM lab_results
            WHERE patient_id = :patient_id
            AND test_date BETWEEN :start_date AND :end_date
            ORDER BY test_date
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        total_tests = len(rows)
        abnormal_count = sum(1 for row in rows if hasattr(row, 'is_abnormal') and row.is_abnormal)
        test_categories = set()
        
        for row in rows:
            if hasattr(row, 'test_category') and row.test_category:
                test_categories.add(row.test_category)
        
        aggregated = {
            'lab_test_count': total_tests,
            'abnormal_lab_count': abnormal_count,
            'abnormal_lab_rate': abnormal_count / total_tests if total_tests > 0 else 0.0,
            'lab_category_count': len(test_categories)
        }
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_device_readings_bp_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract blood pressure monitor device reading features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                systolic,
                diastolic,
                pulse_rate,
                irregular_heartbeat,
                measurement_position,
                recorded_at
            FROM device_readings
            WHERE patient_id = :patient_id
            AND device_type = 'bp_monitor'
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        systolics = []
        diastolics = []
        pulse_rates = []
        irregular_count = 0
        
        for row in rows:
            if hasattr(row, 'systolic') and row.systolic:
                systolics.append(float(row.systolic))
            if hasattr(row, 'diastolic') and row.diastolic:
                diastolics.append(float(row.diastolic))
            if hasattr(row, 'pulse_rate') and row.pulse_rate:
                pulse_rates.append(float(row.pulse_rate))
            if hasattr(row, 'irregular_heartbeat') and row.irregular_heartbeat:
                irregular_count += 1
        
        aggregated = {
            'device_bp_readings_count': len(rows),
            'device_bp_irregular_rate': irregular_count / len(rows) if rows else 0.0
        }
        if systolics:
            aggregated['device_bp_systolic_mean'] = float(np.mean(systolics))
            aggregated['device_bp_systolic_std'] = float(np.std(systolics))
            aggregated['device_bp_systolic_max'] = float(np.max(systolics))
            aggregated['device_bp_systolic_min'] = float(np.min(systolics))
            aggregated['device_bp_systolic_trend'] = float(np.polyfit(range(len(systolics)), systolics, 1)[0]) if len(systolics) > 1 else 0.0
        if diastolics:
            aggregated['device_bp_diastolic_mean'] = float(np.mean(diastolics))
            aggregated['device_bp_diastolic_std'] = float(np.std(diastolics))
            aggregated['device_bp_diastolic_max'] = float(np.max(diastolics))
            aggregated['device_bp_diastolic_min'] = float(np.min(diastolics))
        if pulse_rates:
            aggregated['device_bp_pulse_mean'] = float(np.mean(pulse_rates))
            aggregated['device_bp_pulse_std'] = float(np.std(pulse_rates))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_device_readings_glucose_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract glucose meter device reading features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                glucose_level,
                meal_context,
                ketone_level,
                recorded_at
            FROM device_readings
            WHERE patient_id = :patient_id
            AND device_type = 'glucose_meter'
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        glucose_levels = []
        fasting_glucose = []
        post_meal_glucose = []
        ketone_levels = []
        
        for row in rows:
            if hasattr(row, 'glucose_level') and row.glucose_level:
                glucose_levels.append(float(row.glucose_level))
                meal_context = row.meal_context if hasattr(row, 'meal_context') else None
                if meal_context == 'fasting':
                    fasting_glucose.append(float(row.glucose_level))
                elif meal_context in ['after_meal', 'post_meal']:
                    post_meal_glucose.append(float(row.glucose_level))
            if hasattr(row, 'ketone_level') and row.ketone_level:
                ketone_levels.append(float(row.ketone_level))
        
        aggregated = {
            'device_glucose_readings_count': len(rows)
        }
        if glucose_levels:
            aggregated['device_glucose_mean'] = float(np.mean(glucose_levels))
            aggregated['device_glucose_std'] = float(np.std(glucose_levels))
            aggregated['device_glucose_max'] = float(np.max(glucose_levels))
            aggregated['device_glucose_min'] = float(np.min(glucose_levels))
            aggregated['device_glucose_trend'] = float(np.polyfit(range(len(glucose_levels)), glucose_levels, 1)[0]) if len(glucose_levels) > 1 else 0.0
        if fasting_glucose:
            aggregated['device_glucose_fasting_mean'] = float(np.mean(fasting_glucose))
        if post_meal_glucose:
            aggregated['device_glucose_postmeal_mean'] = float(np.mean(post_meal_glucose))
        if ketone_levels:
            aggregated['device_ketone_mean'] = float(np.mean(ketone_levels))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_device_readings_scale_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract smart scale device reading features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                weight,
                body_fat_percentage,
                muscle_mass,
                bone_mass,
                body_water_percentage,
                visceral_fat,
                bmi,
                bmr,
                metabolic_age,
                recorded_at
            FROM device_readings
            WHERE patient_id = :patient_id
            AND device_type = 'smart_scale'
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        weights = []
        body_fats = []
        muscle_masses = []
        bmis = []
        visceral_fats = []
        
        for row in rows:
            if hasattr(row, 'weight') and row.weight:
                weights.append(float(row.weight))
            if hasattr(row, 'body_fat_percentage') and row.body_fat_percentage:
                body_fats.append(float(row.body_fat_percentage))
            if hasattr(row, 'muscle_mass') and row.muscle_mass:
                muscle_masses.append(float(row.muscle_mass))
            if hasattr(row, 'bmi') and row.bmi:
                bmis.append(float(row.bmi))
            if hasattr(row, 'visceral_fat') and row.visceral_fat:
                visceral_fats.append(float(row.visceral_fat))
        
        aggregated = {
            'device_scale_readings_count': len(rows)
        }
        if weights:
            aggregated['device_weight_mean'] = float(np.mean(weights))
            aggregated['device_weight_std'] = float(np.std(weights))
            aggregated['device_weight_trend'] = float(np.polyfit(range(len(weights)), weights, 1)[0]) if len(weights) > 1 else 0.0
            aggregated['device_weight_latest'] = weights[-1]
        if body_fats:
            aggregated['device_body_fat_mean'] = float(np.mean(body_fats))
            aggregated['device_body_fat_trend'] = float(np.polyfit(range(len(body_fats)), body_fats, 1)[0]) if len(body_fats) > 1 else 0.0
        if muscle_masses:
            aggregated['device_muscle_mass_mean'] = float(np.mean(muscle_masses))
        if bmis:
            aggregated['device_bmi_mean'] = float(np.mean(bmis))
            aggregated['device_bmi_latest'] = bmis[-1]
        if visceral_fats:
            aggregated['device_visceral_fat_mean'] = float(np.mean(visceral_fats))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_device_readings_thermometer_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract thermometer device reading features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                temperature,
                measurement_site,
                recorded_at
            FROM device_readings
            WHERE patient_id = :patient_id
            AND device_type = 'thermometer'
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        temperatures = []
        fever_count = 0
        
        for row in rows:
            if hasattr(row, 'temperature') and row.temperature:
                temp = float(row.temperature)
                temperatures.append(temp)
                if temp >= 38.0:
                    fever_count += 1
        
        aggregated = {
            'device_temp_readings_count': len(rows),
            'device_temp_fever_rate': fever_count / len(rows) if rows else 0.0
        }
        if temperatures:
            aggregated['device_temp_mean'] = float(np.mean(temperatures))
            aggregated['device_temp_std'] = float(np.std(temperatures))
            aggregated['device_temp_max'] = float(np.max(temperatures))
            aggregated['device_temp_min'] = float(np.min(temperatures))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_device_readings_stethoscope_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract digital stethoscope device reading features"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                heart_rate,
                respiratory_rate,
                heart_sounds_abnormal,
                lung_sounds_abnormal,
                murmur_detected,
                arrhythmia_detected,
                recorded_at
            FROM device_readings
            WHERE patient_id = :patient_id
            AND device_type = 'stethoscope'
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        heart_rates = []
        resp_rates = []
        abnormal_heart = 0
        abnormal_lung = 0
        murmur_count = 0
        arrhythmia_count = 0
        
        for row in rows:
            if hasattr(row, 'heart_rate') and row.heart_rate:
                heart_rates.append(float(row.heart_rate))
            if hasattr(row, 'respiratory_rate') and row.respiratory_rate:
                resp_rates.append(float(row.respiratory_rate))
            if hasattr(row, 'heart_sounds_abnormal') and row.heart_sounds_abnormal:
                abnormal_heart += 1
            if hasattr(row, 'lung_sounds_abnormal') and row.lung_sounds_abnormal:
                abnormal_lung += 1
            if hasattr(row, 'murmur_detected') and row.murmur_detected:
                murmur_count += 1
            if hasattr(row, 'arrhythmia_detected') and row.arrhythmia_detected:
                arrhythmia_count += 1
        
        aggregated = {
            'device_stethoscope_readings_count': len(rows),
            'device_stethoscope_abnormal_heart_rate': abnormal_heart / len(rows) if rows else 0.0,
            'device_stethoscope_abnormal_lung_rate': abnormal_lung / len(rows) if rows else 0.0,
            'device_stethoscope_murmur_rate': murmur_count / len(rows) if rows else 0.0,
            'device_stethoscope_arrhythmia_rate': arrhythmia_count / len(rows) if rows else 0.0
        }
        if heart_rates:
            aggregated['device_stethoscope_hr_mean'] = float(np.mean(heart_rates))
            aggregated['device_stethoscope_hr_std'] = float(np.std(heart_rates))
        if resp_rates:
            aggregated['device_stethoscope_rr_mean'] = float(np.mean(resp_rates))
            aggregated['device_stethoscope_rr_std'] = float(np.std(resp_rates))
        
        return {"features": aggregated, "count": len(rows)}

    async def extract_device_readings_smartwatch_features(
        self,
        patient_id: str,
        date_range: Tuple[datetime, datetime]
    ) -> Dict[str, Any]:
        """Extract smartwatch device reading features (50+ metrics)"""
        
        start_date, end_date = date_range
        
        query = text("""
            SELECT 
                heart_rate,
                resting_heart_rate,
                hrv,
                spo2,
                respiratory_rate,
                sleep_score,
                sleep_duration,
                deep_sleep_duration,
                rem_sleep_duration,
                light_sleep_duration,
                awake_duration,
                steps,
                active_minutes,
                calories_burned,
                vo2_max,
                stress_score,
                recovery_score,
                readiness_score,
                body_battery,
                skin_temperature_deviation,
                afib_detected,
                irregular_rhythm_detected,
                blood_pressure_trend,
                menstrual_cycle_day,
                cycle_phase,
                fall_detected,
                ecg_classification,
                training_load,
                training_effect,
                recorded_at
            FROM device_readings
            WHERE patient_id = :patient_id
            AND device_type = 'smartwatch'
            AND recorded_at BETWEEN :start_date AND :end_date
            ORDER BY recorded_at
        """)
        
        try:
            result = await self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            })
            rows = result.fetchall()
        except Exception:
            return {"features": None, "count": 0}
        
        if not rows:
            return {"features": None, "count": 0}
        
        heart_rates = []
        resting_hrs = []
        hrvs = []
        spo2s = []
        resp_rates = []
        sleep_scores = []
        sleep_durations = []
        steps = []
        active_mins = []
        calories = []
        vo2_maxs = []
        stress_scores = []
        recovery_scores = []
        readiness_scores = []
        body_batteries = []
        skin_temp_devs = []
        afib_count = 0
        irregular_count = 0
        fall_count = 0
        
        deep_sleeps = []
        rem_sleeps = []
        light_sleeps = []
        awake_durations = []
        training_loads = []
        training_effects = []
        ecg_normal_count = 0
        ecg_afib_count = 0
        ecg_inconclusive_count = 0
        
        for row in rows:
            if hasattr(row, 'heart_rate') and row.heart_rate:
                heart_rates.append(float(row.heart_rate))
            if hasattr(row, 'resting_heart_rate') and row.resting_heart_rate:
                resting_hrs.append(float(row.resting_heart_rate))
            if hasattr(row, 'hrv') and row.hrv:
                hrvs.append(float(row.hrv))
            if hasattr(row, 'spo2') and row.spo2:
                spo2s.append(float(row.spo2))
            if hasattr(row, 'respiratory_rate') and row.respiratory_rate:
                resp_rates.append(float(row.respiratory_rate))
            if hasattr(row, 'sleep_score') and row.sleep_score:
                sleep_scores.append(float(row.sleep_score))
            if hasattr(row, 'sleep_duration') and row.sleep_duration:
                sleep_durations.append(float(row.sleep_duration))
            if hasattr(row, 'deep_sleep_duration') and row.deep_sleep_duration:
                deep_sleeps.append(float(row.deep_sleep_duration))
            if hasattr(row, 'rem_sleep_duration') and row.rem_sleep_duration:
                rem_sleeps.append(float(row.rem_sleep_duration))
            if hasattr(row, 'light_sleep_duration') and row.light_sleep_duration:
                light_sleeps.append(float(row.light_sleep_duration))
            if hasattr(row, 'awake_duration') and row.awake_duration:
                awake_durations.append(float(row.awake_duration))
            if hasattr(row, 'steps') and row.steps:
                steps.append(float(row.steps))
            if hasattr(row, 'active_minutes') and row.active_minutes:
                active_mins.append(float(row.active_minutes))
            if hasattr(row, 'calories_burned') and row.calories_burned:
                calories.append(float(row.calories_burned))
            if hasattr(row, 'vo2_max') and row.vo2_max:
                vo2_maxs.append(float(row.vo2_max))
            if hasattr(row, 'stress_score') and row.stress_score:
                stress_scores.append(float(row.stress_score))
            if hasattr(row, 'recovery_score') and row.recovery_score:
                recovery_scores.append(float(row.recovery_score))
            if hasattr(row, 'readiness_score') and row.readiness_score:
                readiness_scores.append(float(row.readiness_score))
            if hasattr(row, 'body_battery') and row.body_battery:
                body_batteries.append(float(row.body_battery))
            if hasattr(row, 'skin_temperature_deviation') and row.skin_temperature_deviation:
                skin_temp_devs.append(float(row.skin_temperature_deviation))
            if hasattr(row, 'training_load') and row.training_load:
                training_loads.append(float(row.training_load))
            if hasattr(row, 'training_effect') and row.training_effect:
                training_effects.append(float(row.training_effect))
            if hasattr(row, 'afib_detected') and row.afib_detected:
                afib_count += 1
            if hasattr(row, 'irregular_rhythm_detected') and row.irregular_rhythm_detected:
                irregular_count += 1
            if hasattr(row, 'fall_detected') and row.fall_detected:
                fall_count += 1
            if hasattr(row, 'ecg_classification') and row.ecg_classification:
                ecg = str(row.ecg_classification).lower()
                if 'normal' in ecg or 'sinus' in ecg:
                    ecg_normal_count += 1
                elif 'afib' in ecg or 'fibrillation' in ecg:
                    ecg_afib_count += 1
                else:
                    ecg_inconclusive_count += 1
        
        aggregated = {
            'device_smartwatch_readings_count': len(rows),
            'device_smartwatch_afib_rate': afib_count / len(rows) if rows else 0.0,
            'device_smartwatch_irregular_rhythm_rate': irregular_count / len(rows) if rows else 0.0,
            'device_smartwatch_fall_rate': fall_count / len(rows) if rows else 0.0,
            'device_smartwatch_ecg_normal_rate': ecg_normal_count / len(rows) if rows else 0.0,
            'device_smartwatch_ecg_afib_rate': ecg_afib_count / len(rows) if rows else 0.0,
            'device_smartwatch_ecg_inconclusive_rate': ecg_inconclusive_count / len(rows) if rows else 0.0
        }
        
        if heart_rates:
            aggregated['device_smartwatch_hr_mean'] = float(np.mean(heart_rates))
            aggregated['device_smartwatch_hr_std'] = float(np.std(heart_rates))
            aggregated['device_smartwatch_hr_max'] = float(np.max(heart_rates))
            aggregated['device_smartwatch_hr_min'] = float(np.min(heart_rates))
            aggregated['device_smartwatch_hr_range'] = float(np.max(heart_rates) - np.min(heart_rates))
            aggregated['device_smartwatch_hr_trend'] = float(np.polyfit(range(len(heart_rates)), heart_rates, 1)[0]) if len(heart_rates) > 1 else 0.0
        if resting_hrs:
            aggregated['device_smartwatch_rhr_mean'] = float(np.mean(resting_hrs))
            aggregated['device_smartwatch_rhr_std'] = float(np.std(resting_hrs))
            aggregated['device_smartwatch_rhr_min'] = float(np.min(resting_hrs))
            aggregated['device_smartwatch_rhr_max'] = float(np.max(resting_hrs))
            aggregated['device_smartwatch_rhr_trend'] = float(np.polyfit(range(len(resting_hrs)), resting_hrs, 1)[0]) if len(resting_hrs) > 1 else 0.0
        if hrvs:
            aggregated['device_smartwatch_hrv_mean'] = float(np.mean(hrvs))
            aggregated['device_smartwatch_hrv_std'] = float(np.std(hrvs))
            aggregated['device_smartwatch_hrv_min'] = float(np.min(hrvs))
            aggregated['device_smartwatch_hrv_max'] = float(np.max(hrvs))
            aggregated['device_smartwatch_hrv_trend'] = float(np.polyfit(range(len(hrvs)), hrvs, 1)[0]) if len(hrvs) > 1 else 0.0
            aggregated['device_smartwatch_hrv_coefficient_variation'] = float(np.std(hrvs) / np.mean(hrvs)) if np.mean(hrvs) > 0 else 0.0
        if spo2s:
            aggregated['device_smartwatch_spo2_mean'] = float(np.mean(spo2s))
            aggregated['device_smartwatch_spo2_std'] = float(np.std(spo2s))
            aggregated['device_smartwatch_spo2_min'] = float(np.min(spo2s))
            aggregated['device_smartwatch_spo2_max'] = float(np.max(spo2s))
            aggregated['device_smartwatch_spo2_below_95_rate'] = sum(1 for s in spo2s if s < 95) / len(spo2s)
        if resp_rates:
            aggregated['device_smartwatch_rr_mean'] = float(np.mean(resp_rates))
            aggregated['device_smartwatch_rr_std'] = float(np.std(resp_rates))
            aggregated['device_smartwatch_rr_min'] = float(np.min(resp_rates))
            aggregated['device_smartwatch_rr_max'] = float(np.max(resp_rates))
        if sleep_scores:
            aggregated['device_smartwatch_sleep_score_mean'] = float(np.mean(sleep_scores))
            aggregated['device_smartwatch_sleep_score_std'] = float(np.std(sleep_scores))
            aggregated['device_smartwatch_sleep_score_min'] = float(np.min(sleep_scores))
            aggregated['device_smartwatch_sleep_score_max'] = float(np.max(sleep_scores))
            aggregated['device_smartwatch_sleep_score_trend'] = float(np.polyfit(range(len(sleep_scores)), sleep_scores, 1)[0]) if len(sleep_scores) > 1 else 0.0
        if sleep_durations:
            aggregated['device_smartwatch_sleep_duration_mean'] = float(np.mean(sleep_durations))
            aggregated['device_smartwatch_sleep_duration_std'] = float(np.std(sleep_durations))
            aggregated['device_smartwatch_sleep_duration_total'] = float(sum(sleep_durations))
        if deep_sleeps:
            aggregated['device_smartwatch_deep_sleep_mean'] = float(np.mean(deep_sleeps))
            aggregated['device_smartwatch_deep_sleep_ratio'] = float(np.mean(deep_sleeps)) / float(np.mean(sleep_durations)) if sleep_durations and np.mean(sleep_durations) > 0 else 0.0
        if rem_sleeps:
            aggregated['device_smartwatch_rem_sleep_mean'] = float(np.mean(rem_sleeps))
            aggregated['device_smartwatch_rem_sleep_ratio'] = float(np.mean(rem_sleeps)) / float(np.mean(sleep_durations)) if sleep_durations and np.mean(sleep_durations) > 0 else 0.0
        if light_sleeps:
            aggregated['device_smartwatch_light_sleep_mean'] = float(np.mean(light_sleeps))
        if awake_durations:
            aggregated['device_smartwatch_awake_mean'] = float(np.mean(awake_durations))
            aggregated['device_smartwatch_sleep_efficiency'] = 1.0 - (float(np.mean(awake_durations)) / float(np.mean(sleep_durations))) if sleep_durations and np.mean(sleep_durations) > 0 else 0.0
        if steps:
            aggregated['device_smartwatch_steps_mean'] = float(np.mean(steps))
            aggregated['device_smartwatch_steps_std'] = float(np.std(steps))
            aggregated['device_smartwatch_steps_total'] = float(sum(steps))
            aggregated['device_smartwatch_steps_max'] = float(np.max(steps))
            aggregated['device_smartwatch_steps_trend'] = float(np.polyfit(range(len(steps)), steps, 1)[0]) if len(steps) > 1 else 0.0
        if active_mins:
            aggregated['device_smartwatch_active_mins_mean'] = float(np.mean(active_mins))
            aggregated['device_smartwatch_active_mins_total'] = float(sum(active_mins))
            aggregated['device_smartwatch_active_mins_max'] = float(np.max(active_mins))
        if calories:
            aggregated['device_smartwatch_calories_mean'] = float(np.mean(calories))
            aggregated['device_smartwatch_calories_total'] = float(sum(calories))
        if vo2_maxs:
            aggregated['device_smartwatch_vo2max_mean'] = float(np.mean(vo2_maxs))
            aggregated['device_smartwatch_vo2max_latest'] = vo2_maxs[-1] if vo2_maxs else 0.0
            aggregated['device_smartwatch_vo2max_trend'] = float(np.polyfit(range(len(vo2_maxs)), vo2_maxs, 1)[0]) if len(vo2_maxs) > 1 else 0.0
        if stress_scores:
            aggregated['device_smartwatch_stress_mean'] = float(np.mean(stress_scores))
            aggregated['device_smartwatch_stress_std'] = float(np.std(stress_scores))
            aggregated['device_smartwatch_stress_max'] = float(np.max(stress_scores))
            aggregated['device_smartwatch_stress_min'] = float(np.min(stress_scores))
            aggregated['device_smartwatch_high_stress_rate'] = sum(1 for s in stress_scores if s > 75) / len(stress_scores)
        if recovery_scores:
            aggregated['device_smartwatch_recovery_mean'] = float(np.mean(recovery_scores))
            aggregated['device_smartwatch_recovery_std'] = float(np.std(recovery_scores))
            aggregated['device_smartwatch_recovery_min'] = float(np.min(recovery_scores))
            aggregated['device_smartwatch_recovery_trend'] = float(np.polyfit(range(len(recovery_scores)), recovery_scores, 1)[0]) if len(recovery_scores) > 1 else 0.0
        if readiness_scores:
            aggregated['device_smartwatch_readiness_mean'] = float(np.mean(readiness_scores))
            aggregated['device_smartwatch_readiness_std'] = float(np.std(readiness_scores))
            aggregated['device_smartwatch_readiness_min'] = float(np.min(readiness_scores))
            aggregated['device_smartwatch_low_readiness_rate'] = sum(1 for r in readiness_scores if r < 50) / len(readiness_scores)
        if body_batteries:
            aggregated['device_smartwatch_body_battery_mean'] = float(np.mean(body_batteries))
            aggregated['device_smartwatch_body_battery_std'] = float(np.std(body_batteries))
            aggregated['device_smartwatch_body_battery_min'] = float(np.min(body_batteries))
            aggregated['device_smartwatch_body_battery_max'] = float(np.max(body_batteries))
            aggregated['device_smartwatch_body_battery_range'] = float(np.max(body_batteries) - np.min(body_batteries))
        if skin_temp_devs:
            aggregated['device_smartwatch_skin_temp_dev_mean'] = float(np.mean(skin_temp_devs))
            aggregated['device_smartwatch_skin_temp_dev_std'] = float(np.std(skin_temp_devs))
            aggregated['device_smartwatch_skin_temp_dev_max'] = float(np.max(skin_temp_devs))
            aggregated['device_smartwatch_elevated_temp_rate'] = sum(1 for t in skin_temp_devs if t > 0.5) / len(skin_temp_devs)
        if training_loads:
            aggregated['device_smartwatch_training_load_mean'] = float(np.mean(training_loads))
            aggregated['device_smartwatch_training_load_total'] = float(sum(training_loads))
            aggregated['device_smartwatch_training_load_trend'] = float(np.polyfit(range(len(training_loads)), training_loads, 1)[0]) if len(training_loads) > 1 else 0.0
        if training_effects:
            aggregated['device_smartwatch_training_effect_mean'] = float(np.mean(training_effects))
            aggregated['device_smartwatch_training_effect_max'] = float(np.max(training_effects))
        
        return {"features": aggregated, "count": len(rows)}


class FeatureEngineeringPipeline:
    """Feature engineering and preprocessing pipeline"""
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def normalize_features(
        self,
        features: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """Normalize features using z-score normalization"""
        
        if fit:
            self.feature_stats['mean'] = np.nanmean(features, axis=0)
            self.feature_stats['std'] = np.nanstd(features, axis=0)
            self.feature_stats['std'] = np.where(
                self.feature_stats['std'] == 0,
                1.0,
                self.feature_stats['std']
            )
        
        normalized = (features - self.feature_stats['mean']) / self.feature_stats['std']
        return np.nan_to_num(normalized, nan=0.0)
    
    def create_feature_vector(
        self,
        extracted_features: Dict[str, Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Create a flat feature vector from extracted features"""
        
        vector = []
        names = []
        
        feature_order = [
            'vitals', 'symptoms', 'mental_health',
            'medications', 'wearable', 'lab_results',
            'daily_followup', 'habits', 'wellness',
            'connected_apps', 'wearable_devices',
            'wearable_heart', 'wearable_activity', 'wearable_sleep',
            'wearable_oxygen', 'wearable_stress',
            'environmental_risk', 'medical_history', 'current_conditions',
            'device_readings_bp', 'device_readings_glucose', 'device_readings_scale',
            'device_readings_thermometer', 'device_readings_stethoscope', 'device_readings_smartwatch'
        ]
        
        for category in feature_order:
            if category in extracted_features:
                cat_features = extracted_features[category].get('features', {})
                if cat_features:
                    for key in sorted(cat_features.keys()):
                        value = cat_features[key]
                        if isinstance(value, (int, float)):
                            vector.append(float(value))
                            names.append(f"{category}_{key}")
        
        self.feature_names = names
        return np.array(vector), names
    
    def impute_missing_features(
        self,
        features: np.ndarray,
        strategy: str = 'mean'
    ) -> np.ndarray:
        """Impute missing values in feature matrix"""
        
        result = features.copy()
        
        for i in range(features.shape[1]):
            col = features[:, i]
            mask = np.isnan(col)
            
            if mask.any():
                valid = col[~mask]
                if len(valid) > 0:
                    if strategy == 'mean':
                        fill_value = np.mean(valid)
                    elif strategy == 'median':
                        fill_value = np.median(valid)
                    elif strategy == 'zero':
                        fill_value = 0.0
                    else:
                        fill_value = 0.0
                    
                    result[mask, i] = fill_value
                else:
                    result[mask, i] = 0.0
        
        return result


class TrainingAuditLogger:
    """HIPAA-compliant audit logging for ML training operations"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def log_event(
        self,
        event_type: str,
        event_category: str,
        actor_id: Optional[str],
        actor_type: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        patient_id_hash: Optional[str] = None,
        phi_accessed: bool = False,
        phi_categories: Optional[List[str]] = None,
        event_details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log a training-related audit event"""
        
        try:
            query = text("""
                INSERT INTO ml_training_audit_log (
                    id, event_type, event_category, actor_id, actor_type,
                    resource_type, resource_id, patient_id_hash, phi_accessed,
                    phi_categories, event_details, success, error_message, created_at
                ) VALUES (
                    gen_random_uuid()::text, :event_type, :event_category, :actor_id, :actor_type,
                    :resource_type, :resource_id, :patient_id_hash, :phi_accessed,
                    :phi_categories, :event_details, :success, :error_message, NOW()
                )
            """)
            
            await self.db.execute(query, {
                "event_type": event_type,
                "event_category": event_category,
                "actor_id": actor_id,
                "actor_type": actor_type,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "patient_id_hash": patient_id_hash,
                "phi_accessed": phi_accessed,
                "phi_categories": json.dumps(phi_categories) if phi_categories else None,
                "event_details": json.dumps(event_details) if event_details else None,
                "success": success,
                "error_message": error_message
            })
            await self.db.commit()
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            print(f"[AUDIT_ERROR] Failed to log: {event_type} - {e}")


class MLTrainingPipeline:
    """Main ML Training Pipeline orchestrator"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.consent_service = ConsentVerificationService(db_session)
        self.data_extractor = PatientDataExtractor(db_session)
        self.feature_pipeline = FeatureEngineeringPipeline()
        self.audit_logger = TrainingAuditLogger(db_session)
        self.anonymizer = DataAnonymizer()
    
    async def prepare_training_dataset(
        self,
        config: TrainingJobConfig,
        date_range_days: int = 90
    ) -> TrainingDataset:
        """Prepare a training dataset with consent verification"""
        
        await self.audit_logger.log_event(
            event_type="training_dataset_preparation_started",
            event_category="training",
            actor_id=None,
            actor_type="system",
            resource_type="training_job",
            resource_id=config.job_id,
            event_details={
                "model_name": config.model_name,
                "date_range_days": date_range_days
            }
        )
        
        required_types = config.data_sources.get('patient_data_types', ['vitals', 'symptoms'])
        consented_patients = await self.consent_service.get_consented_patients(required_types)
        
        logger.info(f"Found {len(consented_patients)} patients with required consent")
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=date_range_days)
        date_range = (start_date, end_date)
        
        contributions = []
        all_features = []
        
        for patient in consented_patients:
            try:
                patient_hash = self.anonymizer.hash_patient_id(patient['patient_id'])
                
                extracted = {}
                record_counts = {}
                data_types = patient['data_types']
                
                consented_device_types = []
                skipped_device_types = []
                device_type_keys = [
                    'device_readings_bp', 'device_readings_glucose', 'device_readings_scale',
                    'device_readings_thermometer', 'device_readings_stethoscope', 'device_readings_smartwatch'
                ]
                for dt_key in device_type_keys:
                    camel_key = ''.join(word.capitalize() if i else word for i, word in enumerate(dt_key.split('_')))
                    if data_types.get(dt_key, False) or data_types.get(camel_key, False):
                        consented_device_types.append(dt_key)
                    else:
                        skipped_device_types.append(dt_key)
                
                if skipped_device_types:
                    logger.debug(f"Patient {patient_hash[:8]} skipped device types (no consent): {skipped_device_types}")
                
                if data_types.get('vitals', False):
                    result = await self.data_extractor.extract_vitals_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['vitals'] = result
                        record_counts['vitals'] = result['count']
                
                if data_types.get('symptoms', False):
                    result = await self.data_extractor.extract_symptom_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['symptoms'] = result
                        record_counts['symptoms'] = result['count']
                
                if data_types.get('mentalHealth', False) or data_types.get('mental_health', False):
                    result = await self.data_extractor.extract_mental_health_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['mental_health'] = result
                        record_counts['mental_health'] = result['count']
                
                if data_types.get('medications', False):
                    result = await self.data_extractor.extract_medication_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['medications'] = result
                        record_counts['medications'] = result['count']
                
                if data_types.get('wearableData', False) or data_types.get('wearable', False):
                    result = await self.data_extractor.extract_wearable_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable'] = result
                        record_counts['wearable'] = result['count']
                
                if data_types.get('daily_followup', False):
                    result = await self.data_extractor.extract_daily_followup_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['daily_followup'] = result
                        record_counts['daily_followup'] = result['count']
                
                if data_types.get('habits', False):
                    result = await self.data_extractor.extract_habit_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['habits'] = result
                        record_counts['habits'] = result['count']
                
                if data_types.get('wellness', False):
                    result = await self.data_extractor.extract_wellness_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wellness'] = result
                        record_counts['wellness'] = result['count']
                
                if data_types.get('connected_apps', False):
                    result = await self.data_extractor.extract_connected_apps_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['connected_apps'] = result
                        record_counts['connected_apps'] = result['count']
                
                if data_types.get('wearable_devices', False):
                    result = await self.data_extractor.extract_wearable_device_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable_devices'] = result
                        record_counts['wearable_devices'] = result['count']
                
                if data_types.get('wearable_heart', False):
                    result = await self.data_extractor.extract_wearable_heart_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable_heart'] = result
                        record_counts['wearable_heart'] = result['count']
                
                if data_types.get('wearable_activity', False):
                    result = await self.data_extractor.extract_wearable_activity_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable_activity'] = result
                        record_counts['wearable_activity'] = result['count']
                
                if data_types.get('wearable_sleep', False):
                    result = await self.data_extractor.extract_wearable_sleep_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable_sleep'] = result
                        record_counts['wearable_sleep'] = result['count']
                
                if data_types.get('wearable_oxygen', False):
                    result = await self.data_extractor.extract_wearable_oxygen_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable_oxygen'] = result
                        record_counts['wearable_oxygen'] = result['count']
                
                if data_types.get('wearable_stress', False):
                    result = await self.data_extractor.extract_wearable_stress_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['wearable_stress'] = result
                        record_counts['wearable_stress'] = result['count']
                
                if data_types.get('environmental_risk', False):
                    result = await self.data_extractor.extract_environmental_risk_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['environmental_risk'] = result
                        record_counts['environmental_risk'] = result['count']
                
                if data_types.get('medical_history', False):
                    result = await self.data_extractor.extract_medical_history_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['medical_history'] = result
                        record_counts['medical_history'] = result['count']
                
                if data_types.get('current_conditions', False):
                    result = await self.data_extractor.extract_current_conditions_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['current_conditions'] = result
                        record_counts['current_conditions'] = result['count']
                
                if data_types.get('lab_results', False) or data_types.get('labResults', False):
                    result = await self.data_extractor.extract_lab_results_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['lab_results'] = result
                        record_counts['lab_results'] = result['count']
                
                if data_types.get('device_readings_bp', False) or data_types.get('deviceReadingsBp', False):
                    result = await self.data_extractor.extract_device_readings_bp_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['device_readings_bp'] = result
                        record_counts['device_readings_bp'] = result['count']
                
                if data_types.get('device_readings_glucose', False) or data_types.get('deviceReadingsGlucose', False):
                    result = await self.data_extractor.extract_device_readings_glucose_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['device_readings_glucose'] = result
                        record_counts['device_readings_glucose'] = result['count']
                
                if data_types.get('device_readings_scale', False) or data_types.get('deviceReadingsScale', False):
                    result = await self.data_extractor.extract_device_readings_scale_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['device_readings_scale'] = result
                        record_counts['device_readings_scale'] = result['count']
                
                if data_types.get('device_readings_thermometer', False) or data_types.get('deviceReadingsThermometer', False):
                    result = await self.data_extractor.extract_device_readings_thermometer_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['device_readings_thermometer'] = result
                        record_counts['device_readings_thermometer'] = result['count']
                
                if data_types.get('device_readings_stethoscope', False) or data_types.get('deviceReadingsStethoscope', False):
                    result = await self.data_extractor.extract_device_readings_stethoscope_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['device_readings_stethoscope'] = result
                        record_counts['device_readings_stethoscope'] = result['count']
                
                if data_types.get('device_readings_smartwatch', False) or data_types.get('deviceReadingsSmartwatch', False):
                    result = await self.data_extractor.extract_device_readings_smartwatch_features(
                        patient['patient_id'], date_range
                    )
                    if result['features']:
                        extracted['device_readings_smartwatch'] = result
                        record_counts['device_readings_smartwatch'] = result['count']
                
                if extracted:
                    included_device_types = [k for k in extracted.keys() if k.startswith('device_readings_')]
                    if included_device_types or consented_device_types:
                        await self.audit_logger.log_event(
                            event_type="device_data_consent_verification",
                            event_category="training",
                            actor_id=None,
                            actor_type="system",
                            resource_type="patient_data",
                            patient_id_hash=patient_hash,
                            event_details={
                                "consented_device_types": consented_device_types,
                                "extracted_device_types": included_device_types,
                                "skipped_device_types": skipped_device_types,
                                "device_record_counts": {k: v for k, v in record_counts.items() if k.startswith('device_readings_')}
                            }
                        )
                    
                    feature_vector, feature_names = self.feature_pipeline.create_feature_vector(extracted)
                    
                    contribution = PatientDataContribution(
                        patient_id_hash=patient_hash,
                        consent_id=patient['consent_id'],
                        data_types=ConsentedDataTypes(
                            vitals=data_types.get('vitals', False),
                            symptoms=data_types.get('symptoms', False),
                            mental_health=data_types.get('mentalHealth', False) or data_types.get('mental_health', False),
                            medications=data_types.get('medications', False),
                            wearable_data=data_types.get('wearableData', False) or data_types.get('wearable', False),
                            lab_results=data_types.get('labResults', False) or data_types.get('lab_results', False),
                            daily_followup=data_types.get('daily_followup', False),
                            habits=data_types.get('habits', False),
                            wellness=data_types.get('wellness', False),
                            connected_apps=data_types.get('connected_apps', False),
                            wearable_devices=data_types.get('wearable_devices', False),
                            wearable_heart=data_types.get('wearable_heart', False),
                            wearable_activity=data_types.get('wearable_activity', False),
                            wearable_sleep=data_types.get('wearable_sleep', False),
                            wearable_oxygen=data_types.get('wearable_oxygen', False),
                            wearable_stress=data_types.get('wearable_stress', False),
                            environmental_risk=data_types.get('environmental_risk', False),
                            medical_history=data_types.get('medical_history', False),
                            current_conditions=data_types.get('current_conditions', False),
                            device_readings_bp=data_types.get('device_readings_bp', False) or data_types.get('deviceReadingsBp', False),
                            device_readings_glucose=data_types.get('device_readings_glucose', False) or data_types.get('deviceReadingsGlucose', False),
                            device_readings_scale=data_types.get('device_readings_scale', False) or data_types.get('deviceReadingsScale', False),
                            device_readings_thermometer=data_types.get('device_readings_thermometer', False) or data_types.get('deviceReadingsThermometer', False),
                            device_readings_stethoscope=data_types.get('device_readings_stethoscope', False) or data_types.get('deviceReadingsStethoscope', False),
                            device_readings_smartwatch=data_types.get('device_readings_smartwatch', False) or data_types.get('deviceReadingsSmartwatch', False)
                        ),
                        anonymization_level=patient['anonymization_level'] or 'full',
                        record_counts=record_counts,
                        date_range=date_range
                    )
                    contribution.feature_vectors['combined'] = feature_vector
                    contributions.append(contribution)
                    all_features.append(feature_vector)
                    
                    await self._record_contribution(
                        config.job_id,
                        contribution
                    )
                
            except Exception as e:
                logger.error(f"Error extracting data for patient: {e}")
                await self.audit_logger.log_event(
                    event_type="patient_data_extraction_failed",
                    event_category="training",
                    actor_id=None,
                    actor_type="system",
                    resource_type="patient_data",
                    patient_id_hash=self.anonymizer.hash_patient_id(patient['patient_id']),
                    success=False,
                    error_message=str(e)
                )
        
        if not all_features:
            logger.warning("No training data available from consented patients")
            return TrainingDataset(
                features=np.array([]),
                labels=np.array([]),
                feature_names=[],
                patient_contributions=contributions,
                total_records=0
            )
        
        max_len = max(len(f) for f in all_features)
        padded_features = []
        for f in all_features:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), 'constant', constant_values=0)
                padded_features.append(padded)
            else:
                padded_features.append(f)
        
        feature_matrix = np.array(padded_features)
        
        feature_matrix = self.feature_pipeline.impute_missing_features(feature_matrix)
        feature_matrix = self.feature_pipeline.normalize_features(feature_matrix)
        
        labels = np.zeros(len(feature_matrix))
        
        total_records = sum(
            sum(c.record_counts.values())
            for c in contributions
        )
        
        await self.audit_logger.log_event(
            event_type="training_dataset_prepared",
            event_category="training",
            actor_id=None,
            actor_type="system",
            resource_type="training_job",
            resource_id=config.job_id,
            event_details={
                "patient_count": len(contributions),
                "total_records": total_records,
                "feature_count": feature_matrix.shape[1] if len(feature_matrix) > 0 else 0
            },
            phi_accessed=True,
            phi_categories=required_types
        )
        
        return TrainingDataset(
            features=feature_matrix,
            labels=labels,
            feature_names=self.feature_pipeline.feature_names,
            patient_contributions=contributions,
            total_records=total_records
        )
    
    async def _record_contribution(
        self,
        job_id: str,
        contribution: PatientDataContribution
    ):
        """Record patient contribution in the database"""
        
        try:
            data_types_list = []
            if contribution.data_types.vitals:
                data_types_list.append('vitals')
            if contribution.data_types.symptoms:
                data_types_list.append('symptoms')
            if contribution.data_types.mental_health:
                data_types_list.append('mental_health')
            if contribution.data_types.medications:
                data_types_list.append('medications')
            if contribution.data_types.wearable_data:
                data_types_list.append('wearable')
            
            query = text("""
                INSERT INTO ml_training_contributions (
                    id, patient_id_hash, consent_id, training_job_id,
                    data_types_contributed, record_count,
                    date_range_start, date_range_end,
                    anonymization_level, status, contributed_at
                ) VALUES (
                    gen_random_uuid()::text, :patient_id_hash, :consent_id, :job_id,
                    :data_types, :record_count,
                    :start_date, :end_date,
                    :anonymization_level, 'included', NOW()
                )
            """)
            
            total_records = sum(contribution.record_counts.values())
            start_date, end_date = contribution.date_range
            
            await self.db.execute(query, {
                "patient_id_hash": contribution.patient_id_hash,
                "consent_id": contribution.consent_id,
                "job_id": job_id,
                "data_types": json.dumps(data_types_list),
                "record_count": total_records,
                "start_date": start_date,
                "end_date": end_date,
                "anonymization_level": contribution.anonymization_level
            })
            await self.db.commit()
            
        except Exception as e:
            logger.error(f"Failed to record contribution: {e}")
    
    async def create_training_job(
        self,
        model_name: str,
        model_type: str,
        data_sources: Dict[str, Any],
        hyperparameters: Optional[Dict[str, Any]] = None,
        initiated_by: Optional[str] = None
    ) -> str:
        """Create a new training job entry"""
        
        import uuid
        job_id = str(uuid.uuid4())
        version = f"1.0.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        query = text("""
            INSERT INTO ml_training_jobs (
                id, job_name, model_name, target_version, status,
                priority, data_sources, training_config,
                current_phase, progress_percent, queued_at, initiated_by
            ) VALUES (
                :job_id, :job_name, :model_name, :version, 'queued',
                5, :data_sources, :training_config,
                'queued', 0, NOW(), :initiated_by
            )
        """)
        
        await self.db.execute(query, {
            "job_id": job_id,
            "job_name": f"{model_name}_training_{version}",
            "model_name": model_name,
            "version": version,
            "data_sources": json.dumps(data_sources),
            "training_config": json.dumps(hyperparameters or {}),
            "initiated_by": initiated_by
        })
        await self.db.commit()
        
        await self.audit_logger.log_event(
            event_type="training_job_created",
            event_category="training",
            actor_id=initiated_by,
            actor_type="user" if initiated_by else "system",
            resource_type="training_job",
            resource_id=job_id,
            event_details={
                "model_name": model_name,
                "model_type": model_type,
                "version": version
            }
        )
        
        return job_id
    
    async def update_job_progress(
        self,
        job_id: str,
        phase: str,
        progress: int,
        message: Optional[str] = None
    ):
        """Update training job progress"""
        
        query = text("""
            UPDATE ml_training_jobs
            SET current_phase = :phase,
                progress_percent = :progress,
                progress_message = :message,
                updated_at = NOW()
            WHERE id = :job_id
        """)
        
        await self.db.execute(query, {
            "job_id": job_id,
            "phase": phase,
            "progress": progress,
            "message": message
        })
        await self.db.commit()
    
    async def complete_job(
        self,
        job_id: str,
        model_id: Optional[str] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Mark a training job as complete"""
        
        status = "completed" if success else "failed"
        
        query = text("""
            UPDATE ml_training_jobs
            SET status = :status,
                current_phase = :phase,
                progress_percent = :progress,
                completed_at = NOW(),
                result_model_id = :model_id,
                error_message = :error_message,
                updated_at = NOW()
            WHERE id = :job_id
        """)
        
        await self.db.execute(query, {
            "job_id": job_id,
            "status": status,
            "phase": "completed" if success else "failed",
            "progress": 100 if success else -1,
            "model_id": model_id,
            "error_message": error_message
        })
        await self.db.commit()
        
        await self.audit_logger.log_event(
            event_type=f"training_job_{status}",
            event_category="training",
            actor_id=None,
            actor_type="system",
            resource_type="training_job",
            resource_id=job_id,
            success=success,
            error_message=error_message
        )


def create_training_pipeline(db_session: AsyncSession) -> MLTrainingPipeline:
    """Factory function to create a training pipeline instance"""
    return MLTrainingPipeline(db_session)
