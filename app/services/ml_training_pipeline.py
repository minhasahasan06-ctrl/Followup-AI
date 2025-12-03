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
            'environmental_risk', 'medical_history', 'current_conditions'
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
                
                if extracted:
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
                            current_conditions=data_types.get('current_conditions', False)
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
