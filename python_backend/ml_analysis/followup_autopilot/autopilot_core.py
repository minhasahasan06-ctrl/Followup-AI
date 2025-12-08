"""
Autopilot Core Service

The brain of the Followup Autopilot system. Orchestrates:
1. Risk prediction using all 4 ML models
2. Patient state updates
3. Next follow-up scheduling

All outputs are for wellness monitoring only, NOT medical diagnosis.
"""

import os
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RiskState(str, Enum):
    STABLE = "Stable"
    AT_RISK = "AtRisk"
    WORSENING = "Worsening"
    CRITICAL = "Critical"


RISK_THRESHOLDS = {
    "stable_max": 25,
    "at_risk_max": 50,
    "worsening_max": 75,
}

FOLLOWUP_INTERVALS = {
    RiskState.STABLE: timedelta(days=3),
    RiskState.AT_RISK: timedelta(days=1),
    RiskState.WORSENING: timedelta(hours=12),
    RiskState.CRITICAL: timedelta(hours=6),
}


class AutopilotCore:
    """
    Autopilot Core - Main orchestrator for patient state management.
    
    Responsibilities:
    1. Fetch and prepare patient feature data
    2. Run all 4 ML models
    3. Compute composite risk score and state
    4. Schedule next follow-up
    5. Update patient state in database
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
        
        from .ml_models import get_model_manager
        self.models = get_model_manager()
        
        from .feature_builder import FeatureBuilder
        self.feature_builder = FeatureBuilder(db_session)
    
    def update_patient_state(self, patient_id: str) -> Dict[str, Any]:
        """
        Update patient state using all ML models.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Updated patient state dictionary
        """
        today = date.today()
        
        feature_sequence = self.feature_builder.get_feature_sequence(
            patient_id, today, sequence_length=30
        )
        
        if not feature_sequence or all(sum(day) == 0 for day in feature_sequence):
            return self._create_default_state(patient_id)
        
        risk_result = self.models.get_risk_model().predict(feature_sequence)
        
        today_features = self._sequence_to_features(feature_sequence[-1])
        
        adherence_features = self._build_adherence_features(patient_id, today)
        p_non_adherence = self.models.get_adherence_model().predict(adherence_features)
        
        anomaly_score = self.models.get_anomaly_detector().score(today_features)
        
        risk_score = risk_result["risk_score"]
        if anomaly_score > 0.6:
            risk_score = min(100, risk_score + anomaly_score * 15)
        
        risk_state = self._compute_risk_state(risk_score)
        
        patient_features_for_engagement = {
            "risk_state_numeric": self._state_to_numeric(risk_state),
            "engagement_rate_14d": today_features.get("engagement_rate_14d", 0.5),
            "avg_pain": today_features.get("avg_pain", 3),
            "mh_score": today_features.get("mh_score", 0.3),
        }
        
        existing_state = self._get_existing_state(patient_id)
        preferred_hour = existing_state.get("preferred_contact_hour") if existing_state else None
        
        next_followup = self._compute_next_followup(
            risk_state, patient_features_for_engagement, preferred_hour
        )
        
        state = {
            "patient_id": patient_id,
            "risk_score": round(risk_score, 2),
            "risk_state": risk_state.value,
            "risk_components": {
                "clinical": risk_result["risk_components"]["clinical"],
                "mental_health": risk_result["risk_components"]["mental_health"],
                "adherence": risk_result["risk_components"]["adherence"],
                "anomaly": round(anomaly_score * 100, 1),
            },
            "last_updated": datetime.now(timezone.utc),
            "next_followup_at": next_followup,
            "preferred_contact_hour": preferred_hour,
            "model_version": risk_result.get("model_version", "1.0.0"),
            "inference_confidence": risk_result.get("confidence", 0.5),
            "p_non_adherence_7d": round(p_non_adherence, 4),
            "anomaly_score": round(anomaly_score, 4),
        }
        
        self._save_state(patient_id, state)
        self._audit_log(patient_id, "state_updated", state)
        
        return state
    
    def get_patient_state(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get current patient state without recomputing"""
        return self._get_existing_state(patient_id)
    
    def compute_next_followup_at(
        self,
        risk_state: RiskState,
        patient_features: Dict[str, float],
        preferred_hour: Optional[int] = None
    ) -> datetime:
        """
        Compute next follow-up timestamp using engagement model.
        
        Args:
            risk_state: Current risk state
            patient_features: Current patient features
            preferred_hour: Patient's preferred contact hour
            
        Returns:
            Next follow-up datetime
        """
        return self._compute_next_followup(risk_state, patient_features, preferred_hour)
    
    def _compute_risk_state(self, risk_score: float) -> RiskState:
        """Map risk score to risk state"""
        if risk_score <= RISK_THRESHOLDS["stable_max"]:
            return RiskState.STABLE
        elif risk_score <= RISK_THRESHOLDS["at_risk_max"]:
            return RiskState.AT_RISK
        elif risk_score <= RISK_THRESHOLDS["worsening_max"]:
            return RiskState.WORSENING
        else:
            return RiskState.CRITICAL
    
    def _compute_next_followup(
        self,
        risk_state: RiskState,
        patient_features: Dict[str, float],
        preferred_hour: Optional[int]
    ) -> datetime:
        """Compute next follow-up using engagement model"""
        base_interval = FOLLOWUP_INTERVALS[risk_state]
        
        best_hour = self.models.get_engagement_model().predict_best_hour(
            patient_features, preferred_hour
        )
        
        now = datetime.now(timezone.utc)
        next_date = now + base_interval
        
        next_followup = next_date.replace(
            hour=best_hour, minute=0, second=0, microsecond=0
        )
        
        if next_followup <= now:
            next_followup += timedelta(days=1)
        
        return next_followup
    
    def _sequence_to_features(self, vector: list) -> Dict[str, float]:
        """Convert feature vector back to named features"""
        from .feature_builder import FeatureBuilder
        
        features = {}
        for i, col in enumerate(FeatureBuilder.FEATURE_COLUMNS):
            if i < len(vector):
                features[col] = vector[i]
        return features
    
    def _build_adherence_features(self, patient_id: str, target_date: date) -> Dict[str, float]:
        """Build features for adherence model"""
        if self.db:
            from app.models.followup_autopilot_models import AutopilotDailyFeatures
            from sqlalchemy import func
            
            start_date = target_date - timedelta(days=29)
            rows = self.db.query(AutopilotDailyFeatures).filter(
                AutopilotDailyFeatures.patient_id == patient_id,
                AutopilotDailyFeatures.date >= start_date,
                AutopilotDailyFeatures.date <= target_date
            ).all()
            
            adherence_values = [r.med_adherence_7d or 1.0 for r in rows]
            
            if adherence_values:
                import numpy as np
                return {
                    "adherence_mean_30d": float(np.mean(adherence_values)),
                    "adherence_min_30d": float(np.min(adherence_values)),
                    "adherence_max_30d": float(np.max(adherence_values)),
                    "adherence_trend": float(np.polyfit(range(len(adherence_values)), 
                                                        adherence_values, 1)[0]) if len(adherence_values) > 1 else 0,
                    "med_changes": 0,
                    "engagement_rate": rows[-1].engagement_rate_14d if rows else 0.5,
                    "past_adherence_triggers": 0,
                    "avg_pain": rows[-1].avg_pain if rows else 3,
                    "mh_score": rows[-1].mh_score if rows else 0.3,
                }
        
        return {
            "adherence_mean_30d": 0.9,
            "adherence_min_30d": 0.7,
            "adherence_max_30d": 1.0,
            "adherence_trend": 0,
            "med_changes": 0,
            "engagement_rate": 0.5,
            "past_adherence_triggers": 0,
            "avg_pain": 3,
            "mh_score": 0.3,
        }
    
    def _state_to_numeric(self, state: RiskState) -> int:
        """Convert risk state to numeric value"""
        mapping = {
            RiskState.STABLE: 1,
            RiskState.AT_RISK: 2,
            RiskState.WORSENING: 3,
            RiskState.CRITICAL: 4,
        }
        return mapping.get(state, 1)
    
    def _get_existing_state(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Fetch existing patient state"""
        if self.db:
            from app.models.followup_autopilot_models import AutopilotPatientState
            
            row = self.db.query(AutopilotPatientState).filter(
                AutopilotPatientState.patient_id == patient_id
            ).first()
            
            if row:
                return {
                    "patient_id": row.patient_id,
                    "risk_score": row.risk_score,
                    "risk_state": row.risk_state,
                    "risk_components": row.risk_components,
                    "last_updated": row.last_updated,
                    "last_checkin_at": row.last_checkin_at,
                    "next_followup_at": row.next_followup_at,
                    "preferred_contact_hour": row.preferred_contact_hour,
                }
        return None
    
    def _save_state(self, patient_id: str, state: Dict[str, Any]) -> bool:
        """Save patient state to database"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotPatientState
                
                existing = self.db.query(AutopilotPatientState).filter(
                    AutopilotPatientState.patient_id == patient_id
                ).first()
                
                if existing:
                    existing.risk_score = state["risk_score"]
                    existing.risk_state = state["risk_state"]
                    existing.risk_components = state["risk_components"]
                    existing.next_followup_at = state["next_followup_at"]
                    existing.model_version = state.get("model_version", "1.0.0")
                    existing.inference_confidence = state.get("inference_confidence", 0.5)
                else:
                    new_state = AutopilotPatientState(
                        patient_id=patient_id,
                        risk_score=state["risk_score"],
                        risk_state=state["risk_state"],
                        risk_components=state["risk_components"],
                        next_followup_at=state["next_followup_at"],
                        preferred_contact_hour=state.get("preferred_contact_hour"),
                        model_version=state.get("model_version", "1.0.0"),
                        inference_confidence=state.get("inference_confidence", 0.5),
                    )
                    self.db.add(new_state)
                
                self.db.commit()
                return True
            else:
                return self._save_state_raw(patient_id, state)
                
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            if self.db:
                self.db.rollback()
            return False
    
    def _save_state_raw(self, patient_id: str, state: Dict[str, Any]) -> bool:
        """Direct database save when ORM not available"""
        import psycopg2
        from psycopg2.extras import Json
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            return False
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autopilot_patient_states 
                    (patient_id, risk_score, risk_state, risk_components, 
                     next_followup_at, model_version, inference_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (patient_id) DO UPDATE SET
                    risk_score = EXCLUDED.risk_score,
                    risk_state = EXCLUDED.risk_state,
                    risk_components = EXCLUDED.risk_components,
                    next_followup_at = EXCLUDED.next_followup_at,
                    model_version = EXCLUDED.model_version,
                    inference_confidence = EXCLUDED.inference_confidence,
                    last_updated = NOW()
                """, (
                    patient_id, state["risk_score"], state["risk_state"],
                    Json(state["risk_components"]), state["next_followup_at"],
                    state.get("model_version", "1.0.0"),
                    state.get("inference_confidence", 0.5)
                ))
            conn.commit()
        return True
    
    def _create_default_state(self, patient_id: str) -> Dict[str, Any]:
        """Create default low-risk state for patients with insufficient data"""
        now = datetime.now(timezone.utc)
        
        state = {
            "patient_id": patient_id,
            "risk_score": 15.0,
            "risk_state": RiskState.STABLE.value,
            "risk_components": {
                "clinical": 10.0,
                "mental_health": 10.0,
                "adherence": 10.0,
                "anomaly": 0.0,
            },
            "last_updated": now,
            "next_followup_at": now + timedelta(days=3),
            "preferred_contact_hour": None,
            "model_version": "default",
            "inference_confidence": 0.3,
            "p_non_adherence_7d": 0.1,
            "anomaly_score": 0.0,
        }
        
        self._save_state(patient_id, state)
        return state
    
    def _audit_log(self, patient_id: str, action: str, details: Dict[str, Any]):
        """Log action for HIPAA audit trail"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotAuditLog
                
                safe_details = {
                    k: str(v) if isinstance(v, datetime) else v 
                    for k, v in details.items()
                    if k not in ("patient_id",)
                }
                
                log = AutopilotAuditLog(
                    patient_id=patient_id,
                    action=action,
                    entity_type="patient_state",
                    entity_id=patient_id,
                    new_values=safe_details
                )
                self.db.add(log)
                self.db.commit()
        except Exception as e:
            self.logger.warning(f"Audit log failed: {e}")
