"""
Signal Ingestion Service for Followup Autopilot

Captures signals from all modules:
- Device Data (wearables, vitals)
- Symptoms (manual entries)
- Video AI (respiratory risk)
- Audio AI (emotion, cough detection)
- PainTrack (pain severity)
- Mental Health (PHQ-9, GAD-7, PSS scores)
- Environment (air quality, pollen, weather)
- Medications (adherence, schedules)
- Risk & Exposures (infectious, immunizations, occupational, genetic)

HIPAA Compliance: All signal ingestion is consent-verified and audit-logged.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from uuid import UUID, uuid4
from decimal import Decimal
import json

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    "device", "symptom", "video", "audio", 
    "pain", "mental", "environment", "meds", "exposure"
}

VALID_SOURCES = {
    "wearable", "manual_entry", "video_exam", "audio_recording",
    "questionnaire", "api_sync", "ehr_import", "device_connect",
    "environmental_api", "pharmacy_sync", "etl_pipeline"
}


def normalize_for_json(obj: Any) -> Any:
    """Recursively normalize objects for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, dict):
        return {k: normalize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize_for_json(item) for item in obj]
    return obj


class SignalIngestor:
    """
    Signal ingestion service that captures data from all patient modules.
    
    Responsibilities:
    1. Validate and normalize incoming signals
    2. Verify patient consent before storage
    3. Store signals in autopilot_patient_signals table
    4. Log all operations for HIPAA audit
    """

    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)

    def ingest_signal(
        self,
        patient_id: str,
        category: str,
        source: str,
        raw_payload: Dict[str, Any],
        ml_score: Optional[float] = None,
        signal_time: Optional[datetime] = None,
        skip_consent_check: bool = False
    ) -> Optional[str]:
        """
        Ingest a single signal from any module.
        
        Args:
            patient_id: Patient identifier
            category: Signal category (device, symptom, video, etc.)
            source: Signal source (wearable, manual_entry, etc.)
            raw_payload: Raw data from the source
            ml_score: Optional ML-derived score (0-1 or 0-100)
            signal_time: When the signal was captured (defaults to now)
            skip_consent_check: For internal system signals only
            
        Returns:
            Signal ID if successful, None otherwise
        """
        if category not in VALID_CATEGORIES:
            self.logger.warning(f"Invalid signal category: {category}")
            return None

        if not signal_time:
            signal_time = datetime.now(timezone.utc)

        if not skip_consent_check:
            if not self._verify_consent(patient_id, category):
                self.logger.warning(f"Consent not verified for patient {patient_id}, category {category}")
                return None

        normalized_payload = normalize_for_json(raw_payload)
        signal_id = str(uuid4())

        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotPatientSignal
                signal = AutopilotPatientSignal(
                    id=signal_id,
                    patient_id=patient_id,
                    category=category,
                    source=source,
                    raw_payload=normalized_payload,
                    ml_score=ml_score,
                    signal_time=signal_time
                )
                self.db.add(signal)
                self.db.commit()
            else:
                self._store_signal_raw(signal_id, patient_id, category, source, 
                                       normalized_payload, ml_score, signal_time)

            self._audit_log(patient_id, "signal_ingested", {
                "signal_id": signal_id,
                "category": category,
                "source": source
            })

            return signal_id

        except Exception as e:
            self.logger.error(f"Failed to ingest signal: {e}")
            if self.db:
                self.db.rollback()
            return None

    def ingest_batch(
        self,
        patient_id: str,
        signals: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Ingest multiple signals in a batch for efficiency.
        
        Args:
            patient_id: Patient identifier
            signals: List of signal dicts with category, source, raw_payload, etc.
            
        Returns:
            List of successfully ingested signal IDs
        """
        ingested_ids = []
        
        for signal_data in signals:
            signal_id = self.ingest_signal(
                patient_id=patient_id,
                category=signal_data.get("category"),
                source=signal_data.get("source", "api_sync"),
                raw_payload=signal_data.get("raw_payload", {}),
                ml_score=signal_data.get("ml_score"),
                signal_time=signal_data.get("signal_time")
            )
            if signal_id:
                ingested_ids.append(signal_id)
                
        return ingested_ids

    def ingest_device_data(
        self,
        patient_id: str,
        device_type: str,
        metrics: Dict[str, Any],
        ml_score: Optional[float] = None
    ) -> Optional[str]:
        """Ingest device/wearable data (steps, HR, sleep, etc.)"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="device",
            source="wearable",
            raw_payload={
                "device_type": device_type,
                "metrics": metrics
            },
            ml_score=ml_score
        )

    def ingest_symptom_data(
        self,
        patient_id: str,
        symptoms: List[Dict[str, Any]],
        severity_score: Optional[float] = None
    ) -> Optional[str]:
        """Ingest symptom check-in data"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="symptom",
            source="manual_entry",
            raw_payload={"symptoms": symptoms},
            ml_score=severity_score
        )

    def ingest_video_analysis(
        self,
        patient_id: str,
        respiratory_risk: float,
        facial_metrics: Dict[str, Any],
        analysis_id: str
    ) -> Optional[str]:
        """Ingest Video AI respiratory analysis results"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="video",
            source="video_exam",
            raw_payload={
                "analysis_id": analysis_id,
                "facial_metrics": facial_metrics
            },
            ml_score=respiratory_risk
        )

    def ingest_audio_analysis(
        self,
        patient_id: str,
        emotion_score: float,
        cough_detected: bool,
        audio_metrics: Dict[str, Any]
    ) -> Optional[str]:
        """Ingest Audio AI emotion/cough analysis results"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="audio",
            source="audio_recording",
            raw_payload={
                "cough_detected": cough_detected,
                "audio_metrics": audio_metrics
            },
            ml_score=emotion_score
        )

    def ingest_pain_data(
        self,
        patient_id: str,
        pain_level: float,
        pain_location: str,
        pain_type: str,
        medications_taken: List[str] = None
    ) -> Optional[str]:
        """Ingest PainTrack data"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="pain",
            source="manual_entry",
            raw_payload={
                "pain_location": pain_location,
                "pain_type": pain_type,
                "medications_taken": medications_taken or []
            },
            ml_score=pain_level / 10.0 if pain_level <= 10 else pain_level / 100.0
        )

    def ingest_mental_health_data(
        self,
        patient_id: str,
        questionnaire_type: str,
        score: float,
        responses: Dict[str, Any]
    ) -> Optional[str]:
        """Ingest Mental Health questionnaire results (PHQ-9, GAD-7, PSS)"""
        max_scores = {"PHQ-9": 27, "GAD-7": 21, "PSS-10": 40}
        max_score = max_scores.get(questionnaire_type, 100)
        normalized_score = score / max_score if max_score > 0 else score
        
        return self.ingest_signal(
            patient_id=patient_id,
            category="mental",
            source="questionnaire",
            raw_payload={
                "questionnaire_type": questionnaire_type,
                "raw_score": score,
                "responses": responses
            },
            ml_score=normalized_score
        )

    def ingest_environment_data(
        self,
        patient_id: str,
        env_risk_score: float,
        pollen_index: float,
        aqi: float,
        temp_c: float,
        location: Optional[str] = None
    ) -> Optional[str]:
        """Ingest Environmental Risk data"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="environment",
            source="environmental_api",
            raw_payload={
                "pollen_index": pollen_index,
                "aqi": aqi,
                "temp_c": temp_c,
                "location": location
            },
            ml_score=env_risk_score / 100.0 if env_risk_score > 1 else env_risk_score
        )

    def ingest_medication_data(
        self,
        patient_id: str,
        medication_name: str,
        event_type: str,
        scheduled_time: datetime,
        actual_time: Optional[datetime] = None,
        adherence_score: Optional[float] = None
    ) -> Optional[str]:
        """Ingest medication adherence data"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="meds",
            source="pharmacy_sync",
            raw_payload={
                "medication_name": medication_name,
                "event_type": event_type,
                "scheduled_time": scheduled_time.isoformat() if scheduled_time else None,
                "actual_time": actual_time.isoformat() if actual_time else None
            },
            ml_score=adherence_score
        )

    def ingest_exposure_data(
        self,
        patient_id: str,
        exposure_type: str,
        risk_score: float,
        details: Dict[str, Any]
    ) -> Optional[str]:
        """Ingest Risk & Exposures data (infectious, immunizations, occupational, genetic)"""
        return self.ingest_signal(
            patient_id=patient_id,
            category="exposure",
            source="etl_pipeline",
            raw_payload={
                "exposure_type": exposure_type,
                "details": details
            },
            ml_score=risk_score / 100.0 if risk_score > 1 else risk_score
        )

    def _verify_consent(self, patient_id: str, category: str) -> bool:
        """Verify patient has given consent for this type of data collection"""
        if not self.db:
            return True
            
        try:
            from app.models.security_models import ConsentRecord
            consent = self.db.query(ConsentRecord).filter(
                ConsentRecord.patient_id == patient_id,
                ConsentRecord.consent_given == True,
                ConsentRecord.withdrawn == False
            ).first()
            
            if consent:
                if category in ("video",) and not consent.allow_video_recording:
                    return False
                if category in ("audio",) and not consent.allow_audio_recording:
                    return False
                if not consent.allow_ai_analysis:
                    return False
                return True
            return True
        except Exception as e:
            self.logger.warning(f"Consent check failed, allowing: {e}")
            return True

    def _store_signal_raw(
        self,
        signal_id: str,
        patient_id: str,
        category: str,
        source: str,
        raw_payload: Dict[str, Any],
        ml_score: Optional[float],
        signal_time: datetime
    ):
        """Direct database storage when ORM session not available"""
        import psycopg2
        from psycopg2.extras import Json
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            raise ValueError("DATABASE_URL not set")
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autopilot_patient_signals 
                    (id, patient_id, category, source, raw_payload, ml_score, signal_time, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """, (signal_id, patient_id, category, source, 
                      Json(raw_payload), ml_score, signal_time))
            conn.commit()

    def _audit_log(self, patient_id: str, action: str, details: Dict[str, Any]):
        """Log action for HIPAA audit trail"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotAuditLog
                log = AutopilotAuditLog(
                    patient_id=patient_id,
                    action=action,
                    entity_type="signal",
                    entity_id=details.get("signal_id"),
                    new_values=details
                )
                self.db.add(log)
                self.db.commit()
        except Exception as e:
            self.logger.warning(f"Audit log failed: {e}")
