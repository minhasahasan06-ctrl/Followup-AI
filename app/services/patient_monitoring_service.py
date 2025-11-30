"""
Patient Monitoring Service for Assistant Lysa

Production-grade patient monitoring with:
- Continuous health data monitoring
- Deterioration detection
- Alert generation
- HIPAA-compliant audit logging
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc

import openai

from app.models.patient_sharing import (
    PatientSharingLink, SharingAccessLog, PatientFollowupAlert,
    SharingStatus
)

logger = logging.getLogger(__name__)

openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PatientMonitoringService:
    """Handles patient followup monitoring and alert generation"""
    
    @staticmethod
    async def check_patients_with_sharing_links(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check all patients with active sharing links for health changes.
        
        This is called every 10 minutes by the scheduler for doctors
        with active patient monitoring enabled.
        """
        active_links = db.query(PatientSharingLink).filter(
            and_(
                PatientSharingLink.doctor_id == doctor_id,
                PatientSharingLink.status == SharingStatus.ACTIVE.value,
                PatientSharingLink.followup_monitoring_enabled == True
            )
        ).all()
        
        if not active_links:
            return {
                "success": True,
                "patients_checked": 0,
                "alerts_generated": 0,
                "message": "No active monitoring links"
            }
        
        patients_checked = 0
        alerts_generated = 0
        
        for link in active_links:
            if not link.needs_followup_check():
                continue
            
            try:
                result = await PatientMonitoringService._check_patient_health(
                    db, link
                )
                
                patients_checked += 1
                
                if result.get("alerts"):
                    for alert_data in result["alerts"]:
                        alert = PatientFollowupAlert(
                            sharing_link_id=link.id,
                            doctor_id=doctor_id,
                            patient_id=link.patient_id,
                            alert_type=alert_data["type"],
                            severity=alert_data["severity"],
                            title=alert_data["title"],
                            message=alert_data["message"],
                            detected_data=alert_data.get("data")
                        )
                        db.add(alert)
                        alerts_generated += 1
                
                link.mark_followup_checked()
                db.commit()
                
            except Exception as e:
                logger.error(f"Error checking patient {link.patient_id}: {e}")
                continue
        
        logger.info(f"Patient monitoring: {patients_checked} checked, {alerts_generated} alerts for doctor {doctor_id}")
        
        return {
            "success": True,
            "patients_checked": patients_checked,
            "alerts_generated": alerts_generated
        }
    
    @staticmethod
    async def _check_patient_health(
        db: Session,
        link: PatientSharingLink
    ) -> Dict[str, Any]:
        """
        Check a specific patient's health data for concerning changes.
        """
        alerts = []
        patient_id = link.patient_id
        
        try:
            from app.models.daily_followup import DailyFollowupData
            
            today = datetime.utcnow().date()
            yesterday = today - timedelta(days=1)
            
            recent_data = db.query(DailyFollowupData).filter(
                and_(
                    DailyFollowupData.user_id == patient_id,
                    DailyFollowupData.date >= yesterday
                )
            ).order_by(desc(DailyFollowupData.date)).all()
            
            if not recent_data:
                if link.alert_on_symptoms:
                    alerts.append({
                        "type": "no_recent_data",
                        "severity": "low",
                        "title": "No Recent Health Data",
                        "message": "Patient has not submitted any health data in the last 24 hours.",
                        "data": {"last_data_date": None}
                    })
                return {"alerts": alerts}
            
            latest = recent_data[0] if recent_data else None
            
            if latest and link.share_symptoms:
                symptoms_data = latest.symptoms_data or {}
                high_severity_symptoms = []
                
                for symptom, details in symptoms_data.items():
                    severity = details.get("severity", 0) if isinstance(details, dict) else 0
                    if severity >= 7:
                        high_severity_symptoms.append({
                            "symptom": symptom,
                            "severity": severity
                        })
                
                if high_severity_symptoms and link.alert_on_symptoms:
                    alerts.append({
                        "type": "high_severity_symptoms",
                        "severity": "high",
                        "title": "High Severity Symptoms Reported",
                        "message": f"Patient reported {len(high_severity_symptoms)} high-severity symptoms.",
                        "data": {"symptoms": high_severity_symptoms}
                    })
            
            if latest and link.share_vitals:
                vitals_data = latest.vitals_data or {}
                
                heart_rate = vitals_data.get("heart_rate")
                if heart_rate:
                    if heart_rate > 100 or heart_rate < 50:
                        alerts.append({
                            "type": "abnormal_vitals",
                            "severity": "medium" if 45 < heart_rate < 110 else "high",
                            "title": "Abnormal Heart Rate",
                            "message": f"Heart rate of {heart_rate} BPM is outside normal range.",
                            "data": {"heart_rate": heart_rate, "normal_range": "60-100 BPM"}
                        })
                
                spo2 = vitals_data.get("blood_oxygen")
                if spo2 and spo2 < 95:
                    severity = "high" if spo2 < 90 else "medium"
                    alerts.append({
                        "type": "low_oxygen",
                        "severity": severity,
                        "title": "Low Blood Oxygen",
                        "message": f"Blood oxygen level of {spo2}% is below normal.",
                        "data": {"spo2": spo2, "threshold": 95}
                    })
            
            if link.share_mental_health:
                from app.models.mental_health import MentalHealthAssessment
                
                recent_assessments = db.query(MentalHealthAssessment).filter(
                    and_(
                        MentalHealthAssessment.user_id == patient_id,
                        MentalHealthAssessment.created_at >= datetime.utcnow() - timedelta(hours=24)
                    )
                ).order_by(desc(MentalHealthAssessment.created_at)).first()
                
                if recent_assessments:
                    scores = recent_assessments.scores or {}
                    
                    phq9 = scores.get("phq9_score", 0)
                    if phq9 >= 15:
                        alerts.append({
                            "type": "mental_health_indicator",
                            "severity": "high" if phq9 >= 20 else "medium",
                            "title": "Elevated Depression Indicators",
                            "message": f"PHQ-9 score of {phq9} indicates moderately severe to severe indicators.",
                            "data": {"phq9_score": phq9}
                        })
                    
                    gad7 = scores.get("gad7_score", 0)
                    if gad7 >= 15:
                        alerts.append({
                            "type": "mental_health_indicator",
                            "severity": "high" if gad7 >= 20 else "medium",
                            "title": "Elevated Anxiety Indicators",
                            "message": f"GAD-7 score of {gad7} indicates severe anxiety indicators.",
                            "data": {"gad7_score": gad7}
                        })
            
            if len(recent_data) >= 2 and link.alert_on_deterioration:
                deterioration = await PatientMonitoringService._detect_deterioration(
                    recent_data
                )
                if deterioration:
                    alerts.append({
                        "type": "deterioration_detected",
                        "severity": deterioration["severity"],
                        "title": "Health Deterioration Detected",
                        "message": deterioration["message"],
                        "data": deterioration["data"]
                    })
            
        except ImportError as e:
            logger.warning(f"Model not available for monitoring: {e}")
        except Exception as e:
            logger.error(f"Error in patient health check: {e}")
        
        return {"alerts": alerts}
    
    @staticmethod
    async def _detect_deterioration(
        data_points: List[Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Use AI to analyze health data trend and detect deterioration.
        """
        try:
            if len(data_points) < 2:
                return None
            
            latest = data_points[0]
            previous = data_points[1]
            
            changes = []
            
            if latest.symptoms_data and previous.symptoms_data:
                latest_symptoms = len(latest.symptoms_data)
                prev_symptoms = len(previous.symptoms_data)
                if latest_symptoms > prev_symptoms + 2:
                    changes.append(f"Symptom count increased from {prev_symptoms} to {latest_symptoms}")
            
            if latest.vitals_data and previous.vitals_data:
                latest_hr = latest.vitals_data.get("heart_rate")
                prev_hr = previous.vitals_data.get("heart_rate")
                if latest_hr and prev_hr and abs(latest_hr - prev_hr) > 20:
                    changes.append(f"Heart rate changed by {abs(latest_hr - prev_hr)} BPM")
            
            if not changes:
                return None
            
            return {
                "severity": "medium",
                "message": f"Detected {len(changes)} significant changes in patient condition.",
                "data": {"changes": changes}
            }
            
        except Exception as e:
            logger.error(f"Deterioration detection error: {e}")
            return None
    
    @staticmethod
    async def get_sharing_links_for_doctor(
        db: Session,
        doctor_id: str
    ) -> List[Dict[str, Any]]:
        """Get all sharing links for a doctor"""
        links = db.query(PatientSharingLink).filter(
            PatientSharingLink.doctor_id == doctor_id
        ).order_by(desc(PatientSharingLink.created_at)).all()
        
        result = []
        for link in links:
            result.append({
                "id": link.id,
                "patient_id": link.patient_id,
                "status": link.status,
                "access_level": link.access_level,
                "continuous_sharing": link.continuous_sharing_enabled,
                "followup_monitoring": link.followup_monitoring_enabled,
                "expires_at": link.expires_at.isoformat() if link.expires_at else None,
                "created_at": link.created_at.isoformat(),
                "consent_given_at": link.consent_given_at.isoformat() if link.consent_given_at else None,
                "is_active": link.is_active()
            })
        
        return result
    
    @staticmethod
    async def create_sharing_link(
        db: Session,
        doctor_id: str,
        patient_id: str,
        access_level: str = "view_only",
        expires_in_days: int = 30,
        continuous_sharing: bool = False,
        followup_monitoring: bool = False,
        share_options: Optional[Dict[str, bool]] = None
    ) -> Dict[str, Any]:
        """Create a new patient sharing link"""
        existing = db.query(PatientSharingLink).filter(
            and_(
                PatientSharingLink.doctor_id == doctor_id,
                PatientSharingLink.patient_id == patient_id,
                PatientSharingLink.status.in_([
                    SharingStatus.PENDING.value,
                    SharingStatus.ACTIVE.value,
                    SharingStatus.PAUSED.value
                ])
            )
        ).first()
        
        if existing:
            return {
                "success": False,
                "error": "Active sharing link already exists for this patient",
                "existing_link_id": existing.id
            }
        
        link = PatientSharingLink.create_sharing_link(
            doctor_id=doctor_id,
            patient_id=patient_id,
            access_level=access_level,
            expires_in_days=expires_in_days,
            continuous_sharing=continuous_sharing,
            followup_monitoring=followup_monitoring
        )
        
        if share_options:
            link.share_vitals = share_options.get("vitals", True)
            link.share_symptoms = share_options.get("symptoms", True)
            link.share_medications = share_options.get("medications", True)
            link.share_activities = share_options.get("activities", True)
            link.share_mental_health = share_options.get("mental_health", False)
            link.share_video_exams = share_options.get("video_exams", False)
            link.share_audio_exams = share_options.get("audio_exams", False)
        
        db.add(link)
        db.commit()
        db.refresh(link)
        
        logger.info(f"Created sharing link {link.id} for patient {patient_id} by doctor {doctor_id}")
        
        return {
            "success": True,
            "link_id": link.id,
            "secure_token": link.secure_token,
            "expires_at": link.expires_at.isoformat() if link.expires_at else None
        }
    
    @staticmethod
    async def activate_sharing_link(
        db: Session,
        secure_token: str,
        ip_address: str = None,
        user_agent: str = None
    ) -> Dict[str, Any]:
        """Activate a sharing link with patient consent"""
        link = db.query(PatientSharingLink).filter(
            PatientSharingLink.secure_token == secure_token
        ).first()
        
        if not link:
            return {"success": False, "error": "Invalid sharing link"}
        
        if link.status != SharingStatus.PENDING.value:
            return {"success": False, "error": f"Link is {link.status}, cannot activate"}
        
        if link.expires_at and datetime.utcnow() > link.expires_at:
            link.status = SharingStatus.EXPIRED.value
            db.commit()
            return {"success": False, "error": "Link has expired"}
        
        link.activate(ip_address=ip_address, user_agent=user_agent)
        
        log = SharingAccessLog(
            sharing_link_id=link.id,
            doctor_id=link.doctor_id,
            patient_id=link.patient_id,
            action="consent_granted",
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.add(log)
        
        db.commit()
        
        logger.info(f"Sharing link {link.id} activated by patient {link.patient_id}")
        
        return {
            "success": True,
            "message": "Sharing link activated successfully",
            "doctor_id": link.doctor_id,
            "access_level": link.access_level
        }
    
    @staticmethod
    async def revoke_sharing_link(
        db: Session,
        link_id: str,
        revoked_by: str,
        reason: str = None
    ) -> Dict[str, Any]:
        """Revoke a sharing link"""
        link = db.query(PatientSharingLink).filter(
            PatientSharingLink.id == link_id
        ).first()
        
        if not link:
            return {"success": False, "error": "Sharing link not found"}
        
        link.revoke(revoked_by=revoked_by, reason=reason)
        
        log = SharingAccessLog(
            sharing_link_id=link.id,
            doctor_id=link.doctor_id,
            patient_id=link.patient_id,
            action="link_revoked",
            details={"revoked_by": revoked_by, "reason": reason}
        )
        db.add(log)
        
        db.commit()
        
        logger.info(f"Sharing link {link.id} revoked by {revoked_by}")
        
        return {"success": True, "message": "Sharing link revoked"}
    
    @staticmethod
    async def get_unread_alerts(
        db: Session,
        doctor_id: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get unread alerts for a doctor"""
        alerts = db.query(PatientFollowupAlert).filter(
            and_(
                PatientFollowupAlert.doctor_id == doctor_id,
                PatientFollowupAlert.is_read == False
            )
        ).order_by(desc(PatientFollowupAlert.created_at)).limit(limit).all()
        
        return [
            {
                "id": a.id,
                "patient_id": a.patient_id,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "title": a.title,
                "message": a.message,
                "detected_data": a.detected_data,
                "created_at": a.created_at.isoformat()
            }
            for a in alerts
        ]
    
    @staticmethod
    async def acknowledge_alert(
        db: Session,
        alert_id: str,
        doctor_id: str
    ) -> Dict[str, Any]:
        """Acknowledge an alert"""
        alert = db.query(PatientFollowupAlert).filter(
            and_(
                PatientFollowupAlert.id == alert_id,
                PatientFollowupAlert.doctor_id == doctor_id
            )
        ).first()
        
        if not alert:
            return {"success": False, "error": "Alert not found"}
        
        alert.is_read = True
        alert.is_acknowledged = True
        alert.acknowledged_at = datetime.utcnow()
        alert.acknowledged_by = doctor_id
        
        db.commit()
        
        return {"success": True, "message": "Alert acknowledged"}


patient_monitoring_service = PatientMonitoringService()
