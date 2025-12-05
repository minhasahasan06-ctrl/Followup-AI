"""
Security Monitoring & Anomaly Detection Service - HIPAA Compliance
Detects suspicious activities, anomalies, and potential security threats
"""

from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from collections import defaultdict

from app.models.security_models import SecurityEvent, AuditLog
from app.services.enhanced_audit_logger import EnhancedAuditLogger

logger = logging.getLogger(__name__)


class SecurityMonitoringService:
    """
    Security monitoring and anomaly detection service
    Implements HIPAA-compliant threat detection
    """
    
    def __init__(self):
        self.thresholds = {
            "failed_login_attempts": 5,  # Block after 5 failed attempts
            "phi_access_rate": 100,  # Alert if >100 PHI accesses per hour
            "unusual_access_time": True,  # Alert on access outside business hours
            "multiple_patients_per_hour": 10,  # Alert if accessing >10 patients/hour
            "geographic_anomaly": True,  # Alert on rapid geographic changes
            "privilege_escalation": True,  # Alert on privilege changes
        }
    
    def detect_failed_login_anomaly(
        self,
        db: Session,
        user_id: Optional[str],
        ip_address: str,
        time_window_minutes: int = 15
    ) -> bool:
        """
        Detect excessive failed login attempts (brute force)
        
        Returns:
            True if anomaly detected
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=time_window_minutes)
        
        failed_logins = db.query(AuditLog).filter(
            AuditLog.action_type == "login",
            AuditLog.action_result == "failure",
            AuditLog.ip_address == ip_address,
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        if failed_logins >= self.thresholds["failed_login_attempts"]:
            EnhancedAuditLogger.log_security_event(
                db=db,
                event_type="brute_force_attempt",
                severity="high",
                description=f"Excessive failed login attempts from IP {ip_address}",
                user_id=user_id,
                ip_address=ip_address,
                event_data={"failed_attempts": failed_logins, "time_window": time_window_minutes},
                detection_method="automated_rule",
                confidence_score=0.9,
                action_taken="ip_blocked"
            )
            return True
        
        return False
    
    def detect_phi_access_anomaly(
        self,
        db: Session,
        user_id: str,
        time_window_hours: int = 1
    ) -> bool:
        """
        Detect excessive PHI access (potential data exfiltration)
        
        Returns:
            True if anomaly detected
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        phi_accesses = db.query(AuditLog).filter(
            AuditLog.user_id == user_id,
            AuditLog.phi_accessed == True,
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        if phi_accesses > self.thresholds["phi_access_rate"]:
            EnhancedAuditLogger.log_security_event(
                db=db,
                event_type="excessive_phi_access",
                severity="medium",
                description=f"User {user_id} accessed PHI {phi_accesses} times in {time_window_hours} hour(s)",
                user_id=user_id,
                event_data={"access_count": phi_accesses, "time_window": time_window_hours},
                detection_method="automated_rule",
                confidence_score=0.7,
                action_taken="alert_sent"
            )
            return True
        
        return False
    
    def detect_unusual_access_pattern(
        self,
        db: Session,
        user_id: str,
        patient_id: str,
        access_time: datetime
    ) -> bool:
        """
        Detect unusual access patterns (e.g., accessing patient data outside business hours)
        
        Returns:
            True if anomaly detected
        """
        # Check if access is outside business hours (9 AM - 5 PM)
        hour = access_time.hour
        is_business_hours = 9 <= hour < 17
        
        if not is_business_hours and self.thresholds["unusual_access_time"]:
            # Check if this is a pattern (multiple unusual accesses)
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            unusual_accesses = db.query(AuditLog).filter(
                AuditLog.user_id == user_id,
                AuditLog.patient_id_accessed == patient_id,
                AuditLog.phi_accessed == True,
                AuditLog.timestamp >= cutoff_time
            ).all()
            
            # Count accesses outside business hours
            non_business_accesses = sum(
                1 for log in unusual_accesses
                if not (9 <= log.timestamp.hour < 17)
            )
            
            if non_business_accesses >= 3:  # Pattern detected
                EnhancedAuditLogger.log_security_event(
                    db=db,
                    event_type="unusual_access_pattern",
                    severity="low",
                    description=f"User {user_id} accessing patient {patient_id} outside business hours",
                    user_id=user_id,
                    event_data={
                        "patient_id": patient_id,
                        "access_time": access_time.isoformat(),
                        "non_business_accesses": non_business_accesses
                    },
                    detection_method="automated_rule",
                    confidence_score=0.6,
                    action_taken="logged"
                )
                return True
        
        return False
    
    def detect_multiple_patient_access(
        self,
        db: Session,
        user_id: str,
        time_window_hours: int = 1
    ) -> bool:
        """
        Detect access to multiple patients in short time (potential unauthorized access)
        
        Returns:
            True if anomaly detected
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
        
        # Get distinct patients accessed
        distinct_patients = db.query(AuditLog.patient_id_accessed).filter(
            AuditLog.user_id == user_id,
            AuditLog.phi_accessed == True,
            AuditLog.patient_id_accessed.isnot(None),
            AuditLog.timestamp >= cutoff_time
        ).distinct().count()
        
        if distinct_patients > self.thresholds["multiple_patients_per_hour"]:
            EnhancedAuditLogger.log_security_event(
                db=db,
                event_type="multiple_patient_access",
                severity="medium",
                description=f"User {user_id} accessed {distinct_patients} different patients in {time_window_hours} hour(s)",
                user_id=user_id,
                event_data={
                    "patient_count": distinct_patients,
                    "time_window": time_window_hours
                },
                detection_method="automated_rule",
                confidence_score=0.75,
                action_taken="alert_sent"
            )
            return True
        
        return False
    
    def detect_geographic_anomaly(
        self,
        db: Session,
        user_id: str,
        current_ip: str,
        current_location: Optional[str] = None
    ) -> bool:
        """
        Detect rapid geographic location changes (potential account compromise)
        
        Returns:
            True if anomaly detected
        """
        if not self.thresholds["geographic_anomaly"]:
            return False
        
        # Get recent access logs
        cutoff_time = datetime.utcnow() - timedelta(hours=1)
        
        recent_logs = db.query(AuditLog).filter(
            AuditLog.user_id == user_id,
            AuditLog.timestamp >= cutoff_time
        ).order_by(AuditLog.timestamp.desc()).limit(10).all()
        
        if len(recent_logs) < 2:
            return False
        
        # Check for IP address changes
        unique_ips = set(log.ip_address for log in recent_logs if log.ip_address)
        
        if len(unique_ips) > 3:  # Multiple IPs in short time
            EnhancedAuditLogger.log_security_event(
                db=db,
                event_type="geographic_anomaly",
                severity="high",
                description=f"User {user_id} accessed from {len(unique_ips)} different IPs in 1 hour",
                user_id=user_id,
                ip_address=current_ip,
                event_data={
                    "unique_ips": list(unique_ips),
                    "current_ip": current_ip,
                    "current_location": current_location
                },
                detection_method="automated_rule",
                confidence_score=0.8,
                action_taken="alert_sent"
            )
            return True
        
        return False
    
    def monitor_user_activity(
        self,
        db: Session,
        user_id: str,
        action_type: str,
        resource_type: str,
        phi_accessed: bool = False,
        patient_id: Optional[str] = None,
        ip_address: Optional[str] = None
    ):
        """
        Monitor user activity and detect anomalies
        
        This should be called after each significant user action
        """
        # Run all anomaly detection checks
        if phi_accessed and patient_id:
            self.detect_phi_access_anomaly(db, user_id)
            self.detect_unusual_access_pattern(db, user_id, patient_id, datetime.utcnow())
            self.detect_multiple_patient_access(db, user_id)
        
        if ip_address:
            self.detect_geographic_anomaly(db, user_id, ip_address)
    
    def get_security_dashboard_data(
        self,
        db: Session,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get security dashboard data for compliance reporting
        
        Returns:
            Dictionary with security metrics
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Count security events by severity
        critical_events = db.query(SecurityEvent).filter(
            SecurityEvent.severity == "critical",
            SecurityEvent.occurred_at >= cutoff_time
        ).count()
        
        high_events = db.query(SecurityEvent).filter(
            SecurityEvent.severity == "high",
            SecurityEvent.occurred_at >= cutoff_time
        ).count()
        
        # Count PHI accesses
        phi_accesses = db.query(AuditLog).filter(
            AuditLog.phi_accessed == True,
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        # Count failed logins
        failed_logins = db.query(AuditLog).filter(
            AuditLog.action_type == "login",
            AuditLog.action_result == "failure",
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        return {
            "period_days": days,
            "critical_security_events": critical_events,
            "high_security_events": high_events,
            "total_phi_accesses": phi_accesses,
            "failed_login_attempts": failed_logins,
            "unique_users_accessed_phi": db.query(AuditLog.user_id).filter(
                AuditLog.phi_accessed == True,
                AuditLog.timestamp >= cutoff_time
            ).distinct().count()
        }


# Global singleton instance
_security_monitoring: Optional[SecurityMonitoringService] = None


def get_security_monitoring() -> SecurityMonitoringService:
    """Get or create security monitoring service singleton"""
    global _security_monitoring
    if _security_monitoring is None:
        _security_monitoring = SecurityMonitoringService()
    return _security_monitoring
