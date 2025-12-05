"""
HIPAA Compliance Monitoring & Reporting Service
Automated compliance checks and reporting
"""

from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from app.models.security_models import AuditLog, SecurityEvent, ConsentRecord, DataRetentionPolicy
from app.services.enhanced_audit_logger import EnhancedAuditLogger

logger = logging.getLogger(__name__)


class ComplianceMonitoringService:
    """
    HIPAA compliance monitoring and reporting service
    Provides automated compliance checks and reporting
    """
    
    def check_audit_logging_compliance(
        self,
        db: Session,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Check if audit logging meets HIPAA requirements
        
        Requirements:
        - All PHI access must be logged
        - Logs must be immutable
        - Logs must be retained for minimum 6 years
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Count PHI accesses
        total_phi_accesses = db.query(AuditLog).filter(
            AuditLog.phi_accessed == True,
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        # Count PHI accesses without proper logging
        incomplete_logs = db.query(AuditLog).filter(
            AuditLog.phi_accessed == True,
            AuditLog.timestamp >= cutoff_time
        ).filter(
            (AuditLog.user_id.is_(None)) |
            (AuditLog.patient_id_accessed.is_(None)) |
            (AuditLog.ip_address.is_(None))
        ).count()
        
        compliance_score = 1.0 - (incomplete_logs / max(total_phi_accesses, 1))
        
        return {
            "compliant": incomplete_logs == 0,
            "compliance_score": compliance_score,
            "total_phi_accesses": total_phi_accesses,
            "incomplete_logs": incomplete_logs,
            "period_days": days
        }
    
    def check_access_control_compliance(
        self,
        db: Session,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Check access control compliance
        
        Requirements:
        - All access must be authorized
        - Failed access attempts must be logged
        - Privilege escalations must be monitored
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days)
        
        # Count unauthorized access attempts
        unauthorized_attempts = db.query(AuditLog).filter(
            AuditLog.action_result == "denied",
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        # Count successful PHI accesses
        successful_phi_accesses = db.query(AuditLog).filter(
            AuditLog.phi_accessed == True,
            AuditLog.action_result == "success",
            AuditLog.timestamp >= cutoff_time
        ).count()
        
        # Calculate unauthorized access rate
        total_access_attempts = unauthorized_attempts + successful_phi_accesses
        unauthorized_rate = unauthorized_attempts / max(total_access_attempts, 1)
        
        return {
            "compliant": unauthorized_rate < 0.05,  # Less than 5% unauthorized
            "unauthorized_attempts": unauthorized_attempts,
            "successful_phi_accesses": successful_phi_accesses,
            "unauthorized_rate": unauthorized_rate,
            "period_days": days
        }
    
    def check_consent_compliance(
        self,
        db: Session,
        patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check consent management compliance
        
        Requirements:
        - Patient consent must be obtained before data collection
        - Consent must be documented
        - Consent withdrawal must be honored
        """
        query = db.query(ConsentRecord)
        
        if patient_id:
            query = query.filter(ConsentRecord.patient_id == patient_id)
        
        total_consents = query.count()
        active_consents = query.filter(
            ConsentRecord.consent_given == True,
            ConsentRecord.withdrawn == False
        ).count()
        
        withdrawn_consents = query.filter(
            ConsentRecord.withdrawn == True
        ).count()
        
        return {
            "total_consents": total_consents,
            "active_consents": active_consents,
            "withdrawn_consents": withdrawn_consents,
            "compliant": total_consents > 0
        }
    
    def check_data_retention_compliance(
        self,
        db: Session
    ) -> Dict[str, Any]:
        """
        Check data retention policy compliance
        
        Requirements:
        - Data retention policies must be defined
        - Policies must be enforced
        - Legal holds must be respected
        """
        # Count active retention policies
        active_policies = db.query(DataRetentionPolicy).filter(
            DataRetentionPolicy.is_active == True
        ).count()
        
        # Check if policies cover all data types
        required_data_types = ["video", "audio", "metrics", "alerts", "logs"]
        covered_types = db.query(DataRetentionPolicy.data_type).filter(
            DataRetentionPolicy.is_active == True
        ).distinct().all()
        covered_types_list = [t[0] for t in covered_types]
        
        missing_types = [dt for dt in required_data_types if dt not in covered_types_list]
        
        return {
            "compliant": len(missing_types) == 0,
            "active_policies": active_policies,
            "covered_data_types": covered_types_list,
            "missing_data_types": missing_types
        }
    
    def generate_compliance_report(
        self,
        db: Session,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate comprehensive HIPAA compliance report
        
        Returns:
            Dictionary with all compliance metrics
        """
        audit_compliance = self.check_audit_logging_compliance(db, days)
        access_control_compliance = self.check_access_control_compliance(db, days)
        consent_compliance = self.check_consent_compliance(db)
        retention_compliance = self.check_data_retention_compliance(db)
        
        # Calculate overall compliance score
        compliance_scores = [
            audit_compliance.get("compliance_score", 0),
            1.0 if access_control_compliance.get("compliant") else 0.5,
            1.0 if consent_compliance.get("compliant") else 0.5,
            1.0 if retention_compliance.get("compliant") else 0.5,
        ]
        overall_score = sum(compliance_scores) / len(compliance_scores)
        
        return {
            "report_date": datetime.utcnow().isoformat(),
            "period_days": days,
            "overall_compliance_score": overall_score,
            "overall_compliant": overall_score >= 0.9,
            "audit_logging": audit_compliance,
            "access_control": access_control_compliance,
            "consent_management": consent_compliance,
            "data_retention": retention_compliance,
            "recommendations": self._generate_recommendations(
                audit_compliance,
                access_control_compliance,
                consent_compliance,
                retention_compliance
            )
        }
    
    def _generate_recommendations(
        self,
        audit_compliance: Dict[str, Any],
        access_control_compliance: Dict[str, Any],
        consent_compliance: Dict[str, Any],
        retention_compliance: Dict[str, Any]
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if not audit_compliance.get("compliant"):
            recommendations.append(
                "Improve audit logging: Ensure all PHI accesses include user_id, patient_id, and IP address"
            )
        
        if not access_control_compliance.get("compliant"):
            recommendations.append(
                "Review access controls: High rate of unauthorized access attempts detected"
            )
        
        if not consent_compliance.get("compliant"):
            recommendations.append(
                "Ensure all patients have documented consent records"
            )
        
        if not retention_compliance.get("compliant"):
            recommendations.append(
                f"Define data retention policies for: {', '.join(retention_compliance.get('missing_data_types', []))}"
            )
        
        if not recommendations:
            recommendations.append("All compliance checks passed")
        
        return recommendations


# Global singleton instance
_compliance_monitoring: Optional[ComplianceMonitoringService] = None


def get_compliance_monitoring() -> ComplianceMonitoringService:
    """Get or create compliance monitoring service singleton"""
    global _compliance_monitoring
    if _compliance_monitoring is None:
        _compliance_monitoring = ComplianceMonitoringService()
    return _compliance_monitoring
