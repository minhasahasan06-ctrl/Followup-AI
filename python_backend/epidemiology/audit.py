"""
HIPAA Audit Logging for Epidemiology Research
==============================================
Comprehensive audit logging for all epidemiology data access and analysis.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import psycopg2

logger = logging.getLogger(__name__)


class AuditAction(str, Enum):
    VIEW_SIGNALS = "view_drug_signals"
    VIEW_SUMMARIES = "view_drug_summaries"
    VIEW_INFECTIOUS = "view_infectious_data"
    VIEW_VACCINE = "view_vaccine_data"
    VIEW_OUTBREAK = "view_outbreak_data"
    VIEW_OCCUPATIONAL = "view_occupational_data"
    VIEW_GENETIC = "view_genetic_data"
    RUN_ANALYSIS = "run_epidemiology_analysis"
    EXPORT_DATA = "export_epidemiology_data"
    RUN_SCAN = "run_drug_scan"
    GENERATE_REPORT = "generate_epi_report"
    ACCESS_ML_FEATURES = "access_ml_features"


class EpidemiologyAuditLogger:
    """HIPAA-compliant audit logger for epidemiology research"""
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
    
    def log(
        self,
        action: AuditAction,
        user_id: Optional[str],
        resource_type: str,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None
    ):
        """Log an audit entry"""
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action.value if isinstance(action, AuditAction) else action,
            'user_id': user_id or 'system',
            'resource_type': resource_type,
            'resource_id': resource_id,
            'details': details or {},
            'success': success,
            'error_message': error_message,
            'audit_type': 'HIPAA_EPIDEMIOLOGY'
        }
        
        logger.info(json.dumps(entry))
        
        try:
            self._persist_audit(entry)
        except Exception as e:
            logger.error(f"Failed to persist audit log: {e}")
    
    def _persist_audit(self, entry: Dict[str, Any]):
        """Persist audit entry to database"""
        if not self.db_url:
            return
        
        try:
            conn = psycopg2.connect(self.db_url)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO audit_logs (
                        action_type, user_id, resource_type, resource_id, 
                        request_details, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT DO NOTHING
                """, (
                    entry['action'],
                    entry['user_id'],
                    entry['resource_type'],
                    entry['resource_id'],
                    json.dumps(entry['details'])
                ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not persist audit log to database: {e}")
    
    def log_signal_access(
        self,
        user_id: str,
        drug_code: Optional[str] = None,
        outcome_code: Optional[str] = None,
        location_id: Optional[str] = None,
        scope: str = "all",
        result_count: int = 0
    ):
        """Log access to drug safety signals"""
        self.log(
            action=AuditAction.VIEW_SIGNALS,
            user_id=user_id,
            resource_type='drug_outcome_signals',
            details={
                'drug_code': drug_code,
                'outcome_code': outcome_code,
                'location_id': location_id,
                'scope': scope,
                'result_count': result_count
            }
        )
    
    def log_analysis_run(
        self,
        user_id: str,
        analysis_type: str,
        parameters: Dict[str, Any],
        result_summary: Optional[Dict[str, Any]] = None
    ):
        """Log ML analysis execution"""
        self.log(
            action=AuditAction.RUN_ANALYSIS,
            user_id=user_id,
            resource_type='ml_analysis',
            details={
                'analysis_type': analysis_type,
                'parameters': parameters,
                'result_summary': result_summary
            }
        )
    
    def log_export(
        self,
        user_id: str,
        export_type: str,
        record_count: int,
        filters: Optional[Dict[str, Any]] = None
    ):
        """Log data export"""
        self.log(
            action=AuditAction.EXPORT_DATA,
            user_id=user_id,
            resource_type='export',
            details={
                'export_type': export_type,
                'record_count': record_count,
                'filters': filters
            }
        )
    
    def log_background_scan(
        self,
        scan_type: str,
        drugs_scanned: int,
        outcomes_scanned: int,
        signals_generated: int,
        alerts_created: int,
        duration_seconds: float
    ):
        """Log background drug scanning job"""
        self.log(
            action=AuditAction.RUN_SCAN,
            user_id='system',
            resource_type='background_job',
            details={
                'scan_type': scan_type,
                'drugs_scanned': drugs_scanned,
                'outcomes_scanned': outcomes_scanned,
                'signals_generated': signals_generated,
                'alerts_created': alerts_created,
                'duration_seconds': duration_seconds
            }
        )
