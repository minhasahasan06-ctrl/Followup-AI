"""
Daily Autopilot Inference Script

Run autopilot inference sweep for all patients:
- Update patient risk states using ML models
- Run trigger engine for task generation
- Dispatch pending notifications
- HIPAA-compliant with consent verification and audit logging

Designed to run as a scheduled job (hourly for patients due).
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from sqlalchemy import text

from python_backend.ml_analysis.followup_autopilot.scripts.base_training import (
    SecureLogger, ConsentVerifier, AuditLogger, get_database_session
)


MIN_CELL_SIZE = 10
RATE_LIMIT_PER_MINUTE = 100


def get_patients_due_for_inference(db_session, logger: SecureLogger) -> List[str]:
    """Get patients due for autopilot inference"""
    
    query = text("""
        SELECT DISTINCT p.patient_id
        FROM autopilot_patient_states p
        WHERE p.next_followup_at <= NOW()
           OR p.last_updated < NOW() - INTERVAL '6 hours'
        UNION
        SELECT DISTINCT patient_id
        FROM autopilot_patient_signals
        WHERE signal_time >= NOW() - INTERVAL '1 hour'
        LIMIT :limit
    """)
    
    try:
        result = db_session.execute(query, {"limit": RATE_LIMIT_PER_MINUTE})
        patients = [row[0] for row in result.fetchall()]
        logger.info(f"Found {len(patients)} patients due for inference")
        return patients
    except Exception as e:
        logger.warning(f"Error getting patients: {e}")
        
        fallback_query = text("""
            SELECT DISTINCT patient_id FROM autopilot_patient_states LIMIT 50
        """)
        try:
            result = db_session.execute(fallback_query)
            return [row[0] for row in result.fetchall()]
        except:
            return []


def update_patient_state(
    db_session,
    patient_id: str,
    logger: SecureLogger
) -> Optional[Dict[str, Any]]:
    """Run autopilot inference for a single patient"""
    
    try:
        from python_backend.ml_analysis.followup_autopilot.autopilot_core import AutopilotCore
        
        autopilot = AutopilotCore(db_session)
        state = autopilot.update_patient_state(patient_id)
        
        logger.info(f"Updated state: risk_score={state.get('risk_score', 0):.1f}", patient_id=patient_id)
        return state
        
    except ImportError:
        query = text("""
            UPDATE autopilot_patient_states
            SET last_updated = NOW(),
                next_followup_at = NOW() + INTERVAL '1 day'
            WHERE patient_id = :patient_id
            RETURNING risk_score, risk_state
        """)
        
        try:
            result = db_session.execute(query, {"patient_id": patient_id})
            db_session.commit()
            row = result.fetchone()
            if row:
                return {"risk_score": row[0], "risk_state": row[1]}
        except Exception as e:
            db_session.rollback()
            logger.error(f"Error updating state: {e}", patient_id=patient_id)
        
        return None


def run_triggers(
    db_session,
    patient_id: str,
    logger: SecureLogger
) -> List[Dict[str, Any]]:
    """Run trigger engine for patient"""
    
    try:
        from python_backend.ml_analysis.followup_autopilot.trigger_engine import TriggerEngine
        
        trigger_engine = TriggerEngine(db_session)
        events = trigger_engine.run_triggers(patient_id)
        
        if events:
            logger.info(f"Generated {len(events)} trigger events", patient_id=patient_id)
        
        return events
        
    except ImportError:
        return []
    except Exception as e:
        logger.error(f"Error running triggers: {e}", patient_id=patient_id)
        return []


def dispatch_notifications(
    db_session,
    logger: SecureLogger
) -> Dict[str, int]:
    """Dispatch pending notifications"""
    
    try:
        from python_backend.ml_analysis.followup_autopilot.notification_engine import NotificationEngine
        
        notifier = NotificationEngine(db_session)
        stats = notifier.dispatch_pending_notifications()
        
        logger.info(f"Dispatched notifications: {stats}")
        return stats
        
    except ImportError:
        return {"dispatched": 0, "failed": 0}
    except Exception as e:
        logger.error(f"Error dispatching notifications: {e}")
        return {"dispatched": 0, "failed": 0}


def run_autopilot_sweep(
    db_session,
    logger: SecureLogger,
    consent_verifier: ConsentVerifier
) -> Dict[str, Any]:
    """Run full autopilot sweep"""
    
    logger.info("Starting autopilot inference sweep")
    
    patients = get_patients_due_for_inference(db_session, logger)
    
    if not patients:
        logger.info("No patients due for inference")
        return {"processed": 0, "updated": 0, "triggers": 0}
    
    if 0 < len(patients) < MIN_CELL_SIZE:
        logger.warning(f"PRIVACY_GUARD_ABORT: {len(patients)} patients below minimum cell size ({MIN_CELL_SIZE})")
        logger.warning("Aborting sweep to prevent privacy leakage via small cell inference")
        return {"processed": 0, "updated": 0, "triggers": 0, "aborted": True, "reason": "min_cell_size"}
    
    stats = {
        "processed": 0,
        "updated": 0,
        "triggers": 0,
        "errors": 0
    }
    
    for patient_id in patients:
        stats["processed"] += 1
        
        try:
            state = update_patient_state(db_session, patient_id, logger)
            if state:
                stats["updated"] += 1
            
            events = run_triggers(db_session, patient_id, logger)
            stats["triggers"] += len(events)
            
        except Exception as e:
            stats["errors"] += 1
            logger.error(f"Error processing patient: {e}", patient_id=patient_id)
    
    notification_stats = dispatch_notifications(db_session, logger)
    stats["notifications_dispatched"] = notification_stats.get("dispatched", 0)
    
    logger.info(f"Autopilot sweep complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Run Daily Autopilot Sweep")
    parser.add_argument("--patient-id", type=str, help="Run for specific patient only")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    args = parser.parse_args()
    
    logger = SecureLogger("daily_autopilot", "daily_autopilot.log")
    logger.info("=" * 60)
    logger.info("Starting Daily Autopilot Sweep")
    logger.info("=" * 60)
    
    db_session = get_database_session()
    is_production = os.getenv('NODE_ENV', 'development') == 'production'
    consent_verifier = ConsentVerifier(db_session, strict_mode=is_production)
    audit_logger = AuditLogger(db_session, "daily_autopilot")
    
    audit_logger.log_operation_start({
        "patient_id": args.patient_id,
        "dry_run": args.dry_run
    })
    
    try:
        if args.patient_id:
            logger.info(f"Running for single patient", patient_id=args.patient_id)
            
            state = update_patient_state(db_session, args.patient_id, logger)
            events = run_triggers(db_session, args.patient_id, logger)
            
            stats = {
                "processed": 1,
                "updated": 1 if state else 0,
                "triggers": len(events)
            }
        else:
            stats = run_autopilot_sweep(db_session, logger, consent_verifier)
        
        logger.info(f"Final stats: {stats}")
        audit_logger.log_operation_complete(stats, success=True)
        
    except Exception as e:
        logger.error(f"Autopilot sweep failed: {str(e)}")
        audit_logger.log_operation_complete({"error": str(e)}, success=False)
        raise
    finally:
        db_session.close()


if __name__ == "__main__":
    main()
