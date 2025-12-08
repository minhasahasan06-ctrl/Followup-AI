"""
Task Engine for Followup Autopilot

Creates, manages, and tracks follow-up tasks for patients.
Tasks are linked to specific UI tabs for easy navigation.

All tasks are for wellness monitoring only, NOT medical treatment.
"""

import os
import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

logger = logging.getLogger(__name__)


TASK_TYPES = {
    "symptom_check": {
        "title": "Symptom Check-in",
        "description": "Please log how you're feeling today.",
        "ui_tab": "symptoms",
    },
    "med_adherence_check": {
        "title": "Medication Review",
        "description": "Review your medication schedule and log any doses.",
        "ui_tab": "medications",
    },
    "mh_check": {
        "title": "Mental Wellness Check",
        "description": "Complete a brief mental wellness questionnaire.",
        "ui_tab": "mental_health",
    },
    "resp_symptom_check": {
        "title": "Respiratory Check",
        "description": "Log any breathing-related symptoms.",
        "ui_tab": "symptoms",
    },
    "pain_check": {
        "title": "Pain Assessment",
        "description": "Log your current pain levels.",
        "ui_tab": "paintrack",
    },
    "video_exam": {
        "title": "Video Assessment",
        "description": "Complete a brief video wellness check.",
        "ui_tab": "video_ai",
    },
    "audio_check": {
        "title": "Voice Check-in",
        "description": "Record a brief voice sample for wellness monitoring.",
        "ui_tab": "audio_ai",
    },
    "exposure_check": {
        "title": "Exposure Review",
        "description": "Review your current risk exposures status.",
        "ui_tab": "risk_exposures",
    },
    "urgent_check": {
        "title": "Urgent Wellness Check",
        "description": "IMPORTANT: Please complete this check-in as soon as possible. Contact your care team if you feel unwell.",
        "ui_tab": "symptoms",
    },
    "env_check": {
        "title": "Environmental Check",
        "description": "Review current environmental conditions in your area.",
        "ui_tab": "environmental",
    },
}


class TaskEngine:
    """
    Task creation and management engine.
    
    Responsibilities:
    1. Create follow-up tasks with proper metadata
    2. Handle task deduplication via cooldown
    3. Retrieve pending tasks for patients
    4. Mark tasks as completed
    5. Track task completion for engagement analysis
    """
    
    def __init__(self, db_session=None):
        self.db = db_session
        self.logger = logging.getLogger(__name__)
    
    def create_followup_task(
        self,
        patient_id: str,
        task_type: str,
        priority: str = "medium",
        due_at: Optional[datetime] = None,
        trigger_name: Optional[str] = None,
        reason: Optional[str] = None,
        ui_tab_target: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new follow-up task for a patient.
        
        Args:
            patient_id: Patient identifier
            task_type: Type of task (symptom_check, mh_check, etc.)
            priority: Task priority (low, medium, high, critical)
            due_at: When task is due
            trigger_name: Name of trigger that created this task
            reason: Human-readable reason for the task
            ui_tab_target: Which UI tab this task navigates to
            metadata: Additional task metadata
            
        Returns:
            Task ID if created, None if duplicate or error
        """
        if not due_at:
            due_at = datetime.now(timezone.utc) + timedelta(hours=24)
        
        if self._has_pending_task(patient_id, task_type, trigger_name):
            self.logger.info(f"Skipping duplicate task: {task_type} for {patient_id}")
            return None
        
        task_info = TASK_TYPES.get(task_type, {})
        if not ui_tab_target:
            ui_tab_target = task_info.get("ui_tab", "symptoms")
        
        task_id = str(uuid4())
        
        task_metadata_value = {
            "title": task_info.get("title", task_type.replace("_", " ").title()),
            "description": task_info.get("description", ""),
            **(metadata or {})
        }
        
        task_data = {
            "id": task_id,
            "patient_id": patient_id,
            "task_type": task_type,
            "priority": priority,
            "status": "pending",
            "due_at": due_at,
            "created_by": "autopilot",
            "trigger_name": trigger_name,
            "reason": reason or task_info.get("description", ""),
            "ui_tab_target": ui_tab_target,
            "task_metadata": task_metadata_value
        }
        
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotFollowupTask
                
                task = AutopilotFollowupTask(**task_data)
                self.db.add(task)
                self.db.commit()
            else:
                self._create_task_raw(task_data)
            
            self._audit_log(patient_id, "task_created", task_data)
            
            from .notification_engine import NotificationEngine
            notifier = NotificationEngine(self.db)
            notifier.create_task_notification(patient_id, task_data)
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Failed to create task: {e}")
            if self.db:
                self.db.rollback()
            return None
    
    def get_today_tasks_for_patient(
        self,
        patient_id: str,
        include_overdue: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get pending tasks for a patient that are due today or overdue.
        
        Args:
            patient_id: Patient identifier
            include_overdue: Whether to include overdue tasks
            
        Returns:
            List of task dictionaries
        """
        now = datetime.now(timezone.utc)
        end_of_today = now.replace(hour=23, minute=59, second=59)
        
        if self.db:
            from app.models.followup_autopilot_models import AutopilotFollowupTask
            
            query = self.db.query(AutopilotFollowupTask).filter(
                AutopilotFollowupTask.patient_id == patient_id,
                AutopilotFollowupTask.status == "pending"
            )
            
            if include_overdue:
                query = query.filter(AutopilotFollowupTask.due_at <= end_of_today)
            else:
                start_of_today = now.replace(hour=0, minute=0, second=0)
                query = query.filter(
                    AutopilotFollowupTask.due_at >= start_of_today,
                    AutopilotFollowupTask.due_at <= end_of_today
                )
            
            rows = query.order_by(
                AutopilotFollowupTask.priority.desc(),
                AutopilotFollowupTask.due_at
            ).all()
            
            return [self._row_to_dict(row) for row in rows]
        
        return self._get_tasks_raw(patient_id, end_of_today)
    
    def get_all_pending_tasks(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get all pending tasks for a patient"""
        if self.db:
            from app.models.followup_autopilot_models import AutopilotFollowupTask
            
            rows = self.db.query(AutopilotFollowupTask).filter(
                AutopilotFollowupTask.patient_id == patient_id,
                AutopilotFollowupTask.status == "pending"
            ).order_by(
                AutopilotFollowupTask.priority.desc(),
                AutopilotFollowupTask.due_at
            ).all()
            
            return [self._row_to_dict(row) for row in rows]
        
        return []
    
    def complete_task(
        self,
        task_id: str,
        patient_id: str
    ) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: Task identifier
            patient_id: Patient identifier (for verification)
            
        Returns:
            True if completed successfully
        """
        now = datetime.now(timezone.utc)
        
        try:
            if self.db:
                from app.models.followup_autopilot_models import (
                    AutopilotFollowupTask, AutopilotPatientState
                )
                
                task = self.db.query(AutopilotFollowupTask).filter(
                    AutopilotFollowupTask.id == task_id,
                    AutopilotFollowupTask.patient_id == patient_id
                ).first()
                
                if not task:
                    self.logger.warning(f"Task not found: {task_id}")
                    return False
                
                task.status = "completed"
                task.completed_at = now
                
                state = self.db.query(AutopilotPatientState).filter(
                    AutopilotPatientState.patient_id == patient_id
                ).first()
                
                if state:
                    state.last_checkin_at = now
                
                self.db.commit()
            else:
                self._complete_task_raw(task_id, patient_id, now)
            
            self._audit_log(patient_id, "task_completed", {"task_id": task_id})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to complete task: {e}")
            if self.db:
                self.db.rollback()
            return False
    
    def cancel_task(
        self,
        task_id: str,
        patient_id: str,
        reason: str = "cancelled_by_system"
    ) -> bool:
        """Cancel a pending task"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotFollowupTask
                
                task = self.db.query(AutopilotFollowupTask).filter(
                    AutopilotFollowupTask.id == task_id,
                    AutopilotFollowupTask.patient_id == patient_id
                ).first()
                
                if not task or task.status != "pending":
                    return False
                
                task.status = "cancelled"
                task.task_metadata = {**(task.task_metadata or {}), "cancel_reason": reason}
                self.db.commit()
                
            self._audit_log(patient_id, "task_cancelled", {"task_id": task_id, "reason": reason})
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cancel task: {e}")
            if self.db:
                self.db.rollback()
            return False
    
    def _has_pending_task(
        self,
        patient_id: str,
        task_type: str,
        trigger_name: Optional[str]
    ) -> bool:
        """Check if patient already has a pending task of this type"""
        if self.db:
            from app.models.followup_autopilot_models import AutopilotFollowupTask
            
            query = self.db.query(AutopilotFollowupTask).filter(
                AutopilotFollowupTask.patient_id == patient_id,
                AutopilotFollowupTask.task_type == task_type,
                AutopilotFollowupTask.status == "pending"
            )
            
            if trigger_name:
                query = query.filter(AutopilotFollowupTask.trigger_name == trigger_name)
            
            return query.first() is not None
        
        return False
    
    def _row_to_dict(self, row) -> Dict[str, Any]:
        """Convert database row to dictionary"""
        task_info = TASK_TYPES.get(row.task_type, {})
        
        return {
            "id": str(row.id),
            "patient_id": row.patient_id,
            "task_type": row.task_type,
            "title": (row.task_metadata or {}).get("title", task_info.get("title", row.task_type)),
            "description": (row.task_metadata or {}).get("description", task_info.get("description", "")),
            "priority": row.priority,
            "status": row.status,
            "due_at": row.due_at.isoformat() if row.due_at else None,
            "completed_at": row.completed_at.isoformat() if row.completed_at else None,
            "trigger_name": row.trigger_name,
            "reason": row.reason,
            "ui_tab_target": row.ui_tab_target or task_info.get("ui_tab", "symptoms"),
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "is_overdue": row.due_at < datetime.now(timezone.utc) if row.due_at else False,
        }
    
    def _create_task_raw(self, task_data: Dict[str, Any]):
        """Direct database insert when ORM not available"""
        import psycopg2
        from psycopg2.extras import Json
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            raise ValueError("DATABASE_URL not set")
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO autopilot_followup_tasks 
                    (id, patient_id, task_type, priority, status, due_at, 
                     created_by, trigger_name, reason, ui_tab_target, task_metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                """, (
                    task_data["id"], task_data["patient_id"], task_data["task_type"],
                    task_data["priority"], task_data["status"], task_data["due_at"],
                    task_data["created_by"], task_data.get("trigger_name"),
                    task_data.get("reason"), task_data.get("ui_tab_target"),
                    Json(task_data.get("metadata", {}))
                ))
            conn.commit()
    
    def _get_tasks_raw(
        self,
        patient_id: str,
        end_of_today: datetime
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
                    SELECT * FROM autopilot_followup_tasks
                    WHERE patient_id = %s AND status = 'pending' AND due_at <= %s
                    ORDER BY 
                        CASE priority 
                            WHEN 'critical' THEN 1 
                            WHEN 'high' THEN 2 
                            WHEN 'medium' THEN 3 
                            ELSE 4 
                        END,
                        due_at
                """, (patient_id, end_of_today))
                
                return [dict(row) for row in cur.fetchall()]
    
    def _complete_task_raw(
        self,
        task_id: str,
        patient_id: str,
        completed_at: datetime
    ):
        """Direct database update when ORM not available"""
        import psycopg2
        
        conn_str = os.environ.get("DATABASE_URL")
        if not conn_str:
            return
            
        with psycopg2.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE autopilot_followup_tasks 
                    SET status = 'completed', completed_at = %s
                    WHERE id = %s AND patient_id = %s
                """, (completed_at, task_id, patient_id))
                
                cur.execute("""
                    UPDATE autopilot_patient_states 
                    SET last_checkin_at = %s
                    WHERE patient_id = %s
                """, (completed_at, patient_id))
            conn.commit()
    
    def _audit_log(self, patient_id: str, action: str, details: Dict[str, Any]):
        """Log action for HIPAA audit trail"""
        try:
            if self.db:
                from app.models.followup_autopilot_models import AutopilotAuditLog
                
                safe_details = {}
                for k, v in details.items():
                    if isinstance(v, datetime):
                        safe_details[k] = v.isoformat()
                    elif isinstance(v, (str, int, float, bool, type(None))):
                        safe_details[k] = v
                    else:
                        safe_details[k] = str(v)
                
                log = AutopilotAuditLog(
                    patient_id=patient_id,
                    action=action,
                    entity_type="task",
                    entity_id=details.get("task_id") or details.get("id"),
                    new_values=safe_details
                )
                self.db.add(log)
                self.db.commit()
        except Exception as e:
            self.logger.warning(f"Audit log failed: {e}")
