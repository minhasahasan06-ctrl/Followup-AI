"""
Lysa Automation Engine - Production-Grade Automation Brain

This service orchestrates all automation tasks for Assistant Lysa:
- Job queue management with priority handling
- Execution with retry logic and error handling
- Real-time status tracking and metrics
- Background scheduling and worker coordination

HIPAA-compliant with comprehensive audit logging.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Awaitable
from collections import defaultdict
import uuid
import json

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func

from app.database import get_db
from app.models.automation_models import (
    AutomationJob, AutomationSchedule, AutomationLog, AutomationMetric,
    EmailAutomationConfig, WhatsAppAutomationConfig, 
    AppointmentAutomationConfig, ReminderAutomationConfig,
    ClinicalAutomationConfig, JobStatus, JobType, JobPriority
)

logger = logging.getLogger(__name__)

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        import redis as sync_redis
        aioredis = None
        REDIS_AVAILABLE = False
    except ImportError:
        REDIS_AVAILABLE = False
        logger.warning("Redis not available - using in-memory queue")


class AutomationEngine:
    """
    Core automation engine that manages job execution and scheduling.
    Supports both Redis-backed and in-memory queue modes.
    """
    
    _instance = None
    
    def __init__(self):
        self.redis_client = None
        self.running = False
        self.job_handlers: Dict[str, Callable] = {}
        self.in_memory_queue: List[Dict] = []
        self.current_job: Optional[AutomationJob] = None
        self.worker_count = int(os.getenv("AUTOMATION_WORKERS", "2"))
        self.max_retries = int(os.getenv("AUTOMATION_MAX_RETRIES", "3"))
        self.job_timeout = int(os.getenv("AUTOMATION_JOB_TIMEOUT", "300"))
        self.scheduler_interval = 60
        
    @classmethod
    def get_instance(cls) -> "AutomationEngine":
        """Singleton pattern for automation engine"""
        if cls._instance is None:
            cls._instance = AutomationEngine()
        return cls._instance
    
    async def initialize(self, db_session_factory):
        """Initialize the automation engine with database and optional Redis"""
        self.db_session_factory = db_session_factory
        
        if REDIS_AVAILABLE and aioredis:
            try:
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
                self.redis_client = await aioredis.from_url(
                    redis_url, 
                    decode_responses=True,
                    socket_timeout=5
                )
                await self.redis_client.ping()
                logger.info("✅ Automation Engine connected to Redis")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e} - using in-memory queue")
                self.redis_client = None
        else:
            logger.info("Using in-memory queue for automation jobs")
        
        self._register_default_handlers()
        logger.info("✅ Automation Engine initialized")
    
    def _register_default_handlers(self):
        """Register default job type handlers"""
        from app.services.email_automation_service import EmailAutomationService
        from app.services.whatsapp_automation_service import WhatsAppAutomationService
        from app.services.appointment_automation_service import AppointmentAutomationService
        from app.services.reminder_automation_service import ReminderAutomationService
        from app.services.clinical_automation_service import ClinicalAutomationService
        
        self.job_handlers = {
            JobType.EMAIL_SYNC.value: EmailAutomationService.sync_emails,
            JobType.EMAIL_CLASSIFY.value: EmailAutomationService.classify_email,
            JobType.EMAIL_AUTO_REPLY.value: EmailAutomationService.auto_reply,
            JobType.EMAIL_FORWARD_URGENT.value: EmailAutomationService.forward_urgent,
            JobType.WHATSAPP_SYNC.value: WhatsAppAutomationService.sync_messages,
            JobType.WHATSAPP_AUTO_REPLY.value: WhatsAppAutomationService.auto_reply,
            JobType.WHATSAPP_SEND_TEMPLATE.value: WhatsAppAutomationService.send_template,
            JobType.APPOINTMENT_REQUEST.value: AppointmentAutomationService.process_request,
            JobType.APPOINTMENT_BOOK.value: AppointmentAutomationService.book_appointment,
            JobType.REMINDER_MEDICATION.value: ReminderAutomationService.send_medication_reminder,
            JobType.REMINDER_APPOINTMENT.value: ReminderAutomationService.send_appointment_reminder,
            JobType.REMINDER_FOLLOWUP.value: ReminderAutomationService.send_followup_reminder,
            JobType.CALENDAR_SYNC.value: AppointmentAutomationService.sync_calendar,
            JobType.DIAGNOSIS_SUMMARY.value: ClinicalAutomationService.generate_summary,
            JobType.SOAP_NOTE.value: ClinicalAutomationService.generate_soap_note,
            JobType.ICD10_SUGGEST.value: ClinicalAutomationService.suggest_icd10,
            JobType.DIFFERENTIAL_DIAGNOSIS.value: ClinicalAutomationService.generate_differential,
            JobType.DAILY_REPORT.value: ClinicalAutomationService.generate_daily_report,
        }
    
    def register_handler(self, job_type: str, handler: Callable):
        """Register a custom job handler"""
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")
    
    async def start(self):
        """Start the automation engine workers and scheduler"""
        if self.running:
            logger.warning("Automation engine already running")
            return
        
        self.running = True
        logger.info(f"🚀 Starting Automation Engine with {self.worker_count} workers")
        
        tasks = [
            asyncio.create_task(self._scheduler_loop()),
            asyncio.create_task(self._metrics_aggregator_loop()),
        ]
        
        for i in range(self.worker_count):
            tasks.append(asyncio.create_task(self._worker_loop(f"worker-{i}")))
        
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Gracefully stop the automation engine"""
        logger.info("🛑 Stopping Automation Engine...")
        self.running = False
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Automation Engine stopped")
    
    async def enqueue_job(
        self,
        db: Session,
        doctor_id: str,
        job_type: str,
        input_data: Optional[Dict] = None,
        patient_id: Optional[str] = None,
        priority: str = "normal",
        scheduled_for: Optional[datetime] = None,
        idempotency_key: Optional[str] = None
    ) -> AutomationJob:
        """
        Add a new job to the automation queue.
        
        Args:
            db: Database session
            doctor_id: Doctor's user ID
            job_type: Type of automation job
            input_data: Job-specific input parameters
            patient_id: Optional patient ID for patient-specific jobs
            priority: Job priority (low, normal, high, urgent)
            scheduled_for: Optional future execution time
            idempotency_key: Optional key to prevent duplicate jobs
        
        Returns:
            Created AutomationJob instance
        """
        if idempotency_key:
            existing = db.query(AutomationJob).filter(
                AutomationJob.idempotency_key == idempotency_key
            ).first()
            if existing:
                logger.info(f"Job with idempotency key {idempotency_key} already exists")
                return existing
        
        job = AutomationJob(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id,
            patient_id=patient_id,
            job_type=job_type,
            priority=priority,
            status=JobStatus.PENDING.value if scheduled_for else JobStatus.QUEUED.value,
            input_data=input_data or {},
            scheduled_for=scheduled_for,
            idempotency_key=idempotency_key,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        db.add(job)
        db.commit()
        db.refresh(job)
        
        self._log_action(
            db, job.id, doctor_id, patient_id,
            "info", f"Job created: {job_type}",
            action_type="job_created"
        )
        
        if not scheduled_for:
            await self._add_to_queue(job)
        
        logger.info(f"Job {job.id} enqueued: {job_type} for doctor {doctor_id}")
        return job
    
    async def _add_to_queue(self, job: AutomationJob):
        """Add job to Redis or in-memory queue"""
        job_data = {
            "id": job.id,
            "doctor_id": job.doctor_id,
            "patient_id": job.patient_id,
            "job_type": job.job_type,
            "priority": job.priority,
            "input_data": job.input_data,
            "attempts": job.attempts,
            "created_at": job.created_at.isoformat() if job.created_at else None
        }
        
        if self.redis_client:
            priority_score = self._get_priority_score(job.priority)
            await self.redis_client.zadd(
                "automation:job_queue",
                {json.dumps(job_data): priority_score}
            )
        else:
            self.in_memory_queue.append(job_data)
            self.in_memory_queue.sort(key=lambda x: self._get_priority_score(x["priority"]), reverse=True)
    
    def _get_priority_score(self, priority: str) -> float:
        """Get numeric score for priority ordering (higher = more urgent)"""
        scores = {
            "urgent": 4.0,
            "high": 3.0,
            "normal": 2.0,
            "low": 1.0
        }
        timestamp_offset = datetime.utcnow().timestamp() / 1e10
        return scores.get(priority, 2.0) + timestamp_offset
    
    async def _worker_loop(self, worker_name: str):
        """Main worker loop that processes jobs from the queue"""
        logger.info(f"Worker {worker_name} started")
        
        while self.running:
            try:
                job_data = await self._get_next_job()
                if not job_data:
                    await asyncio.sleep(1)
                    continue
                
                db = next(self.db_session_factory())
                try:
                    await self._execute_job(db, job_data, worker_name)
                finally:
                    db.close()
                    
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(5)
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _get_next_job(self) -> Optional[Dict]:
        """Get the next job from the queue"""
        if self.redis_client:
            result = await self.redis_client.zpopmax("automation:job_queue")
            if result:
                job_json, score = result[0]
                return json.loads(job_json)
        else:
            if self.in_memory_queue:
                return self.in_memory_queue.pop(0)
        return None
    
    async def _execute_job(self, db: Session, job_data: Dict, worker_name: str):
        """Execute a single automation job"""
        job_id = job_data["id"]
        job_type = job_data["job_type"]
        
        job = db.query(AutomationJob).filter(AutomationJob.id == job_id).first()
        if not job:
            logger.warning(f"Job {job_id} not found in database")
            return
        
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.attempts += 1
        db.commit()
        
        self.current_job = job
        start_time = datetime.utcnow()
        
        self._log_action(
            db, job_id, job.doctor_id, job.patient_id,
            "info", f"Job started by {worker_name}",
            action_type="job_started"
        )
        
        try:
            handler = self.job_handlers.get(job_type)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job_type}")
            
            result = await asyncio.wait_for(
                handler(db, job.doctor_id, job.patient_id, job.input_data),
                timeout=self.job_timeout
            )
            
            job.status = JobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            job.output_data = result if isinstance(result, dict) else {"result": str(result)}
            
            duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            self._log_action(
                db, job_id, job.doctor_id, job.patient_id,
                "info", f"Job completed successfully",
                action_type="job_completed",
                duration_ms=duration_ms
            )
            
            logger.info(f"Job {job_id} completed successfully in {duration_ms}ms")
            
        except asyncio.TimeoutError:
            await self._handle_job_failure(db, job, "Job timed out", worker_name)
        except Exception as e:
            await self._handle_job_failure(db, job, str(e), worker_name)
        finally:
            self.current_job = None
            db.commit()
    
    async def _handle_job_failure(self, db: Session, job: AutomationJob, error: str, worker_name: str):
        """Handle job failure with retry logic"""
        job.error_message = error
        job.error_details = {"worker": worker_name, "attempt": job.attempts}
        
        if job.attempts < job.max_attempts:
            job.status = JobStatus.RETRYING.value
            backoff = 2 ** job.attempts * 60
            job.scheduled_for = datetime.utcnow() + timedelta(seconds=backoff)
            
            self._log_action(
                db, job.id, job.doctor_id, job.patient_id,
                "warning", f"Job failed, retrying in {backoff}s: {error}",
                action_type="job_retry_scheduled"
            )
            logger.warning(f"Job {job.id} failed, retry {job.attempts}/{job.max_attempts} in {backoff}s")
        else:
            job.status = JobStatus.FAILED.value
            job.completed_at = datetime.utcnow()
            
            self._log_action(
                db, job.id, job.doctor_id, job.patient_id,
                "error", f"Job failed permanently after {job.attempts} attempts: {error}",
                action_type="job_failed"
            )
            logger.error(f"Job {job.id} failed permanently: {error}")
    
    async def _scheduler_loop(self):
        """Check for scheduled jobs and trigger them"""
        logger.info("Scheduler loop started")
        
        while self.running:
            try:
                db = next(self.db_session_factory())
                try:
                    now = datetime.utcnow()
                    
                    pending_jobs = db.query(AutomationJob).filter(
                        and_(
                            AutomationJob.status.in_([
                                JobStatus.PENDING.value, 
                                JobStatus.RETRYING.value
                            ]),
                            AutomationJob.scheduled_for <= now
                        )
                    ).limit(100).all()
                    
                    for job in pending_jobs:
                        job.status = JobStatus.QUEUED.value
                        await self._add_to_queue(job)
                        logger.info(f"Scheduled job {job.id} queued for execution")
                    
                    db.commit()
                    
                    await self._process_recurring_schedules(db)
                    
                finally:
                    db.close()
                    
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(self.scheduler_interval)
    
    async def _process_recurring_schedules(self, db: Session):
        """Process recurring schedules and create jobs"""
        now = datetime.utcnow()
        
        due_schedules = db.query(AutomationSchedule).filter(
            and_(
                AutomationSchedule.is_enabled == True,
                AutomationSchedule.next_run_at <= now
            )
        ).all()
        
        for schedule in due_schedules:
            try:
                job = await self.enqueue_job(
                    db=db,
                    doctor_id=schedule.doctor_id,
                    job_type=schedule.job_type,
                    input_data=schedule.job_config,
                    priority=schedule.priority,
                    idempotency_key=f"schedule_{schedule.id}_{now.strftime('%Y%m%d%H%M')}"
                )
                
                schedule.last_run_at = now
                schedule.last_run_job_id = job.id
                schedule.run_count += 1
                schedule.next_run_at = self._calculate_next_run(schedule)
                
                logger.info(f"Created job {job.id} from schedule {schedule.id}")
                
            except Exception as e:
                logger.error(f"Failed to create job from schedule {schedule.id}: {e}")
                schedule.failure_count += 1
        
        db.commit()
    
    def _calculate_next_run(self, schedule: AutomationSchedule) -> datetime:
        """Calculate the next run time based on schedule frequency"""
        now = datetime.utcnow()
        
        intervals = {
            "every_minute": timedelta(minutes=1),
            "every_5_minutes": timedelta(minutes=5),
            "every_10_minutes": timedelta(minutes=10),
            "every_15_minutes": timedelta(minutes=15),
            "every_30_minutes": timedelta(minutes=30),
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(weeks=1),
            "monthly": timedelta(days=30),
        }
        
        interval = intervals.get(schedule.frequency)
        if interval:
            return now + interval
        
        return now + timedelta(hours=1)
    
    async def _metrics_aggregator_loop(self):
        """Periodically aggregate automation metrics"""
        logger.info("Metrics aggregator started")
        
        while self.running:
            try:
                db = next(self.db_session_factory())
                try:
                    await self._aggregate_metrics(db)
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Metrics aggregation error: {e}")
            
            await asyncio.sleep(300)
    
    async def _aggregate_metrics(self, db: Session):
        """Aggregate job metrics for the current hour"""
        now = datetime.utcnow()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        result = db.query(
            AutomationJob.doctor_id,
            AutomationJob.job_type,
            func.count(AutomationJob.id).label('total'),
            func.count(AutomationJob.id).filter(
                AutomationJob.status == JobStatus.COMPLETED.value
            ).label('completed'),
            func.count(AutomationJob.id).filter(
                AutomationJob.status == JobStatus.FAILED.value
            ).label('failed')
        ).filter(
            AutomationJob.created_at >= hour_start
        ).group_by(
            AutomationJob.doctor_id,
            AutomationJob.job_type
        ).all()
        
        for row in result:
            existing = db.query(AutomationMetric).filter(
                and_(
                    AutomationMetric.doctor_id == row.doctor_id,
                    AutomationMetric.job_type == row.job_type,
                    AutomationMetric.metric_date == hour_start.date(),
                    AutomationMetric.metric_hour == hour_start.hour
                )
            ).first()
            
            if existing:
                existing.jobs_created = row.total
                existing.jobs_completed = row.completed
                existing.jobs_failed = row.failed
                existing.updated_at = now
            else:
                metric = AutomationMetric(
                    id=str(uuid.uuid4()),
                    doctor_id=row.doctor_id,
                    job_type=row.job_type,
                    metric_date=hour_start.date(),
                    metric_hour=hour_start.hour,
                    jobs_created=row.total,
                    jobs_completed=row.completed,
                    jobs_failed=row.failed
                )
                db.add(metric)
        
        db.commit()
    
    def _log_action(
        self,
        db: Session,
        job_id: str,
        doctor_id: str,
        patient_id: Optional[str],
        level: str,
        message: str,
        action_type: Optional[str] = None,
        details: Optional[Dict] = None,
        duration_ms: Optional[int] = None
    ):
        """Create an audit log entry"""
        log = AutomationLog(
            id=str(uuid.uuid4()),
            job_id=job_id,
            doctor_id=doctor_id,
            patient_id=patient_id,
            log_level=level,
            message=message,
            action_type=action_type,
            details=details,
            duration_ms=duration_ms,
            created_at=datetime.utcnow()
        )
        db.add(log)
    
    async def get_status(self, db: Session, doctor_id: str) -> Dict[str, Any]:
        """Get current automation status for a doctor"""
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        jobs_today = db.query(AutomationJob).filter(
            and_(
                AutomationJob.doctor_id == doctor_id,
                AutomationJob.created_at >= today_start
            )
        ).all()
        
        pending = sum(1 for j in jobs_today if j.status in [
            JobStatus.PENDING.value, JobStatus.QUEUED.value
        ])
        running = sum(1 for j in jobs_today if j.status == JobStatus.RUNNING.value)
        completed = sum(1 for j in jobs_today if j.status == JobStatus.COMPLETED.value)
        failed = sum(1 for j in jobs_today if j.status == JobStatus.FAILED.value)
        
        email_config = db.query(EmailAutomationConfig).filter(
            EmailAutomationConfig.doctor_id == doctor_id
        ).first()
        
        whatsapp_config = db.query(WhatsAppAutomationConfig).filter(
            WhatsAppAutomationConfig.doctor_id == doctor_id
        ).first()
        
        recent_logs = db.query(AutomationLog).filter(
            AutomationLog.doctor_id == doctor_id
        ).order_by(desc(AutomationLog.created_at)).limit(10).all()
        
        return {
            "is_running": self.running,
            "current_job": {
                "id": self.current_job.id,
                "job_type": self.current_job.job_type,
                "status": self.current_job.status
            } if self.current_job and self.current_job.doctor_id == doctor_id else None,
            "jobs_in_queue": pending,
            "jobs_today": len(jobs_today),
            "jobs_completed_today": completed,
            "jobs_failed_today": failed,
            "jobs_running": running,
            "email_sync_status": "active" if email_config and email_config.is_enabled else "disabled",
            "email_last_sync": email_config.last_sync_at if email_config else None,
            "whatsapp_sync_status": "active" if whatsapp_config and whatsapp_config.is_enabled else "disabled",
            "whatsapp_last_sync": whatsapp_config.last_sync_at if whatsapp_config else None,
            "recent_activities": [
                {
                    "id": log.id,
                    "message": log.message,
                    "action_type": log.action_type,
                    "level": log.log_level,
                    "timestamp": log.created_at.isoformat()
                }
                for log in recent_logs
            ]
        }
    
    async def cancel_job(self, db: Session, job_id: str, doctor_id: str) -> bool:
        """Cancel a pending or queued job"""
        job = db.query(AutomationJob).filter(
            and_(
                AutomationJob.id == job_id,
                AutomationJob.doctor_id == doctor_id,
                AutomationJob.status.in_([
                    JobStatus.PENDING.value,
                    JobStatus.QUEUED.value
                ])
            )
        ).first()
        
        if not job:
            return False
        
        job.status = JobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        
        self._log_action(
            db, job_id, doctor_id, job.patient_id,
            "info", "Job cancelled by user",
            action_type="job_cancelled"
        )
        
        db.commit()
        logger.info(f"Job {job_id} cancelled")
        return True


automation_engine = AutomationEngine.get_instance()
