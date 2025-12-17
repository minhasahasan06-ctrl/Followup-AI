"""
Background Job Scheduler for Assistant Lysa Automation Engine

Production-grade scheduler for automated tasks:
- Email sync every minute
- Appointment reminders every morning
- Daily reports
- No-show follow-ups
- Clinical summary generation

Uses asyncio for lightweight scheduling without Celery dependency.
Can be upgraded to Celery/Redis for production scale.
"""

import os
import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, Any, List, Optional, Callable, Awaitable

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.database import SessionLocal
from app.services.automation_engine import automation_engine

logger = logging.getLogger(__name__)


class JobScheduler:
    """
    Lightweight scheduler for automation jobs.
    
    Default schedules:
    - Email sync: Every minute (when enabled)
    - Appointment reminders: 8 AM daily
    - No-show follow-ups: 6 PM daily
    - Daily reports: 11 PM daily
    """
    
    _instance = None
    
    def __init__(self):
        self.running = False
        self.schedules: Dict[str, Dict[str, Any]] = {}
        self.last_run: Dict[str, datetime] = {}
    
    @classmethod
    def get_instance(cls) -> "JobScheduler":
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = JobScheduler()
        return cls._instance
    
    def configure_default_schedules(self):
        """Set up default automation schedules"""
        
        self.schedules = {
            # === EVERY MINUTE TASKS ===
            "email_sync": {
                "job_type": "email_sync",
                "interval_minutes": 1,
                "enabled": True,
                "description": "Sync emails from connected Gmail accounts"
            },
            "whatsapp_sync": {
                "job_type": "whatsapp_sync",
                "interval_minutes": 1,
                "enabled": True,
                "description": "Process WhatsApp messages and auto-reply"
            },
            
            # === EVERY 5 MINUTES TASKS ===
            "email_classify": {
                "job_type": "email_classify",
                "interval_minutes": 5,
                "enabled": True,
                "description": "Classify new unread emails with AI"
            },
            
            # === EVERY 10 MINUTES TASKS ===
            "patient_followup_check": {
                "job_type": "patient_monitor",
                "interval_minutes": 10,
                "enabled": True,
                "description": "Check patients with active sharing links for health changes"
            },
            
            # === EVERY 15 MINUTES TASKS ===
            "calendar_sync": {
                "job_type": "calendar_sync",
                "interval_minutes": 15,
                "enabled": True,
                "description": "Bidirectional sync with Google Calendar"
            },
            
            # === HOURLY TASKS ===
            "clean_stale_tasks": {
                "job_type": "cleanup_stale",
                "interval_minutes": 60,
                "enabled": True,
                "description": "Clean stale/stuck automation tasks"
            },
            
            # === DAILY MORNING TASKS (8 AM) ===
            "morning_reminders": {
                "job_type": "reminder_batch",
                "run_at_hour": 8,
                "run_at_minute": 0,
                "enabled": True,
                "input_data": {"reminder_types": ["medication", "appointment", "followup"]},
                "description": "Send all morning reminders (medication, appointments, follow-ups)"
            },
            "appointment_reminders_24h": {
                "job_type": "reminder_appointment",
                "run_at_hour": 8,
                "run_at_minute": 30,
                "enabled": True,
                "input_data": {"hours_before": 24},
                "description": "Send 24-hour appointment reminders"
            },
            
            # === DAILY AFTERNOON TASKS ===
            "appointment_reminders_2h": {
                "job_type": "reminder_appointment",
                "run_at_hour": 10,
                "run_at_minute": 0,
                "enabled": True,
                "input_data": {"hours_before": 2},
                "description": "Send 2-hour appointment reminders"
            },
            
            # === DAILY EVENING TASKS ===
            "noshow_followups": {
                "job_type": "reminder_noshow",
                "run_at_hour": 18,
                "run_at_minute": 0,
                "enabled": True,
                "description": "Send no-show follow-up messages"
            },
            
            # === MIDNIGHT TASKS ===
            "daily_report": {
                "job_type": "daily_report",
                "run_at_hour": 0,
                "run_at_minute": 0,
                "enabled": True,
                "description": "Generate daily activity reports"
            },
            "daily_cleanup": {
                "job_type": "daily_cleanup",
                "run_at_hour": 0,
                "run_at_minute": 30,
                "enabled": True,
                "description": "Clean up old logs and temporary data"
            },
            
            # === DATA WAREHOUSE AGGREGATION (2 AM) ===
            "warehouse_aggregation": {
                "job_type": "warehouse_aggregation",
                "run_at_hour": 2,
                "run_at_minute": 0,
                "enabled": True,
                "description": "Run nightly data warehouse aggregation (epidemiology, surveillance)"
            }
        }
        
        logger.info(f"✅ Configured {len(self.schedules)} default schedules")
    
    async def start(self):
        """Start the scheduler loop"""
        if self.running:
            logger.warning("Scheduler already running")
            return
        
        self.running = True
        self.configure_default_schedules()
        
        logger.info("🚀 Starting Job Scheduler")
        
        try:
            while self.running:
                await self._check_schedules()
                await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
            self.running = False
    
    async def stop(self):
        """Stop the scheduler"""
        logger.info("🛑 Stopping Job Scheduler")
        self.running = False
    
    async def _check_schedules(self):
        """Check all schedules and trigger jobs if due"""
        now = datetime.utcnow()
        
        for schedule_name, schedule in self.schedules.items():
            if not schedule.get("enabled", True):
                continue
            
            should_run = False
            
            if "interval_minutes" in schedule:
                last_run = self.last_run.get(schedule_name)
                if not last_run:
                    should_run = True
                else:
                    interval = timedelta(minutes=schedule["interval_minutes"])
                    if now - last_run >= interval:
                        should_run = True
            
            elif "run_at_hour" in schedule:
                run_hour = schedule["run_at_hour"]
                run_minute = schedule.get("run_at_minute", 0)
                
                if now.hour == run_hour and now.minute == run_minute:
                    last_run = self.last_run.get(schedule_name)
                    if not last_run or last_run.date() < now.date():
                        should_run = True
            
            if should_run:
                await self._trigger_schedule(schedule_name, schedule)
                self.last_run[schedule_name] = now
    
    async def _trigger_schedule(self, schedule_name: str, schedule: Dict[str, Any]):
        """Trigger jobs for a schedule across all enabled doctors"""
        logger.info(f"⏰ Triggering schedule: {schedule_name}")
        
        try:
            db = SessionLocal()
            
            from app.models.automation_models import (
                EmailAutomationConfig, WhatsAppAutomationConfig,
                AppointmentAutomationConfig, ReminderAutomationConfig,
                ClinicalAutomationConfig
            )
            
            job_type = schedule["job_type"]
            input_data = schedule.get("input_data", {})
            
            # Handle warehouse aggregation separately (no doctor-specific jobs)
            if job_type == "warehouse_aggregation":
                try:
                    from app.services.warehouse_aggregation_jobs import run_nightly_warehouse_aggregation
                    result = await run_nightly_warehouse_aggregation()
                    logger.info(f"✅ Warehouse aggregation completed: {result.get('job', 'unknown')}")
                    db.close()
                    return
                except Exception as e:
                    logger.error(f"❌ Warehouse aggregation failed: {e}")
                    db.close()
                    return
            
            if job_type.startswith("email"):
                configs = db.query(EmailAutomationConfig).filter(
                    EmailAutomationConfig.is_enabled == True
                ).all()
                doctor_ids = [c.doctor_id for c in configs]
            
            elif job_type.startswith("whatsapp"):
                configs = db.query(WhatsAppAutomationConfig).filter(
                    WhatsAppAutomationConfig.is_enabled == True
                ).all()
                doctor_ids = [c.doctor_id for c in configs]
            
            elif job_type.startswith("reminder") or job_type.startswith("noshow"):
                configs = db.query(ReminderAutomationConfig).filter(
                    ReminderAutomationConfig.is_enabled == True
                ).all()
                doctor_ids = [c.doctor_id for c in configs]
            
            elif job_type.startswith("calendar") or job_type.startswith("appointment"):
                configs = db.query(AppointmentAutomationConfig).filter(
                    AppointmentAutomationConfig.is_enabled == True
                ).all()
                doctor_ids = [c.doctor_id for c in configs]
            
            else:
                configs = db.query(EmailAutomationConfig).all()
                doctor_ids = list(set(c.doctor_id for c in configs))
            
            jobs_created = 0
            for doctor_id in doctor_ids:
                try:
                    await automation_engine.enqueue_job(
                        db=db,
                        doctor_id=doctor_id,
                        job_type=job_type,
                        input_data=input_data,
                        priority="normal",
                        idempotency_key=f"{schedule_name}_{doctor_id}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
                    )
                    jobs_created += 1
                except Exception as e:
                    logger.error(f"Failed to create job for doctor {doctor_id}: {e}")
            
            db.close()
            
            logger.info(f"✅ Schedule {schedule_name} created {jobs_created} jobs")
            
        except Exception as e:
            logger.error(f"Schedule trigger error: {e}")
    
    def enable_schedule(self, schedule_name: str):
        """Enable a schedule"""
        if schedule_name in self.schedules:
            self.schedules[schedule_name]["enabled"] = True
            logger.info(f"Schedule {schedule_name} enabled")
    
    def disable_schedule(self, schedule_name: str):
        """Disable a schedule"""
        if schedule_name in self.schedules:
            self.schedules[schedule_name]["enabled"] = False
            logger.info(f"Schedule {schedule_name} disabled")
    
    def add_schedule(
        self,
        name: str,
        job_type: str,
        interval_minutes: Optional[int] = None,
        run_at_hour: Optional[int] = None,
        run_at_minute: int = 0,
        input_data: Optional[Dict] = None,
        description: str = ""
    ):
        """Add a custom schedule"""
        schedule = {
            "job_type": job_type,
            "enabled": True,
            "description": description
        }
        
        if interval_minutes:
            schedule["interval_minutes"] = interval_minutes
        elif run_at_hour is not None:
            schedule["run_at_hour"] = run_at_hour
            schedule["run_at_minute"] = run_at_minute
        
        if input_data:
            schedule["input_data"] = input_data
        
        self.schedules[name] = schedule
        logger.info(f"Added schedule: {name}")
    
    def remove_schedule(self, name: str):
        """Remove a schedule"""
        if name in self.schedules:
            del self.schedules[name]
            logger.info(f"Removed schedule: {name}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        return {
            "running": self.running,
            "schedules": {
                name: {
                    **schedule,
                    "last_run": self.last_run.get(name, None)
                }
                for name, schedule in self.schedules.items()
            }
        }


scheduler = JobScheduler.get_instance()


async def start_automation_services():
    """Start all automation services (engine + scheduler)"""
    from app.database import SessionLocal
    
    await automation_engine.initialize(SessionLocal)
    
    asyncio.create_task(automation_engine.start())
    asyncio.create_task(scheduler.start())
    
    logger.info("✅ All automation services started")


async def stop_automation_services():
    """Stop all automation services"""
    await scheduler.stop()
    await automation_engine.stop()
    logger.info("✅ All automation services stopped")
