"""
Automation API Router for Assistant Lysa

Production-grade API endpoints for:
- Job management (create, cancel, status)
- Schedule management
- Configuration
- Real-time status (SSE)
- Dashboard metrics
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from app.database import get_db
from app.auth import get_current_user
from app.models.automation_models import (
    AutomationJob, AutomationSchedule, AutomationLog, AutomationMetric,
    EmailAutomationConfig, WhatsAppAutomationConfig,
    AppointmentAutomationConfig, ReminderAutomationConfig,
    ClinicalAutomationConfig
)
from app.schemas.automation_schemas import (
    AutomationJobCreate, AutomationJobResponse, AutomationJobListResponse,
    AutomationScheduleCreate, AutomationScheduleResponse,
    AutomationStatusResponse, TriggerJobRequest, AutomationDashboardStats,
    EmailAutomationConfigCreate, EmailAutomationConfigResponse,
    WhatsAppAutomationConfigCreate, WhatsAppAutomationConfigResponse,
    AppointmentAutomationConfigCreate, AppointmentAutomationConfigResponse,
    ReminderAutomationConfigCreate, ReminderAutomationConfigResponse,
    ClinicalAutomationConfigCreate, ClinicalAutomationConfigResponse,
    AutomationLogResponse, AutomationEvent
)
from app.services.automation_engine import automation_engine

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/automation",
    tags=["automation"]
)


@router.get("/status", response_model=AutomationStatusResponse)
async def get_automation_status(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get current automation status for the doctor"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    status = await automation_engine.get_status(db, doctor_id)
    
    return AutomationStatusResponse(
        is_running=status.get("is_running", False),
        current_job=status.get("current_job"),
        jobs_in_queue=status.get("jobs_in_queue", 0),
        jobs_today=status.get("jobs_today", 0),
        jobs_completed_today=status.get("jobs_completed_today", 0),
        jobs_failed_today=status.get("jobs_failed_today", 0),
        email_sync_status=status.get("email_sync_status", "disabled"),
        email_last_sync=status.get("email_last_sync"),
        emails_processed_today=0,
        whatsapp_sync_status=status.get("whatsapp_sync_status", "disabled"),
        whatsapp_last_sync=status.get("whatsapp_last_sync"),
        messages_sent_today=0,
        calendar_sync_status="disabled",
        calendar_last_sync=None,
        appointments_booked_today=0,
        reminders_sent_today=0,
        recent_activities=status.get("recent_activities", [])
    )


@router.get("/dashboard/stats", response_model=AutomationDashboardStats)
async def get_dashboard_stats(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get aggregated automation statistics for the dashboard"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    
    jobs = db.query(AutomationJob).filter(
        and_(
            AutomationJob.doctor_id == doctor_id,
            AutomationJob.created_at >= today_start
        )
    ).all()
    
    total = len(jobs)
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")
    pending = sum(1 for j in jobs if j.status in ["pending", "queued"])
    running = sum(1 for j in jobs if j.status == "running")
    
    job_types = {}
    for job in jobs:
        if job.job_type not in job_types:
            job_types[job.job_type] = {"completed": 0, "total": 0}
        job_types[job.job_type]["total"] += 1
        if job.status == "completed":
            job_types[job.job_type]["completed"] += 1
    
    email_config = db.query(EmailAutomationConfig).filter(
        EmailAutomationConfig.doctor_id == doctor_id
    ).first()
    
    whatsapp_config = db.query(WhatsAppAutomationConfig).filter(
        WhatsAppAutomationConfig.doctor_id == doctor_id
    ).first()
    
    duration_ms = []
    for job in jobs:
        if job.started_at and job.completed_at:
            delta = (job.completed_at - job.started_at).total_seconds() * 1000
            duration_ms.append(delta)
    
    avg_duration = sum(duration_ms) / len(duration_ms) if duration_ms else 0
    success_rate = (completed / total * 100) if total > 0 else 0
    
    return AutomationDashboardStats(
        total_jobs_today=total,
        completed_jobs_today=completed,
        failed_jobs_today=failed,
        pending_jobs=pending,
        running_jobs=running,
        emails_synced=job_types.get("email_sync", {}).get("completed", 0),
        emails_classified=job_types.get("email_classify", {}).get("completed", 0),
        emails_auto_replied=job_types.get("email_auto_reply", {}).get("completed", 0),
        emails_forwarded=job_types.get("email_forward_urgent", {}).get("completed", 0),
        whatsapp_messages_received=0,
        whatsapp_messages_sent=job_types.get("whatsapp_send_template", {}).get("completed", 0),
        whatsapp_auto_replies=job_types.get("whatsapp_auto_reply", {}).get("completed", 0),
        appointments_requested=job_types.get("appointment_request", {}).get("total", 0),
        appointments_booked=job_types.get("appointment_book", {}).get("completed", 0),
        appointments_confirmed=0,
        reminders_medication=job_types.get("reminder_medication", {}).get("completed", 0),
        reminders_appointment=job_types.get("reminder_appointment", {}).get("completed", 0),
        reminders_followup=job_types.get("reminder_followup", {}).get("completed", 0),
        clinical_soap_notes=job_types.get("soap_note", {}).get("completed", 0),
        clinical_diagnoses=job_types.get("differential_diagnosis", {}).get("completed", 0),
        clinical_prescriptions=job_types.get("prescription_generate", {}).get("completed", 0),
        avg_job_duration_ms=avg_duration,
        success_rate=success_rate,
        last_email_sync=email_config.last_sync_at if email_config else None,
        last_whatsapp_sync=whatsapp_config.last_sync_at if whatsapp_config else None,
        last_calendar_sync=None
    )


@router.post("/jobs/trigger")
async def trigger_job(
    request: TriggerJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Manually trigger an automation job"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    job = await automation_engine.enqueue_job(
        db=db,
        doctor_id=doctor_id,
        job_type=request.job_type.value,
        input_data=request.input_data,
        patient_id=request.patient_id,
        priority=request.priority.value
    )
    
    return {
        "success": True,
        "job_id": job.id,
        "status": job.status,
        "message": f"Job {request.job_type.value} queued successfully"
    }


@router.get("/jobs", response_model=AutomationJobListResponse)
async def list_jobs(
    status: Optional[str] = None,
    job_type: Optional[str] = None,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List automation jobs for the current doctor"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    query = db.query(AutomationJob).filter(
        AutomationJob.doctor_id == doctor_id
    )
    
    if status:
        query = query.filter(AutomationJob.status == status)
    if job_type:
        query = query.filter(AutomationJob.job_type == job_type)
    
    total = query.count()
    
    jobs = query.order_by(desc(AutomationJob.created_at)).offset(offset).limit(limit).all()
    
    pending = sum(1 for j in jobs if j.status in ["pending", "queued"])
    running = sum(1 for j in jobs if j.status == "running")
    completed = sum(1 for j in jobs if j.status == "completed")
    failed = sum(1 for j in jobs if j.status == "failed")
    
    return AutomationJobListResponse(
        jobs=[AutomationJobResponse.model_validate(j) for j in jobs],
        total=total,
        pending=pending,
        running=running,
        completed=completed,
        failed=failed
    )


@router.get("/jobs/{job_id}", response_model=AutomationJobResponse)
async def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific automation job"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    job = db.query(AutomationJob).filter(
        and_(
            AutomationJob.id == job_id,
            AutomationJob.doctor_id == doctor_id
        )
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return AutomationJobResponse.model_validate(job)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Cancel a pending or queued job"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    success = await automation_engine.cancel_job(db, job_id, doctor_id)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (not found or already running)"
        )
    
    return {"success": True, "message": "Job cancelled successfully"}


@router.get("/jobs/{job_id}/logs", response_model=List[AutomationLogResponse])
async def get_job_logs(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get logs for a specific job"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    job = db.query(AutomationJob).filter(
        and_(
            AutomationJob.id == job_id,
            AutomationJob.doctor_id == doctor_id
        )
    ).first()
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logs = db.query(AutomationLog).filter(
        AutomationLog.job_id == job_id
    ).order_by(AutomationLog.created_at).all()
    
    return [AutomationLogResponse.model_validate(log) for log in logs]


@router.get("/schedules", response_model=List[AutomationScheduleResponse])
async def list_schedules(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List all automation schedules for the current doctor"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    schedules = db.query(AutomationSchedule).filter(
        AutomationSchedule.doctor_id == doctor_id
    ).order_by(AutomationSchedule.next_run_at).all()
    
    return [AutomationScheduleResponse.model_validate(s) for s in schedules]


@router.post("/schedules", response_model=AutomationScheduleResponse)
async def create_schedule(
    request: AutomationScheduleCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new automation schedule"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    import uuid
    
    schedule = AutomationSchedule(
        id=str(uuid.uuid4()),
        doctor_id=doctor_id,
        name=request.name,
        description=request.description,
        job_type=request.job_type.value,
        job_config=request.job_config,
        frequency=request.frequency.value,
        cron_expression=request.cron_expression,
        timezone=request.timezone,
        priority=request.priority.value,
        is_enabled=request.is_enabled,
        next_run_at=datetime.utcnow() + timedelta(minutes=1)
    )
    
    db.add(schedule)
    db.commit()
    db.refresh(schedule)
    
    return AutomationScheduleResponse.model_validate(schedule)


@router.patch("/schedules/{schedule_id}")
async def update_schedule(
    schedule_id: str,
    is_enabled: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a schedule (enable/disable)"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    schedule = db.query(AutomationSchedule).filter(
        and_(
            AutomationSchedule.id == schedule_id,
            AutomationSchedule.doctor_id == doctor_id
        )
    ).first()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    if is_enabled is not None:
        schedule.is_enabled = is_enabled
    
    db.commit()
    
    return {"success": True, "is_enabled": schedule.is_enabled}


@router.delete("/schedules/{schedule_id}")
async def delete_schedule(
    schedule_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a schedule"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    schedule = db.query(AutomationSchedule).filter(
        and_(
            AutomationSchedule.id == schedule_id,
            AutomationSchedule.doctor_id == doctor_id
        )
    ).first()
    
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    db.delete(schedule)
    db.commit()
    
    return {"success": True, "message": "Schedule deleted"}


@router.get("/config/email", response_model=EmailAutomationConfigResponse)
async def get_email_config(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get email automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(EmailAutomationConfig).filter(
        EmailAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = EmailAutomationConfig(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return EmailAutomationConfigResponse.model_validate(config)


@router.put("/config/email", response_model=EmailAutomationConfigResponse)
async def update_email_config(
    request: EmailAutomationConfigCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update email automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(EmailAutomationConfig).filter(
        EmailAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = EmailAutomationConfig(id=str(uuid.uuid4()), doctor_id=doctor_id)
        db.add(config)
    
    for field, value in request.model_dump().items():
        setattr(config, field, value)
    
    config.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(config)
    
    return EmailAutomationConfigResponse.model_validate(config)


@router.get("/config/whatsapp", response_model=WhatsAppAutomationConfigResponse)
async def get_whatsapp_config(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get WhatsApp automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(WhatsAppAutomationConfig).filter(
        WhatsAppAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = WhatsAppAutomationConfig(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return WhatsAppAutomationConfigResponse.model_validate(config)


@router.put("/config/whatsapp", response_model=WhatsAppAutomationConfigResponse)
async def update_whatsapp_config(
    request: WhatsAppAutomationConfigCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update WhatsApp automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(WhatsAppAutomationConfig).filter(
        WhatsAppAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = WhatsAppAutomationConfig(id=str(uuid.uuid4()), doctor_id=doctor_id)
        db.add(config)
    
    for field, value in request.model_dump().items():
        setattr(config, field, value)
    
    config.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(config)
    
    return WhatsAppAutomationConfigResponse.model_validate(config)


@router.get("/config/appointments", response_model=AppointmentAutomationConfigResponse)
async def get_appointment_config(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get appointment automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(AppointmentAutomationConfig).filter(
        AppointmentAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = AppointmentAutomationConfig(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return AppointmentAutomationConfigResponse.model_validate(config)


@router.get("/config/reminders", response_model=ReminderAutomationConfigResponse)
async def get_reminder_config(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get reminder automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(ReminderAutomationConfig).filter(
        ReminderAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = ReminderAutomationConfig(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return ReminderAutomationConfigResponse.model_validate(config)


@router.get("/config/clinical", response_model=ClinicalAutomationConfigResponse)
async def get_clinical_config(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get clinical automation configuration"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(ClinicalAutomationConfig).filter(
        ClinicalAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        import uuid
        config = ClinicalAutomationConfig(
            id=str(uuid.uuid4()),
            doctor_id=doctor_id
        )
        db.add(config)
        db.commit()
        db.refresh(config)
    
    return ClinicalAutomationConfigResponse.model_validate(config)


@router.get("/events/stream")
async def automation_events_stream(
    current_user: dict = Depends(get_current_user)
):
    """Server-Sent Events stream for real-time automation updates"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    async def event_generator():
        """Generate SSE events"""
        while True:
            event = AutomationEvent(
                event_type="heartbeat",
                job_id=None,
                job_type=None,
                status="running",
                message="Automation engine active",
                timestamp=datetime.utcnow(),
                data={"doctor_id": doctor_id}
            )
            
            yield f"data: {json.dumps(event.model_dump(mode='json'))}\n\n"
            
            await asyncio.sleep(30)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
