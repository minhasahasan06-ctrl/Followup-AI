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
    ClinicalAutomationConfig, RxTemplate
)
from app.schemas.automation_schemas import (
    AutomationJobCreate, AutomationJobResponse, AutomationJobListResponse,
    AutomationScheduleCreate, AutomationScheduleResponse,
    AutomationStatusResponse, TriggerJobRequest, AutomationDashboardStats,
    EmailAutomationConfigCreate, EmailAutomationConfigResponse,
    WhatsAppAutomationConfigCreate, WhatsAppAutomationConfigResponse,
    AppointmentAutomationConfigCreate, AppointmentAutomationConfigResponse,
    ReminderAutomationConfigCreate, ReminderAutomationConfigResponse,
    ClinicalAutomationConfigCreate, ClinicalAutomationConfigUpdate, ClinicalAutomationConfigResponse,
    AutomationLogResponse, AutomationEvent,
    RxTemplateCreate, RxTemplateUpdate, RxTemplateResponse
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


@router.put("/config/clinical", response_model=ClinicalAutomationConfigResponse)
async def update_clinical_config(
    request: ClinicalAutomationConfigUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update clinical automation configuration with partial updates"""
    import uuid
    
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    config = db.query(ClinicalAutomationConfig).filter(
        ClinicalAutomationConfig.doctor_id == doctor_id
    ).first()
    
    if not config:
        config = ClinicalAutomationConfig(id=str(uuid.uuid4()), doctor_id=doctor_id)
        db.add(config)
    
    update_data = request.model_dump(exclude_unset=True)
    
    chronic_refill_will_be_enabled = update_data.get('chronic_refill_enabled', config.chronic_refill_enabled)
    
    if chronic_refill_will_be_enabled:
        current_threshold = update_data.get('chronic_refill_adherence_threshold')
        current_days = update_data.get('chronic_refill_days_before_expiry')
        
        final_threshold = current_threshold if current_threshold is not None else (config.chronic_refill_adherence_threshold or 80)
        final_days = current_days if current_days is not None else (config.chronic_refill_days_before_expiry or 7)
        
        if not (0 <= final_threshold <= 100):
            raise HTTPException(
                status_code=400,
                detail="Chronic refill adherence threshold must be between 0 and 100"
            )
        if not (1 <= final_days <= 90):
            raise HTTPException(
                status_code=400,
                detail="Chronic refill days before expiry must be between 1 and 90"
            )
        
        config.chronic_refill_adherence_threshold = final_threshold
        config.chronic_refill_days_before_expiry = final_days
    
    for field, value in update_data.items():
        if hasattr(config, field) and value is not None:
            setattr(config, field, value)
    
    config.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(config)
    
    logger.info(f"Updated clinical automation config for doctor {doctor_id}")
    
    return ClinicalAutomationConfigResponse.model_validate(config)


@router.get("/config")
async def get_unified_config(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get unified automation configuration for all channels"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    email_config = db.query(EmailAutomationConfig).filter(
        EmailAutomationConfig.doctor_id == doctor_id
    ).first()
    
    whatsapp_config = db.query(WhatsAppAutomationConfig).filter(
        WhatsAppAutomationConfig.doctor_id == doctor_id
    ).first()
    
    appointment_config = db.query(AppointmentAutomationConfig).filter(
        AppointmentAutomationConfig.doctor_id == doctor_id
    ).first()
    
    return {
        "email": {
            "enabled": email_config.is_enabled if email_config else False,
            "auto_reply": email_config.auto_reply_enabled if email_config else False,
            "auto_classify": email_config.auto_classify if email_config else True,
            "forward_urgent": email_config.forward_urgent_enabled if email_config else True,
            "sync_interval_minutes": email_config.sync_frequency_minutes if email_config else 5
        },
        "whatsapp": {
            "enabled": whatsapp_config.is_enabled if whatsapp_config else False,
            "auto_reply": whatsapp_config.auto_reply_enabled if whatsapp_config else False,
            "business_hours_only": whatsapp_config.business_hours_only if whatsapp_config else True,
            "welcome_message": whatsapp_config.greeting_template if whatsapp_config else "",
            "away_message": whatsapp_config.out_of_hours_template if whatsapp_config else ""
        },
        "calendar": {
            "enabled": appointment_config.is_enabled if appointment_config else False,
            "bidirectional_sync": appointment_config.calendar_sync_enabled if appointment_config else True,
            "sync_interval_minutes": appointment_config.calendar_sync_frequency_minutes if appointment_config else 15,
            "auto_book_appointments": appointment_config.auto_book_enabled if appointment_config else False
        }
    }


@router.patch("/config")
async def update_unified_config(
    request: dict,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update unified automation configuration"""
    import uuid
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    updated = {}
    
    if "email" in request:
        email_data = request["email"]
        config = db.query(EmailAutomationConfig).filter(
            EmailAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config:
            config = EmailAutomationConfig(id=str(uuid.uuid4()), doctor_id=doctor_id)
            db.add(config)
        
        if "enabled" in email_data:
            config.is_enabled = email_data["enabled"]
        if "auto_reply" in email_data:
            config.auto_reply_enabled = email_data["auto_reply"]
        if "auto_classify" in email_data:
            config.auto_classify = email_data["auto_classify"]
        if "forward_urgent" in email_data:
            config.forward_urgent_enabled = email_data["forward_urgent"]
        if "sync_interval_minutes" in email_data:
            config.sync_frequency_minutes = email_data["sync_interval_minutes"]
        
        config.updated_at = datetime.utcnow()
        updated["email"] = True
    
    if "whatsapp" in request:
        whatsapp_data = request["whatsapp"]
        config = db.query(WhatsAppAutomationConfig).filter(
            WhatsAppAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config:
            config = WhatsAppAutomationConfig(id=str(uuid.uuid4()), doctor_id=doctor_id)
            db.add(config)
        
        if "enabled" in whatsapp_data:
            config.is_enabled = whatsapp_data["enabled"]
        if "auto_reply" in whatsapp_data:
            config.auto_reply_enabled = whatsapp_data["auto_reply"]
        if "business_hours_only" in whatsapp_data:
            config.business_hours_only = whatsapp_data["business_hours_only"]
        if "welcome_message" in whatsapp_data:
            config.greeting_template = whatsapp_data["welcome_message"]
        if "away_message" in whatsapp_data:
            config.out_of_hours_template = whatsapp_data["away_message"]
        
        config.updated_at = datetime.utcnow()
        updated["whatsapp"] = True
    
    if "calendar" in request:
        calendar_data = request["calendar"]
        config = db.query(AppointmentAutomationConfig).filter(
            AppointmentAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config:
            config = AppointmentAutomationConfig(id=str(uuid.uuid4()), doctor_id=doctor_id)
            db.add(config)
        
        if "enabled" in calendar_data:
            config.is_enabled = calendar_data["enabled"]
        if "bidirectional_sync" in calendar_data:
            config.calendar_sync_enabled = calendar_data["bidirectional_sync"]
        if "sync_interval_minutes" in calendar_data:
            config.calendar_sync_frequency_minutes = calendar_data["sync_interval_minutes"]
        if "auto_book_appointments" in calendar_data:
            config.auto_book_enabled = calendar_data["auto_book_appointments"]
        
        config.updated_at = datetime.utcnow()
        updated["calendar"] = True
    
    db.commit()
    
    return {
        "success": True,
        "updated": updated,
        "message": "Configuration updated successfully"
    }


@router.post("/engine/start")
async def start_engine(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Start the automation engine"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    await automation_engine.start()
    
    return {
        "success": True,
        "status": "running",
        "message": "Automation engine started successfully"
    }


@router.post("/engine/pause")
async def pause_engine(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Pause the automation engine"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    await automation_engine.stop()
    
    return {
        "success": True,
        "status": "paused",
        "message": "Automation engine paused successfully"
    }


@router.post("/sync/{channel}")
async def trigger_sync(
    channel: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Trigger a manual sync for a specific channel"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    if channel not in ["email", "whatsapp", "calendar"]:
        raise HTTPException(status_code=400, detail=f"Invalid channel: {channel}")
    
    job_type_map = {
        "email": "email_sync",
        "whatsapp": "whatsapp_sync",
        "calendar": "calendar_sync"
    }
    
    job = await automation_engine.enqueue_job(
        db=db,
        doctor_id=doctor_id,
        job_type=job_type_map[channel],
        input_data={"triggered_by": "manual"},
        priority="high"
    )
    
    return {
        "success": True,
        "job_id": job.id,
        "channel": channel,
        "message": f"{channel.capitalize()} sync triggered successfully"
    }


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


@router.get("/rx-templates", response_model=List[RxTemplateResponse])
async def get_rx_templates(
    active_only: bool = Query(True, description="Filter to active templates only"),
    condition: Optional[str] = Query(None, description="Filter by condition"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all prescription templates for the doctor"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    query = db.query(RxTemplate).filter(RxTemplate.doctor_id == doctor_id)
    
    if active_only:
        query = query.filter(RxTemplate.is_active == True)
    
    if condition:
        query = query.filter(RxTemplate.condition.ilike(f"%{condition}%"))
    
    templates = query.order_by(desc(RxTemplate.usage_count)).all()
    
    return [RxTemplateResponse.model_validate(t) for t in templates]


@router.post("/rx-templates", response_model=RxTemplateResponse, status_code=201)
async def create_rx_template(
    request: RxTemplateCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new prescription template"""
    import uuid
    
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    existing = db.query(RxTemplate).filter(
        and_(
            RxTemplate.doctor_id == doctor_id,
            RxTemplate.name == request.name
        )
    ).first()
    
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Template with name '{request.name}' already exists"
        )
    
    template = RxTemplate(
        id=str(uuid.uuid4()),
        doctor_id=doctor_id,
        name=request.name,
        condition=request.condition,
        medication_name=request.medication_name,
        dosage=request.dosage,
        frequency=request.frequency,
        duration=request.duration,
        route=request.route,
        instructions=request.instructions,
        is_active=request.is_active,
        usage_count=0
    )
    
    db.add(template)
    db.commit()
    db.refresh(template)
    
    logger.info(f"Created Rx template '{request.name}' for doctor {doctor_id}")
    
    return RxTemplateResponse.model_validate(template)


@router.get("/rx-templates/{template_id}", response_model=RxTemplateResponse)
async def get_rx_template(
    template_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific prescription template"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    template = db.query(RxTemplate).filter(
        and_(
            RxTemplate.id == template_id,
            RxTemplate.doctor_id == doctor_id
        )
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    return RxTemplateResponse.model_validate(template)


@router.put("/rx-templates/{template_id}", response_model=RxTemplateResponse)
async def update_rx_template(
    template_id: str,
    request: RxTemplateUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a prescription template with partial updates"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    template = db.query(RxTemplate).filter(
        and_(
            RxTemplate.id == template_id,
            RxTemplate.doctor_id == doctor_id
        )
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    update_data = request.model_dump(exclude_unset=True)
    
    if 'name' in update_data and update_data['name'] != template.name:
        name_conflict = db.query(RxTemplate).filter(
            and_(
                RxTemplate.doctor_id == doctor_id,
                RxTemplate.name == update_data['name'],
                RxTemplate.id != template_id
            )
        ).first()
        
        if name_conflict:
            raise HTTPException(
                status_code=400,
                detail=f"Another template with name '{update_data['name']}' already exists"
            )
    
    for field, value in update_data.items():
        if hasattr(template, field) and value is not None:
            setattr(template, field, value)
    
    template.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(template)
    
    logger.info(f"Updated Rx template '{template.name}' for doctor {doctor_id}")
    
    return RxTemplateResponse.model_validate(template)


@router.delete("/rx-templates/{template_id}")
async def delete_rx_template(
    template_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a prescription template"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    template = db.query(RxTemplate).filter(
        and_(
            RxTemplate.id == template_id,
            RxTemplate.doctor_id == doctor_id
        )
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template_name = template.name
    db.delete(template)
    db.commit()
    
    logger.info(f"Deleted Rx template '{template_name}' for doctor {doctor_id}")
    
    return {"success": True, "message": f"Template '{template_name}' deleted"}


@router.post("/rx-templates/{template_id}/use", response_model=RxTemplateResponse)
async def use_rx_template(
    template_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Increment usage count when a template is used"""
    doctor_id = current_user.get("id") or current_user.get("sub")
    
    if not doctor_id:
        raise HTTPException(status_code=401, detail="User ID not found")
    
    template = db.query(RxTemplate).filter(
        and_(
            RxTemplate.id == template_id,
            RxTemplate.doctor_id == doctor_id
        )
    ).first()
    
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    
    template.usage_count = (template.usage_count or 0) + 1
    template.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(template)
    
    return RxTemplateResponse.model_validate(template)
