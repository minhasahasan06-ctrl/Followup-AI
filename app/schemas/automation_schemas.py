"""
Pydantic schemas for Lysa Automation Engine API.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class JobType(str, Enum):
    EMAIL_SYNC = "email_sync"
    EMAIL_CLASSIFY = "email_classify"
    EMAIL_AUTO_REPLY = "email_auto_reply"
    EMAIL_FORWARD_URGENT = "email_forward_urgent"
    WHATSAPP_SYNC = "whatsapp_sync"
    WHATSAPP_AUTO_REPLY = "whatsapp_auto_reply"
    WHATSAPP_SEND_TEMPLATE = "whatsapp_send_template"
    APPOINTMENT_REQUEST = "appointment_request"
    APPOINTMENT_BOOK = "appointment_book"
    APPOINTMENT_CANCEL = "appointment_cancel"
    APPOINTMENT_RESCHEDULE = "appointment_reschedule"
    REMINDER_MEDICATION = "reminder_medication"
    REMINDER_APPOINTMENT = "reminder_appointment"
    REMINDER_FOLLOWUP = "reminder_followup"
    REMINDER_NOSHOW = "reminder_noshow"
    CALENDAR_SYNC = "calendar_sync"
    PATIENT_LOOKUP = "patient_lookup"
    DIAGNOSIS_SUMMARY = "diagnosis_summary"
    SOAP_NOTE = "soap_note"
    ICD10_SUGGEST = "icd10_suggest"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    PRESCRIPTION_GENERATE = "prescription_generate"
    DAILY_REPORT = "daily_report"
    WEEKLY_DIGEST = "weekly_digest"
    ALERT_PROCESS = "alert_process"
    PATIENT_MONITOR = "patient_monitor"


class ScheduleFrequency(str, Enum):
    ONCE = "once"
    EVERY_MINUTE = "every_minute"
    EVERY_5_MINUTES = "every_5_minutes"
    EVERY_10_MINUTES = "every_10_minutes"
    EVERY_15_MINUTES = "every_15_minutes"
    EVERY_30_MINUTES = "every_30_minutes"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM_CRON = "custom_cron"


class AutomationJobCreate(BaseModel):
    job_type: JobType
    priority: JobPriority = JobPriority.NORMAL
    patient_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    scheduled_for: Optional[datetime] = None
    idempotency_key: Optional[str] = None


class AutomationJobUpdate(BaseModel):
    status: Optional[JobStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class AutomationJobResponse(BaseModel):
    id: str
    doctor_id: str
    patient_id: Optional[str]
    job_type: str
    priority: str
    status: str
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    attempts: int
    max_attempts: int
    scheduled_for: Optional[datetime]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AutomationJobListResponse(BaseModel):
    jobs: List[AutomationJobResponse]
    total: int
    pending: int
    running: int
    completed: int
    failed: int


class AutomationScheduleCreate(BaseModel):
    name: str
    description: Optional[str] = None
    job_type: JobType
    job_config: Optional[Dict[str, Any]] = None
    frequency: ScheduleFrequency
    cron_expression: Optional[str] = None
    timezone: str = "UTC"
    priority: JobPriority = JobPriority.NORMAL
    is_enabled: bool = True


class AutomationScheduleUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    job_config: Optional[Dict[str, Any]] = None
    frequency: Optional[ScheduleFrequency] = None
    cron_expression: Optional[str] = None
    timezone: Optional[str] = None
    priority: Optional[JobPriority] = None
    is_enabled: Optional[bool] = None


class AutomationScheduleResponse(BaseModel):
    id: str
    doctor_id: str
    name: str
    description: Optional[str]
    job_type: str
    job_config: Optional[Dict[str, Any]]
    frequency: str
    cron_expression: Optional[str]
    timezone: str
    is_enabled: bool
    priority: str
    last_run_at: Optional[datetime]
    next_run_at: Optional[datetime]
    last_run_status: Optional[str]
    run_count: int
    success_count: int
    failure_count: int
    created_at: datetime

    class Config:
        from_attributes = True


class AutomationLogResponse(BaseModel):
    id: str
    job_id: str
    doctor_id: str
    patient_id: Optional[str]
    log_level: str
    message: str
    details: Optional[Dict[str, Any]]
    action_type: Optional[str]
    resource_type: Optional[str]
    resource_id: Optional[str]
    duration_ms: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True


class AutomationStatusResponse(BaseModel):
    """Real-time automation status for dashboard"""
    is_running: bool
    current_job: Optional[AutomationJobResponse]
    jobs_in_queue: int
    jobs_today: int
    jobs_completed_today: int
    jobs_failed_today: int
    
    email_sync_status: str
    email_last_sync: Optional[datetime]
    emails_processed_today: int
    
    whatsapp_sync_status: str
    whatsapp_last_sync: Optional[datetime]
    messages_sent_today: int
    
    calendar_sync_status: str
    calendar_last_sync: Optional[datetime]
    
    appointments_booked_today: int
    reminders_sent_today: int
    
    recent_activities: List[Dict[str, Any]]


class EmailAutomationConfigCreate(BaseModel):
    is_enabled: bool = True
    auto_classify: bool = True
    auto_reply_enabled: bool = False
    forward_urgent_enabled: bool = True
    forward_urgent_to: Optional[str] = None
    auto_reply_template: Optional[str] = None
    auto_reply_conditions: Optional[Dict[str, Any]] = None
    classification_rules: Optional[Dict[str, Any]] = None
    priority_keywords: Optional[List[str]] = None
    sync_frequency_minutes: int = 5


class EmailAutomationConfigResponse(BaseModel):
    id: str
    doctor_id: str
    is_enabled: bool
    auto_classify: bool
    auto_reply_enabled: bool
    forward_urgent_enabled: bool
    forward_urgent_to: Optional[str]
    auto_reply_template: Optional[str]
    sync_frequency_minutes: int
    last_sync_at: Optional[datetime]

    class Config:
        from_attributes = True


class WhatsAppAutomationConfigCreate(BaseModel):
    is_enabled: bool = True
    auto_reply_enabled: bool = False
    greeting_template: Optional[str] = None
    out_of_hours_template: Optional[str] = None
    appointment_confirmation_template: Optional[str] = None
    reminder_template: Optional[str] = None
    business_hours_start: str = "09:00"
    business_hours_end: str = "17:00"
    business_days: List[str] = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    sync_frequency_minutes: int = 5


class WhatsAppAutomationConfigResponse(BaseModel):
    id: str
    doctor_id: str
    is_enabled: bool
    auto_reply_enabled: bool
    greeting_template: Optional[str]
    business_hours_start: str
    business_hours_end: str
    sync_frequency_minutes: int
    last_sync_at: Optional[datetime]

    class Config:
        from_attributes = True


class AppointmentAutomationConfigCreate(BaseModel):
    is_enabled: bool = True
    auto_book_enabled: bool = False
    auto_confirm_enabled: bool = True
    default_duration_minutes: int = 30
    buffer_minutes: int = 15
    available_hours_start: str = "09:00"
    available_hours_end: str = "17:00"
    available_days: List[str] = ["monday", "tuesday", "wednesday", "thursday", "friday"]
    confirmation_email_enabled: bool = True
    confirmation_whatsapp_enabled: bool = False
    reminder_enabled: bool = True
    reminder_hours_before: List[int] = [24, 2]
    calendar_sync_enabled: bool = True


class AppointmentAutomationConfigResponse(BaseModel):
    id: str
    doctor_id: str
    is_enabled: bool
    auto_book_enabled: bool
    auto_confirm_enabled: bool
    default_duration_minutes: int
    available_hours_start: str
    available_hours_end: str
    reminder_enabled: bool
    calendar_sync_enabled: bool

    class Config:
        from_attributes = True


class ReminderAutomationConfigCreate(BaseModel):
    is_enabled: bool = True
    medication_reminders_enabled: bool = True
    appointment_reminders_enabled: bool = True
    followup_reminders_enabled: bool = True
    noshow_followup_enabled: bool = True
    email_enabled: bool = True
    whatsapp_enabled: bool = False
    sms_enabled: bool = False
    quiet_hours_start: str = "21:00"
    quiet_hours_end: str = "08:00"


class ReminderAutomationConfigResponse(BaseModel):
    id: str
    doctor_id: str
    is_enabled: bool
    medication_reminders_enabled: bool
    appointment_reminders_enabled: bool
    followup_reminders_enabled: bool
    email_enabled: bool
    whatsapp_enabled: bool
    sms_enabled: bool

    class Config:
        from_attributes = True


class ClinicalAutomationConfigCreate(BaseModel):
    is_enabled: bool = True
    auto_soap_notes: bool = True
    auto_icd10_suggest: bool = True
    auto_differential_diagnosis: bool = True
    prescription_assist_enabled: bool = True
    require_prescription_approval: bool = True
    use_patient_history: bool = True
    drug_interaction_check: bool = True
    contraindication_alerts: bool = True
    auto_dosage_recommendations: bool = True
    chronic_refill_enabled: bool = False
    chronic_refill_adherence_threshold: int = 80
    chronic_refill_days_before_expiry: int = 7
    chronic_refill_require_approval: bool = True


class ClinicalAutomationConfigUpdate(BaseModel):
    """Partial update schema for clinical automation config"""
    is_enabled: Optional[bool] = None
    auto_soap_notes: Optional[bool] = None
    auto_icd10_suggest: Optional[bool] = None
    auto_differential_diagnosis: Optional[bool] = None
    prescription_assist_enabled: Optional[bool] = None
    require_prescription_approval: Optional[bool] = None
    use_patient_history: Optional[bool] = None
    drug_interaction_check: Optional[bool] = None
    contraindication_alerts: Optional[bool] = None
    auto_dosage_recommendations: Optional[bool] = None
    chronic_refill_enabled: Optional[bool] = None
    chronic_refill_adherence_threshold: Optional[int] = Field(None, ge=0, le=100)
    chronic_refill_days_before_expiry: Optional[int] = Field(None, ge=1, le=90)
    chronic_refill_require_approval: Optional[bool] = None
    
    @model_validator(mode='after')
    def validate_chronic_refill_params(self) -> 'ClinicalAutomationConfigUpdate':
        """Ensure chronic refill parameters have safe values when enabling"""
        if self.chronic_refill_enabled is True:
            if self.chronic_refill_adherence_threshold is None:
                self.chronic_refill_adherence_threshold = 80
            if self.chronic_refill_days_before_expiry is None:
                self.chronic_refill_days_before_expiry = 7
        return self


class ClinicalAutomationConfigResponse(BaseModel):
    id: str
    doctor_id: str
    is_enabled: bool
    auto_soap_notes: bool
    auto_icd10_suggest: bool
    auto_differential_diagnosis: bool
    prescription_assist_enabled: bool
    require_prescription_approval: bool
    use_patient_history: bool = True
    drug_interaction_check: bool = True
    contraindication_alerts: bool = True
    auto_dosage_recommendations: bool = True
    chronic_refill_enabled: bool = False
    chronic_refill_adherence_threshold: int = 80
    chronic_refill_days_before_expiry: int = 7
    chronic_refill_require_approval: bool = True

    class Config:
        from_attributes = True


class RxTemplateCreate(BaseModel):
    name: str
    condition: Optional[str] = None
    medication_name: str
    dosage: str
    frequency: str = "once_daily"
    duration: str = "30 days"
    route: str = "oral"
    instructions: Optional[str] = None
    is_active: bool = True


class RxTemplateUpdate(BaseModel):
    """Partial update schema for Rx templates"""
    name: Optional[str] = None
    condition: Optional[str] = None
    medication_name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    duration: Optional[str] = None
    route: Optional[str] = None
    instructions: Optional[str] = None
    is_active: Optional[bool] = None


class RxTemplateResponse(BaseModel):
    id: str
    doctor_id: str
    name: str
    condition: Optional[str]
    medication_name: str
    dosage: str
    frequency: str
    duration: str
    route: str
    instructions: Optional[str]
    is_active: bool
    usage_count: int = 0

    class Config:
        from_attributes = True


class TriggerJobRequest(BaseModel):
    """Request to manually trigger an automation job"""
    job_type: JobType
    patient_id: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    priority: JobPriority = JobPriority.NORMAL


class AutomationDashboardStats(BaseModel):
    """Aggregated stats for the automation dashboard"""
    total_jobs_today: int
    completed_jobs_today: int
    failed_jobs_today: int
    pending_jobs: int
    running_jobs: int
    
    emails_synced: int
    emails_classified: int
    emails_auto_replied: int
    emails_forwarded: int
    
    whatsapp_messages_received: int
    whatsapp_messages_sent: int
    whatsapp_auto_replies: int
    
    appointments_requested: int
    appointments_booked: int
    appointments_confirmed: int
    
    reminders_medication: int
    reminders_appointment: int
    reminders_followup: int
    
    clinical_soap_notes: int
    clinical_diagnoses: int
    clinical_prescriptions: int
    
    avg_job_duration_ms: float
    success_rate: float
    
    last_email_sync: Optional[datetime]
    last_whatsapp_sync: Optional[datetime]
    last_calendar_sync: Optional[datetime]


class AutomationEvent(BaseModel):
    """Real-time automation event for SSE streaming"""
    event_type: str
    job_id: Optional[str]
    job_type: Optional[str]
    status: str
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]]
