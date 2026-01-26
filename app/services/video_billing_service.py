"""
Video Billing Service - Phase 12
================================

Production-grade billing service for video consultations with:
- Per-appointment usage tracking
- Per-doctor monthly invoice generation
- Chronological overage allocation
- Scalable design for 1M+ doctors
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from decimal import Decimal
import logging

from sqlalchemy.orm import Session
from sqlalchemy import func, and_

from app.config import settings
from app.models.video_billing_models import (
    DoctorVideoSettings, AppointmentVideo, VideoUsageEvent,
    VideoUsageSession, VideoUsageLedger, DoctorSubscription,
    DoctorMonthlyInvoice
)
from app.services.access_control import HIPAAAuditLogger, PHICategory
from app.services.daily_video_service import VideoUsageCalculator

logger = logging.getLogger(__name__)


class VideoBillingService:
    """
    Production-grade video billing service.
    
    Features:
    - Per-appointment usage ledger updates
    - Monthly invoice generation with overage calculation
    - Chronological overage allocation
    - Efficient queries for 1M+ doctor scale
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_or_create_subscription(self, doctor_id: str) -> DoctorSubscription:
        """Get or create a subscription for a doctor"""
        sub = self.db.query(DoctorSubscription).filter(
            DoctorSubscription.doctor_id == doctor_id
        ).first()
        
        if not sub:
            sub = DoctorSubscription(
                doctor_id=doctor_id,
                plan="TRIAL",
                status="active",
                included_participant_minutes=settings.PLAN_TRIAL_INCLUDED_PM,
                overage_rate_usd_per_pm=Decimal(settings.OVERAGE_RATE_USD),
                period_start=datetime.utcnow()
            )
            self.db.add(sub)
            self.db.commit()
            self.db.refresh(sub)
            logger.info(f"Created TRIAL subscription for doctor {doctor_id}")
        
        return sub
    
    def update_ledger_for_appointment(
        self,
        appointment_id: str,
        doctor_id: str,
        billing_month: str
    ) -> VideoUsageLedger:
        """
        Update or create ledger entry for an appointment.
        Called after webhook processes a participant_left event.
        """
        sessions = self.db.query(VideoUsageSession).filter(
            and_(
                VideoUsageSession.appointment_id == appointment_id,
                VideoUsageSession.left_at.isnot(None)
            )
        ).all()
        
        total_minutes = sum(
            VideoUsageCalculator.calculate_session_minutes(s.duration_seconds or 0)
            for s in sessions
        )
        
        cost_usd = VideoUsageCalculator.calculate_cost(total_minutes)
        
        ledger = self.db.query(VideoUsageLedger).filter(
            VideoUsageLedger.appointment_id == appointment_id
        ).first()
        
        if ledger:
            ledger.participant_minutes = total_minutes
            ledger.cost_usd = cost_usd
            ledger.billing_month = billing_month
        else:
            ledger = VideoUsageLedger(
                appointment_id=appointment_id,
                doctor_id=doctor_id,
                billing_month=billing_month,
                participant_minutes=total_minutes,
                cost_usd=cost_usd
            )
            self.db.add(ledger)
        
        self.db.commit()
        self.db.refresh(ledger)
        
        return ledger
    
    def recompute_doctor_invoice(
        self,
        doctor_id: str,
        billing_month: str
    ) -> DoctorMonthlyInvoice:
        """
        Recompute monthly invoice for a doctor.
        
        Algorithm:
        1. Get subscription for the doctor
        2. Sum participant_minutes from ledger for the month
        3. Calculate overage based on plan
        4. Update or create invoice
        5. Allocate overage across appointments chronologically
        """
        sub = self.get_or_create_subscription(doctor_id)
        
        result = self.db.query(
            func.coalesce(func.sum(VideoUsageLedger.participant_minutes), 0)
        ).filter(
            and_(
                VideoUsageLedger.doctor_id == doctor_id,
                VideoUsageLedger.billing_month == billing_month
            )
        ).scalar()
        
        total_minutes = int(result) if result else 0
        included_minutes = sub.included_participant_minutes or 0
        overage_minutes, overage_cost = VideoUsageCalculator.calculate_overage(
            total_minutes,
            included_minutes,
            sub.overage_rate_usd_per_pm
        )
        
        invoice = self.db.query(DoctorMonthlyInvoice).filter(
            and_(
                DoctorMonthlyInvoice.doctor_id == doctor_id,
                DoctorMonthlyInvoice.billing_month == billing_month
            )
        ).first()
        
        if invoice:
            invoice.total_participant_minutes = total_minutes
            invoice.included_participant_minutes = included_minutes
            invoice.overage_minutes = overage_minutes
            invoice.amount_due_usd = overage_cost
        else:
            invoice = DoctorMonthlyInvoice(
                doctor_id=doctor_id,
                billing_month=billing_month,
                total_participant_minutes=total_minutes,
                included_participant_minutes=included_minutes,
                overage_minutes=overage_minutes,
                amount_due_usd=overage_cost,
                status="pending"
            )
            self.db.add(invoice)
        
        self.db.commit()
        self.db.refresh(invoice)
        
        self._allocate_overage_to_appointments(doctor_id, billing_month, included_minutes, sub.overage_rate_usd_per_pm)
        
        logger.info(f"Recomputed invoice for doctor {doctor_id}, month {billing_month}: "
                   f"{total_minutes} mins, ${overage_cost} overage")
        
        return invoice
    
    def _allocate_overage_to_appointments(
        self,
        doctor_id: str,
        billing_month: str,
        included_minutes: int,
        overage_rate: Decimal
    ):
        """
        Allocate overage across appointments chronologically.
        
        First appointments in the month consume included minutes.
        Later appointments may incur overage charges.
        """
        ledgers = self.db.query(VideoUsageLedger).filter(
            and_(
                VideoUsageLedger.doctor_id == doctor_id,
                VideoUsageLedger.billing_month == billing_month,
                VideoUsageLedger.finalized == False
            )
        ).order_by(VideoUsageLedger.updated_at).all()
        
        remaining_included = included_minutes
        
        for ledger in ledgers:
            appt_minutes = ledger.participant_minutes or 0
            
            if remaining_included >= appt_minutes:
                ledger.overage_billable_minutes = 0
                ledger.billed_to_doctor_usd = Decimal("0")
                remaining_included -= appt_minutes
            else:
                overage_for_appt = appt_minutes - remaining_included
                ledger.overage_billable_minutes = overage_for_appt
                ledger.billed_to_doctor_usd = Decimal(overage_for_appt) * overage_rate
                remaining_included = 0
        
        self.db.commit()
    
    def finalize_month(self, billing_month: str) -> int:
        """
        Mark all ledger entries for a month as finalized.
        Called by end-of-month scheduled job.
        
        Returns:
            Number of entries finalized
        """
        now = datetime.utcnow()
        
        result = self.db.query(VideoUsageLedger).filter(
            and_(
                VideoUsageLedger.billing_month == billing_month,
                VideoUsageLedger.finalized == False
            )
        ).update({
            "finalized": True,
            "finalized_at": now
        })
        
        self.db.commit()
        
        logger.info(f"Finalized {result} ledger entries for {billing_month}")
        return result
    
    def get_doctor_usage_summary(
        self,
        doctor_id: str,
        billing_month: str = None
    ) -> Dict[str, Any]:
        """Get usage summary for a doctor"""
        if billing_month is None:
            billing_month = VideoUsageCalculator.get_billing_month()
        
        sub = self.get_or_create_subscription(doctor_id)
        
        invoice = self.db.query(DoctorMonthlyInvoice).filter(
            and_(
                DoctorMonthlyInvoice.doctor_id == doctor_id,
                DoctorMonthlyInvoice.billing_month == billing_month
            )
        ).first()
        
        if not invoice:
            result = self.db.query(
                func.coalesce(func.sum(VideoUsageLedger.participant_minutes), 0)
            ).filter(
                and_(
                    VideoUsageLedger.doctor_id == doctor_id,
                    VideoUsageLedger.billing_month == billing_month
                )
            ).scalar()
            
            total_minutes = int(result) if result else 0
            included_minutes = sub.included_participant_minutes or 0
            overage_minutes = max(0, total_minutes - included_minutes)
            overage_cost = Decimal(overage_minutes) * (sub.overage_rate_usd_per_pm or Decimal("0.008"))
        else:
            total_minutes = invoice.total_participant_minutes or 0
            included_minutes = invoice.included_participant_minutes or 0
            overage_minutes = invoice.overage_minutes or 0
            overage_cost = invoice.amount_due_usd or Decimal("0")
        
        return {
            "billing_month": billing_month,
            "plan": sub.plan,
            "status": sub.status,
            "total_participant_minutes": total_minutes,
            "included_participant_minutes": included_minutes,
            "used_percentage": round((total_minutes / max(included_minutes, 1)) * 100, 1),
            "overage_minutes": overage_minutes,
            "amount_due_usd": float(overage_cost),
            "invoice_status": invoice.status if invoice else "not_generated"
        }
    
    def get_doctor_invoices(
        self,
        doctor_id: str,
        limit: int = 12
    ) -> List[Dict[str, Any]]:
        """Get recent invoices for a doctor"""
        invoices = self.db.query(DoctorMonthlyInvoice).filter(
            DoctorMonthlyInvoice.doctor_id == doctor_id
        ).order_by(
            DoctorMonthlyInvoice.billing_month.desc()
        ).limit(limit).all()
        
        return [
            {
                "billing_month": inv.billing_month,
                "total_participant_minutes": inv.total_participant_minutes,
                "included_participant_minutes": inv.included_participant_minutes,
                "overage_minutes": inv.overage_minutes,
                "amount_due_usd": float(inv.amount_due_usd or 0),
                "status": inv.status,
                "generated_at": inv.generated_at.isoformat() if inv.generated_at else None,
                "paid_at": inv.paid_at.isoformat() if inv.paid_at else None
            }
            for inv in invoices
        ]


class VideoSettingsService:
    """Manage doctor video settings"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_settings(self, doctor_id: str) -> Optional[DoctorVideoSettings]:
        """Get video settings for a doctor"""
        return self.db.query(DoctorVideoSettings).filter(
            DoctorVideoSettings.doctor_id == doctor_id
        ).first()
    
    def get_or_create_settings(self, doctor_id: str) -> DoctorVideoSettings:
        """Get or create default settings for a doctor"""
        settings_obj = self.get_settings(doctor_id)
        
        if not settings_obj:
            settings_obj = DoctorVideoSettings(
                doctor_id=doctor_id,
                allow_external_video=False,
                default_video_provider="daily"
            )
            self.db.add(settings_obj)
            self.db.commit()
            self.db.refresh(settings_obj)
        
        return settings_obj
    
    def update_settings(
        self,
        doctor_id: str,
        allow_external_video: bool = None,
        zoom_join_url: str = None,
        meet_join_url: str = None,
        default_video_provider: str = None,
        enable_recording: bool = None,
        enable_chat: bool = None,
        max_participants: int = None
    ) -> DoctorVideoSettings:
        """Update video settings for a doctor"""
        from app.services.daily_video_service import ExternalVideoProvider
        
        settings_obj = self.get_or_create_settings(doctor_id)
        
        if allow_external_video is not None:
            if allow_external_video:
                has_zoom = zoom_join_url or settings_obj.zoom_join_url
                has_meet = meet_join_url or settings_obj.meet_join_url
                if not has_zoom and not has_meet:
                    raise ValueError("At least one external video URL required when enabling external video")
            settings_obj.allow_external_video = allow_external_video
        
        if zoom_join_url is not None:
            if zoom_join_url and not ExternalVideoProvider.validate_zoom_url(zoom_join_url):
                raise ValueError("Invalid Zoom URL format")
            settings_obj.zoom_join_url = zoom_join_url
        
        if meet_join_url is not None:
            if meet_join_url and not ExternalVideoProvider.validate_meet_url(meet_join_url):
                raise ValueError("Invalid Google Meet URL format")
            settings_obj.meet_join_url = meet_join_url
        
        if default_video_provider is not None:
            if default_video_provider not in ["daily", "zoom", "meet"]:
                raise ValueError("Invalid video provider")
            settings_obj.default_video_provider = default_video_provider
        
        if enable_recording is not None:
            settings_obj.enable_recording = enable_recording
        
        if enable_chat is not None:
            settings_obj.enable_chat = enable_chat
        
        if max_participants is not None:
            settings_obj.max_participants = max(2, min(max_participants, 10))
        
        self.db.commit()
        self.db.refresh(settings_obj)
        
        return settings_obj


class AppointmentVideoService:
    """Manage per-appointment video configuration"""
    
    def __init__(self, db: Session):
        self.db = db
        self.settings_service = VideoSettingsService(db)
    
    def get_config(self, appointment_id: str) -> Optional[AppointmentVideo]:
        """Get video config for an appointment"""
        return self.db.query(AppointmentVideo).filter(
            AppointmentVideo.appointment_id == appointment_id
        ).first()
    
    def configure_appointment(
        self,
        appointment_id: str,
        doctor_id: str,
        video_provider: str = "daily"
    ) -> AppointmentVideo:
        """Configure video for an appointment"""
        from app.services.daily_video_service import ExternalVideoProvider
        
        doctor_settings = self.settings_service.get_or_create_settings(doctor_id)
        
        if video_provider in ["zoom", "meet"]:
            if not doctor_settings.allow_external_video:
                raise ValueError("External video not enabled for this doctor")
            
            if video_provider == "zoom":
                if not doctor_settings.zoom_join_url:
                    raise ValueError("Zoom URL not configured")
                external_url = doctor_settings.zoom_join_url
            else:
                if not doctor_settings.meet_join_url:
                    raise ValueError("Google Meet URL not configured")
                external_url = doctor_settings.meet_join_url
        else:
            video_provider = "daily"
            external_url = None
        
        config = self.get_config(appointment_id)
        
        if config:
            config.video_provider = video_provider
            config.external_join_url = external_url
        else:
            config = AppointmentVideo(
                appointment_id=appointment_id,
                video_provider=video_provider,
                external_join_url=external_url
            )
            self.db.add(config)
        
        self.db.commit()
        self.db.refresh(config)
        
        return config
