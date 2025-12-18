"""
Webhook Router for Assistant Lysa Automation

Production-grade webhook endpoints for:
- Gmail Push Notifications (Pub/Sub)
- WhatsApp Cloud API webhooks
- HIPAA-compliant audit logging
"""

import os
import time
import json
import base64
import logging
import hmac
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any
from fastapi import APIRouter, Request, HTTPException, Query, BackgroundTasks, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.database import get_db
from app.models.whatsapp import (
    WhatsAppConversation, WhatsAppMessage, 
    WhatsAppWebhookLog, GmailWebhookLog
)
from app.models.automation_models import (
    WhatsAppAutomationConfig, EmailAutomationConfig, AutomationJob
)
from app.models.calendar_sync import GmailSync
from app.services.email_automation_service import EmailAutomationService
from app.services.whatsapp_automation_service import WhatsAppAutomationService
from app.services.automation_engine import automation_engine

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/webhooks",
    tags=["webhooks"]
)


@router.post("/gmail/push")
async def gmail_push_notification(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Handle Gmail Push Notification from Google Cloud Pub/Sub.
    
    This endpoint receives notifications when new emails arrive in 
    connected Gmail accounts. It triggers the email sync and 
    classification pipeline for real-time email processing.
    
    Headers from Pub/Sub:
    - Authorization: Bearer token (for verification)
    - X-Goog-Resource-State: Type of notification
    """
    start_time = time.time()
    
    try:
        body = await request.json()
        
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            logger.warning("Gmail push notification without authorization header")
        
        if 'message' in body:
            pubsub_message = body['message']
            data = pubsub_message.get('data', '')
            
            if data:
                decoded_data = base64.b64decode(data).decode('utf-8')
                message_data = json.loads(decoded_data)
            else:
                message_data = {}
            
            email_address = message_data.get('emailAddress', '')
            history_id = message_data.get('historyId', '')
            
            gmail_sync = db.query(GmailSync).filter(
                GmailSync.gmail_address == email_address
            ).first()
            
            if gmail_sync is not None and gmail_sync.sync_enabled is True:
                doctor_id = str(gmail_sync.doctor_id)
                
                job = await automation_engine.enqueue_job(
                    db=db,
                    doctor_id=doctor_id,
                    job_type="email_sync",
                    input_data={
                        "history_id": history_id,
                        "triggered_by": "gmail_push",
                        "query": "is:unread",
                        "max_results": 20
                    },
                    priority="high"
                )
                
                webhook_log = GmailWebhookLog(
                    doctor_id=doctor_id,
                    event_type="push_notification",
                    history_id=history_id,
                    request_body=body,
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
                db.add(webhook_log)
                db.commit()
                
                logger.info(f"Gmail push notification processed for {email_address}, job {job.id}")
                
                return {"success": True, "job_id": job.id}
            else:
                logger.warning(f"Gmail push notification for unregistered email: {email_address}")
                
                webhook_log = GmailWebhookLog(
                    event_type="push_notification_ignored",
                    request_body=body,
                    error=f"Email not registered: {email_address}",
                    processing_time_ms=int((time.time() - start_time) * 1000)
                )
                db.add(webhook_log)
                db.commit()
                
                return {"success": False, "error": "Email not registered"}
        
        return {"success": True, "message": "No message data"}
        
    except Exception as e:
        logger.error(f"Gmail webhook error: {e}")
        
        webhook_log = GmailWebhookLog(
            event_type="push_notification_error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        db.add(webhook_log)
        db.commit()
        
        return {"success": False, "error": str(e)}


@router.get("/whatsapp")
async def whatsapp_verify(
    hub_mode: Optional[str] = Query(None, alias="hub.mode"),
    hub_challenge: Optional[str] = Query(None, alias="hub.challenge"),
    hub_verify_token: Optional[str] = Query(None, alias="hub.verify_token")
):
    """
    WhatsApp Webhook Verification Endpoint.
    
    Meta sends a GET request to verify webhook ownership.
    We respond with the challenge token if verification token matches.
    """
    expected_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "lysa_webhook_verify_token")
    
    if hub_mode == "subscribe" and hub_verify_token == expected_token:
        logger.info("WhatsApp webhook verified successfully")
        return PlainTextResponse(content=hub_challenge, status_code=200)
    
    logger.warning(f"WhatsApp webhook verification failed: mode={hub_mode}, token mismatch")
    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Handle incoming WhatsApp Cloud API webhook events.
    
    Processes:
    - Incoming messages (text, media, interactive)
    - Message status updates (sent, delivered, read)
    - Triggers auto-reply when configured
    
    Webhook Structure:
    {
        "object": "whatsapp_business_account",
        "entry": [{
            "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
            "changes": [{
                "value": {
                    "messaging_product": "whatsapp",
                    "metadata": {"display_phone_number": "...", "phone_number_id": "..."},
                    "contacts": [...],
                    "messages": [...],
                    "statuses": [...]
                },
                "field": "messages"
            }]
        }]
    }
    """
    start_time = time.time()
    
    try:
        body = await request.json()
        
        x_hub_signature = request.headers.get("X-Hub-Signature-256", "")
        if x_hub_signature:
            app_secret = os.getenv("WHATSAPP_APP_SECRET", "")
            if app_secret:
                body_bytes = await request.body()
                expected_signature = "sha256=" + hmac.new(
                    app_secret.encode('utf-8'),
                    body_bytes,
                    hashlib.sha256
                ).hexdigest()
                
                if not hmac.compare_digest(x_hub_signature, expected_signature):
                    logger.warning("WhatsApp webhook signature verification failed")
                    raise HTTPException(status_code=403, detail="Invalid signature")
        
        if body.get("object") != "whatsapp_business_account":
            return {"success": True, "message": "Non-WhatsApp event ignored"}
        
        results = {
            "messages_received": 0,
            "messages_stored": 0,
            "auto_replies_sent": 0,
            "statuses_processed": 0
        }
        
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id")
                
                messages = value.get("messages", [])
                contacts = value.get("contacts", [])
                statuses = value.get("statuses", [])
                
                contact_map = {c.get("wa_id"): c for c in contacts}
                
                for message in messages:
                    results["messages_received"] += 1
                    
                    from_number = message.get("from")
                    message_id = message.get("id")
                    message_type = message.get("type")
                    timestamp = message.get("timestamp")
                    
                    content = ""
                    media_url = None
                    media_type_value = None
                    
                    if message_type == "text":
                        content = message.get("text", {}).get("body", "")
                    elif message_type == "image":
                        media_info = message.get("image", {})
                        content = media_info.get("caption", "[Image]")
                        media_url = media_info.get("id")
                        media_type_value = "image"
                    elif message_type == "audio":
                        content = "[Audio message]"
                        media_url = message.get("audio", {}).get("id")
                        media_type_value = "audio"
                    elif message_type == "document":
                        doc_info = message.get("document", {})
                        content = doc_info.get("caption", f"[Document: {doc_info.get('filename', 'Unknown')}]")
                        media_url = doc_info.get("id")
                        media_type_value = "document"
                    elif message_type == "interactive":
                        interactive = message.get("interactive", {})
                        if interactive.get("type") == "button_reply":
                            content = interactive.get("button_reply", {}).get("title", "")
                        elif interactive.get("type") == "list_reply":
                            content = interactive.get("list_reply", {}).get("title", "")
                    
                    contact_info = contact_map.get(from_number, {})
                    contact_name = contact_info.get("profile", {}).get("name", "Unknown")
                    
                    doctor_config = db.query(WhatsAppAutomationConfig).filter(
                        WhatsAppAutomationConfig.is_enabled == True
                    ).first()
                    
                    if doctor_config is None:
                        continue
                    
                    doctor_id = str(doctor_config.doctor_id)
                    
                    conversation = db.query(WhatsAppConversation).filter(
                        and_(
                            WhatsAppConversation.doctor_id == doctor_id,
                            WhatsAppConversation.phone_number == from_number
                        )
                    ).first()
                    
                    if not conversation:
                        conversation = WhatsAppConversation(
                            doctor_id=doctor_id,
                            phone_number=from_number,
                            patient_name=contact_name,
                            status="active",
                            message_count=0,
                            unread_count=0,
                            auto_reply_enabled=True
                        )
                        db.add(conversation)
                        db.flush()
                    
                    existing_msg = db.query(WhatsAppMessage).filter(
                        WhatsAppMessage.whatsapp_message_id == message_id
                    ).first()
                    
                    if not existing_msg:
                        received_dt = datetime.fromtimestamp(int(timestamp)) if timestamp else datetime.utcnow()
                        
                        wa_message = WhatsAppMessage(
                            conversation_id=conversation.id,
                            whatsapp_message_id=message_id,
                            direction="inbound",
                            message_type=message_type,
                            content=content[:5000] if content else "",
                            media_url=media_url,
                            media_type=media_type_value,
                            from_number=from_number,
                            status="received",
                            received_at=received_dt,
                            metadata={"raw_message": message}
                        )
                        db.add(wa_message)
                        
                        conversation.message_count += 1
                        conversation.unread_count += 1
                        conversation.is_read = False
                        conversation.last_message_at = received_dt
                        conversation.last_message_preview = content[:200] if content else f"[{message_type}]"
                        conversation.last_message_direction = "inbound"
                        
                        if str(conversation.patient_name) == "Unknown" and contact_name != "Unknown":
                            conversation.patient_name = contact_name
                        
                        db.commit()
                        results["messages_stored"] += 1
                        
                        if doctor_config.auto_reply_enabled is True and content:
                            job = await automation_engine.enqueue_job(
                                db=db,
                                doctor_id=doctor_id,
                                job_type="whatsapp_auto_reply",
                                input_data={
                                    "phone_number": from_number,
                                    "message": content,
                                    "conversation_id": conversation.id,
                                    "message_id": wa_message.id
                                },
                                priority="high"
                            )
                            logger.info(f"Auto-reply job queued: {job.id}")
                            results["auto_replies_sent"] += 1
                
                for status in statuses:
                    status_id = status.get("id")
                    status_value = status.get("status")
                    recipient_id = status.get("recipient_id")
                    timestamp_str = status.get("timestamp")
                    
                    msg = db.query(WhatsAppMessage).filter(
                        WhatsAppMessage.whatsapp_message_id == status_id
                    ).first()
                    
                    if msg:
                        msg.status = status_value
                        msg.status_updated_at = datetime.utcnow()
                        
                        if status_value == "delivered":
                            msg.delivered_at = datetime.fromtimestamp(int(timestamp_str)) if timestamp_str else datetime.utcnow()
                        elif status_value == "read":
                            msg.read_at = datetime.fromtimestamp(int(timestamp_str)) if timestamp_str else datetime.utcnow()
                        elif status_value == "failed":
                            errors = status.get("errors", [])
                            if errors:
                                msg.error_code = errors[0].get("code")
                                msg.error_message = errors[0].get("message")
                        
                        db.commit()
                    
                    results["statuses_processed"] += 1
        
        processing_time = int((time.time() - start_time) * 1000)
        
        webhook_log = WhatsAppWebhookLog(
            event_type="message_webhook",
            request_body=body,
            response_status=200,
            messages_processed=results["messages_received"],
            auto_replies_sent=results["auto_replies_sent"],
            processing_time_ms=processing_time
        )
        db.add(webhook_log)
        db.commit()
        
        logger.info(f"WhatsApp webhook processed: {results}")
        
        return {"success": True, **results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"WhatsApp webhook error: {e}")
        
        webhook_log = WhatsAppWebhookLog(
            event_type="webhook_error",
            error=str(e),
            processing_time_ms=int((time.time() - start_time) * 1000)
        )
        db.add(webhook_log)
        db.commit()
        
        return {"success": False, "error": str(e)}


@router.post("/gmail/setup-push")
async def setup_gmail_push_notifications(
    doctor_id: str,
    db: Session = Depends(get_db)
):
    """
    Set up Gmail Push Notifications for a doctor's Gmail account.
    
    This creates a watch on the Gmail inbox to receive real-time
    notifications via Google Cloud Pub/Sub.
    
    Requires:
    - GOOGLE_CLOUD_PROJECT_ID
    - GOOGLE_PUBSUB_TOPIC
    - Gmail account connected and authorized
    """
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    
    gmail_sync = db.query(GmailSync).filter(
        GmailSync.doctor_id == doctor_id
    ).first()
    
    if not gmail_sync or not gmail_sync.sync_enabled:
        raise HTTPException(status_code=400, detail="Gmail not connected")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    topic_name = os.getenv("GOOGLE_PUBSUB_TOPIC", "gmail-push-notifications")
    
    if not project_id:
        raise HTTPException(status_code=500, detail="Google Cloud project not configured")
    
    try:
        creds = Credentials(
            token=gmail_sync.access_token,
            refresh_token=gmail_sync.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.getenv("GOOGLE_CLIENT_ID"),
            client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
        )
        
        service = build('gmail', 'v1', credentials=creds)
        
        topic = f"projects/{project_id}/topics/{topic_name}"
        
        watch_result = service.users().watch(
            userId='me',
            body={
                'topicName': topic,
                'labelIds': ['INBOX']
            }
        ).execute()
        
        history_id = watch_result.get('historyId')
        expiration = watch_result.get('expiration')
        
        gmail_sync.history_id = history_id
        gmail_sync.watch_expiration = datetime.fromtimestamp(int(expiration) / 1000) if expiration else None
        db.commit()
        
        logger.info(f"Gmail push notifications set up for doctor {doctor_id}")
        
        return {
            "success": True,
            "history_id": history_id,
            "expiration": expiration
        }
        
    except Exception as e:
        logger.error(f"Gmail push setup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_webhook_status(
    db: Session = Depends(get_db)
):
    """
    Get webhook health status and recent activity.
    """
    now = datetime.utcnow()
    
    recent_gmail = db.query(GmailWebhookLog).filter(
        GmailWebhookLog.created_at >= datetime.utcnow().replace(hour=0, minute=0)
    ).count()
    
    recent_whatsapp = db.query(WhatsAppWebhookLog).filter(
        WhatsAppWebhookLog.created_at >= datetime.utcnow().replace(hour=0, minute=0)
    ).count()
    
    gmail_errors = db.query(GmailWebhookLog).filter(
        and_(
            GmailWebhookLog.created_at >= datetime.utcnow().replace(hour=0, minute=0),
            GmailWebhookLog.error != None
        )
    ).count()
    
    whatsapp_errors = db.query(WhatsAppWebhookLog).filter(
        and_(
            WhatsAppWebhookLog.created_at >= datetime.utcnow().replace(hour=0, minute=0),
            WhatsAppWebhookLog.error != None
        )
    ).count()
    
    gmail_connected = db.query(GmailSync).filter(
        GmailSync.sync_enabled == True
    ).first() is not None
    
    last_gmail_log = db.query(GmailWebhookLog).order_by(
        GmailWebhookLog.created_at.desc()
    ).first()
    
    last_whatsapp_log = db.query(WhatsAppWebhookLog).order_by(
        WhatsAppWebhookLog.created_at.desc()
    ).first()
    
    whatsapp_connected = db.query(WhatsAppAutomationConfig).filter(
        WhatsAppAutomationConfig.is_enabled == True
    ).first() is not None
    
    whatsapp_verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN") is not None
    
    return {
        "status": "healthy",
        "gmail": {
            "connected": gmail_connected,
            "events_today": recent_gmail,
            "errors_today": gmail_errors,
            "last_notification": last_gmail_log.created_at.isoformat() if last_gmail_log else None
        },
        "whatsapp": {
            "connected": whatsapp_connected,
            "verify_token_set": whatsapp_verify_token,
            "events_today": recent_whatsapp,
            "errors_today": whatsapp_errors,
            "last_webhook": last_whatsapp_log.created_at.isoformat() if last_whatsapp_log else None
        },
        "timestamp": now.isoformat()
    }


# =============================================================================
# DAILY.CO VIDEO WEBHOOKS - Phase 12
# =============================================================================

@router.post("/daily")
async def handle_daily_webhook(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    Handle Daily.co webhook events for video usage tracking.
    
    Events processed:
    - meeting.participant-joined: Create open session
    - meeting.participant-left: Close session, update billing ledger
    
    Security:
    - HMAC signature validation (if DAILY_WEBHOOK_SECRET set)
    - Idempotent processing via event_id
    """
    from app.services.daily_video_service import DailyVideoService, VideoUsageCalculator
    from app.services.video_billing_service import VideoBillingService
    from app.models.video_billing_models import VideoUsageEvent, VideoUsageSession, AppointmentVideo
    from fastapi import Header
    
    raw_body = await request.body()
    
    daily_service = DailyVideoService()
    
    signature = request.headers.get("X-Webhook-Signature")
    if signature:
        if not daily_service.validate_webhook_signature(raw_body, signature):
            logger.warning("Invalid Daily webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    
    event_type = body.get("type", "")
    event_id = body.get("id")
    event_ts = body.get("event_ts", 0)
    payload = body.get("payload", {})
    
    if event_id:
        existing = db.query(VideoUsageEvent).filter(
            VideoUsageEvent.event_id == event_id
        ).first()
        
        if existing:
            logger.info(f"Duplicate Daily event ignored: {event_id}")
            return {"status": "duplicate", "event_id": event_id}
    
    room_name = payload.get("room_name", "")
    appointment_id = DailyVideoService.extract_appointment_id(room_name)
    
    if not appointment_id:
        logger.warning(f"Could not extract appointment_id from room: {room_name}")
        return {"status": "ignored", "reason": "unknown_room"}
    
    doctor_id = "unknown"
    
    def determine_role(p: Dict) -> str:
        user_name = p.get("user_name", "").lower()
        is_owner = p.get("is_owner", False)
        return "doctor" if is_owner or "doctor" in user_name else "patient"
    
    event = VideoUsageEvent(
        appointment_id=appointment_id,
        doctor_id=doctor_id,
        provider="daily",
        event_type=event_type,
        event_id=event_id,
        participant_id=payload.get("participant_id", "unknown"),
        participant_name=payload.get("user_name"),
        participant_role=determine_role(payload),
        event_ts=datetime.fromtimestamp(event_ts) if event_ts else datetime.utcnow(),
        payload=payload
    )
    db.add(event)
    db.commit()
    
    if event_type == "meeting.participant-joined":
        session = VideoUsageSession(
            appointment_id=appointment_id,
            doctor_id=doctor_id,
            participant_id=payload.get("participant_id", "unknown"),
            participant_role=determine_role(payload),
            joined_at=datetime.fromtimestamp(event_ts) if event_ts else datetime.utcnow()
        )
        db.add(session)
        db.commit()
        
    elif event_type == "meeting.participant-left":
        participant_id = payload.get("participant_id", "unknown")
        left_at = datetime.fromtimestamp(event_ts) if event_ts else datetime.utcnow()
        
        session = db.query(VideoUsageSession).filter(
            and_(
                VideoUsageSession.appointment_id == appointment_id,
                VideoUsageSession.participant_id == participant_id,
                VideoUsageSession.left_at.is_(None)
            )
        ).order_by(VideoUsageSession.joined_at.desc()).first()
        
        if session:
            session.left_at = left_at
            session.duration_seconds = int((left_at - session.joined_at).total_seconds())
            session.billing_month = VideoUsageCalculator.get_billing_month(left_at)
            db.commit()
            
            billing_service = VideoBillingService(db)
            billing_service.update_ledger_for_appointment(
                appointment_id=appointment_id,
                doctor_id=doctor_id,
                billing_month=session.billing_month
            )
    
    logger.info(f"Processed Daily webhook: {event_type} for appointment {appointment_id}")
    
    return {"status": "processed", "event_type": event_type, "appointment_id": appointment_id}


@router.get("/daily/test")
async def test_daily_webhook():
    """Test endpoint to verify Daily webhook configuration"""
    return {
        "status": "ok",
        "message": "Daily webhook endpoint is configured",
        "expected_events": [
            "meeting.participant-joined",
            "meeting.participant-left",
            "room.deleted"
        ]
    }
