"""
WhatsApp Business Automation Service for Assistant Lysa

Production-grade WhatsApp automation with:
- WhatsApp Cloud API integration
- Message template management
- Auto-reply with AI
- Patient conversation sync
- HIPAA-compliant messaging
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import httpx

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

import openai

from app.models.automation_models import WhatsAppAutomationConfig

logger = logging.getLogger(__name__)

openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

WHATSAPP_API_URL = "https://graph.facebook.com/v18.0"


class WhatsAppAutomationService:
    """Handles all WhatsApp Business automation tasks"""
    
    @staticmethod
    async def sync_messages(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sync messages from WhatsApp Business API.
        
        Note: WhatsApp uses webhooks for real-time messages.
        This method handles any queued/cached messages.
        """
        config = db.query(WhatsAppAutomationConfig).filter(
            WhatsAppAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.is_enabled:
            return {
                "success": False,
                "error": "WhatsApp automation not enabled",
                "messages_synced": 0
            }
        
        config.last_sync_at = datetime.utcnow()
        db.commit()
        
        return {
            "success": True,
            "messages_synced": 0,
            "note": "WhatsApp uses webhooks for real-time message delivery",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    async def auto_reply(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate and send an AI-powered auto-reply via WhatsApp.
        """
        phone_number = input_data.get("phone_number")
        incoming_message = input_data.get("message")
        
        if not phone_number or not incoming_message:
            return {
                "success": False,
                "error": "phone_number and message required"
            }
        
        config = db.query(WhatsAppAutomationConfig).filter(
            WhatsAppAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.auto_reply_enabled:
            return {
                "success": False,
                "error": "WhatsApp auto-reply not enabled"
            }
        
        now = datetime.utcnow()
        current_hour = now.hour
        
        try:
            start_hour = int(config.business_hours_start.split(':')[0])
            end_hour = int(config.business_hours_end.split(':')[0])
        except:
            start_hour = 9
            end_hour = 17
        
        is_business_hours = start_hour <= current_hour < end_hour
        current_day = now.strftime('%A').lower()
        is_business_day = current_day in (config.business_days or [])
        
        if not is_business_hours or not is_business_day:
            reply_text = config.out_of_hours_template or (
                "Thank you for your message. Our office is currently closed. "
                "We will respond during business hours: "
                f"{config.business_hours_start} - {config.business_hours_end}, "
                f"Monday - Friday. For medical emergencies, please call 911."
            )
        else:
            prompt = f"""Generate a brief, professional WhatsApp reply for a medical office.

Patient message: {incoming_message}

Guidelines:
1. Be professional and empathetic
2. Do NOT provide medical advice
3. Do NOT reference specific health conditions
4. Keep response under 160 characters for SMS-like readability
5. Guide them to call the office or book an appointment if needed

Reply text only:"""

            try:
                response = await openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a professional medical office WhatsApp assistant. Be helpful but never give medical advice."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=100
                )
                
                reply_text = response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"AI reply generation error: {e}")
                reply_text = config.greeting_template or (
                    "Thank you for contacting our office. "
                    "We will respond to your message shortly. "
                    "For urgent matters, please call us directly."
                )
        
        send_result = await WhatsAppAutomationService._send_message(
            db, doctor_id, phone_number, reply_text
        )
        
        return {
            "success": send_result.get("success", False),
            "phone_number": phone_number,
            "reply_text": reply_text,
            "is_business_hours": is_business_hours and is_business_day,
            "message_id": send_result.get("message_id")
        }
    
    @staticmethod
    async def send_template(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Send a WhatsApp message template.
        
        Templates are pre-approved messages for:
        - Appointment confirmations
        - Appointment reminders
        - Medication reminders
        - Follow-up reminders
        """
        phone_number = input_data.get("phone_number")
        template_type = input_data.get("template_type")
        template_data = input_data.get("template_data", {})
        
        if not phone_number or not template_type:
            return {
                "success": False,
                "error": "phone_number and template_type required"
            }
        
        config = db.query(WhatsAppAutomationConfig).filter(
            WhatsAppAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.is_enabled:
            return {
                "success": False,
                "error": "WhatsApp not enabled"
            }
        
        templates = {
            "appointment_confirmation": config.appointment_confirmation_template or (
                "Your appointment has been confirmed for {date} at {time}. "
                "Location: {location}. Reply YES to confirm or call us to reschedule."
            ),
            "appointment_reminder": config.reminder_template or (
                "Reminder: You have an appointment tomorrow, {date} at {time}. "
                "Please arrive 10 minutes early. Reply YES to confirm."
            ),
            "medication_reminder": (
                "Medication Reminder: It's time to take your {medication}. "
                "If you have any questions, please contact your doctor."
            ),
            "followup_reminder": (
                "Hi! This is a friendly reminder to schedule your follow-up appointment. "
                "Please call our office or reply to this message."
            )
        }
        
        template = templates.get(template_type)
        if not template:
            return {
                "success": False,
                "error": f"Unknown template type: {template_type}"
            }
        
        try:
            message_text = template.format(**template_data)
        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing template variable: {e}"
            }
        
        send_result = await WhatsAppAutomationService._send_message(
            db, doctor_id, phone_number, message_text
        )
        
        return {
            "success": send_result.get("success", False),
            "phone_number": phone_number,
            "template_type": template_type,
            "message_text": message_text,
            "message_id": send_result.get("message_id")
        }
    
    @staticmethod
    async def _send_message(
        db: Session,
        doctor_id: str,
        phone_number: str,
        message_text: str
    ) -> Dict[str, Any]:
        """
        Send a WhatsApp message using the Cloud API.
        
        Requires WhatsApp Business API credentials:
        - WHATSAPP_ACCESS_TOKEN
        - WHATSAPP_PHONE_NUMBER_ID
        """
        access_token = os.getenv("WHATSAPP_ACCESS_TOKEN")
        phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        
        if not access_token or not phone_number_id:
            logger.warning("WhatsApp API credentials not configured")
            return {
                "success": False,
                "error": "WhatsApp API credentials not configured",
                "simulated": True
            }
        
        formatted_phone = phone_number.replace("+", "").replace("-", "").replace(" ", "")
        
        url = f"{WHATSAPP_API_URL}/{phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": formatted_phone,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": message_text
            }
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    message_id = data.get("messages", [{}])[0].get("id")
                    
                    logger.info(f"WhatsApp message sent to {phone_number}: {message_id}")
                    
                    return {
                        "success": True,
                        "message_id": message_id
                    }
                else:
                    error_data = response.json()
                    logger.error(f"WhatsApp API error: {error_data}")
                    return {
                        "success": False,
                        "error": error_data.get("error", {}).get("message", "Unknown error")
                    }
                    
        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def handle_webhook(
        db: Session,
        doctor_id: str,
        webhook_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle incoming WhatsApp webhook events.
        
        Processes:
        - Incoming messages
        - Message status updates
        - Read receipts
        """
        try:
            entry = webhook_data.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            
            messages = value.get("messages", [])
            statuses = value.get("statuses", [])
            
            results = {
                "messages_processed": 0,
                "statuses_processed": 0,
                "auto_replies_sent": 0
            }
            
            for message in messages:
                from_number = message.get("from")
                message_type = message.get("type")
                
                if message_type == "text":
                    text = message.get("text", {}).get("body", "")
                    
                    config = db.query(WhatsAppAutomationConfig).filter(
                        WhatsAppAutomationConfig.doctor_id == doctor_id
                    ).first()
                    
                    if config and config.auto_reply_enabled:
                        reply_result = await WhatsAppAutomationService.auto_reply(
                            db, doctor_id, None, {
                                "phone_number": from_number,
                                "message": text
                            }
                        )
                        if reply_result.get("success"):
                            results["auto_replies_sent"] += 1
                    
                    results["messages_processed"] += 1
            
            for status in statuses:
                results["statuses_processed"] += 1
            
            return {
                "success": True,
                **results
            }
            
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
