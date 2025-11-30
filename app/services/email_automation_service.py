"""
Email Automation Service for Assistant Lysa

Production-grade email automation with:
- Gmail API integration for email sync
- AI-powered email classification
- Smart auto-reply generation
- Urgent email forwarding
- HIPAA-compliant PHI handling
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import base64
from email.mime.text import MIMEText

from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import openai

from app.models.automation_models import EmailAutomationConfig
from app.models.email import EmailThread, EmailMessage
from app.models.calendar_sync import GmailSync

logger = logging.getLogger(__name__)

openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EmailAutomationService:
    """Handles all email automation tasks for Assistant Lysa"""
    
    @staticmethod
    async def sync_emails(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sync emails from connected Gmail account.
        
        Args:
            db: Database session
            doctor_id: Doctor's user ID
            patient_id: Not used for email sync
            input_data: Optional parameters (max_results, since_date)
        
        Returns:
            Sync result with email counts
        """
        sync_config = db.query(GmailSync).filter(
            GmailSync.doctor_id == doctor_id
        ).first()
        
        if not sync_config or not sync_config.sync_enabled:
            return {
                "success": False,
                "error": "Gmail sync not enabled",
                "emails_synced": 0
            }
        
        try:
            creds = Credentials(
                token=sync_config.access_token,
                refresh_token=sync_config.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
            )
            
            service = build('gmail', 'v1', credentials=creds)
            
            max_results = input_data.get("max_results", 50)
            query = input_data.get("query", "is:unread")
            
            results = service.users().messages().list(
                userId='me',
                maxResults=max_results,
                q=query
            ).execute()
            
            messages = results.get('messages', [])
            synced_count = 0
            new_emails = []
            
            for msg in messages:
                msg_detail = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='full'
                ).execute()
                
                existing = db.query(EmailMessage).filter(
                    EmailMessage.message_id == msg['id']
                ).first()
                
                if existing:
                    continue
                
                headers = {h['name']: h['value'] for h in msg_detail['payload']['headers']}
                
                body = ""
                if 'parts' in msg_detail['payload']:
                    for part in msg_detail['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            body = base64.urlsafe_b64decode(
                                part['body'].get('data', '')
                            ).decode('utf-8', errors='ignore')
                            break
                elif 'body' in msg_detail['payload'] and msg_detail['payload']['body'].get('data'):
                    body = base64.urlsafe_b64decode(
                        msg_detail['payload']['body']['data']
                    ).decode('utf-8', errors='ignore')
                
                gmail_thread_id = msg_detail.get('threadId', msg['id'])
                thread = db.query(EmailThread).filter(
                    EmailThread.thread_id == gmail_thread_id
                ).first()
                
                import uuid
                if not thread:
                    thread = EmailThread(
                        id=str(uuid.uuid4()),
                        doctor_id=doctor_id,
                        thread_id=gmail_thread_id,
                        subject=headers.get('Subject', 'No Subject'),
                        category='uncategorized',
                        priority='normal',
                        message_count=0,
                        last_message_at=datetime.utcnow()
                    )
                    db.add(thread)
                    db.flush()
                
                email_msg = EmailMessage(
                    id=str(uuid.uuid4()),
                    thread_id=thread.id,
                    doctor_id=doctor_id,
                    message_id=msg['id'],
                    from_email=headers.get('From', ''),
                    to_emails=headers.get('To', ''),
                    subject=headers.get('Subject', 'No Subject'),
                    body=body[:10000],
                    is_read='UNREAD' not in msg_detail.get('labelIds', []),
                    received_at=datetime.fromtimestamp(
                        int(msg_detail['internalDate']) / 1000
                    )
                )
                db.add(email_msg)
                
                thread.message_count += 1
                thread.last_message_at = email_msg.received_at
                
                new_emails.append({
                    "id": msg['id'],
                    "subject": headers.get('Subject', ''),
                    "from": headers.get('From', '')
                })
                synced_count += 1
            
            email_config = db.query(EmailAutomationConfig).filter(
                EmailAutomationConfig.doctor_id == doctor_id
            ).first()
            if email_config:
                email_config.last_sync_at = datetime.utcnow()
            
            db.commit()
            
            logger.info(f"Synced {synced_count} emails for doctor {doctor_id}")
            
            return {
                "success": True,
                "emails_synced": synced_count,
                "new_emails": new_emails,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except HttpError as e:
            logger.error(f"Gmail API error: {e}")
            return {
                "success": False,
                "error": f"Gmail API error: {str(e)}",
                "emails_synced": 0
            }
        except Exception as e:
            logger.error(f"Email sync error: {e}")
            return {
                "success": False,
                "error": str(e),
                "emails_synced": 0
            }
    
    @staticmethod
    async def classify_email(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify an email using AI.
        
        Categories: appointment_request, lab_results, prescription_refill,
                   urgent_medical, billing, general_inquiry, spam
        
        Priority: urgent, high, normal, low
        """
        email_id = input_data.get("email_id")
        if not email_id:
            return {"success": False, "error": "email_id required"}
        
        email = db.query(EmailMessage).filter(
            EmailMessage.id == email_id
        ).first()
        
        if not email:
            return {"success": False, "error": "Email not found"}
        
        config = db.query(EmailAutomationConfig).filter(
            EmailAutomationConfig.doctor_id == doctor_id
        ).first()
        
        priority_keywords = config.priority_keywords if config else [
            "urgent", "emergency", "asap", "immediately", "critical"
        ]
        
        prompt = f"""Classify this medical office email. Respond with JSON only.

Subject: {email.subject}
From: {email.from_email}
Body (first 1000 chars): {email.body[:1000] if email.body else 'No body'}

Categories:
- appointment_request: Scheduling, rescheduling, cancellation
- lab_results: Test results, blood work, imaging
- prescription_refill: Medication refills, pharmacy
- urgent_medical: Symptoms, pain, emergencies
- billing: Insurance, payments, statements
- general_inquiry: General questions
- spam: Marketing, unrelated

Priority indicators: {', '.join(priority_keywords)}

Respond with:
{{"category": "...", "priority": "urgent|high|normal|low", "summary": "one line summary", "suggested_action": "...", "is_patient_email": true/false, "requires_doctor_review": true/false}}"""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical office email classifier. Be HIPAA-aware and concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            thread = db.query(EmailThread).filter(
                EmailThread.id == email.thread_id
            ).first()
            
            if thread:
                thread.category = result.get("category", "general_inquiry")
                thread.priority = result.get("priority", "normal")
                thread.ai_summary = result.get("summary", "")
                thread.requires_action = result.get("requires_doctor_review", False)
            
            email.category = result.get("category")
            email.priority = result.get("priority")
            email.ai_classification = result
            
            db.commit()
            
            logger.info(f"Classified email {email_id}: {result.get('category')}")
            
            return {
                "success": True,
                "email_id": email_id,
                "classification": result
            }
            
        except Exception as e:
            logger.error(f"Email classification error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def auto_reply(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate and optionally send an auto-reply to an email.
        """
        email_id = input_data.get("email_id")
        send_reply = input_data.get("send_reply", False)
        
        if not email_id:
            return {"success": False, "error": "email_id required"}
        
        email = db.query(EmailMessage).filter(
            EmailMessage.id == email_id
        ).first()
        
        if not email:
            return {"success": False, "error": "Email not found"}
        
        config = db.query(EmailAutomationConfig).filter(
            EmailAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.auto_reply_enabled:
            return {
                "success": False,
                "error": "Auto-reply not enabled for this doctor"
            }
        
        template = config.auto_reply_template or """Thank you for contacting our office. 

We have received your message regarding "{subject}" and will respond within 1-2 business days.

If this is a medical emergency, please call 911 or go to your nearest emergency room.

For urgent matters, please call our office directly.

Best regards,
{doctor_name}'s Office"""
        
        category = email.category or "general"
        
        prompt = f"""Generate a professional, HIPAA-compliant auto-reply for this medical office email.

Original Email:
Subject: {email.subject}
From: {email.from_email}
Category: {category}

Template to use as base:
{template}

Requirements:
1. Be professional and empathetic
2. Do NOT include any specific medical advice
3. Do NOT reference specific health information
4. Provide appropriate next steps based on email category
5. Keep the response concise (under 150 words)

Generate the reply text only, no subject line needed."""

        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a professional medical office assistant. Generate HIPAA-compliant auto-replies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            
            reply_text = response.choices[0].message.content.strip()
            
            result = {
                "success": True,
                "email_id": email_id,
                "reply_text": reply_text,
                "sent": False
            }
            
            if send_reply:
                sync_config = db.query(GmailSync).filter(
                    GmailSync.doctor_id == doctor_id
                ).first()
                
                if sync_config and sync_config.sync_enabled:
                    try:
                        creds = Credentials(
                            token=sync_config.access_token,
                            refresh_token=sync_config.refresh_token,
                            token_uri="https://oauth2.googleapis.com/token",
                            client_id=os.getenv("GOOGLE_CLIENT_ID"),
                            client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
                        )
                        
                        service = build('gmail', 'v1', credentials=creds)
                        
                        message = MIMEText(reply_text)
                        message['To'] = email.from_email
                        message['Subject'] = f"Re: {email.subject}"
                        
                        raw = base64.urlsafe_b64encode(
                            message.as_bytes()
                        ).decode('utf-8')
                        
                        service.users().messages().send(
                            userId='me',
                            body={'raw': raw, 'threadId': email.message_id}
                        ).execute()
                        
                        result["sent"] = True
                        email.auto_replied = True
                        email.auto_replied_at = datetime.utcnow()
                        db.commit()
                        
                        logger.info(f"Auto-reply sent for email {email_id}")
                        
                    except Exception as e:
                        logger.error(f"Failed to send auto-reply: {e}")
                        result["send_error"] = str(e)
            
            return result
            
        except Exception as e:
            logger.error(f"Auto-reply generation error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def forward_urgent(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Forward urgent emails to the doctor's configured email address.
        """
        email_id = input_data.get("email_id")
        
        if not email_id:
            return {"success": False, "error": "email_id required"}
        
        email = db.query(EmailMessage).filter(
            EmailMessage.id == email_id
        ).first()
        
        if not email:
            return {"success": False, "error": "Email not found"}
        
        config = db.query(EmailAutomationConfig).filter(
            EmailAutomationConfig.doctor_id == doctor_id
        ).first()
        
        if not config or not config.forward_urgent_enabled:
            return {
                "success": False,
                "error": "Urgent forwarding not enabled"
            }
        
        forward_to = config.forward_urgent_to
        if not forward_to:
            return {
                "success": False,
                "error": "No forwarding email configured"
            }
        
        sync_config = db.query(GmailSync).filter(
            GmailSync.doctor_id == doctor_id
        ).first()
        
        if not sync_config or not sync_config.sync_enabled:
            return {
                "success": False,
                "error": "Gmail not connected"
            }
        
        try:
            creds = Credentials(
                token=sync_config.access_token,
                refresh_token=sync_config.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=os.getenv("GOOGLE_CLIENT_ID"),
                client_secret=os.getenv("GOOGLE_CLIENT_SECRET")
            )
            
            service = build('gmail', 'v1', credentials=creds)
            
            forward_text = f"""[URGENT - Forwarded by Lysa AI Assistant]

Original From: {email.from_email}
Original Subject: {email.subject}
Received: {email.received_at}

---

{email.body}

---
This email was automatically forwarded because it was classified as urgent.
"""
            
            message = MIMEText(forward_text)
            message['To'] = forward_to
            message['Subject'] = f"[URGENT] FWD: {email.subject}"
            
            raw = base64.urlsafe_b64encode(
                message.as_bytes()
            ).decode('utf-8')
            
            service.users().messages().send(
                userId='me',
                body={'raw': raw}
            ).execute()
            
            email.forwarded = True
            email.forwarded_to = forward_to
            email.forwarded_at = datetime.utcnow()
            db.commit()
            
            logger.info(f"Urgent email {email_id} forwarded to {forward_to}")
            
            return {
                "success": True,
                "email_id": email_id,
                "forwarded_to": forward_to
            }
            
        except Exception as e:
            logger.error(f"Forward urgent error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def process_inbox(
        db: Session,
        doctor_id: str,
        patient_id: Optional[str],
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Full inbox processing pipeline:
        1. Sync new emails
        2. Classify each email
        3. Auto-reply to applicable emails
        4. Forward urgent emails
        """
        results = {
            "synced": 0,
            "classified": 0,
            "auto_replied": 0,
            "forwarded": 0,
            "errors": []
        }
        
        sync_result = await EmailAutomationService.sync_emails(
            db, doctor_id, None, input_data
        )
        results["synced"] = sync_result.get("emails_synced", 0)
        
        if not sync_result.get("success"):
            results["errors"].append(f"Sync failed: {sync_result.get('error')}")
            return results
        
        unclassified = db.query(EmailMessage).join(
            EmailThread
        ).filter(
            and_(
                EmailThread.doctor_id == doctor_id,
                EmailMessage.category == None
            )
        ).limit(50).all()
        
        config = db.query(EmailAutomationConfig).filter(
            EmailAutomationConfig.doctor_id == doctor_id
        ).first()
        
        for email in unclassified:
            try:
                class_result = await EmailAutomationService.classify_email(
                    db, doctor_id, None, {"email_id": email.id}
                )
                
                if class_result.get("success"):
                    results["classified"] += 1
                    classification = class_result.get("classification", {})
                    
                    if (config and config.auto_reply_enabled and 
                        classification.get("category") in ["appointment_request", "general_inquiry"]):
                        reply_result = await EmailAutomationService.auto_reply(
                            db, doctor_id, None, {"email_id": email.id, "send_reply": True}
                        )
                        if reply_result.get("sent"):
                            results["auto_replied"] += 1
                    
                    if (config and config.forward_urgent_enabled and
                        classification.get("priority") == "urgent"):
                        fwd_result = await EmailAutomationService.forward_urgent(
                            db, doctor_id, None, {"email_id": email.id}
                        )
                        if fwd_result.get("success"):
                            results["forwarded"] += 1
                            
            except Exception as e:
                results["errors"].append(f"Error processing email {email.id}: {str(e)}")
        
        logger.info(f"Inbox processing complete for doctor {doctor_id}: {results}")
        
        return {
            "success": True,
            **results
        }
