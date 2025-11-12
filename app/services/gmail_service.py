from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app.models.calendar_sync import GmailSync
from app.models.email import EmailThread, EmailMessage
from datetime import datetime


class GmailService:
    def __init__(self, db: Session):
        self.db = db
    
    def get_sync_config(self, doctor_id: str) -> Optional[GmailSync]:
        return self.db.query(GmailSync).filter(
            GmailSync.doctor_id == doctor_id
        ).first()
    
    def create_credentials(self, sync_config: GmailSync) -> Credentials:
        return Credentials(
            token=sync_config.access_token,
            refresh_token=sync_config.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=None,
            client_secret=None
        )
    
    def fetch_emails(self, doctor_id: str, max_results: int = 50) -> Dict:
        sync_config = self.get_sync_config(doctor_id)
        if not sync_config or not sync_config.sync_enabled:
            return {"error": "Gmail sync not enabled"}
        
        creds = self.create_credentials(sync_config)
        service = build('gmail', 'v1', credentials=creds)
        
        results = service.users().messages().list(
            userId='me',
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        
        return {
            "success": True,
            "email_count": len(messages),
            "messages": messages
        }
    
    def categorize_email(self, subject: str, body: str) -> Dict:
        subject_lower = subject.lower()
        
        if "appointment" in subject_lower or "schedule" in subject_lower:
            return {"category": "appointment", "priority": "high"}
        elif "urgent" in subject_lower or "emergency" in subject_lower:
            return {"category": "urgent", "priority": "high"}
        elif "lab" in subject_lower or "test result" in subject_lower:
            return {"category": "lab_results", "priority": "medium"}
        else:
            return {"category": "general", "priority": "normal"}
