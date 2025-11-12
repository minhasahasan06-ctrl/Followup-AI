from sqlalchemy import Column, String, DateTime, Text, Boolean
from sqlalchemy.sql import func
from app.database import Base


class GoogleCalendarSync(Base):
    __tablename__ = "google_calendar_sync"

    id = Column(String, primary_key=True)
    doctor_id = Column(String, nullable=False, unique=True, index=True)
    
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    token_expiry = Column(DateTime, nullable=False)
    
    calendar_id = Column(String, nullable=False)
    sync_enabled = Column(Boolean, default=True)
    
    last_sync_at = Column(DateTime, nullable=True)
    webhook_id = Column(String, nullable=True)
    webhook_resource_id = Column(String, nullable=True)
    webhook_expiration = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class GmailSync(Base):
    __tablename__ = "gmail_sync"

    id = Column(String, primary_key=True)
    doctor_id = Column(String, nullable=False, unique=True, index=True)
    
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text, nullable=False)
    token_expiry = Column(DateTime, nullable=False)
    
    sync_enabled = Column(Boolean, default=True)
    last_sync_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
