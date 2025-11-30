from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.sql import func
from app.database import Base
import uuid


class EmailThread(Base):
    __tablename__ = "email_threads"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    thread_id = Column(String, nullable=False, unique=True, index=True)
    subject = Column(String, nullable=False)
    
    participants = Column(Text, nullable=True)
    message_count = Column(Integer, default=1)
    
    category = Column(String, default="general")
    priority = Column(String, default="normal")
    status = Column(String, default="new")
    
    is_read = Column(Boolean, default=False)
    ai_summary = Column(Text, nullable=True)
    requires_action = Column(Boolean, default=False)
    
    contains_phi = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    
    last_message_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class EmailMessage(Base):
    __tablename__ = "email_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    thread_id = Column(String, ForeignKey("email_threads.id"), nullable=False, index=True)
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    message_id = Column(String, nullable=False, unique=True, index=True)
    from_email = Column(String, nullable=False)
    to_emails = Column(Text, nullable=True)
    
    subject = Column(String, nullable=False)
    body = Column(Text, nullable=True)
    snippet = Column(String, nullable=True)
    
    category = Column(String, nullable=True)
    priority = Column(String, nullable=True)
    ai_classification = Column(JSON, nullable=True)
    
    is_read = Column(Boolean, default=False)
    contains_phi = Column(Boolean, default=False)
    
    auto_replied = Column(Boolean, default=False)
    auto_replied_at = Column(DateTime, nullable=True)
    
    forwarded = Column(Boolean, default=False)
    forwarded_to = Column(String, nullable=True)
    forwarded_at = Column(DateTime, nullable=True)
    
    received_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
