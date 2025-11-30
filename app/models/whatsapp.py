"""
WhatsApp Message Models for Assistant Lysa

HIPAA-compliant message storage for:
- WhatsApp conversations with patients
- Message history tracking
- Auto-reply audit logs
- Template message tracking
"""

from sqlalchemy import (
    Column, String, Text, Integer, Boolean, DateTime, Float,
    ForeignKey, JSON, Index, func
)
from sqlalchemy.orm import relationship
from app.database import Base
from datetime import datetime
import uuid


class WhatsAppConversation(Base):
    """
    WhatsApp conversation thread with a patient.
    Groups all messages from/to a single phone number.
    """
    __tablename__ = "whatsapp_conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    patient_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    phone_number = Column(String(20), nullable=False, index=True)
    patient_name = Column(String(255), nullable=True)
    
    status = Column(String(20), default="active")
    category = Column(String(50), default="general")
    priority = Column(String(20), default="normal")
    
    is_read = Column(Boolean, default=False)
    requires_action = Column(Boolean, default=False)
    ai_summary = Column(Text, nullable=True)
    
    message_count = Column(Integer, default=0)
    unread_count = Column(Integer, default=0)
    
    last_message_at = Column(DateTime, nullable=True)
    last_message_preview = Column(String(200), nullable=True)
    last_message_direction = Column(String(10), default="inbound")
    
    auto_reply_enabled = Column(Boolean, default=True)
    last_auto_reply_at = Column(DateTime, nullable=True)
    
    consent_received = Column(Boolean, default=False)
    consent_at = Column(DateTime, nullable=True)
    
    extra_data = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        Index("ix_wa_conv_doctor_phone", "doctor_id", "phone_number", unique=True),
        Index("ix_wa_conv_doctor_unread", "doctor_id", "is_read"),
        Index("ix_wa_conv_doctor_action", "doctor_id", "requires_action"),
        Index("ix_wa_conv_last_message", "doctor_id", "last_message_at"),
    )


class WhatsAppMessage(Base):
    """
    Individual WhatsApp message within a conversation.
    Stores both inbound and outbound messages.
    """
    __tablename__ = "whatsapp_messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("whatsapp_conversations.id"), nullable=False, index=True)
    
    whatsapp_message_id = Column(String(255), nullable=True, unique=True, index=True)
    
    direction = Column(String(10), nullable=False, index=True)
    message_type = Column(String(20), default="text")
    
    content = Column(Text, nullable=True)
    media_url = Column(String(500), nullable=True)
    media_type = Column(String(50), nullable=True)
    
    from_number = Column(String(20), nullable=True)
    to_number = Column(String(20), nullable=True)
    
    status = Column(String(20), default="sent")
    status_updated_at = Column(DateTime, nullable=True)
    error_code = Column(String(50), nullable=True)
    error_message = Column(Text, nullable=True)
    
    is_auto_reply = Column(Boolean, default=False)
    is_template = Column(Boolean, default=False)
    template_name = Column(String(100), nullable=True)
    
    ai_classification = Column(JSON, nullable=True)
    is_urgent = Column(Boolean, default=False)
    contains_phi = Column(Boolean, default=False)
    
    metadata = Column(JSON, nullable=True)
    
    received_at = Column(DateTime, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True)
    read_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("ix_wa_msg_conv_created", "conversation_id", "created_at"),
        Index("ix_wa_msg_direction", "conversation_id", "direction"),
        Index("ix_wa_msg_status", "status"),
    )


class WhatsAppWebhookLog(Base):
    """
    Audit log for all WhatsApp webhook events.
    HIPAA-compliant tracking of all API interactions.
    """
    __tablename__ = "whatsapp_webhook_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    event_type = Column(String(50), nullable=False, index=True)
    phone_number_id = Column(String(50), nullable=True)
    
    request_body = Column(JSON, nullable=True)
    response_status = Column(Integer, nullable=True)
    
    messages_processed = Column(Integer, default=0)
    auto_replies_sent = Column(Integer, default=0)
    
    processing_time_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("ix_wa_webhook_event_time", "event_type", "created_at"),
    )


class GmailWebhookLog(Base):
    """
    Audit log for Gmail Push Notification webhooks.
    HIPAA-compliant tracking of email sync events.
    """
    __tablename__ = "gmail_webhook_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doctor_id = Column(String, ForeignKey("users.id"), nullable=True, index=True)
    
    event_type = Column(String(50), nullable=False, index=True)
    history_id = Column(String(50), nullable=True)
    
    request_body = Column(JSON, nullable=True)
    
    emails_synced = Column(Integer, default=0)
    emails_classified = Column(Integer, default=0)
    auto_replies_sent = Column(Integer, default=0)
    
    processing_time_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    __table_args__ = (
        Index("ix_gmail_webhook_doctor_time", "doctor_id", "created_at"),
    )
