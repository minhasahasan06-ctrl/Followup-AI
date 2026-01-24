"""
Payment & Wallet Models for Stripe Integration
SQLAlchemy models for payments ledger, wallet transactions, and Stripe Connect.
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, Boolean, ForeignKey, Index, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.database import Base
import enum


class TransactionType(enum.Enum):
    """Type of wallet transaction."""
    CREDIT_PURCHASE = "credit_purchase"
    DEBIT_CHARGE = "debit_charge"
    WITHDRAWAL = "withdrawal"
    PAYOUT = "payout"
    REFUND = "refund"


class WithdrawStatus(enum.Enum):
    """Status of withdrawal request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PAID = "paid"


class StripeAccountStatus(enum.Enum):
    """Status of Stripe Connect account."""
    PENDING = "pending"
    ACTIVE = "active"
    RESTRICTED = "restricted"
    DISABLED = "disabled"


class PaymentsLedger(Base):
    """Ledger for all wallet transactions - single source of truth for balance."""
    __tablename__ = "payments_ledger"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    type = Column(String, nullable=False)
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="usd")
    balance_after_cents = Column(Integer, nullable=False)
    
    reference = Column(Text)
    payment_metadata = Column(JSONB, default=dict)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('idx_ledger_user', 'user_id'),
        Index('idx_ledger_user_created', 'user_id', 'created_at'),
        Index('idx_ledger_type', 'type'),
    )


class WalletWithdrawRequest(Base):
    """Withdrawal requests from doctor wallet to their bank account."""
    __tablename__ = "wallet_withdraw_requests"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    
    amount_cents = Column(Integer, nullable=False)
    currency = Column(String(3), nullable=False, default="usd")
    
    status = Column(String, nullable=False, default="pending")
    payout_reference = Column(String)
    
    requested_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True))
    
    kyc_verification_status = Column(String)
    notes = Column(Text)
    
    __table_args__ = (
        Index('idx_withdraw_doctor', 'doctor_id'),
        Index('idx_withdraw_status', 'status'),
        Index('idx_withdraw_requested', 'requested_at'),
    )


class DoctorStripeAccount(Base):
    """Stripe Connect accounts for doctor payouts."""
    __tablename__ = "doctor_stripe_accounts"
    
    id = Column(String, primary_key=True, server_default=func.gen_random_uuid())
    doctor_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    stripe_account_id = Column(String, nullable=False, unique=True)
    account_status = Column(String, nullable=False, default="pending")
    kyc_status = Column(String, default="pending")
    
    connected_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('idx_stripe_doctor', 'doctor_id'),
        Index('idx_stripe_account', 'stripe_account_id'),
    )
