"""
Wallet Service - Credit balance and transaction management
Single source of truth for user credit balances via payments_ledger.
"""

import logging
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.models.user import User

logger = logging.getLogger(__name__)


@dataclass
class WalletBalance:
    """User wallet balance."""
    user_id: str
    balance_cents: int
    currency: str = "usd"


@dataclass
class Transaction:
    """Wallet transaction."""
    id: str
    type: str
    amount_cents: int
    balance_after_cents: int
    reference: Optional[str]
    created_at: datetime


@dataclass
class WalletResult:
    """Result of a wallet operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class WalletService:
    """
    Wallet service for:
    - Balance queries
    - Credit purchases
    - Debit charges
    - Transaction history
    - Withdrawal requests
    """
    
    def get_balance(self, db: Session, user_id: str) -> WalletBalance:
        """Get current wallet balance from ledger."""
        result = db.execute(
            text("""
                SELECT balance_after_cents 
                FROM payments_ledger 
                WHERE user_id = :uid 
                ORDER BY created_at DESC 
                LIMIT 1
            """),
            {"uid": user_id}
        ).fetchone()
        
        balance = result[0] if result else 0
        return WalletBalance(user_id=user_id, balance_cents=balance)
    
    def get_transactions(
        self,
        db: Session,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Transaction]:
        """Get transaction history for a user."""
        rows = db.execute(
            text("""
                SELECT id, type, amount_cents, balance_after_cents, reference, created_at
                FROM payments_ledger
                WHERE user_id = :uid
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
            """),
            {"uid": user_id, "limit": limit, "offset": offset}
        ).fetchall()
        
        return [
            Transaction(
                id=row[0],
                type=row[1],
                amount_cents=row[2],
                balance_after_cents=row[3],
                reference=row[4],
                created_at=row[5]
            )
            for row in rows
        ]
    
    def add_credits(
        self,
        db: Session,
        user_id: str,
        amount_cents: int,
        reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WalletResult:
        """Add credits to wallet (credit purchase, refund)."""
        try:
            current = self.get_balance(db, user_id)
            new_balance = current.balance_cents + amount_cents
            
            db.execute(
                text("""
                    INSERT INTO payments_ledger 
                    (user_id, type, amount_cents, balance_after_cents, reference, payment_metadata)
                    VALUES (:uid, 'credit_purchase', :amount, :balance, :ref, :meta::jsonb)
                """),
                {
                    "uid": user_id,
                    "amount": amount_cents,
                    "balance": new_balance,
                    "ref": reference,
                    "meta": json.dumps(metadata or {})
                }
            )
            db.commit()
            
            db.execute(
                text("UPDATE users SET credit_balance = :bal WHERE id = :uid"),
                {"bal": new_balance, "uid": user_id}
            )
            db.commit()
            
            return WalletResult(success=True, data={
                "balance_cents": new_balance,
                "credits_added": amount_cents
            })
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to add credits: {e}")
            return WalletResult(success=False, error=str(e))
    
    def debit_credits(
        self,
        db: Session,
        user_id: str,
        amount_cents: int,
        reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WalletResult:
        """Debit credits from wallet (charge for service)."""
        try:
            current = self.get_balance(db, user_id)
            
            if current.balance_cents < amount_cents:
                return WalletResult(
                    success=False,
                    error="Insufficient balance",
                    data={"balance_cents": current.balance_cents, "required": amount_cents}
                )
            
            new_balance = current.balance_cents - amount_cents
            
            db.execute(
                text("""
                    INSERT INTO payments_ledger 
                    (user_id, type, amount_cents, balance_after_cents, reference, payment_metadata)
                    VALUES (:uid, 'debit_charge', :amount, :balance, :ref, :meta::jsonb)
                """),
                {
                    "uid": user_id,
                    "amount": -amount_cents,
                    "balance": new_balance,
                    "ref": reference,
                    "meta": json.dumps(metadata or {})
                }
            )
            db.commit()
            
            db.execute(
                text("UPDATE users SET credit_balance = :bal WHERE id = :uid"),
                {"bal": new_balance, "uid": user_id}
            )
            db.commit()
            
            return WalletResult(success=True, data={
                "balance_cents": new_balance,
                "credits_debited": amount_cents
            })
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to debit credits: {e}")
            return WalletResult(success=False, error=str(e))
    
    def request_withdrawal(
        self,
        db: Session,
        doctor_id: str,
        amount_cents: int
    ) -> WalletResult:
        """Request withdrawal for a doctor."""
        try:
            current = self.get_balance(db, doctor_id)
            
            if current.balance_cents < amount_cents:
                return WalletResult(
                    success=False,
                    error="Insufficient balance",
                    data={"balance_cents": current.balance_cents}
                )
            
            result = db.execute(
                text("""
                    INSERT INTO wallet_withdraw_requests 
                    (doctor_id, amount_cents, currency, status)
                    VALUES (:did, :amount, 'usd', 'pending')
                    RETURNING id
                """),
                {"did": doctor_id, "amount": amount_cents}
            )
            request_id = result.fetchone()[0]
            db.commit()
            
            return WalletResult(success=True, data={
                "request_id": request_id,
                "amount_cents": amount_cents,
                "status": "pending"
            })
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to create withdrawal request: {e}")
            return WalletResult(success=False, error=str(e))
    
    def process_withdrawal(
        self,
        db: Session,
        request_id: str,
        payout_reference: str,
        approved: bool = True
    ) -> WalletResult:
        """Process a withdrawal request (admin/system)."""
        try:
            request = db.execute(
                text("""
                    SELECT doctor_id, amount_cents, status 
                    FROM wallet_withdraw_requests 
                    WHERE id = :rid
                """),
                {"rid": request_id}
            ).fetchone()
            
            if not request:
                return WalletResult(success=False, error="Request not found")
            
            if request[2] != "pending":
                return WalletResult(success=False, error="Request already processed")
            
            doctor_id = request[0]
            amount = request[1]
            new_status = "paid" if approved else "rejected"
            
            if approved:
                current = self.get_balance(db, doctor_id)
                new_balance = current.balance_cents - amount
                
                db.execute(
                    text("""
                        INSERT INTO payments_ledger 
                        (user_id, type, amount_cents, balance_after_cents, reference, payment_metadata)
                        VALUES (:uid, 'withdrawal', :amount, :balance, :ref, :meta::jsonb)
                    """),
                    {
                        "uid": doctor_id,
                        "amount": -amount,
                        "balance": new_balance,
                        "ref": payout_reference,
                        "meta": json.dumps({"withdraw_request_id": request_id})
                    }
                )
                
                db.execute(
                    text("UPDATE users SET credit_balance = :bal WHERE id = :uid"),
                    {"bal": new_balance, "uid": doctor_id}
                )
            
            db.execute(
                text("""
                    UPDATE wallet_withdraw_requests 
                    SET status = :status, payout_reference = :ref, processed_at = NOW()
                    WHERE id = :rid
                """),
                {"status": new_status, "ref": payout_reference, "rid": request_id}
            )
            db.commit()
            
            return WalletResult(success=True, data={
                "request_id": request_id,
                "status": new_status
            })
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to process withdrawal: {e}")
            return WalletResult(success=False, error=str(e))


_wallet_service: Optional[WalletService] = None


def get_wallet_service() -> WalletService:
    """Get singleton wallet service instance."""
    global _wallet_service
    if _wallet_service is None:
        _wallet_service = WalletService()
    return _wallet_service
