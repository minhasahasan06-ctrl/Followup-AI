"""
Wallet Router - Credit balance and transaction management
Endpoints for balance queries, credit purchases, and withdrawals.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user, get_current_doctor
from app.models.user import User
from app.services.stripe_service import get_stripe_service
from app.services.wallet_service import get_wallet_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/wallet", tags=["wallet"])


class PurchaseRequest(BaseModel):
    credits: int
    price_cents: int
    success_url: str
    cancel_url: str


class WithdrawRequest(BaseModel):
    amount_cents: int


@router.get("/balance")
async def get_balance(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get current wallet balance."""
    wallet_service = get_wallet_service()
    
    balance = wallet_service.get_balance(db, current_user.id)
    
    return {
        "user_id": balance.user_id,
        "balance_cents": balance.balance_cents,
        "balance_dollars": balance.balance_cents / 100,
        "currency": balance.currency
    }


@router.get("/transactions")
async def get_transactions(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get transaction history."""
    wallet_service = get_wallet_service()
    
    transactions = wallet_service.get_transactions(
        db=db,
        user_id=current_user.id,
        limit=min(limit, 100),
        offset=offset
    )
    
    return {
        "transactions": [
            {
                "id": t.id,
                "type": t.type,
                "amount_cents": t.amount_cents,
                "amount_dollars": t.amount_cents / 100,
                "balance_after_cents": t.balance_after_cents,
                "reference": t.reference,
                "created_at": t.created_at.isoformat()
            }
            for t in transactions
        ],
        "limit": limit,
        "offset": offset
    }


@router.post("/purchase")
async def purchase_credits(
    request: PurchaseRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a Stripe Checkout session for credit purchase."""
    if request.credits <= 0:
        raise HTTPException(status_code=400, detail="Credits must be positive")
    
    if request.price_cents <= 0:
        raise HTTPException(status_code=400, detail="Price must be positive")
    
    stripe_service = get_stripe_service()
    
    result = stripe_service.create_checkout_session(
        user=current_user,
        credits=request.credits,
        price_cents=request.price_cents,
        success_url=request.success_url,
        cancel_url=request.cancel_url
    )
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "session_id": result.data["session_id"],
        "url": result.data["url"]
    }


@router.post("/withdraw")
async def request_withdrawal(
    request: WithdrawRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_doctor)
):
    """Request a withdrawal (doctors only)."""
    if request.amount_cents <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    
    if request.amount_cents < 1000:
        raise HTTPException(status_code=400, detail="Minimum withdrawal is $10")
    
    wallet_service = get_wallet_service()
    
    result = wallet_service.request_withdrawal(
        db=db,
        doctor_id=current_user.id,
        amount_cents=request.amount_cents
    )
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "request_id": result.data["request_id"],
        "amount_cents": result.data["amount_cents"],
        "status": result.data["status"]
    }


@router.get("/withdraw/requests")
async def get_withdrawal_requests(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_doctor)
):
    """Get withdrawal request history (doctors only)."""
    from sqlalchemy import text
    
    rows = db.execute(
        text("""
            SELECT id, amount_cents, currency, status, payout_reference, 
                   requested_at, processed_at, notes
            FROM wallet_withdraw_requests
            WHERE doctor_id = :did
            ORDER BY requested_at DESC
            LIMIT 50
        """),
        {"did": current_user.id}
    ).fetchall()
    
    return {
        "requests": [
            {
                "id": row[0],
                "amount_cents": row[1],
                "amount_dollars": row[1] / 100,
                "currency": row[2],
                "status": row[3],
                "payout_reference": row[4],
                "requested_at": row[5].isoformat() if row[5] else None,
                "processed_at": row[6].isoformat() if row[6] else None,
                "notes": row[7]
            }
            for row in rows
        ]
    }
