"""
Payments Router - Stripe integration endpoints
Handles customer creation, subscriptions, webhooks, and Connect onboarding.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user, require_m2m_auth
from app.models.user import User
from app.services.stripe_service import get_stripe_service
from app.services.wallet_service import get_wallet_service
from app.services.access_control import HIPAAAuditLogger

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/payments", tags=["payments"])


class CreateSubscriptionRequest(BaseModel):
    price_id: str
    payment_method_id: Optional[str] = None


class PurchaseCreditsRequest(BaseModel):
    credits: int
    price_cents: int
    success_url: str
    cancel_url: str


class ConnectOnboardRequest(BaseModel):
    refresh_url: str
    return_url: str


@router.post("/create-customer")
async def create_customer(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a Stripe customer for the current user."""
    stripe_service = get_stripe_service()
    
    result = stripe_service.create_customer(db, current_user)
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {"success": True, "customer_id": result.data["customer_id"]}


@router.post("/create-subscription")
async def create_subscription(
    request: CreateSubscriptionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create a subscription for the current user."""
    stripe_service = get_stripe_service()
    
    result = stripe_service.create_subscription(
        db=db,
        user=current_user,
        price_id=request.price_id,
        payment_method_id=request.payment_method_id
    )
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "subscription_id": result.data["subscription_id"],
        "status": result.data["status"],
        "client_secret": result.data.get("client_secret")
    }


@router.post("/stripe-webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="Stripe-Signature"),
    db: Session = Depends(get_db)
):
    """Handle Stripe webhook events."""
    stripe_service = get_stripe_service()
    wallet_service = get_wallet_service()
    
    payload = await request.body()
    
    result = stripe_service.verify_webhook(payload, stripe_signature)
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    event = result.data["event"]
    event_type = event["type"]
    data = event["data"]["object"]
    
    logger.info(f"[Stripe Webhook] Received event: {event_type}")
    
    if event_type == "checkout.session.completed":
        metadata = data.get("metadata", {})
        if metadata.get("type") == "credit_purchase":
            user_id = metadata.get("user_id")
            credits = int(metadata.get("credits", 0))
            amount = data.get("amount_total", 0)
            
            if user_id and credits > 0:
                wallet_service.add_credits(
                    db=db,
                    user_id=user_id,
                    amount_cents=credits * 100,
                    reference=data.get("id"),
                    metadata={"checkout_session_id": data.get("id")}
                )
                logger.info(f"[Wallet] Added {credits} credits to user {user_id}")
    
    elif event_type == "invoice.payment_succeeded":
        subscription_id = data.get("subscription")
        customer_id = data.get("customer")
        
        from sqlalchemy import text
        db.execute(
            text("UPDATE users SET subscription_status = 'active' WHERE stripe_customer_id = :cid"),
            {"cid": customer_id}
        )
        db.commit()
        logger.info(f"[Subscription] Updated status to active for customer {customer_id}")
    
    elif event_type == "invoice.payment_failed":
        customer_id = data.get("customer")
        
        from sqlalchemy import text
        db.execute(
            text("UPDATE users SET subscription_status = 'past_due' WHERE stripe_customer_id = :cid"),
            {"cid": customer_id}
        )
        db.commit()
        logger.warning(f"[Subscription] Payment failed for customer {customer_id}")
    
    elif event_type == "account.updated":
        account_id = data.get("id")
        charges_enabled = data.get("charges_enabled", False)
        payouts_enabled = data.get("payouts_enabled", False)
        
        status = "active" if charges_enabled and payouts_enabled else "pending"
        
        from sqlalchemy import text
        db.execute(
            text("UPDATE doctor_stripe_accounts SET account_status = :status WHERE stripe_account_id = :aid"),
            {"status": status, "aid": account_id}
        )
        db.commit()
        logger.info(f"[Connect] Updated account {account_id} status to {status}")
    
    elif event_type == "payout.paid":
        metadata = data.get("metadata", {})
        doctor_id = metadata.get("doctor_id")
        withdraw_request_id = metadata.get("withdraw_request_id")
        
        if withdraw_request_id:
            wallet_service.process_withdrawal(
                db=db,
                request_id=withdraw_request_id,
                payout_reference=data.get("id"),
                approved=True
            )
            logger.info(f"[Payout] Completed payout for request {withdraw_request_id}")
    
    return {"received": True}


@router.post("/connect/onboard")
async def connect_onboard(
    request: ConnectOnboardRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create Stripe Connect onboarding link for a doctor."""
    if current_user.role != "doctor":
        raise HTTPException(status_code=403, detail="Only doctors can onboard to Connect")
    
    stripe_service = get_stripe_service()
    
    result = stripe_service.create_connect_account_link(
        db=db,
        doctor=current_user,
        refresh_url=request.refresh_url,
        return_url=request.return_url
    )
    
    if not result.success:
        raise HTTPException(status_code=400, detail=result.error)
    
    return {
        "success": True,
        "account_id": result.data["account_id"],
        "url": result.data["url"]
    }
