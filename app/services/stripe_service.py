"""
Stripe Payment Service - HIPAA-compliant payment processing
Handles subscriptions, wallet, and Connect for doctor payouts.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from datetime import datetime

import stripe
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.config import settings
from app.models.user import User
from app.models.payments import PaymentsLedger, WalletWithdrawRequest, DoctorStripeAccount

logger = logging.getLogger(__name__)


@dataclass
class PaymentResult:
    """Result of a payment operation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class StripeService:
    """
    Stripe payment service for:
    - Customer management
    - Subscriptions
    - Wallet credit purchases
    - Doctor Connect payouts
    """
    
    def __init__(self):
        self.api_key = settings.STRIPE_API_KEY
        self.webhook_secret = settings.STRIPE_WEBHOOK_SECRET
        self.connect_client_id = settings.STRIPE_CONNECT_CLIENT_ID
        
        if self.api_key:
            stripe.api_key = self.api_key
            logger.info("[Stripe] Service initialized")
        else:
            logger.warning("[Stripe] API key not configured")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)
    
    def create_customer(self, db: Session, user: User) -> PaymentResult:
        """Create a Stripe customer for a user."""
        if not self.is_configured:
            return PaymentResult(success=False, error="Stripe not configured")
        
        if user.stripe_customer_id:
            return PaymentResult(success=True, data={"customer_id": user.stripe_customer_id})
        
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=f"{user.first_name} {user.last_name}",
                metadata={
                    "user_id": user.id,
                    "role": user.role
                }
            )
            
            db.execute(
                text("UPDATE users SET stripe_customer_id = :cid WHERE id = :uid"),
                {"cid": customer.id, "uid": user.id}
            )
            db.commit()
            
            return PaymentResult(success=True, data={"customer_id": customer.id})
            
        except stripe.StripeError as e:
            logger.error(f"Stripe customer creation failed: {e}")
            return PaymentResult(success=False, error=str(e))
    
    def create_subscription(
        self,
        db: Session,
        user: User,
        price_id: str,
        payment_method_id: Optional[str] = None
    ) -> PaymentResult:
        """Create a subscription for a user."""
        if not self.is_configured:
            return PaymentResult(success=False, error="Stripe not configured")
        
        if not user.stripe_customer_id:
            result = self.create_customer(db, user)
            if not result.success:
                return result
            customer_id = result.data["customer_id"]
        else:
            customer_id = user.stripe_customer_id
        
        try:
            subscription_params = {
                "customer": customer_id,
                "items": [{"price": price_id}],
                "payment_behavior": "default_incomplete",
                "expand": ["latest_invoice.payment_intent"]
            }
            
            if payment_method_id:
                stripe.PaymentMethod.attach(payment_method_id, customer=customer_id)
                stripe.Customer.modify(
                    customer_id,
                    invoice_settings={"default_payment_method": payment_method_id}
                )
            
            subscription = stripe.Subscription.create(**subscription_params)
            
            db.execute(
                text("""
                    UPDATE users 
                    SET stripe_subscription_id = :sid, subscription_status = :status 
                    WHERE id = :uid
                """),
                {"sid": subscription.id, "status": subscription.status, "uid": user.id}
            )
            db.commit()
            
            return PaymentResult(success=True, data={
                "subscription_id": subscription.id,
                "status": subscription.status,
                "client_secret": subscription.latest_invoice.payment_intent.client_secret
                if subscription.latest_invoice and subscription.latest_invoice.payment_intent
                else None
            })
            
        except stripe.StripeError as e:
            logger.error(f"Subscription creation failed: {e}")
            return PaymentResult(success=False, error=str(e))
    
    def create_checkout_session(
        self,
        user: User,
        credits: int,
        price_cents: int,
        success_url: str,
        cancel_url: str
    ) -> PaymentResult:
        """Create a Checkout session for credit purchase."""
        if not self.is_configured:
            return PaymentResult(success=False, error="Stripe not configured")
        
        try:
            session = stripe.checkout.Session.create(
                mode="payment",
                customer=user.stripe_customer_id if user.stripe_customer_id else None,
                customer_email=user.email if not user.stripe_customer_id else None,
                line_items=[{
                    "price_data": {
                        "currency": "usd",
                        "unit_amount": price_cents,
                        "product_data": {
                            "name": f"{credits} Credits",
                            "description": f"Purchase {credits} credits for Followup AI"
                        }
                    },
                    "quantity": 1
                }],
                metadata={
                    "user_id": user.id,
                    "credits": str(credits),
                    "type": "credit_purchase"
                },
                success_url=success_url,
                cancel_url=cancel_url
            )
            
            return PaymentResult(success=True, data={
                "session_id": session.id,
                "url": session.url
            })
            
        except stripe.StripeError as e:
            logger.error(f"Checkout session creation failed: {e}")
            return PaymentResult(success=False, error=str(e))
    
    def verify_webhook(self, payload: bytes, signature: str) -> PaymentResult:
        """Verify a Stripe webhook signature."""
        if not self.webhook_secret:
            return PaymentResult(success=False, error="Webhook secret not configured")
        
        try:
            event = stripe.Webhook.construct_event(
                payload, signature, self.webhook_secret
            )
            return PaymentResult(success=True, data={"event": event})
        except stripe.SignatureVerificationError as e:
            logger.error(f"Webhook signature verification failed: {e}")
            return PaymentResult(success=False, error="Invalid signature")
        except Exception as e:
            logger.error(f"Webhook processing error: {e}")
            return PaymentResult(success=False, error=str(e))
    
    def create_connect_account_link(
        self,
        db: Session,
        doctor: User,
        refresh_url: str,
        return_url: str
    ) -> PaymentResult:
        """Create a Stripe Connect onboarding link for a doctor."""
        if not self.is_configured:
            return PaymentResult(success=False, error="Stripe not configured")
        
        try:
            existing = db.execute(
                text("SELECT stripe_account_id FROM doctor_stripe_accounts WHERE doctor_id = :did"),
                {"did": doctor.id}
            ).fetchone()
            
            if existing and existing[0]:
                account_id = existing[0]
            else:
                account = stripe.Account.create(
                    type="express",
                    country="US",
                    email=doctor.email,
                    capabilities={
                        "card_payments": {"requested": True},
                        "transfers": {"requested": True}
                    },
                    metadata={
                        "doctor_id": doctor.id
                    }
                )
                account_id = account.id
                
                db.execute(
                    text("""
                        INSERT INTO doctor_stripe_accounts (doctor_id, stripe_account_id, account_status)
                        VALUES (:did, :aid, 'pending')
                        ON CONFLICT (doctor_id) DO UPDATE SET stripe_account_id = :aid
                    """),
                    {"did": doctor.id, "aid": account_id}
                )
                db.commit()
            
            account_link = stripe.AccountLink.create(
                account=account_id,
                refresh_url=refresh_url,
                return_url=return_url,
                type="account_onboarding"
            )
            
            return PaymentResult(success=True, data={
                "account_id": account_id,
                "url": account_link.url
            })
            
        except stripe.StripeError as e:
            logger.error(f"Connect account link creation failed: {e}")
            return PaymentResult(success=False, error=str(e))
    
    def create_payout(
        self,
        db: Session,
        doctor: User,
        amount_cents: int,
        withdraw_request_id: str
    ) -> PaymentResult:
        """Create a payout to a doctor's connected account."""
        if not self.is_configured:
            return PaymentResult(success=False, error="Stripe not configured")
        
        try:
            account_row = db.execute(
                text("""
                    SELECT stripe_account_id, account_status 
                    FROM doctor_stripe_accounts 
                    WHERE doctor_id = :did
                """),
                {"did": doctor.id}
            ).fetchone()
            
            if not account_row or account_row[1] != "active":
                return PaymentResult(success=False, error="No active Stripe Connect account")
            
            transfer = stripe.Transfer.create(
                amount=amount_cents,
                currency="usd",
                destination=account_row[0],
                metadata={
                    "doctor_id": doctor.id,
                    "withdraw_request_id": withdraw_request_id
                }
            )
            
            return PaymentResult(success=True, data={
                "transfer_id": transfer.id,
                "amount": transfer.amount
            })
            
        except stripe.StripeError as e:
            logger.error(f"Payout creation failed: {e}")
            return PaymentResult(success=False, error=str(e))


_stripe_service: Optional[StripeService] = None


def get_stripe_service() -> StripeService:
    """Get singleton Stripe service instance."""
    global _stripe_service
    if _stripe_service is None:
        _stripe_service = StripeService()
    return _stripe_service
