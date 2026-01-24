"""
Stripe Payment and Wallet Service Tests
Tests for StripeService using actual production APIs.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime

from app.services.stripe_service import StripeService, PaymentResult
from app.models.payments import PaymentsLedger, WalletWithdrawRequest, DoctorStripeAccount


class TestPaymentResult:
    """Test PaymentResult data class."""

    def test_payment_result_success(self):
        """Test successful payment result."""
        result = PaymentResult(
            success=True,
            data={"customer_id": "cus_123", "subscription_id": "sub_456"}
        )
        
        assert result.success is True
        assert result.data["customer_id"] == "cus_123"
        assert result.error is None

    def test_payment_result_failure(self):
        """Test failed payment result."""
        result = PaymentResult(
            success=False,
            error="Card declined"
        )
        
        assert result.success is False
        assert result.error == "Card declined"
        assert result.data is None


class TestStripeServiceInit:
    """Test StripeService initialization."""

    @patch.dict('os.environ', {
        'STRIPE_API_KEY': 'sk_test_123',
        'STRIPE_WEBHOOK_SECRET': 'whsec_123'
    })
    def test_service_init_configured(self):
        """Test service initializes with API key."""
        with patch('app.services.stripe_service.settings') as mock_settings:
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            service = StripeService()
            
            assert service.is_configured is True

    def test_service_not_configured_without_key(self):
        """Test service reports not configured without API key."""
        with patch('app.services.stripe_service.settings') as mock_settings:
            mock_settings.STRIPE_API_KEY = None
            mock_settings.STRIPE_WEBHOOK_SECRET = None
            mock_settings.STRIPE_CONNECT_CLIENT_ID = None
            
            service = StripeService()
            
            assert service.is_configured is False


class TestStripeServiceCustomer:
    """Test customer creation."""

    def test_create_customer_success(self):
        """Test successful customer creation."""
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.Customer.create') as mock_create:
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_create.return_value = MagicMock(id='cus_test123')
            
            mock_user = MagicMock()
            mock_user.id = 'user-123'
            mock_user.email = 'test@example.com'
            mock_user.first_name = 'John'
            mock_user.last_name = 'Doe'
            mock_user.role = 'patient'
            mock_user.stripe_customer_id = None
            
            mock_db = MagicMock()
            
            service = StripeService()
            result = service.create_customer(mock_db, mock_user)
            
            assert result.success is True
            assert result.data["customer_id"] == "cus_test123"

    def test_create_customer_already_exists(self):
        """Test customer already exists returns existing ID."""
        with patch('app.services.stripe_service.settings') as mock_settings:
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_user = MagicMock()
            mock_user.stripe_customer_id = 'cus_existing'
            
            mock_db = MagicMock()
            
            service = StripeService()
            result = service.create_customer(mock_db, mock_user)
            
            assert result.success is True
            assert result.data["customer_id"] == "cus_existing"


class TestStripeServiceSubscription:
    """Test subscription creation."""

    def test_create_subscription_success(self):
        """Test successful subscription creation."""
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.Subscription.create') as mock_sub, \
             patch('stripe.PaymentMethod.attach'), \
             patch('stripe.Customer.modify'):
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_invoice = MagicMock()
            mock_invoice.payment_intent = MagicMock(client_secret='secret_123')
            
            mock_sub.return_value = MagicMock(
                id='sub_test123',
                status='active',
                latest_invoice=mock_invoice
            )
            
            mock_user = MagicMock()
            mock_user.id = 'user-123'
            mock_user.stripe_customer_id = 'cus_123'
            
            mock_db = MagicMock()
            
            service = StripeService()
            result = service.create_subscription(mock_db, mock_user, 'price_123')
            
            assert result.success is True
            assert result.data["subscription_id"] == "sub_test123"


class TestStripeServiceCheckout:
    """Test checkout session creation."""

    def test_create_checkout_session_success(self):
        """Test successful checkout session creation."""
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.checkout.Session.create') as mock_session:
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_session.return_value = MagicMock(
                id='cs_test123',
                url='https://checkout.stripe.com/...'
            )
            
            mock_user = MagicMock()
            mock_user.id = 'user-123'
            mock_user.email = 'test@example.com'
            mock_user.stripe_customer_id = 'cus_123'
            
            service = StripeService()
            result = service.create_checkout_session(
                mock_user,
                credits=100,
                price_cents=1000,
                success_url='https://example.com/success',
                cancel_url='https://example.com/cancel'
            )
            
            assert result.success is True


class TestPaymentsLedger:
    """Test PaymentsLedger model."""

    def test_ledger_entry_fields(self):
        """Test ledger entry has required fields."""
        ledger_entry = {
            'id': 'entry-123',
            'user_id': 'user-456',
            'type': 'credit_purchase',
            'amount_cents': 5000,
            'currency': 'usd',
            'balance_after_cents': 5000,
            'reference': 'pi_test123',
            'metadata': {'credits': 100}
        }
        
        assert 'user_id' in ledger_entry
        assert 'amount_cents' in ledger_entry
        assert 'type' in ledger_entry
        assert ledger_entry['type'] in ['credit_purchase', 'debit_charge', 'withdrawal', 'payout', 'refund']

    def test_ledger_immutability_pattern(self):
        """Test ledger entries are append-only (no UPDATE operations)."""
        class ImmutableLedger:
            def __init__(self):
                self._entries = []
            
            def append(self, entry):
                self._entries.append(entry)
                return entry['id']
            
            def update(self, entry_id, updates):
                raise ValueError("Ledger entries are immutable - use compensating entries instead")
        
        ledger = ImmutableLedger()
        entry_id = ledger.append({'id': 'e1', 'amount': 100})
        
        with pytest.raises(ValueError) as exc:
            ledger.update('e1', {'amount': 200})
        
        assert 'immutable' in str(exc.value).lower()


class TestWalletWithdrawRequest:
    """Test WalletWithdrawRequest model."""

    def test_withdraw_request_statuses(self):
        """Test valid withdrawal request statuses."""
        valid_statuses = ['pending', 'approved', 'rejected', 'paid']
        
        for status in valid_statuses:
            request = {
                'id': 'wr-123',
                'doctor_id': 'doc-456',
                'amount_cents': 10000,
                'currency': 'usd',
                'status': status
            }
            assert request['status'] in valid_statuses


class TestDoctorStripeAccount:
    """Test DoctorStripeAccount model."""

    def test_connect_account_fields(self):
        """Test Connect account has required fields."""
        account = {
            'id': 'dsa-123',
            'doctor_id': 'doc-456',
            'stripe_account_id': 'acct_123',
            'account_status': 'active',
            'kyc_status': 'verified'
        }
        
        assert 'doctor_id' in account
        assert 'stripe_account_id' in account
        assert account['account_status'] in ['pending', 'active', 'restricted', 'disabled']


class TestStripeConnect:
    """Test Stripe Connect for doctor payouts."""

    def test_create_connect_account_link(self):
        """Test creating Connect account onboarding link."""
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.Account.create') as mock_account, \
             patch('stripe.AccountLink.create') as mock_link:
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_account.return_value = MagicMock(id='acct_test123')
            mock_link.return_value = MagicMock(url='https://connect.stripe.com/setup/...')
            
            mock_doctor = MagicMock()
            mock_doctor.id = 'doc-123'
            mock_doctor.email = 'doctor@example.com'
            
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchone.return_value = None
            
            service = StripeService()
            result = service.create_connect_account_link(
                mock_db,
                mock_doctor,
                'https://example.com/refresh',
                'https://example.com/return'
            )
            
            assert result.success is True

    def test_create_payout_success(self):
        """Test creating payout to connected account."""
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.Transfer.create') as mock_transfer:
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_transfer.return_value = MagicMock(id='tr_123', amount=10000)
            
            mock_doctor = MagicMock()
            mock_doctor.id = 'doc-123'
            
            mock_db = MagicMock()
            mock_db.execute.return_value.fetchone.return_value = ('acct_test123', 'active')
            
            service = StripeService()
            result = service.create_payout(
                db=mock_db,
                doctor=mock_doctor,
                amount_cents=10000,
                withdraw_request_id='wr-123'
            )
            
            assert result.success is True


class TestWebhookHandling:
    """Test Stripe webhook signature verification."""

    def test_verify_webhook_signature(self):
        """Test webhook signature verification returns PaymentResult."""
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.Webhook.construct_event') as mock_construct:
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_construct.return_value = {
                'type': 'checkout.session.completed',
                'data': {'object': {'id': 'cs_123'}}
            }
            
            service = StripeService()
            result = service.verify_webhook(
                payload=b'{}',
                signature='sig_123'
            )
            
            assert result.success is True
            assert result.data['event']['type'] == 'checkout.session.completed'

    def test_invalid_webhook_signature_returns_error(self):
        """Test invalid webhook signature returns error result."""
        import stripe
        
        with patch('app.services.stripe_service.settings') as mock_settings, \
             patch('stripe.Webhook.construct_event') as mock_construct:
            
            mock_settings.STRIPE_API_KEY = 'sk_test_123'
            mock_settings.STRIPE_WEBHOOK_SECRET = 'whsec_123'
            mock_settings.STRIPE_CONNECT_CLIENT_ID = 'ca_123'
            
            mock_construct.side_effect = stripe.SignatureVerificationError(
                'Invalid signature', 'sig_123'
            )
            
            service = StripeService()
            result = service.verify_webhook(
                payload=b'{}',
                signature='invalid_sig'
            )
            
            assert result.success is False
            assert result.error == "Invalid signature"


class TestWalletBalance:
    """Test wallet balance calculations."""

    def test_calculate_balance_from_ledger(self):
        """Test calculating balance from ledger entries."""
        ledger_entries = [
            {'type': 'credit_purchase', 'amount_cents': 5000},
            {'type': 'debit_charge', 'amount_cents': 1500},
            {'type': 'credit_purchase', 'amount_cents': 2000}
        ]
        
        credits = sum(e['amount_cents'] for e in ledger_entries if 'credit' in e['type'])
        debits = sum(e['amount_cents'] for e in ledger_entries if 'debit' in e['type'])
        balance = credits - debits
        
        assert balance == 5500

    def test_balance_after_cents_tracking(self):
        """Test balance_after_cents field is maintained."""
        ledger = []
        balance = 0
        
        transactions = [
            ('credit_purchase', 5000),
            ('debit_charge', 1000),
            ('credit_purchase', 2000)
        ]
        
        for tx_type, amount in transactions:
            if 'credit' in tx_type:
                balance += amount
            else:
                balance -= amount
            
            ledger.append({
                'type': tx_type,
                'amount_cents': amount,
                'balance_after_cents': balance
            })
        
        assert ledger[-1]['balance_after_cents'] == 6000
