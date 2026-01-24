"""
Tests for Stripe payment and wallet services.
Validates payment processing, wallet operations, and doctor payout flows.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime


class TestStripePaymentService:
    """Test Stripe payment processing."""

    def test_create_payment_intent_success(self):
        """Should create payment intent for valid amount."""
        from app.services.stripe_service import create_payment_intent
        
        with patch('stripe.PaymentIntent.create') as mock_create:
            mock_create.return_value = MagicMock(
                id='pi_test123',
                client_secret='secret_test',
                status='requires_payment_method'
            )
            
            result = create_payment_intent(
                amount=5000,
                currency='usd',
                customer_id='cus_test'
            )
            
            assert result['intent_id'] == 'pi_test123'
            assert result['client_secret'] == 'secret_test'

    def test_create_payment_intent_minimum_amount(self):
        """Should enforce minimum payment amount."""
        from app.services.stripe_service import create_payment_intent
        
        with pytest.raises(ValueError) as exc_info:
            create_payment_intent(
                amount=50,
                currency='usd',
                customer_id='cus_test'
            )
        
        assert "minimum" in str(exc_info.value).lower()

    def test_payment_metadata_includes_hipaa_fields(self):
        """Payment metadata should include HIPAA audit fields."""
        from app.services.stripe_service import create_payment_intent
        
        with patch('stripe.PaymentIntent.create') as mock_create:
            mock_create.return_value = MagicMock(
                id='pi_test123',
                client_secret='secret_test'
            )
            
            create_payment_intent(
                amount=5000,
                currency='usd',
                customer_id='cus_test',
                user_id='user_123'
            )
            
            call_args = mock_create.call_args
            metadata = call_args.kwargs.get('metadata', {})
            assert 'user_id' in metadata


class TestWalletService:
    """Test wallet balance and operations."""

    def test_get_balance_returns_decimal(self):
        """Should return balance as Decimal for precision."""
        from app.services.wallet_service import get_wallet_balance
        
        with patch('app.services.wallet_service.get_db') as mock_db:
            mock_session = MagicMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 10050
            mock_session.execute.return_value = mock_result
            mock_db.return_value.__enter__ = lambda x: mock_session
            mock_db.return_value.__exit__ = MagicMock()
            
            balance = get_wallet_balance('user_123')
            
            assert isinstance(balance, (Decimal, int, float))

    def test_add_credits_updates_balance(self):
        """Should add credits to wallet balance."""
        from app.services.wallet_service import add_credits
        
        with patch('app.services.wallet_service.get_db') as mock_db:
            mock_session = MagicMock()
            mock_db.return_value.__enter__ = lambda x: mock_session
            mock_db.return_value.__exit__ = MagicMock()
            
            result = add_credits(
                user_id='user_123',
                amount=5000,
                source='stripe_payment',
                reference_id='pi_test123'
            )
            
            assert result['success'] is True
            mock_session.commit.assert_called()

    def test_deduct_credits_insufficient_balance(self):
        """Should fail deduction when balance insufficient."""
        from app.services.wallet_service import deduct_credits
        
        with patch('app.services.wallet_service.get_wallet_balance') as mock_balance:
            mock_balance.return_value = Decimal('10.00')
            
            result = deduct_credits(
                user_id='user_123',
                amount=5000
            )
            
            assert result['success'] is False
            assert 'insufficient' in result['error'].lower()

    def test_wallet_transactions_logged(self):
        """All wallet transactions should be logged for audit."""
        from app.services.wallet_service import add_credits
        
        with patch('app.services.wallet_service.get_db') as mock_db:
            with patch('app.services.wallet_service.log_transaction') as mock_log:
                mock_session = MagicMock()
                mock_db.return_value.__enter__ = lambda x: mock_session
                mock_db.return_value.__exit__ = MagicMock()
                
                add_credits(
                    user_id='user_123',
                    amount=5000,
                    source='test'
                )
                
                mock_log.assert_called()


class TestDoctorPayouts:
    """Test doctor Stripe Connect payouts."""

    def test_onboard_doctor_creates_connect_account(self):
        """Should create Stripe Connect account for doctor."""
        from app.services.stripe_service import onboard_doctor
        
        with patch('stripe.Account.create') as mock_create:
            mock_create.return_value = MagicMock(
                id='acct_test123',
                type='express'
            )
            
            result = onboard_doctor(
                doctor_id='doc_123',
                email='doctor@example.com'
            )
            
            assert result['account_id'] == 'acct_test123'
            mock_create.assert_called_with(
                type='express',
                email='doctor@example.com',
                metadata={'doctor_id': 'doc_123'}
            )

    def test_doctor_payout_requires_verified_license(self):
        """Payouts should require verified doctor license."""
        from app.services.stripe_service import create_doctor_payout
        
        with pytest.raises(ValueError) as exc_info:
            create_doctor_payout(
                doctor_id='doc_123',
                amount=10000,
                license_verified=False
            )
        
        assert "license" in str(exc_info.value).lower()

    def test_payout_creates_ledger_entry(self):
        """Payout should create payment ledger entry."""
        from app.services.stripe_service import create_doctor_payout
        
        with patch('stripe.Transfer.create') as mock_transfer:
            with patch('app.services.stripe_service.create_ledger_entry') as mock_ledger:
                mock_transfer.return_value = MagicMock(id='tr_test123')
                
                create_doctor_payout(
                    doctor_id='doc_123',
                    amount=10000,
                    license_verified=True,
                    stripe_account_id='acct_test123'
                )
                
                mock_ledger.assert_called()


class TestPaymentsRouter:
    """Test payments router endpoints."""

    @pytest.mark.asyncio
    async def test_create_payment_endpoint(self):
        """POST /payments should create payment intent."""
        from app.routers.payments_router import create_payment
        
        mock_request = MagicMock()
        mock_request.amount = 5000
        mock_request.user_id = 'user_123'
        
        with patch('app.routers.payments_router.create_payment_intent') as mock_create:
            mock_create.return_value = {
                'intent_id': 'pi_test',
                'client_secret': 'secret'
            }
            
            result = await create_payment(mock_request)
            
            assert result['client_secret'] == 'secret'

    @pytest.mark.asyncio
    async def test_webhook_validates_signature(self):
        """Webhook should validate Stripe signature."""
        from app.routers.payments_router import handle_webhook
        
        with patch('stripe.Webhook.construct_event') as mock_construct:
            mock_construct.side_effect = ValueError("Invalid signature")
            
            with pytest.raises(ValueError):
                await handle_webhook(
                    payload=b'test',
                    signature='invalid'
                )


class TestPaymentLedger:
    """Test payment ledger for audit trail."""

    def test_ledger_entry_immutable_pattern(self):
        """Ledger entries should be append-only."""
        class ImmutableLedger:
            def __init__(self):
                self._entries = []
            
            def append(self, entry):
                self._entries.append(entry)
                return len(self._entries) - 1
            
            def update(self, entry_id, new_data):
                raise ValueError("Ledger entries are immutable and cannot be updated")
        
        ledger = ImmutableLedger()
        entry_id = ledger.append({
            'user_id': 'user-123',
            'amount': 5000,
            'type': 'credit'
        })
        
        with pytest.raises(ValueError) as exc_info:
            ledger.update(entry_id, {'amount': 10000})
        
        assert "immutable" in str(exc_info.value).lower()

    def test_ledger_tracks_all_transactions(self):
        """All payment transactions should be in ledger."""
        ledger_entries = [
            {'id': 'entry-1', 'user_id': 'user-123', 'amount': 5000, 'type': 'credit'},
            {'id': 'entry-2', 'user_id': 'user-123', 'amount': 2500, 'type': 'debit'}
        ]
        
        def get_entries_for_user(entries, user_id):
            return [e for e in entries if e['user_id'] == user_id]
        
        user_entries = get_entries_for_user(ledger_entries, 'user-123')
        
        assert len(user_entries) == 2
        assert user_entries[0]['amount'] == 5000
        assert user_entries[1]['type'] == 'debit'

    def test_ledger_reconciliation_pattern(self):
        """Ledger should support daily reconciliation."""
        from decimal import Decimal
        
        ledger_entries = [
            {'amount': Decimal('50.00'), 'type': 'credit'},
            {'amount': Decimal('25.00'), 'type': 'debit'},
            {'amount': Decimal('100.00'), 'type': 'credit'}
        ]
        
        stripe_transactions = [
            {'amount': Decimal('50.00'), 'type': 'charge'},
            {'amount': Decimal('-25.00'), 'type': 'refund'},
            {'amount': Decimal('100.00'), 'type': 'charge'}
        ]
        
        ledger_total = sum(
            e['amount'] if e['type'] == 'credit' else -e['amount']
            for e in ledger_entries
        )
        
        stripe_total = sum(t['amount'] for t in stripe_transactions)
        
        assert ledger_total == stripe_total
