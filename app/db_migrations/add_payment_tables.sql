-- Payment & Wallet Tables Migration
-- Adds payments_ledger, wallet_withdraw_requests, and doctor_stripe_accounts

-- Payments Ledger - Single source of truth for wallet balance
CREATE TABLE IF NOT EXISTS payments_ledger (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    user_id VARCHAR NOT NULL REFERENCES users(id),
    
    type VARCHAR NOT NULL,
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'usd',
    balance_after_cents INTEGER NOT NULL,
    
    reference TEXT,
    payment_metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ledger_user ON payments_ledger(user_id);
CREATE INDEX IF NOT EXISTS idx_ledger_user_created ON payments_ledger(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_ledger_type ON payments_ledger(type);

COMMENT ON TABLE payments_ledger IS 'Immutable transaction ledger for wallet credits';

-- Wallet Withdraw Requests - Doctor payout requests
CREATE TABLE IF NOT EXISTS wallet_withdraw_requests (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    doctor_id VARCHAR NOT NULL REFERENCES users(id),
    
    amount_cents INTEGER NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'usd',
    
    status VARCHAR NOT NULL DEFAULT 'pending',
    payout_reference VARCHAR,
    
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    
    kyc_verification_status VARCHAR,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_withdraw_doctor ON wallet_withdraw_requests(doctor_id);
CREATE INDEX IF NOT EXISTS idx_withdraw_status ON wallet_withdraw_requests(status);
CREATE INDEX IF NOT EXISTS idx_withdraw_requested ON wallet_withdraw_requests(requested_at);

COMMENT ON TABLE wallet_withdraw_requests IS 'Doctor withdrawal requests for Connect payouts';

-- Doctor Stripe Accounts - Stripe Connect for payouts
CREATE TABLE IF NOT EXISTS doctor_stripe_accounts (
    id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
    doctor_id VARCHAR NOT NULL UNIQUE REFERENCES users(id),
    
    stripe_account_id VARCHAR NOT NULL UNIQUE,
    account_status VARCHAR NOT NULL DEFAULT 'pending',
    kyc_status VARCHAR DEFAULT 'pending',
    
    connected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_stripe_doctor ON doctor_stripe_accounts(doctor_id);
CREATE INDEX IF NOT EXISTS idx_stripe_account ON doctor_stripe_accounts(stripe_account_id);

COMMENT ON TABLE doctor_stripe_accounts IS 'Stripe Connect accounts for doctor payouts';
