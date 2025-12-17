-- HIPAA Audit Logs Table (Phase 8)
-- Tracks all PHI access with full audit trail for 7-year retention
-- HIPAA Required: who, what, when, why, from-where

CREATE TABLE IF NOT EXISTS hipaa_audit_logs (
    id VARCHAR PRIMARY KEY,
    
    -- WHO - Actor information
    actor_id VARCHAR NOT NULL,
    actor_role VARCHAR NOT NULL,
    
    -- WHAT - PHI access details
    patient_id VARCHAR NOT NULL,
    action VARCHAR NOT NULL,
    phi_categories JSONB,
    resource_type VARCHAR NOT NULL,
    resource_id VARCHAR,
    
    -- WHY - Access justification
    access_scope VARCHAR,
    access_reason VARCHAR,
    consent_verified BOOLEAN DEFAULT FALSE,
    assignment_id VARCHAR,
    
    -- FROM WHERE - Request context
    ip_address VARCHAR,
    user_agent TEXT,
    request_path VARCHAR,
    
    -- Result
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    additional_context JSONB,
    
    -- WHEN - Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Indexes for efficient querying (7-year retention requires fast lookup)
CREATE INDEX IF NOT EXISTS idx_hipaa_patient_actor ON hipaa_audit_logs(patient_id, actor_id);
CREATE INDEX IF NOT EXISTS idx_hipaa_action ON hipaa_audit_logs(action, created_at);
CREATE INDEX IF NOT EXISTS idx_hipaa_phi_access ON hipaa_audit_logs(patient_id, created_at);
CREATE INDEX IF NOT EXISTS idx_hipaa_actor_time ON hipaa_audit_logs(actor_id, created_at);

-- Comment for documentation
COMMENT ON TABLE hipaa_audit_logs IS 'HIPAA-compliant PHI access audit logging with 7-year retention policy';
