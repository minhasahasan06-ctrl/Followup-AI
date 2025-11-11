-- Gmail Sync tables (HIPAA CRITICAL)
-- Created: 2025-01-11
-- PRODUCTION REQUIREMENTS:
-- 1. Encrypt OAuth tokens with AWS KMS
-- 2. Google Workspace BAA required (not regular Gmail)
-- 3. PHI detection/redaction before sync
-- 4. Patient consent workflow
-- 5. Immutable audit logging

CREATE TABLE IF NOT EXISTS gmail_sync (
  id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
  doctor_id VARCHAR NOT NULL UNIQUE REFERENCES users(id),
  
  -- OAuth tokens (MUST encrypt in production)
  access_token TEXT,
  refresh_token TEXT,
  token_expiry TIMESTAMP,
  token_scopes TEXT[],
  
  -- Google Workspace validation
  google_workspace_domain VARCHAR,
  google_workspace_baa_confirmed BOOLEAN DEFAULT false,
  
  -- Sync configuration
  sync_enabled BOOLEAN DEFAULT false,
  last_sync_at TIMESTAMP,
  last_sync_status VARCHAR,
  last_sync_error TEXT,
  
  -- Statistics
  total_emails_synced INTEGER DEFAULT 0,
  
  -- PHI protection
  phi_redaction_enabled BOOLEAN DEFAULT true,
  consent_confirmed BOOLEAN DEFAULT false,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS gmail_sync_doctor_idx ON gmail_sync(doctor_id);

-- Gmail sync audit logs
CREATE TABLE IF NOT EXISTS gmail_sync_logs (
  id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
  doctor_id VARCHAR NOT NULL REFERENCES users(id),
  sync_id VARCHAR REFERENCES gmail_sync(id),
  
  -- Audit details
  action VARCHAR NOT NULL,
  status VARCHAR NOT NULL,
  emails_fetched INTEGER DEFAULT 0,
  phi_detected BOOLEAN DEFAULT false,
  
  -- Error tracking
  error TEXT,
  error_details JSONB,
  
  -- Audit metadata
  ip_address VARCHAR,
  user_agent TEXT,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS gmail_sync_logs_doctor_idx ON gmail_sync_logs(doctor_id);
CREATE INDEX IF NOT EXISTS gmail_sync_logs_created_idx ON gmail_sync_logs(created_at);
CREATE INDEX IF NOT EXISTS gmail_sync_logs_action_idx ON gmail_sync_logs(action);
