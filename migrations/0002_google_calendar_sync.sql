-- Google Calendar Sync tables
-- Created: 2025-01-11

-- Google Calendar sync tracking table
CREATE TABLE IF NOT EXISTS google_calendar_sync (
  id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
  doctor_id VARCHAR NOT NULL UNIQUE REFERENCES users(id),
  
  -- OAuth tokens (encrypted in production)
  access_token TEXT,
  refresh_token TEXT,
  token_expiry TIMESTAMP,
  
  -- Calendar details
  calendar_id VARCHAR,
  calendar_name VARCHAR,
  
  -- Sync status
  sync_enabled BOOLEAN DEFAULT true,
  sync_direction VARCHAR DEFAULT 'bidirectional',
  last_sync_at TIMESTAMP,
  last_sync_status VARCHAR,
  last_sync_error TEXT,
  
  -- Incremental sync tokens
  sync_token TEXT,
  page_token TEXT,
  
  -- Sync statistics
  total_events_synced INTEGER DEFAULT 0,
  last_event_synced_at TIMESTAMP,
  
  -- Conflict resolution
  conflict_resolution VARCHAR DEFAULT 'google_wins',
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index on doctor_id
CREATE INDEX IF NOT EXISTS google_calendar_sync_doctor_idx ON google_calendar_sync(doctor_id);

-- Google Calendar sync logs (audit trail)
CREATE TABLE IF NOT EXISTS google_calendar_sync_logs (
  id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
  doctor_id VARCHAR NOT NULL REFERENCES users(id),
  sync_id VARCHAR REFERENCES google_calendar_sync(id),
  
  -- Sync details
  sync_type VARCHAR NOT NULL,
  sync_direction VARCHAR NOT NULL,
  
  -- Results
  status VARCHAR NOT NULL,
  events_created INTEGER DEFAULT 0,
  events_updated INTEGER DEFAULT 0,
  events_deleted INTEGER DEFAULT 0,
  conflicts_detected INTEGER DEFAULT 0,
  
  -- Error tracking
  error TEXT,
  error_details JSONB,
  
  -- Performance
  duration_ms INTEGER,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS google_calendar_sync_logs_doctor_idx ON google_calendar_sync_logs(doctor_id);
CREATE INDEX IF NOT EXISTS google_calendar_sync_logs_created_idx ON google_calendar_sync_logs(created_at);
