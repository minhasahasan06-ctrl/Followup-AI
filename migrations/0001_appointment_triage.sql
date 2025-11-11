-- Migration: Add symptom triage system for appointments
-- Created: 2025-11-11
-- Description: Adds AI-powered symptom triage fields to appointments table and creates audit log table

-- Add triage fields to appointments table
ALTER TABLE appointments 
ADD COLUMN IF NOT EXISTS symptoms TEXT,
ADD COLUMN IF NOT EXISTS urgency_level VARCHAR DEFAULT 'routine',
ADD COLUMN IF NOT EXISTS triage_assessment JSONB,
ADD COLUMN IF NOT EXISTS triage_assessed_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS clinician_override BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS clinician_override_reason TEXT,
ADD COLUMN IF NOT EXISTS clinician_override_by VARCHAR;

-- Create urgency queue index for efficient appointment sorting
CREATE INDEX IF NOT EXISTS appointments_urgency_queue_idx 
ON appointments(doctor_id, urgency_level, start_time);

-- Create appointment_triage_logs table for audit trail
CREATE TABLE IF NOT EXISTS appointment_triage_logs (
  id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid(),
  appointment_id VARCHAR REFERENCES appointments(id),
  patient_id VARCHAR NOT NULL REFERENCES users(id),
  
  -- Input
  symptoms TEXT NOT NULL,
  patient_self_assessment VARCHAR,
  
  -- AI Assessment
  urgency_level VARCHAR NOT NULL,
  urgency_score INTEGER NOT NULL,
  recommended_timeframe VARCHAR NOT NULL,
  red_flags TEXT[],
  confidence DECIMAL(3, 2) NOT NULL,
  assessment_method VARCHAR NOT NULL,
  
  -- Clinician Review
  clinician_reviewed BOOLEAN DEFAULT FALSE,
  clinician_agreed BOOLEAN,
  clinician_override_level VARCHAR,
  clinician_override_reason TEXT,
  reviewed_by VARCHAR REFERENCES users(id),
  reviewed_at TIMESTAMP,
  
  -- Risk Alert Integration
  risk_alert_created BOOLEAN DEFAULT FALSE,
  risk_alert_id VARCHAR,
  
  -- Metadata
  model_version VARCHAR,
  processing_time_ms INTEGER,
  
  created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for triage logs
CREATE INDEX IF NOT EXISTS triage_logs_patient_idx ON appointment_triage_logs(patient_id, created_at);
CREATE INDEX IF NOT EXISTS triage_logs_urgency_idx ON appointment_triage_logs(urgency_level, created_at);
