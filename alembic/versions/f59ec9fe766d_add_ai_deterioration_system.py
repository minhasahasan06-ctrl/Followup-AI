"""add ai deterioration system - video audio trend alert

Revision ID: f59ec9fe766d
Revises: ml_inference_001
Create Date: 2025-11-16

This migration creates the comprehensive AI deterioration detection system tables
for Followup AI, including:
- Media analysis (video/audio upload and processing)
- Video AI metrics (respiratory rate, skin pallor, jaundice, facial swelling, tremor)
- Audio AI metrics (breath cycles, speech pace, cough detection, wheeze signatures)
- Trend prediction (time-series risk modeling, Bayesian updates, patient baselines)
- Alert orchestration (multi-channel notifications, rule engine)
- Security & compliance (HIPAA consent management, PHI audit logging, retention policies)
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f59ec9fe766d'
down_revision = 'ml_inference_001'
branch_labels = None
depends_on = None


def upgrade():
    # Create media_sessions table (video/audio uploads with S3 encryption)
    op.create_table(
        'media_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('session_type', sa.String(), nullable=False),  # 'video' or 'audio'
        sa.Column('s3_key', sa.String(), nullable=False),
        sa.Column('s3_bucket', sa.String(), nullable=False),
        sa.Column('kms_key_id', sa.String(), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('duration_seconds', sa.Float(), nullable=True),
        sa.Column('processing_status', sa.String(), nullable=False, server_default='pending'),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('uploaded_by', sa.String(), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_media_sessions_id', 'media_sessions', ['id'])
    op.create_index('ix_media_sessions_patient_id', 'media_sessions', ['patient_id'])
    op.create_index('idx_media_patient_type', 'media_sessions', ['patient_id', 'session_type'])
    op.create_index('idx_media_status', 'media_sessions', ['processing_status'])

    # Create video_metrics table (10+ AI-powered metrics from video analysis)
    op.create_table(
        'video_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('respiratory_rate_bpm', sa.Float(), nullable=True),
        sa.Column('respiratory_variability', sa.Float(), nullable=True),
        sa.Column('skin_pallor_score', sa.Float(), nullable=True),
        sa.Column('eye_sclera_yellowness', sa.Float(), nullable=True),
        sa.Column('facial_swelling_score', sa.Float(), nullable=True),
        sa.Column('head_movement_score', sa.Float(), nullable=True),
        sa.Column('head_stability_score', sa.Float(), nullable=True),
        sa.Column('tremor_detected', sa.Boolean(), nullable=True),
        sa.Column('tremor_frequency_hz', sa.Float(), nullable=True),
        sa.Column('lighting_quality', sa.Float(), nullable=True),
        sa.Column('frame_quality', sa.Float(), nullable=True),
        sa.Column('analysis_confidence', sa.Float(), nullable=True),
        sa.Column('analysis_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['media_sessions.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_video_metrics_id', 'video_metrics', ['id'])
    op.create_index('ix_video_metrics_session_id', 'video_metrics', ['session_id'])
    op.create_index('ix_video_metrics_patient_id', 'video_metrics', ['patient_id'])
    op.create_index('idx_video_patient_timestamp', 'video_metrics', ['patient_id', 'analysis_timestamp'])

    # Create audio_metrics table (10+ AI-powered metrics from audio analysis)
    op.create_table(
        'audio_metrics',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('breath_cycle_count', sa.Integer(), nullable=True),
        sa.Column('avg_breath_duration_sec', sa.Float(), nullable=True),
        sa.Column('breath_irregularity', sa.Float(), nullable=True),
        sa.Column('speech_pace_wpm', sa.Float(), nullable=True),
        sa.Column('speech_pace_variability', sa.Float(), nullable=True),
        sa.Column('pause_frequency', sa.Float(), nullable=True),
        sa.Column('cough_count', sa.Integer(), nullable=True),
        sa.Column('cough_severity', sa.Float(), nullable=True),
        sa.Column('wheeze_detected', sa.Boolean(), nullable=True),
        sa.Column('wheeze_frequency_hz', sa.Float(), nullable=True),
        sa.Column('voice_hoarseness', sa.Float(), nullable=True),
        sa.Column('voice_tremor', sa.Float(), nullable=True),
        sa.Column('background_noise_level', sa.Float(), nullable=True),
        sa.Column('audio_quality', sa.Float(), nullable=True),
        sa.Column('analysis_confidence', sa.Float(), nullable=True),
        sa.Column('analysis_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['session_id'], ['media_sessions.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audio_metrics_id', 'audio_metrics', ['id'])
    op.create_index('ix_audio_metrics_session_id', 'audio_metrics', ['session_id'])
    op.create_index('ix_audio_metrics_patient_id', 'audio_metrics', ['patient_id'])
    op.create_index('idx_audio_patient_timestamp', 'audio_metrics', ['patient_id', 'analysis_timestamp'])

    # Create patient_baselines table (7-day rolling stats for personalized risk scoring)
    op.create_table(
        'patient_baselines',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('metric_name', sa.String(), nullable=False),
        sa.Column('baseline_mean', sa.Float(), nullable=False),
        sa.Column('baseline_std', sa.Float(), nullable=False),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('window_start', sa.DateTime(timezone=True), nullable=False),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=False),
        sa.Column('calculated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_patient_baselines_id', 'patient_baselines', ['id'])
    op.create_index('ix_patient_baselines_patient_id', 'patient_baselines', ['patient_id'])
    op.create_index('idx_baseline_patient_metric', 'patient_baselines', ['patient_id', 'metric_name'], unique=True)

    # Create trend_snapshots table (time-series risk assessments with Bayesian updates)
    op.create_table(
        'trend_snapshots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('risk_score', sa.Float(), nullable=False),
        sa.Column('risk_level', sa.String(), nullable=False),  # 'green', 'yellow', 'red'
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('anomaly_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('deviation_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('contributing_factors', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('wellness_recommendations', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('snapshot_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_trend_snapshots_id', 'trend_snapshots', ['id'])
    op.create_index('ix_trend_snapshots_patient_id', 'trend_snapshots', ['patient_id'])
    op.create_index('idx_trend_patient_time', 'trend_snapshots', ['patient_id', 'snapshot_timestamp'])
    op.create_index('idx_trend_risk_level', 'trend_snapshots', ['risk_level'])

    # Create risk_events table (critical risk level transitions for alert triggering)
    op.create_table(
        'risk_events',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('event_type', sa.String(), nullable=False),  # 'risk_increase', 'risk_decrease', 'anomaly_detected'
        sa.Column('previous_risk_level', sa.String(), nullable=True),
        sa.Column('new_risk_level', sa.String(), nullable=False),
        sa.Column('risk_delta', sa.Float(), nullable=True),
        sa.Column('triggered_alert', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('event_details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('event_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_risk_events_id', 'risk_events', ['id'])
    op.create_index('ix_risk_events_patient_id', 'risk_events', ['patient_id'])
    op.create_index('idx_risk_event_patient_time', 'risk_events', ['patient_id', 'event_timestamp'])
    op.create_index('idx_risk_event_type', 'risk_events', ['event_type'])

    # Create alert_rules table (configurable alert rules for doctors)
    op.create_table(
        'alert_rules',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('doctor_id', sa.String(), nullable=False),
        sa.Column('rule_name', sa.String(), nullable=False),
        sa.Column('rule_type', sa.String(), nullable=False),  # 'risk_threshold', 'metric_deviation', 'trend_change'
        sa.Column('conditions', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('notification_channels', postgresql.ARRAY(sa.String()), nullable=False),  # ['dashboard', 'email', 'sms']
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_alert_rules_id', 'alert_rules', ['id'])
    op.create_index('idx_alert_rule_doctor', 'alert_rules', ['doctor_id'])
    op.create_index('idx_alert_rule_type', 'alert_rules', ['rule_type'])
    op.create_index('idx_alert_rule_active', 'alert_rules', ['is_active'])

    # Create alerts table (generated alerts with delivery tracking)
    op.create_table(
        'alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('rule_id', sa.Integer(), nullable=True),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('doctor_id', sa.String(), nullable=False),
        sa.Column('alert_type', sa.String(), nullable=False),
        sa.Column('severity', sa.String(), nullable=False),  # 'low', 'medium', 'high', 'critical'
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('status', sa.String(), nullable=False, server_default='pending'),  # 'pending', 'sent', 'acknowledged', 'resolved'
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('acknowledged_by', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['rule_id'], ['alert_rules.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_alerts_id', 'alerts', ['id'])
    op.create_index('ix_alerts_rule_id', 'alerts', ['rule_id'])
    op.create_index('ix_alerts_patient_id', 'alerts', ['patient_id'])
    op.create_index('idx_alert_doctor', 'alerts', ['doctor_id'])
    op.create_index('idx_alert_status', 'alerts', ['status'])
    op.create_index('idx_alert_severity', 'alerts', ['severity'])

    # Create consent_records table (HIPAA-compliant granular consent management)
    op.create_table(
        'consent_records',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('consent_type', sa.String(), nullable=False),  # 'video_analysis', 'audio_analysis', 'ai_processing', 'data_sharing'
        sa.Column('consent_given', sa.Boolean(), nullable=False),
        sa.Column('consent_timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('withdrawn', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('withdrawn_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('consent_version', sa.String(), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_consent_records_id', 'consent_records', ['id'])
    op.create_index('ix_consent_records_patient_id', 'consent_records', ['patient_id'])
    op.create_index('idx_consent_patient_type', 'consent_records', ['patient_id', 'consent_type'])
    op.create_index('idx_consent_status', 'consent_records', ['consent_given'])
    op.create_index('idx_consent_withdrawn', 'consent_records', ['withdrawn'])

    # Create audit_logs table (comprehensive PHI access logging for HIPAA compliance)
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('user_role', sa.String(), nullable=True),
        sa.Column('action_type', sa.String(), nullable=False),  # 'view', 'create', 'update', 'delete', 'export'
        sa.Column('resource_type', sa.String(), nullable=False),
        sa.Column('resource_id', sa.String(), nullable=True),
        sa.Column('patient_id_accessed', sa.String(), nullable=True),
        sa.Column('phi_accessed', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('data_fields_accessed', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('access_justification', sa.Text(), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('request_details', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_audit_logs_id', 'audit_logs', ['id'])
    op.create_index('ix_audit_logs_user_id', 'audit_logs', ['user_id'])
    op.create_index('ix_audit_logs_action_type', 'audit_logs', ['action_type'])
    op.create_index('ix_audit_logs_patient_id_accessed', 'audit_logs', ['patient_id_accessed'])
    op.create_index('ix_audit_logs_phi_accessed', 'audit_logs', ['phi_accessed'])
    op.create_index('ix_audit_logs_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('ix_audit_logs_ip_address', 'audit_logs', ['ip_address'])
    op.create_index('idx_audit_user_action', 'audit_logs', ['user_id', 'action_type'])
    op.create_index('idx_audit_patient_access', 'audit_logs', ['patient_id_accessed', 'timestamp'])
    op.create_index('idx_audit_phi_access', 'audit_logs', ['phi_accessed', 'timestamp'])
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'])

    # Create data_retention_policies table (HIPAA retention schedule management)
    op.create_table(
        'data_retention_policies',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('data_type', sa.String(), nullable=False),
        sa.Column('retention_days', sa.Integer(), nullable=False),
        sa.Column('deletion_method', sa.String(), nullable=False),  # 'soft_delete', 'hard_delete', 'archive'
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_data_retention_policies_id', 'data_retention_policies', ['id'])
    op.create_index('idx_retention_data_type', 'data_retention_policies', ['data_type'], unique=True)
    op.create_index('idx_retention_active', 'data_retention_policies', ['is_active'])


def downgrade():
    # Drop tables in reverse order to respect foreign key constraints
    op.drop_index('idx_retention_active', table_name='data_retention_policies')
    op.drop_index('idx_retention_data_type', table_name='data_retention_policies')
    op.drop_index('ix_data_retention_policies_id', table_name='data_retention_policies')
    op.drop_table('data_retention_policies')

    op.drop_index('idx_audit_timestamp', table_name='audit_logs')
    op.drop_index('idx_audit_phi_access', table_name='audit_logs')
    op.drop_index('idx_audit_patient_access', table_name='audit_logs')
    op.drop_index('idx_audit_user_action', table_name='audit_logs')
    op.drop_index('ix_audit_logs_ip_address', table_name='audit_logs')
    op.drop_index('ix_audit_logs_timestamp', table_name='audit_logs')
    op.drop_index('ix_audit_logs_phi_accessed', table_name='audit_logs')
    op.drop_index('ix_audit_logs_patient_id_accessed', table_name='audit_logs')
    op.drop_index('ix_audit_logs_action_type', table_name='audit_logs')
    op.drop_index('ix_audit_logs_user_id', table_name='audit_logs')
    op.drop_index('ix_audit_logs_id', table_name='audit_logs')
    op.drop_table('audit_logs')

    op.drop_index('idx_consent_withdrawn', table_name='consent_records')
    op.drop_index('idx_consent_status', table_name='consent_records')
    op.drop_index('idx_consent_patient_type', table_name='consent_records')
    op.drop_index('ix_consent_records_patient_id', table_name='consent_records')
    op.drop_index('ix_consent_records_id', table_name='consent_records')
    op.drop_table('consent_records')

    op.drop_index('idx_alert_severity', table_name='alerts')
    op.drop_index('idx_alert_status', table_name='alerts')
    op.drop_index('idx_alert_doctor', table_name='alerts')
    op.drop_index('ix_alerts_patient_id', table_name='alerts')
    op.drop_index('ix_alerts_rule_id', table_name='alerts')
    op.drop_index('ix_alerts_id', table_name='alerts')
    op.drop_table('alerts')

    op.drop_index('idx_alert_rule_active', table_name='alert_rules')
    op.drop_index('idx_alert_rule_type', table_name='alert_rules')
    op.drop_index('idx_alert_rule_doctor', table_name='alert_rules')
    op.drop_index('ix_alert_rules_id', table_name='alert_rules')
    op.drop_table('alert_rules')

    op.drop_index('idx_risk_event_type', table_name='risk_events')
    op.drop_index('idx_risk_event_patient_time', table_name='risk_events')
    op.drop_index('ix_risk_events_patient_id', table_name='risk_events')
    op.drop_index('ix_risk_events_id', table_name='risk_events')
    op.drop_table('risk_events')

    op.drop_index('idx_trend_risk_level', table_name='trend_snapshots')
    op.drop_index('idx_trend_patient_time', table_name='trend_snapshots')
    op.drop_index('ix_trend_snapshots_patient_id', table_name='trend_snapshots')
    op.drop_index('ix_trend_snapshots_id', table_name='trend_snapshots')
    op.drop_table('trend_snapshots')

    op.drop_index('idx_baseline_patient_metric', table_name='patient_baselines')
    op.drop_index('ix_patient_baselines_patient_id', table_name='patient_baselines')
    op.drop_index('ix_patient_baselines_id', table_name='patient_baselines')
    op.drop_table('patient_baselines')

    op.drop_index('idx_audio_patient_timestamp', table_name='audio_metrics')
    op.drop_index('ix_audio_metrics_patient_id', table_name='audio_metrics')
    op.drop_index('ix_audio_metrics_session_id', table_name='audio_metrics')
    op.drop_index('ix_audio_metrics_id', table_name='audio_metrics')
    op.drop_table('audio_metrics')

    op.drop_index('idx_video_patient_timestamp', table_name='video_metrics')
    op.drop_index('ix_video_metrics_patient_id', table_name='video_metrics')
    op.drop_index('ix_video_metrics_session_id', table_name='video_metrics')
    op.drop_index('ix_video_metrics_id', table_name='video_metrics')
    op.drop_table('video_metrics')

    op.drop_index('idx_media_status', table_name='media_sessions')
    op.drop_index('idx_media_patient_type', table_name='media_sessions')
    op.drop_index('ix_media_sessions_patient_id', table_name='media_sessions')
    op.drop_index('ix_media_sessions_id', table_name='media_sessions')
    op.drop_table('media_sessions')
