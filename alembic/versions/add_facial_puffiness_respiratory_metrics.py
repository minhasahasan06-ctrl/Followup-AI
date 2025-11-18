"""add facial puffiness and respiratory metrics tables

Revision ID: fps_resp_001
Revises: f59ec9fe766d
Create Date: 2025-11-18

This migration creates tables for:
- Facial Puffiness Score (FPS) system with MediaPipe Face Mesh tracking
- Respiratory metrics with temporal analytics and baseline tracking
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'fps_resp_001'
down_revision = 'f59ec9fe766d'
branch_labels = None
depends_on = None


def upgrade():
    # Create facial_puffiness_baselines table
    op.create_table(
        'facial_puffiness_baselines',
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('baseline_eye_area', sa.Float(), nullable=False),
        sa.Column('baseline_cheek_width', sa.Float(), nullable=False),
        sa.Column('baseline_cheek_projection', sa.Float(), nullable=False),
        sa.Column('baseline_jawline_width', sa.Float(), nullable=False),
        sa.Column('baseline_forehead_width', sa.Float(), nullable=False),
        sa.Column('baseline_face_perimeter', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('source', sa.String(), server_default='auto'),
        sa.Column('last_calibration_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('patient_id'),
        sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create facial_puffiness_metrics table
    op.create_table(
        'facial_puffiness_metrics',
        sa.Column('id', sa.String(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String()),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('facial_puffiness_score', sa.Float()),
        sa.Column('fps_risk_level', sa.String()),
        sa.Column('fps_periorbital', sa.Float()),
        sa.Column('fps_cheek', sa.Float()),
        sa.Column('fps_jawline', sa.Float()),
        sa.Column('fps_forehead', sa.Float()),
        sa.Column('fps_overall_contour', sa.Float()),
        sa.Column('facial_asymmetry_score', sa.Float()),
        sa.Column('asymmetry_detected', sa.Boolean(), server_default='false'),
        sa.Column('raw_eye_area', sa.Float()),
        sa.Column('raw_cheek_width', sa.Float()),
        sa.Column('raw_cheek_projection', sa.Float()),
        sa.Column('raw_jawline_width', sa.Float()),
        sa.Column('raw_forehead_width', sa.Float()),
        sa.Column('raw_face_perimeter', sa.Float()),
        sa.Column('detection_confidence', sa.Float(), server_default='0.0'),
        sa.Column('frames_analyzed', sa.Integer(), server_default='0'),
        sa.Column('metrics_metadata', postgresql.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['video_exam_sessions.id'], ondelete='CASCADE')
    )
    op.create_index('idx_fps_patient_time', 'facial_puffiness_metrics', ['patient_id', 'recorded_at'])
    
    # Create respiratory_baselines table
    op.create_table(
        'respiratory_baselines',
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('baseline_rr_bpm', sa.Float(), nullable=False),
        sa.Column('baseline_rr_std', sa.Float(), nullable=False, server_default='2.0'),
        sa.Column('sample_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('source', sa.String(), server_default='auto'),
        sa.Column('last_calibration_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('patient_id'),
        sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ondelete='CASCADE')
    )
    
    # Create respiratory_metrics table
    op.create_table(
        'respiratory_metrics',
        sa.Column('id', sa.String(), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String()),
        sa.Column('recorded_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('rr_bpm', sa.Float()),
        sa.Column('rr_confidence', sa.Float(), server_default='0.0'),
        sa.Column('breath_interval_std', sa.Float()),
        sa.Column('variability_index', sa.Float()),
        sa.Column('accessory_muscle_score', sa.Float()),
        sa.Column('chest_expansion_amplitude', sa.Float()),
        sa.Column('gasping_detected', sa.Boolean(), server_default='false'),
        sa.Column('chest_shape_asymmetry', sa.Float()),
        sa.Column('thoracoabdominal_synchrony', sa.Float()),
        sa.Column('z_score_vs_baseline', sa.Float()),
        sa.Column('rolling_daily_avg', sa.Float()),
        sa.Column('rolling_three_day_slope', sa.Float()),
        sa.Column('metrics_metadata', postgresql.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['video_exam_sessions.id'], ondelete='CASCADE')
    )
    op.create_index('idx_resp_patient_time', 'respiratory_metrics', ['patient_id', 'recorded_at'])


def downgrade():
    # Drop tables in reverse order (metrics before baselines)
    op.drop_index('idx_resp_patient_time', 'respiratory_metrics')
    op.drop_table('respiratory_metrics')
    op.drop_table('respiratory_baselines')
    
    op.drop_index('idx_fps_patient_time', 'facial_puffiness_metrics')
    op.drop_table('facial_puffiness_metrics')
    op.drop_table('facial_puffiness_baselines')
