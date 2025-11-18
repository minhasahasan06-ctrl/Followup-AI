"""add skin analysis metrics and baselines

Revision ID: a0cc11c8c80e
Revises: fps_resp_001
Create Date: 2025-11-18 19:10:45.672454

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'a0cc11c8c80e'
down_revision = 'fps_resp_001'
branch_labels = None
depends_on = None


def upgrade():
    # Create skin_analysis_metrics table
    op.create_table(
        'skin_analysis_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, primary_key=True),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('session_id', sa.String(), nullable=True),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        
        # LAB Color Space Metrics - Facial
        sa.Column('facial_l_lightness', sa.Float()),
        sa.Column('facial_a_red_green', sa.Float()),
        sa.Column('facial_b_yellow_blue', sa.Float()),
        sa.Column('facial_perfusion_index', sa.Float()),
        
        # LAB Color Space Metrics - Palmar
        sa.Column('palmar_l_lightness', sa.Float()),
        sa.Column('palmar_a_red_green', sa.Float()),
        sa.Column('palmar_b_yellow_blue', sa.Float()),
        sa.Column('palmar_perfusion_index', sa.Float()),
        
        # LAB Color Space Metrics - Nailbed
        sa.Column('nailbed_l_lightness', sa.Float()),
        sa.Column('nailbed_a_red_green', sa.Float()),
        sa.Column('nailbed_b_yellow_blue', sa.Float()),
        sa.Column('nailbed_color_index', sa.Float()),
        
        # Clinical Color Changes - Pallor
        sa.Column('pallor_detected', sa.Boolean(), server_default='false'),
        sa.Column('pallor_severity', sa.Float()),
        sa.Column('pallor_region', sa.String()),
        
        # Clinical Color Changes - Cyanosis
        sa.Column('cyanosis_detected', sa.Boolean(), server_default='false'),
        sa.Column('cyanosis_severity', sa.Float()),
        sa.Column('cyanosis_region', sa.String()),
        
        # Clinical Color Changes - Jaundice
        sa.Column('jaundice_detected', sa.Boolean(), server_default='false'),
        sa.Column('jaundice_severity', sa.Float()),
        sa.Column('jaundice_region', sa.String()),
        
        # Capillary Refill
        sa.Column('capillary_refill_time_sec', sa.Float()),
        sa.Column('capillary_refill_method', sa.String()),
        sa.Column('capillary_refill_quality', sa.Float()),
        sa.Column('capillary_refill_abnormal', sa.Boolean(), server_default='false'),
        
        # Nailbed Analysis
        sa.Column('nail_clubbing_detected', sa.Boolean(), server_default='false'),
        sa.Column('nail_clubbing_severity', sa.Float()),
        sa.Column('nail_pitting_detected', sa.Boolean(), server_default='false'),
        sa.Column('nail_pitting_count', sa.Integer()),
        sa.Column('nail_abnormalities', postgresql.JSON()),
        
        # Texture & Temperature
        sa.Column('skin_texture_score', sa.Float()),
        sa.Column('hydration_status', sa.String()),
        sa.Column('temperature_proxy', sa.String()),
        
        # Rash/Lesion Detection
        sa.Column('rash_detected', sa.Boolean(), server_default='false'),
        sa.Column('rash_distribution', sa.String()),
        sa.Column('lesions_bruises_detected', sa.Boolean(), server_default='false'),
        sa.Column('lesion_details', postgresql.JSON()),
        
        # Baseline Comparison
        sa.Column('z_score_perfusion_vs_baseline', sa.Float()),
        sa.Column('z_score_capillary_vs_baseline', sa.Float()),
        sa.Column('z_score_nailbed_vs_baseline', sa.Float()),
        
        # Rolling Temporal Analytics
        sa.Column('rolling_24hr_avg_perfusion', sa.Float()),
        sa.Column('rolling_3day_perfusion_slope', sa.Float()),
        sa.Column('rolling_24hr_avg_capillary_refill', sa.Float()),
        
        # Detection Quality
        sa.Column('detection_confidence', sa.Float(), server_default='0.0'),
        sa.Column('frames_analyzed', sa.Integer(), server_default='0'),
        sa.Column('facial_roi_detected', sa.Boolean(), server_default='false'),
        sa.Column('palmar_roi_detected', sa.Boolean(), server_default='false'),
        sa.Column('nailbed_roi_detected', sa.Boolean(), server_default='false'),
        
        # Metadata
        sa.Column('metrics_metadata', postgresql.JSON()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['session_id'], ['video_exam_sessions.id'], ondelete='CASCADE')
    )
    
    # Create indices for efficient time-series queries
    op.create_index('idx_skin_patient_time', 'skin_analysis_metrics', ['patient_id', 'recorded_at'])
    op.create_index('idx_skin_session', 'skin_analysis_metrics', ['session_id'])
    
    # Create skin_baselines table
    op.create_table(
        'skin_baselines',
        sa.Column('patient_id', sa.String(), primary_key=True),
        
        # Baseline LAB Color Vectors - Facial
        sa.Column('baseline_facial_l', sa.Float(), nullable=False),
        sa.Column('baseline_facial_a', sa.Float(), nullable=False),
        sa.Column('baseline_facial_b', sa.Float(), nullable=False),
        sa.Column('baseline_facial_perfusion', sa.Float(), nullable=False),
        
        # Baseline LAB Color Vectors - Palmar
        sa.Column('baseline_palmar_l', sa.Float(), nullable=False),
        sa.Column('baseline_palmar_a', sa.Float(), nullable=False),
        sa.Column('baseline_palmar_b', sa.Float(), nullable=False),
        sa.Column('baseline_palmar_perfusion', sa.Float(), nullable=False),
        
        # Baseline LAB Color Vectors - Nailbed
        sa.Column('baseline_nailbed_l', sa.Float(), nullable=False),
        sa.Column('baseline_nailbed_a', sa.Float(), nullable=False),
        sa.Column('baseline_nailbed_b', sa.Float(), nullable=False),
        sa.Column('baseline_nailbed_color_index', sa.Float(), nullable=False),
        
        # Capillary Refill Baseline
        sa.Column('baseline_capillary_refill_sec', sa.Float(), nullable=False, server_default='1.5'),
        sa.Column('baseline_capillary_refill_std', sa.Float(), nullable=False, server_default='0.3'),
        
        # Texture/Hydration Baseline
        sa.Column('baseline_texture_score', sa.Float()),
        sa.Column('baseline_hydration_status', sa.String()),
        
        # Baseline Quality
        sa.Column('sample_size', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        
        # Metadata
        sa.Column('source', sa.String(), server_default='auto'),
        sa.Column('last_calibration_at', sa.DateTime(timezone=True)),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), onupdate=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        
        sa.PrimaryKeyConstraint('patient_id'),
        sa.ForeignKeyConstraint(['patient_id'], ['users.id'], ondelete='CASCADE')
    )


def downgrade():
    op.drop_index('idx_skin_session', table_name='skin_analysis_metrics')
    op.drop_index('idx_skin_patient_time', table_name='skin_analysis_metrics')
    op.drop_table('skin_baselines')
    op.drop_table('skin_analysis_metrics')
