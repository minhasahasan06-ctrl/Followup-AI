"""add_hepatic_anemia_color_metrics

Revision ID: 9aebc6a2f848
Revises: a0cc11c8c80e
Create Date: 2025-11-19 06:59:06.026346

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9aebc6a2f848'
down_revision = 'a0cc11c8c80e'
branch_labels = None
depends_on = None


def upgrade():
    """Add hepatic/anemia color metrics to video_metrics table"""
    
    # Scleral Metrics (Jaundice Detection)
    op.add_column('video_metrics', sa.Column('scleral_chromaticity_index', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('scleral_skin_delta', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('scleral_l_lightness', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('scleral_a_red_green', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('scleral_b_yellow_blue', sa.Float(), nullable=True))
    
    # Conjunctival Metrics (Anemia Detection from Inner Eyelid)
    op.add_column('video_metrics', sa.Column('conjunctival_pallor_index', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('conjunctival_red_saturation', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('conjunctival_l_lightness', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('conjunctival_a_red_green', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('conjunctival_b_yellow_blue', sa.Float(), nullable=True))
    
    # Palmar Metrics (LAB-based Pallor Index)
    op.add_column('video_metrics', sa.Column('palmar_pallor_lab_index', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('palmar_l_lightness', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('palmar_a_red_green', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('palmar_b_yellow_blue', sa.Float(), nullable=True))
    
    # Tongue Color Metrics (LAB Color Space)
    op.add_column('video_metrics', sa.Column('tongue_color_index', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('tongue_color_l', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('tongue_color_a', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('tongue_color_b', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('tongue_coating_detected', sa.Boolean(), nullable=True))
    op.add_column('video_metrics', sa.Column('tongue_coating_color', sa.String(), nullable=True))
    
    # Lip Color and Hydration Metrics
    op.add_column('video_metrics', sa.Column('lip_hydration_score', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('lip_color_l', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('lip_color_a', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('lip_color_b', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('lip_dryness_score', sa.Float(), nullable=True))
    op.add_column('video_metrics', sa.Column('lip_cyanosis_detected', sa.Boolean(), nullable=True))
    
    # Detection Quality Indicators
    op.add_column('video_metrics', sa.Column('scleral_roi_detected', sa.Boolean(), nullable=True))
    op.add_column('video_metrics', sa.Column('conjunctival_roi_detected', sa.Boolean(), nullable=True))
    op.add_column('video_metrics', sa.Column('tongue_roi_detected', sa.Boolean(), nullable=True))
    op.add_column('video_metrics', sa.Column('lip_roi_detected', sa.Boolean(), nullable=True))
    op.add_column('video_metrics', sa.Column('palmar_roi_detected', sa.Boolean(), nullable=True))
    
    # Exam Session Metadata
    op.add_column('video_metrics', sa.Column('guided_exam_session_id', sa.String(), nullable=True))
    op.add_column('video_metrics', sa.Column('exam_stage', sa.String(), nullable=True))


def downgrade():
    """Remove hepatic/anemia color metrics from video_metrics table"""
    
    # Scleral Metrics
    op.drop_column('video_metrics', 'scleral_chromaticity_index')
    op.drop_column('video_metrics', 'scleral_skin_delta')
    op.drop_column('video_metrics', 'scleral_l_lightness')
    op.drop_column('video_metrics', 'scleral_a_red_green')
    op.drop_column('video_metrics', 'scleral_b_yellow_blue')
    
    # Conjunctival Metrics
    op.drop_column('video_metrics', 'conjunctival_pallor_index')
    op.drop_column('video_metrics', 'conjunctival_red_saturation')
    op.drop_column('video_metrics', 'conjunctival_l_lightness')
    op.drop_column('video_metrics', 'conjunctival_a_red_green')
    op.drop_column('video_metrics', 'conjunctival_b_yellow_blue')
    
    # Palmar Metrics
    op.drop_column('video_metrics', 'palmar_pallor_lab_index')
    op.drop_column('video_metrics', 'palmar_l_lightness')
    op.drop_column('video_metrics', 'palmar_a_red_green')
    op.drop_column('video_metrics', 'palmar_b_yellow_blue')
    
    # Tongue Metrics
    op.drop_column('video_metrics', 'tongue_color_index')
    op.drop_column('video_metrics', 'tongue_color_l')
    op.drop_column('video_metrics', 'tongue_color_a')
    op.drop_column('video_metrics', 'tongue_color_b')
    op.drop_column('video_metrics', 'tongue_coating_detected')
    op.drop_column('video_metrics', 'tongue_coating_color')
    
    # Lip Metrics
    op.drop_column('video_metrics', 'lip_hydration_score')
    op.drop_column('video_metrics', 'lip_color_l')
    op.drop_column('video_metrics', 'lip_color_a')
    op.drop_column('video_metrics', 'lip_color_b')
    op.drop_column('video_metrics', 'lip_dryness_score')
    op.drop_column('video_metrics', 'lip_cyanosis_detected')
    
    # Detection Quality
    op.drop_column('video_metrics', 'scleral_roi_detected')
    op.drop_column('video_metrics', 'conjunctival_roi_detected')
    op.drop_column('video_metrics', 'tongue_roi_detected')
    op.drop_column('video_metrics', 'lip_roi_detected')
    op.drop_column('video_metrics', 'palmar_roi_detected')
    
    # Exam Metadata
    op.drop_column('video_metrics', 'guided_exam_session_id')
    op.drop_column('video_metrics', 'exam_stage')
