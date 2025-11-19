"""create_video_exam_sessions_table

Revision ID: f2181dcadde9
Revises: 9aebc6a2f848
Create Date: 2025-11-19 07:14:58.133092

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON


# revision identifiers, used by Alembic.
revision = 'f2181dcadde9'
down_revision = '9aebc6a2f848'
branch_labels = None
depends_on = None


def upgrade():
    # Create video_exam_sessions table for guided video examination workflow
    op.create_table(
        'video_exam_sessions',
        sa.Column('id', sa.String(), nullable=False, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('patient_id', sa.String(), nullable=False),
        
        # Session lifecycle
        sa.Column('status', sa.String(), nullable=False, server_default='in_progress'),  # in_progress, completed, failed
        sa.Column('current_stage', sa.String(), nullable=True),  # eyes, palm, tongue, lips
        
        # Frame storage per stage (S3 URIs)
        sa.Column('eyes_frame_s3_uri', sa.String(), nullable=True),
        sa.Column('palm_frame_s3_uri', sa.String(), nullable=True),
        sa.Column('tongue_frame_s3_uri', sa.String(), nullable=True),
        sa.Column('lips_frame_s3_uri', sa.String(), nullable=True),
        
        # Stage completion tracking
        sa.Column('eyes_stage_completed', sa.Boolean(), default=False),
        sa.Column('palm_stage_completed', sa.Boolean(), default=False),
        sa.Column('tongue_stage_completed', sa.Boolean(), default=False),
        sa.Column('lips_stage_completed', sa.Boolean(), default=False),
        
        # Quality scores per stage (0-100)
        sa.Column('eyes_quality_score', sa.Float(), nullable=True),
        sa.Column('palm_quality_score', sa.Float(), nullable=True),
        sa.Column('tongue_quality_score', sa.Float(), nullable=True),
        sa.Column('lips_quality_score', sa.Float(), nullable=True),
        
        # Overall session quality
        sa.Column('overall_quality_score', sa.Float(), nullable=True),
        
        # ML analysis reference (links to VideoMetrics after completion)
        sa.Column('video_metrics_id', sa.Integer(), nullable=True),
        
        # Exam metadata
        sa.Column('prep_time_seconds', sa.Integer(), default=30),  # 30-second prep screens
        sa.Column('total_duration_seconds', sa.Float(), nullable=True),
        sa.Column('device_info', JSON(), nullable=True),  # Browser/device metadata
        
        # Error tracking
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indices for efficient queries
    op.create_index('idx_video_exam_patient_id', 'video_exam_sessions', ['patient_id'])
    op.create_index('idx_video_exam_status', 'video_exam_sessions', ['status'])
    op.create_index('idx_video_exam_created_at', 'video_exam_sessions', ['created_at'])


def downgrade():
    op.drop_index('idx_video_exam_created_at', table_name='video_exam_sessions')
    op.drop_index('idx_video_exam_status', table_name='video_exam_sessions')
    op.drop_index('idx_video_exam_patient_id', table_name='video_exam_sessions')
    op.drop_table('video_exam_sessions')
