"""add ml inference tables

Revision ID: ml_inference_001
Revises: 
Create Date: 2025-11-16

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'ml_inference_001'
down_revision = None  # TODO: Update with your latest migration ID
branch_labels = None
depends_on = None


def upgrade():
    # Create ml_models table
    op.create_table(
        'ml_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('version', sa.String(), nullable=False),
        sa.Column('model_type', sa.String(), nullable=False),
        sa.Column('task_type', sa.String(), nullable=False),
        sa.Column('file_path', sa.String(), nullable=True),
        sa.Column('input_schema', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('output_schema', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True, default=True),
        sa.Column('is_deployed', sa.Boolean(), nullable=True, default=False),
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ml_models_id'), 'ml_models', ['id'], unique=False)
    op.create_index(op.f('ix_ml_models_name'), 'ml_models', ['name'], unique=False)
    op.create_index('idx_ml_model_name_version', 'ml_models', ['name', 'version'], unique=False)

    # Create ml_predictions table
    op.create_table(
        'ml_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('patient_id', sa.String(), nullable=False),
        sa.Column('prediction_type', sa.String(), nullable=False),
        sa.Column('input_data', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('prediction_result', postgresql.JSON(astext_type=sa.Text()), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('inference_time_ms', sa.Float(), nullable=True),
        sa.Column('cache_hit', sa.Boolean(), nullable=True, default=False),
        sa.Column('predicted_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('ip_address', sa.String(), nullable=True),
        sa.Column('user_agent', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ml_predictions_id'), 'ml_predictions', ['id'], unique=False)
    op.create_index(op.f('ix_ml_predictions_model_id'), 'ml_predictions', ['model_id'], unique=False)
    op.create_index(op.f('ix_ml_predictions_patient_id'), 'ml_predictions', ['patient_id'], unique=False)
    op.create_index(op.f('ix_ml_predictions_predicted_at'), 'ml_predictions', ['predicted_at'], unique=False)
    op.create_index('idx_ml_prediction_patient_type', 'ml_predictions', ['patient_id', 'prediction_type'], unique=False)
    op.create_index('idx_ml_prediction_created', 'ml_predictions', ['predicted_at'], unique=False)

    # Create ml_performance_logs table
    op.create_table(
        'ml_performance_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('metric_name', sa.String(), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(), nullable=True),
        sa.Column('sample_size', sa.Integer(), nullable=True),
        sa.Column('time_window_minutes', sa.Integer(), nullable=True),
        sa.Column('aggregation_type', sa.String(), nullable=True),
        sa.Column('measured_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('window_start', sa.DateTime(timezone=True), nullable=True),
        sa.Column('window_end', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ml_performance_logs_id'), 'ml_performance_logs', ['id'], unique=False)
    op.create_index(op.f('ix_ml_performance_logs_model_id'), 'ml_performance_logs', ['model_id'], unique=False)
    op.create_index(op.f('ix_ml_performance_logs_measured_at'), 'ml_performance_logs', ['measured_at'], unique=False)
    op.create_index('idx_ml_perf_model_metric', 'ml_performance_logs', ['model_id', 'metric_name'], unique=False)
    op.create_index('idx_ml_perf_measured', 'ml_performance_logs', ['measured_at'], unique=False)

    # Create ml_batch_jobs table
    op.create_table(
        'ml_batch_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('job_name', sa.String(), nullable=False),
        sa.Column('job_type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False, default='pending'),
        sa.Column('total_items', sa.Integer(), nullable=True),
        sa.Column('processed_items', sa.Integer(), nullable=True, default=0),
        sa.Column('failed_items', sa.Integer(), nullable=True, default=0),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('estimated_completion', sa.DateTime(timezone=True), nullable=True),
        sa.Column('results_summary', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('error_log', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_by', sa.String(), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ml_batch_jobs_id'), 'ml_batch_jobs', ['id'], unique=False)
    op.create_index(op.f('ix_ml_batch_jobs_model_id'), 'ml_batch_jobs', ['model_id'], unique=False)
    op.create_index(op.f('ix_ml_batch_jobs_created_at'), 'ml_batch_jobs', ['created_at'], unique=False)
    op.create_index('idx_ml_batch_status', 'ml_batch_jobs', ['status'], unique=False)
    op.create_index('idx_ml_batch_created', 'ml_batch_jobs', ['created_at'], unique=False)


def downgrade():
    # Drop tables in reverse order
    op.drop_index('idx_ml_batch_created', table_name='ml_batch_jobs')
    op.drop_index('idx_ml_batch_status', table_name='ml_batch_jobs')
    op.drop_index(op.f('ix_ml_batch_jobs_created_at'), table_name='ml_batch_jobs')
    op.drop_index(op.f('ix_ml_batch_jobs_model_id'), table_name='ml_batch_jobs')
    op.drop_index(op.f('ix_ml_batch_jobs_id'), table_name='ml_batch_jobs')
    op.drop_table('ml_batch_jobs')
    
    op.drop_index('idx_ml_perf_measured', table_name='ml_performance_logs')
    op.drop_index('idx_ml_perf_model_metric', table_name='ml_performance_logs')
    op.drop_index(op.f('ix_ml_performance_logs_measured_at'), table_name='ml_performance_logs')
    op.drop_index(op.f('ix_ml_performance_logs_model_id'), table_name='ml_performance_logs')
    op.drop_index(op.f('ix_ml_performance_logs_id'), table_name='ml_performance_logs')
    op.drop_table('ml_performance_logs')
    
    op.drop_index('idx_ml_prediction_created', table_name='ml_predictions')
    op.drop_index('idx_ml_prediction_patient_type', table_name='ml_predictions')
    op.drop_index(op.f('ix_ml_predictions_predicted_at'), table_name='ml_predictions')
    op.drop_index(op.f('ix_ml_predictions_patient_id'), table_name='ml_predictions')
    op.drop_index(op.f('ix_ml_predictions_model_id'), table_name='ml_predictions')
    op.drop_index(op.f('ix_ml_predictions_id'), table_name='ml_predictions')
    op.drop_table('ml_predictions')
    
    op.drop_index('idx_ml_model_name_version', table_name='ml_models')
    op.drop_index(op.f('ix_ml_models_name'), table_name='ml_models')
    op.drop_index(op.f('ix_ml_models_id'), table_name='ml_models')
    op.drop_table('ml_models')
