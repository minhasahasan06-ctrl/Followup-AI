"""add_whatsapp_business_hours_only

Revision ID: b1c2d3e4f5g6
Revises: 9aebc6a2f848
Create Date: 2025-11-30 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'b1c2d3e4f5g6'
down_revision = 'f2181dcadde9'
branch_labels = None
depends_on = None


def upgrade():
    """Add business_hours_only column to whatsapp_automation_configs table if it exists"""
    from sqlalchemy import inspect
    from alembic import context
    
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    
    if 'whatsapp_automation_configs' in tables:
        columns = [col['name'] for col in inspector.get_columns('whatsapp_automation_configs')]
        if 'business_hours_only' not in columns:
            op.add_column('whatsapp_automation_configs', 
                sa.Column('business_hours_only', sa.Boolean(), nullable=True, server_default='true'))
            op.execute("UPDATE whatsapp_automation_configs SET business_hours_only = true WHERE business_hours_only IS NULL")


def downgrade():
    """Remove business_hours_only column from whatsapp_automation_configs table if it exists"""
    from sqlalchemy import inspect
    
    bind = op.get_bind()
    inspector = inspect(bind)
    tables = inspector.get_table_names()
    
    if 'whatsapp_automation_configs' in tables:
        columns = [col['name'] for col in inspector.get_columns('whatsapp_automation_configs')]
        if 'business_hours_only' in columns:
            op.drop_column('whatsapp_automation_configs', 'business_hours_only')
