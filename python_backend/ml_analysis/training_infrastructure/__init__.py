"""
ML Training Infrastructure
===========================
Production-grade ML training infrastructure with:
- Training Job Queue - Redis-backed job queue with priority support
- Training Job Worker - Background worker for processing jobs  
- Consent Enforcer - Patient consent verification for data pipelines
- Governance Hooks - Pre/post-build validation and PHI detection
- Artifact Storage - Secure model artifact storage with versioning
- Model Versioning - Semantic versioning with deployment tracking
- Training API - FastAPI router for all training operations
- Training Security - JWT authentication, structured failures, audit logging

All components are HIPAA-compliant with:
- JWT-based admin authentication
- Database-backed audit logging to autopilot_audit_logs
- Structured failure types for consent/governance errors
"""

from .training_job_queue import TrainingJobQueue, TrainingJob, JobStatus
from .training_job_worker import TrainingJobWorker
from .consent_enforcer import ConsentEnforcer, ConsentLevel, DataCategory, ConsentCheckResult
from .governance_hooks import GovernanceHooks, GovernanceAction, GovernanceResult, GovernanceCheckResult
from .artifact_storage import ArtifactStorage, ModelArtifact
from .model_versioning import ModelVersionManager, ModelVersion, DeploymentStatus
from .training_api import router as training_router
from .training_security import (
    TrainingAuditLogger,
    TrainingAuditAction,
    TrainingFailure,
    FailureType,
    AdminUser,
    get_audit_logger,
    verify_admin_token
)

__all__ = [
    # Job Queue
    'TrainingJobQueue',
    'TrainingJob', 
    'JobStatus',
    
    # Worker
    'TrainingJobWorker',
    
    # Consent
    'ConsentEnforcer',
    'ConsentLevel',
    'DataCategory',
    'ConsentCheckResult',
    
    # Governance
    'GovernanceHooks',
    'GovernanceAction',
    'GovernanceResult',
    'GovernanceCheckResult',
    
    # Storage
    'ArtifactStorage',
    'ModelArtifact',
    
    # Versioning
    'ModelVersionManager',
    'ModelVersion',
    'DeploymentStatus',
    
    # Security & Audit
    'TrainingAuditLogger',
    'TrainingAuditAction',
    'TrainingFailure',
    'FailureType',
    'AdminUser',
    'get_audit_logger',
    'verify_admin_token',
    
    # API
    'training_router',
]
