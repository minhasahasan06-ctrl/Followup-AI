"""
Research Storage Service
Production-grade storage service for research artifacts, datasets, and exports.
Reuses existing S3Service with research-specific functionality.
"""

import hashlib
import logging
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, BinaryIO
from uuid import uuid4
import asyncio

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.services.s3_service import s3_service
from app.services.access_control import HIPAAAuditLogger
from app.models.research_models import (
    AnalysisArtifact, 
    ResearchDataset, 
    DatasetLineage,
    ResearchExport,
    PHILevel,
    VisibilityScope,
    JobStatus,
)

logger = logging.getLogger(__name__)

K_ANONYMITY_THRESHOLD = 5
SIGNED_URL_EXPIRY_SECONDS = 900


class ResearchStorageService:
    """
    Unified storage service for research artifacts, datasets, and exports.
    Provides S3/local storage with HIPAA compliance and audit logging.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.s3 = s3_service
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA-256 checksum for data integrity"""
        return hashlib.sha256(data).hexdigest()
    
    def _log_storage_access(
        self,
        actor_id: str,
        actor_role: str,
        action: str,
        resource_type: str,
        resource_id: str,
        phi_level: str = "de_identified",
        additional_context: Optional[Dict] = None
    ) -> str:
        """Log storage operation for HIPAA audit trail"""
        return HIPAAAuditLogger.log_phi_access(
            actor_id=actor_id,
            actor_role=actor_role,
            patient_id="aggregate",
            action=action,
            phi_categories=["research_data"],
            resource_type=resource_type,
            resource_id=resource_id,
            access_scope="research",
            access_reason="research_storage",
            consent_verified=True,
            additional_context=additional_context or {},
        )
    
    async def upload_artifact(
        self,
        job_id: Optional[str],
        study_id: str,
        file_data: bytes,
        filename: str,
        artifact_type: str,
        format: str,
        created_by: str,
        phi_level: str = PHILevel.DE_IDENTIFIED.value,
        visibility_scope: str = VisibilityScope.STUDY.value,
        metadata: Optional[Dict] = None,
        retention_days: int = 365,
    ) -> AnalysisArtifact:
        """
        Upload a research artifact to storage.
        
        Args:
            job_id: Optional study job ID that produced this artifact
            study_id: Study this artifact belongs to
            file_data: Raw bytes of the artifact
            filename: Original filename
            artifact_type: Type of artifact (plot, model, report, table)
            format: File format (png, json, csv, pkl, pdf)
            created_by: User ID creating the artifact
            phi_level: PHI classification level
            visibility_scope: Who can access this artifact
            metadata: Additional metadata dict
            retention_days: Days before artifact expires
        
        Returns:
            Created AnalysisArtifact record
        """
        artifact_id = str(uuid4())
        checksum = self._compute_checksum(file_data)
        
        s3_key = f"research/artifacts/{study_id}/{artifact_type}/{artifact_id}/{filename}"
        
        content_type = self._get_content_type(format)
        
        storage_uri = await self.s3.upload_file(
            file_data=file_data,
            s3_key=s3_key,
            content_type=content_type,
            metadata={
                "artifact_id": artifact_id,
                "study_id": study_id,
                "phi_level": phi_level,
                **(metadata or {})
            }
        )
        
        artifact = AnalysisArtifact(
            id=artifact_id,
            job_id=job_id,
            study_id=study_id,
            artifact_type=artifact_type,
            format=format,
            filename=filename,
            storage_uri=storage_uri,
            size_bytes=len(file_data),
            checksum=checksum,
            phi_level=phi_level,
            visibility_scope=visibility_scope,
            retention_days=retention_days,
            expires_at=datetime.utcnow() + timedelta(days=retention_days),
            metadata_json=metadata,
            created_by=created_by,
        )
        
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        
        self._log_storage_access(
            actor_id=created_by,
            actor_role="researcher",
            action="upload_artifact",
            resource_type="analysis_artifact",
            resource_id=artifact_id,
            phi_level=phi_level,
            additional_context={
                "study_id": study_id,
                "artifact_type": artifact_type,
                "size_bytes": len(file_data),
            }
        )
        
        logger.info(f"Uploaded artifact {artifact_id} for study {study_id}")
        return artifact
    
    def get_artifact(self, artifact_id: str) -> Optional[AnalysisArtifact]:
        """Get artifact by ID"""
        return self.db.query(AnalysisArtifact).filter(
            and_(
                AnalysisArtifact.id == artifact_id,
                AnalysisArtifact.deleted_at.is_(None)
            )
        ).first()
    
    def list_artifacts(
        self,
        study_id: Optional[str] = None,
        job_id: Optional[str] = None,
        artifact_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[AnalysisArtifact]:
        """List artifacts with optional filters"""
        query = self.db.query(AnalysisArtifact).filter(
            AnalysisArtifact.deleted_at.is_(None)
        )
        
        if study_id:
            query = query.filter(AnalysisArtifact.study_id == study_id)
        if job_id:
            query = query.filter(AnalysisArtifact.job_id == job_id)
        if artifact_type:
            query = query.filter(AnalysisArtifact.artifact_type == artifact_type)
        
        return query.order_by(AnalysisArtifact.created_at.desc()).limit(limit).all()
    
    def generate_artifact_download_url(
        self,
        artifact_id: str,
        user_id: str,
        user_role: str,
        expiry_seconds: int = SIGNED_URL_EXPIRY_SECONDS,
    ) -> Optional[str]:
        """
        Generate a signed download URL for an artifact.
        
        Args:
            artifact_id: Artifact to download
            user_id: User requesting download
            user_role: Role of requesting user
            expiry_seconds: URL expiration time
        
        Returns:
            Signed URL or None if not available
        """
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return None
        
        if artifact.storage_uri.startswith("s3://"):
            parts = artifact.storage_uri.replace("s3://", "").split("/", 1)
            s3_key = parts[1]
            url = self.s3.generate_presigned_url(s3_key, expiry_seconds)
        else:
            url = artifact.storage_uri
        
        self._log_storage_access(
            actor_id=user_id,
            actor_role=user_role,
            action="download_artifact",
            resource_type="analysis_artifact",
            resource_id=artifact_id,
            phi_level=artifact.phi_level,
            additional_context={"expiry_seconds": expiry_seconds}
        )
        
        return url
    
    def soft_delete_artifact(self, artifact_id: str, deleted_by: str) -> bool:
        """Soft delete an artifact"""
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return False
        
        artifact.deleted_at = datetime.utcnow()
        self.db.commit()
        
        self._log_storage_access(
            actor_id=deleted_by,
            actor_role="researcher",
            action="delete_artifact",
            resource_type="analysis_artifact",
            resource_id=artifact_id,
        )
        
        return True
    
    async def upload_dataset(
        self,
        study_id: str,
        cohort_snapshot_id: Optional[str],
        name: str,
        file_data: bytes,
        format: str,
        columns_json: List[Dict],
        row_count: int,
        created_by: str,
        description: Optional[str] = None,
        pii_classification: str = PHILevel.DE_IDENTIFIED.value,
    ) -> ResearchDataset:
        """
        Upload a new versioned dataset.
        
        Automatically increments version number for the study.
        """
        dataset_id = str(uuid4())
        checksum = self._compute_checksum(file_data)
        schema_hash = hashlib.sha256(
            json.dumps(columns_json, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        existing_versions = self.db.query(ResearchDataset).filter(
            ResearchDataset.study_id == study_id
        ).count()
        version = existing_versions + 1
        
        filename = f"dataset_v{version}.{format}"
        s3_key = f"research/datasets/{study_id}/{dataset_id}/{filename}"
        
        content_type = self._get_content_type(format)
        
        storage_uri = await self.s3.upload_file(
            file_data=file_data,
            s3_key=s3_key,
            content_type=content_type,
            metadata={
                "dataset_id": dataset_id,
                "study_id": study_id,
                "version": str(version),
                "pii_classification": pii_classification,
            }
        )
        
        dataset = ResearchDataset(
            id=dataset_id,
            study_id=study_id,
            cohort_snapshot_id=cohort_snapshot_id,
            name=name,
            description=description,
            version=version,
            storage_uri=storage_uri,
            format=format,
            row_count=row_count,
            column_count=len(columns_json),
            columns_json=columns_json,
            schema_hash=schema_hash,
            checksum=checksum,
            pii_classification=pii_classification,
            created_by=created_by,
        )
        
        self.db.add(dataset)
        self.db.commit()
        self.db.refresh(dataset)
        
        self._log_storage_access(
            actor_id=created_by,
            actor_role="researcher",
            action="upload_dataset",
            resource_type="research_dataset",
            resource_id=dataset_id,
            phi_level=pii_classification,
            additional_context={
                "study_id": study_id,
                "version": version,
                "row_count": row_count,
            }
        )
        
        logger.info(f"Uploaded dataset {dataset_id} v{version} for study {study_id}")
        return dataset
    
    def get_dataset(self, dataset_id: str) -> Optional[ResearchDataset]:
        """Get dataset by ID"""
        return self.db.query(ResearchDataset).filter(
            ResearchDataset.id == dataset_id
        ).first()
    
    def list_datasets(
        self,
        study_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ResearchDataset]:
        """List datasets with optional filters"""
        query = self.db.query(ResearchDataset)
        
        if study_id:
            query = query.filter(ResearchDataset.study_id == study_id)
        
        return query.order_by(ResearchDataset.created_at.desc()).limit(limit).all()
    
    def get_dataset_versions(self, study_id: str) -> List[ResearchDataset]:
        """Get all versions of datasets for a study"""
        return self.db.query(ResearchDataset).filter(
            ResearchDataset.study_id == study_id
        ).order_by(ResearchDataset.version.desc()).all()
    
    def add_lineage(
        self,
        parent_dataset_id: str,
        child_dataset_id: str,
        transformation_type: str,
        transformation_params: Optional[Dict] = None,
    ) -> DatasetLineage:
        """Record lineage between datasets"""
        lineage = DatasetLineage(
            parent_dataset_id=parent_dataset_id,
            child_dataset_id=child_dataset_id,
            transformation_type=transformation_type,
            transformation_params=transformation_params,
        )
        
        self.db.add(lineage)
        self.db.commit()
        self.db.refresh(lineage)
        
        return lineage
    
    def get_lineage(self, dataset_id: str) -> Dict[str, List[DatasetLineage]]:
        """Get lineage graph for a dataset"""
        parents = self.db.query(DatasetLineage).filter(
            DatasetLineage.child_dataset_id == dataset_id
        ).all()
        
        children = self.db.query(DatasetLineage).filter(
            DatasetLineage.parent_dataset_id == dataset_id
        ).all()
        
        return {
            "parents": parents,
            "children": children,
        }
    
    def generate_dataset_download_url(
        self,
        dataset_id: str,
        user_id: str,
        user_role: str,
        expiry_seconds: int = SIGNED_URL_EXPIRY_SECONDS,
    ) -> Optional[str]:
        """Generate a signed download URL for a dataset"""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        if dataset.storage_uri.startswith("s3://"):
            parts = dataset.storage_uri.replace("s3://", "").split("/", 1)
            s3_key = parts[1]
            url = self.s3.generate_presigned_url(s3_key, expiry_seconds)
        else:
            url = dataset.storage_uri
        
        self._log_storage_access(
            actor_id=user_id,
            actor_role=user_role,
            action="download_dataset",
            resource_type="research_dataset",
            resource_id=dataset_id,
            phi_level=dataset.pii_classification,
            additional_context={"expiry_seconds": expiry_seconds}
        )
        
        return url
    
    async def download_artifact_data(self, artifact_id: str) -> Optional[bytes]:
        """Download artifact file data"""
        artifact = self.get_artifact(artifact_id)
        if not artifact:
            return None
        
        return await self.s3.download_file(artifact.storage_uri)
    
    async def download_dataset_data(self, dataset_id: str) -> Optional[bytes]:
        """Download dataset file data"""
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        return await self.s3.download_file(dataset.storage_uri)
    
    def validate_checksum(self, artifact_id: str) -> bool:
        """Validate artifact checksum for data integrity"""
        artifact = self.get_artifact(artifact_id)
        if not artifact or not artifact.checksum:
            return False
        
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(self.s3.download_file(artifact.storage_uri))
        
        computed = self._compute_checksum(data)
        return computed == artifact.checksum
    
    def cleanup_expired_artifacts(self) -> int:
        """Remove expired artifacts (for scheduled job)"""
        now = datetime.utcnow()
        
        expired = self.db.query(AnalysisArtifact).filter(
            and_(
                AnalysisArtifact.expires_at <= now,
                AnalysisArtifact.deleted_at.is_(None)
            )
        ).all()
        
        count = 0
        for artifact in expired:
            artifact.deleted_at = now
            count += 1
        
        self.db.commit()
        
        if count > 0:
            logger.info(f"Cleaned up {count} expired artifacts")
            self._log_storage_access(
                actor_id="system",
                actor_role="scheduler",
                action="cleanup_expired_artifacts",
                resource_type="analysis_artifact",
                resource_id="batch",
                additional_context={"count": count}
            )
        
        return count
    
    def _get_content_type(self, format: str) -> str:
        """Get MIME type for file format"""
        content_types = {
            "json": "application/json",
            "csv": "text/csv",
            "parquet": "application/octet-stream",
            "pkl": "application/octet-stream",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "pdf": "application/pdf",
            "html": "text/html",
            "txt": "text/plain",
        }
        return content_types.get(format.lower(), "application/octet-stream")


def get_research_storage_service(db: Session) -> ResearchStorageService:
    """Factory function for dependency injection"""
    return ResearchStorageService(db)
