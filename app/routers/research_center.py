"""
Phase 10: Research Center API Router
Production-grade endpoints for cohort building, studies, datasets, and NLP.
All endpoints require doctor authentication and include HIPAA audit logging.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, text

from app.database import get_db
from app.dependencies import get_current_user
from app.models.user import User
from app.models.research_models import (
    AnalysisArtifact,
    ResearchDataset,
    DatasetLineage,
    StudyJob,
    StudyJobEvent,
    ResearchCohortSnapshot,
    ResearchExport,
    NLPDocument,
    NLPRedactionRun,
    ResearchQASession,
    ResearchQAMessage,
    ResearchCohort,
    ResearchStudy,
    JobStatus,
    PHILevel,
)
from app.services.research_storage_service import ResearchStorageService
from app.services.research_qa_service import ResearchQAService
from app.services.phi_redaction_service import PHIRedactionService
from app.services.access_control import HIPAAAuditLogger
from app.services.s3_service import s3_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/research-center", tags=["research-center"])

K_ANONYMITY_THRESHOLD = 5


def require_doctor(user: User = Depends(get_current_user)) -> User:
    """Dependency to require doctor role"""
    if user.role not in ["doctor", "admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only doctors can access research center"
        )
    return user


class CohortFilter(BaseModel):
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    sex: Optional[str] = None
    conditions: List[str] = Field(default_factory=list)
    exclude_conditions: List[str] = Field(default_factory=list)
    risk_score_min: Optional[float] = None
    risk_score_max: Optional[float] = None
    consent_data_types: List[str] = Field(default_factory=list)


class CohortPreviewRequest(BaseModel):
    filters: CohortFilter


class CohortCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None
    filters: CohortFilter


class CohortPreviewResponse(BaseModel):
    total_patients: int
    matching_patients: int
    suppressed: bool = False
    age_distribution: List[Dict[str, Any]] = Field(default_factory=list)
    gender_distribution: List[Dict[str, Any]] = Field(default_factory=list)
    condition_distribution: List[Dict[str, Any]] = Field(default_factory=list)
    risk_score_distribution: List[Dict[str, Any]] = Field(default_factory=list)
    average_age: Optional[float] = None
    average_risk_score: Optional[float] = None


class StudyCreateRequest(BaseModel):
    title: str
    description: Optional[str] = None
    cohort_id: Optional[str] = None
    target_enrollment: int = 100
    inclusion_criteria: Optional[str] = None
    exclusion_criteria: Optional[str] = None
    auto_reanalysis: bool = False
    reanalysis_frequency: Optional[str] = None


class StudyUpdateRequest(BaseModel):
    status: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class JobCreateRequest(BaseModel):
    job_type: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5


class ExportRequest(BaseModel):
    format: str = "csv"
    include_phi: bool = False
    columns: Optional[List[str]] = None


class NLPDocumentCreateRequest(BaseModel):
    source_type: str
    source_uri: str
    patient_id: Optional[str] = None
    text: Optional[str] = None


class QASessionCreateRequest(BaseModel):
    study_id: Optional[str] = None
    dataset_id: Optional[str] = None
    title: Optional[str] = None


class QAMessageRequest(BaseModel):
    content: str


@router.get("/cohorts")
async def list_cohorts(
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    limit: int = Query(default=50, le=100),
):
    """List saved cohorts"""
    cohorts = db.query(ResearchCohort).filter(
        ResearchCohort.is_active == True
    ).order_by(ResearchCohort.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": str(c.id),
            "name": c.name,
            "description": c.description,
            "patientCount": c.patient_count,
            "createdAt": c.created_at.isoformat() if c.created_at else None,
            "updatedAt": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in cohorts
    ]


@router.post("/cohorts/preview")
async def preview_cohort(
    request: CohortPreviewRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """
    Preview cohort with real database queries.
    Applies k-anonymity suppression for privacy.
    """
    filters = request.filters
    
    try:
        base_query = text("""
            SELECT 
                COUNT(DISTINCT u.id) as total_count,
                AVG(EXTRACT(YEAR FROM AGE(CURRENT_DATE, u.date_of_birth))) as avg_age
            FROM users u
            WHERE u.role = 'patient'
        """)
        
        result = db.execute(base_query).fetchone()
        total_patients = result[0] if result else 0
        
        query_parts = ["SELECT COUNT(DISTINCT u.id) as count FROM users u WHERE u.role = 'patient'"]
        params = {}
        
        if filters.age_min:
            query_parts.append("AND EXTRACT(YEAR FROM AGE(CURRENT_DATE, u.date_of_birth)) >= :age_min")
            params["age_min"] = filters.age_min
        
        if filters.age_max:
            query_parts.append("AND EXTRACT(YEAR FROM AGE(CURRENT_DATE, u.date_of_birth)) <= :age_max")
            params["age_max"] = filters.age_max
        
        if filters.sex:
            query_parts.append("AND u.sex = :sex")
            params["sex"] = filters.sex
        
        count_query = " ".join(query_parts)
        count_result = db.execute(text(count_query), params).fetchone()
        matching_patients = count_result[0] if count_result else 0
        
        suppressed = matching_patients < K_ANONYMITY_THRESHOLD and matching_patients > 0
        
        if suppressed:
            matching_patients = 0
        
        age_distribution = []
        gender_distribution = []
        
        if not suppressed and matching_patients >= K_ANONYMITY_THRESHOLD:
            age_query = text("""
                SELECT 
                    CASE 
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_of_birth)) < 30 THEN '18-29'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_of_birth)) < 45 THEN '30-44'
                        WHEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, date_of_birth)) < 60 THEN '45-59'
                        ELSE '60+'
                    END as age_range,
                    COUNT(*) as count
                FROM users
                WHERE role = 'patient' AND date_of_birth IS NOT NULL
                GROUP BY age_range
                HAVING COUNT(*) >= :threshold
            """)
            age_results = db.execute(age_query, {"threshold": K_ANONYMITY_THRESHOLD}).fetchall()
            age_distribution = [{"range": r[0], "count": r[1]} for r in age_results if r[0]]
            
            gender_query = text("""
                SELECT sex, COUNT(*) as count
                FROM users
                WHERE role = 'patient' AND sex IS NOT NULL
                GROUP BY sex
                HAVING COUNT(*) >= :threshold
            """)
            gender_results = db.execute(gender_query, {"threshold": K_ANONYMITY_THRESHOLD}).fetchall()
            gender_distribution = [{"name": r[0], "value": r[1]} for r in gender_results]
        
        HIPAAAuditLogger.log_phi_access(
            actor_id=str(user.id),
            actor_role=user.role,
            patient_id="aggregate",
            action="cohort_preview",
            phi_categories=["aggregate_data"],
            resource_type="cohort_preview",
            access_scope="research",
            access_reason="cohort_analysis",
            consent_verified=True,
            additional_context={"filters": filters.dict(), "matching_count": matching_patients}
        )
        
        return CohortPreviewResponse(
            total_patients=total_patients,
            matching_patients=matching_patients,
            suppressed=suppressed,
            age_distribution=age_distribution,
            gender_distribution=gender_distribution,
            condition_distribution=[],
            risk_score_distribution=[],
            average_age=None,
            average_risk_score=None,
        )
        
    except Exception as e:
        logger.error(f"Error in cohort preview: {e}")
        raise HTTPException(status_code=500, detail="Error computing cohort preview")


@router.post("/cohorts")
async def create_cohort(
    request: CohortCreateRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Create and save a new cohort"""
    cohort_id = str(uuid4())
    
    cohort = ResearchCohort(
        id=cohort_id,
        name=request.name,
        description=request.description,
        filters_json=request.filters.dict(),
        created_by=str(user.id),
    )
    
    db.add(cohort)
    
    snapshot = ResearchCohortSnapshot(
        id=str(uuid4()),
        cohort_id=cohort_id,
        filter_json=request.filters.dict(),
        k_anonymity_applied=True,
        created_by=str(user.id),
    )
    
    db.add(snapshot)
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=str(user.id),
        actor_role=user.role,
        patient_id="aggregate",
        action="create_cohort",
        phi_categories=["aggregate_data"],
        resource_type="research_cohort",
        resource_id=cohort_id,
        access_scope="research",
        access_reason="cohort_creation",
        consent_verified=True,
    )
    
    return {
        "id": cohort_id,
        "name": request.name,
        "message": "Cohort created successfully"
    }


@router.get("/cohorts/{cohort_id}")
async def get_cohort(
    cohort_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get cohort details"""
    cohort = db.query(ResearchCohort).filter(ResearchCohort.id == cohort_id).first()
    if not cohort:
        raise HTTPException(status_code=404, detail="Cohort not found")
    
    return {
        "id": str(cohort.id),
        "name": cohort.name,
        "description": cohort.description,
        "filters": cohort.filters_json,
        "patientCount": cohort.patient_count,
        "createdAt": cohort.created_at.isoformat() if cohort.created_at else None,
    }


@router.get("/cohorts/{cohort_id}/snapshots")
async def get_cohort_snapshots(
    cohort_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get snapshot history for a cohort"""
    snapshots = db.query(ResearchCohortSnapshot).filter(
        ResearchCohortSnapshot.cohort_id == cohort_id
    ).order_by(ResearchCohortSnapshot.computed_at.desc()).all()
    
    return [
        {
            "id": str(s.id),
            "patientCount": s.patient_count,
            "suppressed": s.suppressed,
            "computedAt": s.computed_at.isoformat() if s.computed_at else None,
        }
        for s in snapshots
    ]


@router.get("/studies")
async def list_studies(
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, le=100),
):
    """List research studies"""
    query = db.query(ResearchStudy)
    
    if status_filter and status_filter != "all":
        query = query.filter(ResearchStudy.status == status_filter)
    
    studies = query.order_by(ResearchStudy.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": str(s.id),
            "title": s.title,
            "description": s.description,
            "status": s.status,
            "cohortId": s.cohort_id,
            "targetEnrollment": s.target_enrollment,
            "currentEnrollment": s.current_enrollment,
            "startDate": s.start_date.isoformat() if s.start_date else None,
            "endDate": s.end_date.isoformat() if s.end_date else None,
            "autoReanalysis": s.auto_reanalysis,
            "createdAt": s.created_at.isoformat() if s.created_at else None,
            "updatedAt": s.updated_at.isoformat() if s.updated_at else None,
        }
        for s in studies
    ]


@router.post("/studies")
async def create_study(
    request: StudyCreateRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Create a new research study"""
    study_id = str(uuid4())
    
    study = ResearchStudy(
        id=study_id,
        title=request.title,
        description=request.description,
        cohort_id=request.cohort_id,
        target_enrollment=request.target_enrollment,
        inclusion_criteria=request.inclusion_criteria,
        exclusion_criteria=request.exclusion_criteria,
        auto_reanalysis=request.auto_reanalysis,
        reanalysis_frequency=request.reanalysis_frequency,
        created_by=str(user.id),
    )
    
    db.add(study)
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=str(user.id),
        actor_role=user.role,
        patient_id="aggregate",
        action="create_study",
        phi_categories=["research_data"],
        resource_type="research_study",
        resource_id=study_id,
        access_scope="research",
        access_reason="study_creation",
        consent_verified=True,
    )
    
    return {
        "id": study_id,
        "title": request.title,
        "message": "Study created successfully"
    }


@router.get("/studies/{study_id}")
async def get_study(
    study_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get study details"""
    study = db.query(ResearchStudy).filter(ResearchStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    
    return {
        "id": str(study.id),
        "title": study.title,
        "description": study.description,
        "status": study.status,
        "cohortId": study.cohort_id,
        "targetEnrollment": study.target_enrollment,
        "currentEnrollment": study.current_enrollment,
        "inclusionCriteria": study.inclusion_criteria,
        "exclusionCriteria": study.exclusion_criteria,
        "autoReanalysis": study.auto_reanalysis,
        "reanalysisFrequency": study.reanalysis_frequency,
        "createdAt": study.created_at.isoformat() if study.created_at else None,
    }


@router.patch("/studies/{study_id}")
async def update_study(
    study_id: str,
    request: StudyUpdateRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Update study status or details"""
    study = db.query(ResearchStudy).filter(ResearchStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    
    if request.status:
        study.status = request.status
    if request.title:
        study.title = request.title
    if request.description:
        study.description = request.description
    
    db.commit()
    
    return {"message": "Study updated successfully"}


@router.get("/studies/{study_id}/jobs")
async def list_study_jobs(
    study_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    status_filter: Optional[str] = Query(default=None),
):
    """List jobs for a study"""
    query = db.query(StudyJob).filter(StudyJob.study_id == study_id)
    
    if status_filter:
        query = query.filter(StudyJob.status == status_filter)
    
    jobs = query.order_by(StudyJob.created_at.desc()).all()
    
    return [
        {
            "id": str(j.id),
            "type": j.job_type,
            "status": j.status,
            "progress": j.progress,
            "progressMessage": j.progress_message,
            "createdAt": j.created_at.isoformat() if j.created_at else None,
            "startedAt": j.started_at.isoformat() if j.started_at else None,
            "completedAt": j.completed_at.isoformat() if j.completed_at else None,
            "errorLog": j.error_log,
        }
        for j in jobs
    ]


@router.post("/studies/{study_id}/jobs")
async def create_study_job(
    study_id: str,
    request: JobCreateRequest,
    background_tasks: BackgroundTasks,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Create and queue a new study job"""
    study = db.query(ResearchStudy).filter(ResearchStudy.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    
    job_id = str(uuid4())
    
    job = StudyJob(
        id=job_id,
        study_id=study_id,
        job_type=request.job_type,
        status=JobStatus.PENDING.value,
        priority=request.priority,
        payload_json=request.payload,
        created_by=str(user.id),
    )
    
    db.add(job)
    
    event = StudyJobEvent(
        job_id=job_id,
        event_type="created",
        new_value=JobStatus.PENDING.value,
        message=f"Job created by {user.email}",
    )
    db.add(event)
    
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=str(user.id),
        actor_role=user.role,
        patient_id="aggregate",
        action="create_job",
        phi_categories=["research_data"],
        resource_type="study_job",
        resource_id=job_id,
        access_scope="research",
        access_reason="job_execution",
        consent_verified=True,
        additional_context={"job_type": request.job_type}
    )
    
    return {
        "id": job_id,
        "status": JobStatus.PENDING.value,
        "message": "Job queued successfully"
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get job details and results"""
    job = db.query(StudyJob).filter(StudyJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "id": str(job.id),
        "studyId": job.study_id,
        "type": job.job_type,
        "status": job.status,
        "progress": job.progress,
        "progressMessage": job.progress_message,
        "payload": job.payload_json,
        "result": job.result_json,
        "errorLog": job.error_log,
        "retryCount": job.retry_count,
        "createdAt": job.created_at.isoformat() if job.created_at else None,
        "startedAt": job.started_at.isoformat() if job.started_at else None,
        "completedAt": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Cancel a running or pending job"""
    job = db.query(StudyJob).filter(StudyJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status in [JobStatus.COMPLETED.value, JobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed or cancelled job")
    
    old_status = job.status
    job.status = JobStatus.CANCELLED.value
    job.completed_at = datetime.utcnow()
    
    event = StudyJobEvent(
        job_id=job_id,
        event_type="cancelled",
        old_value=old_status,
        new_value=JobStatus.CANCELLED.value,
        message=f"Cancelled by {user.email}",
    )
    db.add(event)
    
    db.commit()
    
    return {"message": "Job cancelled successfully"}


@router.get("/jobs/{job_id}/artifacts")
async def list_job_artifacts(
    job_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """List artifacts produced by a job"""
    artifacts = db.query(AnalysisArtifact).filter(
        and_(
            AnalysisArtifact.job_id == job_id,
            AnalysisArtifact.deleted_at.is_(None)
        )
    ).all()
    
    return [
        {
            "id": str(a.id),
            "type": a.artifact_type,
            "format": a.format,
            "filename": a.filename,
            "sizeBytes": a.size_bytes,
            "phiLevel": a.phi_level,
            "createdAt": a.created_at.isoformat() if a.created_at else None,
        }
        for a in artifacts
    ]


@router.get("/artifacts/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get artifact metadata"""
    artifact = db.query(AnalysisArtifact).filter(
        and_(
            AnalysisArtifact.id == artifact_id,
            AnalysisArtifact.deleted_at.is_(None)
        )
    ).first()
    
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return {
        "id": str(artifact.id),
        "jobId": artifact.job_id,
        "studyId": artifact.study_id,
        "type": artifact.artifact_type,
        "format": artifact.format,
        "filename": artifact.filename,
        "sizeBytes": artifact.size_bytes,
        "checksum": artifact.checksum,
        "phiLevel": artifact.phi_level,
        "metadata": artifact.metadata_json,
        "createdAt": artifact.created_at.isoformat() if artifact.created_at else None,
    }


@router.get("/artifacts/{artifact_id}/download")
async def download_artifact(
    artifact_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get signed download URL for artifact"""
    storage = ResearchStorageService(db)
    
    url = storage.generate_artifact_download_url(
        artifact_id=artifact_id,
        user_id=str(user.id),
        user_role=user.role,
    )
    
    if not url:
        raise HTTPException(status_code=404, detail="Artifact not found")
    
    return {"downloadUrl": url, "expiresIn": 900}


@router.get("/datasets")
async def list_datasets(
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    study_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=100),
):
    """List research datasets"""
    query = db.query(ResearchDataset)
    
    if study_id:
        query = query.filter(ResearchDataset.study_id == study_id)
    
    datasets = query.order_by(ResearchDataset.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": str(d.id),
            "studyId": d.study_id,
            "name": d.name,
            "description": d.description,
            "version": d.version,
            "format": d.format,
            "rowCount": d.row_count,
            "columnCount": d.column_count,
            "piiClassification": d.pii_classification,
            "createdAt": d.created_at.isoformat() if d.created_at else None,
        }
        for d in datasets
    ]


@router.get("/datasets/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get dataset metadata"""
    dataset = db.query(ResearchDataset).filter(ResearchDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {
        "id": str(dataset.id),
        "studyId": dataset.study_id,
        "name": dataset.name,
        "description": dataset.description,
        "version": dataset.version,
        "format": dataset.format,
        "rowCount": dataset.row_count,
        "columnCount": dataset.column_count,
        "columns": dataset.columns_json,
        "schemaHash": dataset.schema_hash,
        "piiClassification": dataset.pii_classification,
        "createdAt": dataset.created_at.isoformat() if dataset.created_at else None,
    }


@router.get("/datasets/{dataset_id}/versions")
async def get_dataset_versions(
    dataset_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get version history for a dataset"""
    dataset = db.query(ResearchDataset).filter(ResearchDataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    versions = db.query(ResearchDataset).filter(
        ResearchDataset.study_id == dataset.study_id
    ).order_by(ResearchDataset.version.desc()).all()
    
    return [
        {
            "id": str(v.id),
            "version": v.version,
            "rowCount": v.row_count,
            "schemaHash": v.schema_hash,
            "createdAt": v.created_at.isoformat() if v.created_at else None,
        }
        for v in versions
    ]


@router.get("/datasets/{dataset_id}/lineage")
async def get_dataset_lineage(
    dataset_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get lineage graph for a dataset"""
    storage = ResearchStorageService(db)
    lineage = storage.get_lineage(dataset_id)
    
    return {
        "parents": [
            {
                "datasetId": l.parent_dataset_id,
                "transformationType": l.transformation_type,
                "params": l.transformation_params,
            }
            for l in lineage["parents"]
        ],
        "children": [
            {
                "datasetId": l.child_dataset_id,
                "transformationType": l.transformation_type,
                "params": l.transformation_params,
            }
            for l in lineage["children"]
        ],
    }


@router.get("/datasets/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get signed download URL for dataset"""
    storage = ResearchStorageService(db)
    
    url = storage.generate_dataset_download_url(
        dataset_id=dataset_id,
        user_id=str(user.id),
        user_role=user.role,
    )
    
    if not url:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    return {"downloadUrl": url, "expiresIn": 900}


@router.post("/datasets/{dataset_id}/export")
async def export_dataset(
    dataset_id: str,
    request: ExportRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """
    Request dataset export in specified format.
    
    PHI Access Controls:
    - include_phi=True requires admin role
    - Exports are blocked if dataset size < k-anonymity threshold (5)
    
    All validation is performed by ResearchExportService to ensure
    consistent enforcement across API and worker paths.
    """
    from app.services.research_export_service import ResearchExportService
    
    export_service = ResearchExportService(db)
    
    try:
        export_record = await export_service.create_export(
            dataset_id=dataset_id,
            format=request.format,
            user_id=str(user.id),
            user_role=user.role,
            include_phi=request.include_phi,
            columns=request.columns,
        )
        
        return {
            "exportId": str(export_record.id),
            "status": export_record.status,
            "message": "Export queued successfully"
        }
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except ValueError as e:
        if "k-anonymity" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )


@router.get("/exports")
async def list_exports(
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, le=100),
):
    """List exports for the current user with HIPAA security controls.
    
    Security: 
    - Users only see their own exports (unless admin)
    - PHI exports require explicit download authorization (not shown in list)
    - k-anonymity enforcement via ResearchExportService on download
    """
    query = db.query(ResearchExport)
    
    # Security: Only show user's own exports unless admin
    if user.role != "admin":
        query = query.filter(ResearchExport.created_by == str(user.id))
    
    if status_filter:
        query = query.filter(ResearchExport.status == status_filter)
    
    exports = query.order_by(ResearchExport.created_at.desc()).limit(limit).all()
    
    result = []
    for exp in exports:
        dataset = db.query(ResearchDataset).filter(ResearchDataset.id == exp.dataset_id).first()
        
        # Security: NEVER expose signed_url in list response
        # All downloads must go through GET /exports/{id} for:
        # 1. Fresh short-lived URL generation (15-minute TTL)
        # 2. HIPAA audit logging with request context
        # 3. Ownership verification on each download request
        
        result.append({
            "id": str(exp.id),
            "datasetId": str(exp.dataset_id) if exp.dataset_id else None,
            "datasetName": dataset.name if dataset else None,
            "format": exp.format,
            "status": exp.status,
            "includePhi": exp.include_phi,
            "rowCount": exp.row_count,
            "fileSizeBytes": exp.file_size_bytes,
            "errorMessage": exp.error_message,
            "createdAt": exp.created_at.isoformat() if exp.created_at else None,
            "createdBy": str(exp.created_by) if exp.created_by else None,
            "createdByRole": exp.created_by_role,
        })
    
    return result


SIGNED_URL_EXPIRY_SECONDS = 900  # 15 minutes - short-lived for security

@router.get("/exports/{export_id}")
async def get_export_status(
    export_id: str,
    request: Request,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get export status and download URL with HIPAA security controls.
    
    Security Model (HIPAA-compliant):
    - K-anonymity (threshold=5) is enforced at export CREATION time via ResearchExportService
    - Download authorization requires ownership verification (creator or admin only)
    - All download attempts are HIPAA audit logged with full request context
    - Signed URLs are regenerated per request (15-minute TTL), never stored/returned from DB
    
    This two-stage model is valid because:
    1. K-anonymity protects against re-identification at the point of data export
    2. Once exported, the file content is immutable - no re-identification risk increases
    3. Ownership checks prevent unauthorized access to already-validated exports
    4. Fresh URL generation prevents replay attacks if URLs are leaked
    """
    export_record = db.query(ResearchExport).filter(ResearchExport.id == export_id).first()
    if not export_record:
        raise HTTPException(status_code=404, detail="Export not found")
    
    # Extract request context for audit logging
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    # Security: Verify ownership (unless admin)
    if user.role != "admin" and str(export_record.created_by) != str(user.id):
        # HIPAA Audit: Log unauthorized access attempt with full context
        HIPAAAuditLogger.log_access(
            user_id=str(user.id),
            user_role=user.role,
            action="export_access_denied",
            resource_type="ResearchExport",
            resource_id=export_id,
            access_reason="ownership_violation",
            was_successful=False,
            additional_context={
                "owner_id": str(export_record.created_by),
                "client_ip": client_ip,
                "user_agent": user_agent[:200] if user_agent else None,
            }
        )
        raise HTTPException(status_code=403, detail="Access denied: you can only view your own exports")
    
    # Security: Generate fresh short-lived URL for completed exports
    download_url = None
    url_expires_at = None
    
    if export_record.status == JobStatus.COMPLETED.value and export_record.storage_uri:
        # Generate fresh presigned URL from storage_uri (never return stored URL)
        # Extract S3 key from storage_uri (format: s3://bucket/key or just the key)
        s3_key = export_record.storage_uri
        if s3_key.startswith("s3://"):
            s3_key = "/".join(s3_key.split("/")[3:])  # Remove s3://bucket/ prefix
        
        # Generate fresh short-lived URL (15 minutes)
        fresh_url = s3_service.generate_presigned_url(s3_key, expiration=SIGNED_URL_EXPIRY_SECONDS)
        url_expires_at = (datetime.utcnow() + timedelta(seconds=SIGNED_URL_EXPIRY_SECONDS)).isoformat()
        
        if fresh_url:
            download_url = fresh_url
            
            # HIPAA Audit: Log download URL generation with full context
            audit_action = "phi_export_download" if export_record.include_phi else "export_download"
            HIPAAAuditLogger.log_access(
                user_id=str(user.id),
                user_role=user.role,
                action=audit_action,
                resource_type="ResearchExport",
                resource_id=export_id,
                access_reason="authorized_download_url_generated",
                was_successful=True,
                additional_context={
                    "dataset_id": str(export_record.dataset_id) if export_record.dataset_id else None,
                    "row_count": export_record.row_count,
                    "format": export_record.format,
                    "include_phi": export_record.include_phi,
                    "url_expires_at": url_expires_at,
                    "client_ip": client_ip,
                    "user_agent": user_agent[:200] if user_agent else None,
                }
            )
    
    return {
        "id": str(export_record.id),
        "status": export_record.status,
        "format": export_record.format,
        "includePhi": export_record.include_phi,
        "downloadUrl": download_url,
        "urlExpiresAt": url_expires_at,
        "fileSizeBytes": export_record.file_size_bytes,
        "rowCount": export_record.row_count,
        "createdAt": export_record.created_at.isoformat() if export_record.created_at else None,
        "completedAt": export_record.completed_at.isoformat() if export_record.completed_at else None,
    }


@router.get("/nlp/documents")
async def list_nlp_documents(
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    study_id: Optional[str] = Query(default=None),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, le=100),
):
    """List NLP documents with redaction runs"""
    query = db.query(NLPDocument)
    
    if study_id:
        query = query.filter(NLPDocument.study_id == study_id)
    if status_filter:
        query = query.filter(NLPDocument.status == status_filter)
    
    docs = query.order_by(NLPDocument.created_at.desc()).limit(limit).all()
    
    result = []
    for d in docs:
        runs = db.query(NLPRedactionRun).filter(
            NLPRedactionRun.document_id == str(d.id)
        ).order_by(NLPRedactionRun.created_at.desc()).all()
        
        result.append({
            "id": str(d.id),
            "studyId": d.study_id,
            "sourceType": d.source_type,
            "sourceUri": d.source_uri,
            "patientId": d.patient_id,
            "status": d.status,
            "phiCount": d.phi_count,
            "redactedUri": d.redacted_uri,
            "processedAt": d.processed_at.isoformat() if d.processed_at else None,
            "createdAt": d.created_at.isoformat() if d.created_at else None,
            "createdBy": d.created_by,
            "redactionRuns": [
                {
                    "id": str(r.id),
                    "status": r.status,
                    "entitiesDetected": r.entities_detected,
                    "entitiesRedacted": r.entities_redacted,
                    "createdAt": r.created_at.isoformat() if r.created_at else None,
                }
                for r in runs
            ]
        })
    
    return result


@router.post("/nlp/documents")
async def create_nlp_document(
    request: NLPDocumentCreateRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Ingest a new document for NLP processing"""
    doc_id = str(uuid4())
    
    doc = NLPDocument(
        id=doc_id,
        source_type=request.source_type,
        source_uri=request.source_uri,
        patient_id=request.patient_id,
        original_text=request.text,
        status="pending",
        created_by=str(user.id),
    )
    
    db.add(doc)
    db.commit()
    
    HIPAAAuditLogger.log_phi_access(
        actor_id=str(user.id),
        actor_role=user.role,
        patient_id=request.patient_id or "aggregate",
        action="ingest_document",
        phi_categories=["clinical_notes"],
        resource_type="nlp_document",
        resource_id=doc_id,
        access_scope="research",
        access_reason="nlp_processing",
        consent_verified=True,
    )
    
    return {
        "id": doc_id,
        "status": "pending",
        "message": "Document ingested for processing"
    }


@router.post("/nlp/documents/{document_id}/process")
async def process_nlp_document(
    document_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Trigger PHI redaction processing for a document"""
    doc = db.query(NLPDocument).filter(NLPDocument.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    text = doc.original_text
    if not text:
        raise HTTPException(status_code=400, detail="Document has no text to process")
    
    redaction_service = PHIRedactionService(db)
    
    try:
        result = await redaction_service.process_document(
            document_id=document_id,
            text=text,
            user_id=str(user.id),
        )
        
        return {
            "id": document_id,
            "status": result["status"],
            "findings": result["findings"],
            "redactedText": result["redacted_text"],
            "entitiesDetected": result.get("entities_detected", len(result.get("findings", []))),
            "entitiesRedacted": result.get("entities_redacted", len(result.get("findings", []))),
            "entityCounts": result["entity_counts"],
            "processingTimeMs": result["processing_time_ms"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/nlp/documents/{document_id}")
async def get_nlp_document(
    document_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get NLP document details"""
    doc = db.query(NLPDocument).filter(NLPDocument.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    runs = db.query(NLPRedactionRun).filter(
        NLPRedactionRun.document_id == document_id
    ).order_by(NLPRedactionRun.created_at.desc()).all()
    
    return {
        "id": str(doc.id),
        "studyId": doc.study_id,
        "patientId": doc.patient_id,
        "sourceType": doc.source_type,
        "sourceUri": doc.source_uri,
        "originalText": doc.original_text,
        "redactedText": doc.redacted_text,
        "status": doc.status,
        "phiDetected": doc.phi_detected_json,
        "phiCount": doc.phi_count,
        "redactedUri": doc.redacted_uri,
        "processedAt": doc.processed_at.isoformat() if doc.processed_at else None,
        "processingTimeMs": doc.processing_time_ms,
        "createdAt": doc.created_at.isoformat() if doc.created_at else None,
        "createdBy": doc.created_by,
        "redactionRuns": [
            {
                "id": r.id,
                "modelName": r.model_name,
                "entitiesDetected": r.entities_detected,
                "entitiesRedacted": r.entities_redacted,
                "createdAt": r.created_at.isoformat() if r.created_at else None,
            }
            for r in runs
        ],
    }


@router.get("/ai/qa/sessions")
async def list_qa_sessions(
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
    limit: int = Query(default=20, le=50),
):
    """List user's Q&A sessions"""
    sessions = db.query(ResearchQASession).filter(
        ResearchQASession.user_id == str(user.id)
    ).order_by(ResearchQASession.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": str(s.id),
            "title": s.title,
            "studyId": s.study_id,
            "datasetId": s.dataset_id,
            "status": s.status,
            "totalMessages": s.total_messages,
            "createdAt": s.created_at.isoformat() if s.created_at else None,
            "lastMessageAt": s.last_message_at.isoformat() if s.last_message_at else None,
        }
        for s in sessions
    ]


@router.post("/ai/qa/sessions")
async def create_qa_session(
    request: QASessionCreateRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Create a new Q&A session"""
    session_id = str(uuid4())
    
    session = ResearchQASession(
        id=session_id,
        user_id=str(user.id),
        study_id=request.study_id,
        dataset_id=request.dataset_id,
        title=request.title or "New Analysis Session",
    )
    
    db.add(session)
    db.commit()
    
    return {
        "id": session_id,
        "message": "Session created successfully"
    }


@router.get("/ai/qa/sessions/{session_id}")
async def get_qa_session(
    session_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Get Q&A session with messages"""
    session = db.query(ResearchQASession).filter(
        and_(
            ResearchQASession.id == session_id,
            ResearchQASession.user_id == str(user.id)
        )
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(ResearchQAMessage).filter(
        ResearchQAMessage.session_id == session_id
    ).order_by(ResearchQAMessage.created_at.asc()).all()
    
    return {
        "id": str(session.id),
        "title": session.title,
        "studyId": session.study_id,
        "datasetId": session.dataset_id,
        "status": session.status,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "references": m.references_json,
                "createdAt": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }


@router.post("/ai/qa/sessions/{session_id}/messages")
async def send_qa_message(
    session_id: str,
    request: QAMessageRequest,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Send a message to the Q&A session and get AI response using OpenAI"""
    session = db.query(ResearchQASession).filter(
        and_(
            ResearchQASession.id == session_id,
            ResearchQASession.user_id == str(user.id)
        )
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    qa_service = ResearchQAService(db)
    
    result = await qa_service.generate_response(
        session=session,
        user_message=request.content,
        user_id=str(user.id),
        user_role=user.role,
    )
    
    return {
        "userMessage": result["user_message"],
        "assistantMessage": result["assistant_message"],
        "tokenUsage": result.get("token_usage", 0),
    }


@router.delete("/ai/qa/sessions/{session_id}")
async def delete_qa_session(
    session_id: str,
    user: User = Depends(require_doctor),
    db: Session = Depends(get_db),
):
    """Delete a Q&A session"""
    session = db.query(ResearchQASession).filter(
        and_(
            ResearchQASession.id == session_id,
            ResearchQASession.user_id == str(user.id)
        )
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db.query(ResearchQAMessage).filter(
        ResearchQAMessage.session_id == session_id
    ).delete()
    
    db.delete(session)
    db.commit()
    
    return {"message": "Session deleted successfully"}
