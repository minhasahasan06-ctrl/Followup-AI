from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.dependencies import get_current_doctor
from app.models.user import User
from app.services.research_service import ResearchService
from pydantic import BaseModel
from typing import Dict

router = APIRouter(prefix="/api/v1/research", tags=["research"])


class FHIRQueryRequest(BaseModel):
    resourceType: str
    parameters: Dict[str, str]


class ResearchReportRequest(BaseModel):
    study_type: str
    parameters: Dict


@router.post("/fhir/query")
async def query_fhir(
    request: FHIRQueryRequest,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = ResearchService(db)
    result = service.query_fhir_data(request.resourceType, request.parameters)
    return result


@router.get("/epidemiology/{condition}")
async def get_epidemiology(
    condition: str,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = ResearchService(db)
    result = service.get_epidemiological_data(condition)
    return result


@router.get("/population-health")
async def get_population_health(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = ResearchService(db)
    result = service.get_population_health_metrics(current_user.id)
    return result


@router.post("/reports/generate")
async def generate_report(
    request: ResearchReportRequest,
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db)
):
    service = ResearchService(db)
    result = service.generate_research_report(request.study_type, request.parameters)
    return result
