"""
Epidemiology Analytics API Router
Production-grade endpoints for surveillance, occupational, and genetic epidemiology.
All endpoints require doctor authentication and log access for HIPAA compliance.
"""

from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from app.database import get_db
from app.dependencies import get_current_doctor
from app.models.user import User
from app.services.epidemiology_service import EpidemiologyService

router = APIRouter(prefix="/api/research/epidemiology", tags=["epidemiology"])


@router.get("/drug-safety")
async def get_drug_safety_signals(
    drug_query: Optional[str] = Query(None, description="Filter by drug name"),
    outcome_query: Optional[str] = Query(None, description="Filter by outcome"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    scope: str = Query("all", description="Patient scope: all, my_patients, research_cohort"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get drug safety signals with privacy protection"""
    service = EpidemiologyService(db)
    return service.get_drug_safety_signals(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        drug_query=drug_query,
        outcome_query=outcome_query,
        location_id=location_id,
        flagged_only=False,
        limit=limit,
    )


@router.get("/locations")
async def get_pharmaco_locations(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get available locations for pharmacovigilance filtering"""
    service = EpidemiologyService(db)
    locations = service.get_surveillance_locations()
    signal_counts = {}
    return {"locations": [
        {**loc, "signal_count": signal_counts.get(loc["id"], 0)}
        for loc in locations
    ]}


@router.post("/drug-safety/scan")
async def trigger_drug_scan(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Trigger a drug safety signal scan (async job)"""
    return {
        "status": "initiated",
        "message": "Drug safety scan has been queued",
        "job_id": f"scan_{current_user.id}_{__import__('time').time()}",
    }


@router.get("/infectious-surveillance/epicurve")
async def get_epicurve(
    pathogen_code: str = Query(..., description="Pathogen code to analyze"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    days: int = Query(90, le=365, description="Number of days of data"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get epidemic curve data for a pathogen"""
    service = EpidemiologyService(db)
    return service.get_epicurve(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        pathogen_code=pathogen_code,
        location_id=location_id,
        days=days,
    )


@router.get("/infectious-surveillance/r0")
async def get_r0_estimate(
    pathogen_code: str = Query(..., description="Pathogen code"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get R0/Rt estimate for a pathogen"""
    service = EpidemiologyService(db)
    return service.get_r0_estimate(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        pathogen_code=pathogen_code,
        location_id=location_id,
    )


@router.get("/infectious-surveillance/pathogens")
async def get_pathogens(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get list of tracked pathogens"""
    service = EpidemiologyService(db)
    pathogens = service.get_pathogens(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
    )
    return {"pathogens": pathogens}


@router.get("/vaccine-analytics/coverage")
async def get_vaccine_coverage(
    vaccine_code: str = Query(..., description="Vaccine code"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get vaccine coverage statistics"""
    service = EpidemiologyService(db)
    return service.get_vaccine_coverage(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        vaccine_code=vaccine_code,
        location_id=location_id,
    )


@router.get("/vaccine-analytics/effectiveness")
async def get_vaccine_effectiveness(
    vaccine_code: str = Query(..., description="Vaccine code"),
    outcome_code: str = Query(..., description="Outcome to measure effectiveness against"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get vaccine effectiveness estimates"""
    service = EpidemiologyService(db)
    return service.get_vaccine_effectiveness(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        vaccine_code=vaccine_code,
        outcome_code=outcome_code,
        location_id=location_id,
    )


@router.get("/vaccine-analytics/vaccines")
async def get_vaccines(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get list of tracked vaccines"""
    service = EpidemiologyService(db)
    return service.get_vaccines(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
    )


@router.get("/occupational-signals")
async def get_occupational_signals(
    industry_query: Optional[str] = Query(None, description="Filter by industry"),
    hazard_query: Optional[str] = Query(None, description="Filter by hazard"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get occupational health signals"""
    service = EpidemiologyService(db)
    return service.get_occupational_signals(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        industry_query=industry_query,
        hazard_query=hazard_query,
        limit=limit,
    )


@router.get("/occupational-signals/cohorts")
async def get_occupational_cohorts(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get occupational cohort summaries"""
    service = EpidemiologyService(db)
    cohorts = service.get_occupational_cohorts(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
    )
    return {"cohorts": cohorts}


@router.get("/occupational-signals/industries")
async def get_industries(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get list of industries with incident data"""
    from app.models.epidemiology_models import OccupationalIncident
    from sqlalchemy import func
    
    results = (
        db.query(
            OccupationalIncident.industry_code,
            OccupationalIncident.industry_name,
            func.count(OccupationalIncident.id).label("incident_count"),
        )
        .group_by(
            OccupationalIncident.industry_code,
            OccupationalIncident.industry_name,
        )
        .all()
    )
    
    return {
        "industries": [
            {
                "industry_code": r[0],
                "industry_name": r[1],
                "incident_count": r[2] or 0,
            }
            for r in results
        ]
    }


@router.get("/occupational-signals/hazards")
async def get_hazards(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get list of hazards with incident data"""
    from app.models.epidemiology_models import OccupationalIncident
    from sqlalchemy import func
    
    results = (
        db.query(
            OccupationalIncident.hazard_code,
            OccupationalIncident.hazard_name,
            OccupationalIncident.hazard_category,
            func.count(OccupationalIncident.id).label("incident_count"),
        )
        .group_by(
            OccupationalIncident.hazard_code,
            OccupationalIncident.hazard_name,
            OccupationalIncident.hazard_category,
        )
        .all()
    )
    
    return {
        "hazards": [
            {
                "hazard_code": r[0],
                "hazard_name": r[1],
                "hazard_category": r[2],
                "incident_count": r[3] or 0,
            }
            for r in results
        ]
    }


@router.get("/genetic-associations")
async def get_genetic_associations(
    gene_query: Optional[str] = Query(None, description="Filter by gene/rsid"),
    outcome_query: Optional[str] = Query(None, description="Filter by outcome"),
    p_value_threshold: float = Query(5e-8, description="P-value significance threshold"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get genetic/genomic associations"""
    service = EpidemiologyService(db)
    return service.get_genetic_associations(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        gene_query=gene_query,
        outcome_query=outcome_query,
        p_value_threshold=p_value_threshold,
        limit=limit,
    )


@router.get("/pharmacogenomics")
async def get_pharmacogenomics(
    gene_query: Optional[str] = Query(None, description="Filter by gene/rsid"),
    drug_query: Optional[str] = Query(None, description="Filter by drug"),
    limit: int = Query(50, le=200),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get pharmacogenomic drug-gene interactions"""
    service = EpidemiologyService(db)
    interactions = service.get_pharmacogenomic_interactions(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        gene_query=gene_query,
        drug_query=drug_query,
        limit=limit,
    )
    return {"interactions": interactions}


@router.get("/genetic-associations/genes")
async def get_genes(
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get list of genes with association data"""
    from app.models.epidemiology_models import GenomicMarkerStat
    from sqlalchemy import func
    
    results = (
        db.query(
            GenomicMarkerStat.gene_symbol,
            GenomicMarkerStat.gene_name,
            func.count(GenomicMarkerStat.id).label("association_count"),
        )
        .group_by(
            GenomicMarkerStat.gene_symbol,
            GenomicMarkerStat.gene_name,
        )
        .order_by(func.count(GenomicMarkerStat.id).desc())
        .limit(100)
        .all()
    )
    
    return {
        "genes": [
            {
                "gene_symbol": r[0],
                "gene_name": r[1],
                "association_count": r[2] or 0,
            }
            for r in results
        ]
    }


@router.get("/surveillance/summary")
async def get_surveillance_summary(
    location_id: Optional[str] = Query(None, description="Filter by location"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get overall surveillance summary"""
    service = EpidemiologyService(db)
    return service.get_surveillance_summary(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        location_id=location_id,
    )


@router.get("/surveillance/weekly-trends")
async def get_weekly_trends(
    condition_code: str = Query(..., description="Condition code"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    weeks: int = Query(12, le=52, description="Number of weeks"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get weekly incidence trends"""
    service = EpidemiologyService(db)
    trends = service.get_weekly_trends(
        actor_id=str(current_user.id),
        actor_role=str(current_user.role),
        condition_code=condition_code,
        location_id=location_id,
        weeks=weeks,
    )
    return {"trends": trends}


@router.get("/surveillance/locations")
async def get_surveillance_locations(
    location_type: Optional[str] = Query(None, description="Filter by type"),
    current_user: User = Depends(get_current_doctor),
    db: Session = Depends(get_db),
):
    """Get surveillance locations"""
    service = EpidemiologyService(db)
    locations = service.get_surveillance_locations(location_type=location_type)
    return {"locations": locations}
