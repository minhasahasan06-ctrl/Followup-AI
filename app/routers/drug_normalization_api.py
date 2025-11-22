"""
Drug Normalization API Router
===============================

Exposes drug normalization service via REST API for medication creation workflows.
Automatically normalizes medication names against RxNorm and creates drug records.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.drug_normalization_service import DrugNormalizationService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/drug-normalization",
    tags=["drug-normalization"]
)


class DrugNormalizationRequest(BaseModel):
    """Request model for drug normalization"""
    medication_name: str


class DrugNormalizationResponse(BaseModel):
    """Response model for drug normalization"""
    drug_id: str | None
    rxcui: str | None
    drug_name: str | None
    generic_name: str | None
    confidence_score: float
    match_source: str
    matched_by: str | None
    message: str


@router.post("/normalize", response_model=DrugNormalizationResponse)
async def normalize_drug(
    request: DrugNormalizationRequest,
    db: Session = Depends(get_db)
):
    """
    Normalize a medication name against RxNorm database
    
    This endpoint:
    1. Searches for the medication in the local drugs table
    2. If not found, queries RxNorm API for exact/approximate match
    3. Creates a new drug record if RxNorm match found
    4. Returns drug_id and confidence scoring
    
    Use this when creating new medication records to ensure standardized drug mapping.
    """
    try:
        logger.info(f"[NORMALIZE] Normalizing medication: {request.medication_name}")
        
        service = DrugNormalizationService(db)
        result = service.normalize_medication(request.medication_name)
        
        if result["drug_id"]:
            logger.info(
                f"[NORMALIZE] ✓ Successfully normalized '{request.medication_name}' "
                f"to drug_id={result['drug_id']} (RxCUI: {result.get('rxcui')})"
            )
            return DrugNormalizationResponse(
                drug_id=result["drug_id"],
                rxcui=result.get("rxcui"),
                drug_name=result.get("drug_name"),
                generic_name=result.get("generic_name"),
                confidence_score=result.get("confidence_score", 0.0),
                match_source=result.get("match_source", "unknown"),
                matched_by=result.get("matched_by"),
                message=f"Successfully normalized to drug: {result.get('drug_name')}"
            )
        else:
            logger.warning(
                f"[NORMALIZE] ✗ Could not normalize '{request.medication_name}' "
                f"(not found in RxNorm)"
            )
            return DrugNormalizationResponse(
                drug_id=None,
                rxcui=None,
                drug_name=None,
                generic_name=None,
                confidence_score=0.0,
                match_source="none",
                matched_by=None,
                message=f"Medication '{request.medication_name}' not found in RxNorm database"
            )
    
    except Exception as e:
        logger.error(f"[NORMALIZE] Error normalizing medication: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to normalize medication: {str(e)}"
        )
