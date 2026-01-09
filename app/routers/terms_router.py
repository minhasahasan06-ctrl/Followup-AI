"""
Terms & Conditions Router
Handles terms acceptance recording with audit logging
HIPAA-compliant with proper authentication
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel
import logging

from app.database import get_db
from app.models.terms_audit import TermsAcceptance
from app.schemas.terms_audit_schemas import TermsAcceptResponse
from app.services.user_audit_service import log_user_audit, AuditEventType
from app.auth.auth0 import get_current_token, TokenPayload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/terms", tags=["Terms"])


class TermsAcceptRequest(BaseModel):
    """Terms acceptance request - user_id derived from auth context"""
    terms_version: str = "v2025-01"
    research_consent: bool = False


@router.post("/accept", response_model=TermsAcceptResponse)
async def accept_terms(
    payload: TermsAcceptRequest,
    request: Request,
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Record authenticated user's acceptance of Terms & Conditions
    User ID is derived from authentication token, not client request
    Captures IP, user agent, and timestamp for HIPAA compliance
    """
    try:
        user_id = token.user_id
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent", "")[:500]
        
        accepted_at = datetime.now(timezone.utc)
        
        terms_record = TermsAcceptance(
            user_id=user_id,
            terms_version=payload.terms_version,
            accepted_at=accepted_at,
            ip_address=ip_address,
            user_agent=user_agent,
            research_consent=payload.research_consent
        )
        
        db.add(terms_record)
        db.commit()
        
        log_user_audit(
            user_id=user_id,
            event_type=AuditEventType.TERMS_ACCEPTED,
            event_data={
                "terms_version": payload.terms_version,
                "research_consent": payload.research_consent
            },
            ip_address=ip_address,
            user_agent=user_agent,
            db=db
        )
        
        logger.info(f"Terms accepted by user {user_id}, version {payload.terms_version}")
        
        return TermsAcceptResponse(
            status="ok",
            recorded_at=accepted_at
        )
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to record terms acceptance: {e}")
        raise HTTPException(status_code=500, detail="Failed to record terms acceptance")


@router.get("/status")
async def get_terms_status(
    token: TokenPayload = Depends(get_current_token),
    db: Session = Depends(get_db)
):
    """
    Check if authenticated user has accepted current terms version
    """
    user_id = token.user_id
    current_version = "v2025-01"
    
    latest = (
        db.query(TermsAcceptance)
        .filter(TermsAcceptance.user_id == user_id)
        .filter(TermsAcceptance.terms_version == current_version)
        .first()
    )
    
    return {
        "user_id": user_id,
        "current_version": current_version,
        "accepted": latest is not None,
        "accepted_at": latest.accepted_at.isoformat() if latest else None,
        "research_consent": latest.research_consent if latest else False
    }
