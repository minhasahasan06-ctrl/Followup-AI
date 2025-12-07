"""
Occupational Epidemiology API Router
=====================================
Endpoints for workplace exposure tracking, industry risk analysis, and hazard signals.
All endpoints return aggregated, privacy-protected data only.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
from decimal import Decimal

from .privacy import PrivacyGuard, PrivacyConfig, RoleBasedAccessControl, MIN_CELL_SIZE


from .audit import EpidemiologyAuditLogger, AuditAction
from .auth import verify_epidemiology_auth, AuthenticatedUser


def normalize_row(row: dict) -> dict:
    """Convert Decimal and other non-JSON-serializable types to native Python types."""
    result = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            result[key] = float(value)
        else:
            result[key] = value
    return result

router = APIRouter(prefix="/api/v1/occupational", tags=["Occupational Epidemiology"])

DB_URL = os.environ.get('DATABASE_URL')
audit_logger = EpidemiologyAuditLogger()


class IndustryRiskSignal(BaseModel):
    id: int
    industry_code: str
    industry_name: str
    occupation_code: Optional[str]
    occupation_name: Optional[str]
    hazard_code: str
    hazard_name: str
    outcome_code: str
    outcome_name: str
    location_id: Optional[str]
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    signal_strength: float
    n_workers: int
    n_events: int
    mean_exposure_years: Optional[float]
    flagged: bool
    suppressed: bool = False


class IndustryRiskResponse(BaseModel):
    signals: List[Dict[str, Any]]
    total_count: int
    suppressed_count: int
    privacy_note: str


class IndustrySummary(BaseModel):
    industry_code: str
    industry_name: str
    worker_count: int
    signal_count: int
    high_risk_hazards: int


class HazardSummary(BaseModel):
    hazard_code: str
    hazard_name: str
    affected_industries: int
    total_workers: int
    avg_exposure_years: float


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DB_URL)


@router.get("/signals", response_model=IndustryRiskResponse)
async def get_occupational_signals(
    industry_query: Optional[str] = Query(None, description="Industry code or name search"),
    hazard_query: Optional[str] = Query(None, description="Hazard code or name search"),
    outcome_query: Optional[str] = Query(None, description="Outcome code or name search"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    flagged_only: bool = Query(False, description="Only show flagged signals"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get occupational risk signals with privacy protection.
    
    Returns aggregated industry-hazard-outcome associations.
    All results are subject to minimum cell-size suppression.
    """
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT * FROM occupational_risk_signals
            WHERE 1=1
        """
        params: List[Any] = []
        
        if industry_query:
            query += " AND (industry_code ILIKE %s OR industry_name ILIKE %s)"
            params.extend([f"%{industry_query}%", f"%{industry_query}%"])
        
        if hazard_query:
            query += " AND (hazard_code ILIKE %s OR hazard_name ILIKE %s)"
            params.extend([f"%{hazard_query}%", f"%{hazard_query}%"])
        
        if outcome_query:
            query += " AND (outcome_code ILIKE %s OR outcome_name ILIKE %s)"
            params.extend([f"%{outcome_query}%", f"%{outcome_query}%"])
        
        if location_id:
            query += " AND location_id = %s"
            params.append(location_id)
        
        if flagged_only:
            query += " AND flagged = TRUE"
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM ({query}) AS sub"
        cur.execute(count_query, params)
        count_result = cur.fetchone()
        total_count = count_result['count'] if count_result else 0
        
        # Get paginated results
        query += " ORDER BY signal_strength DESC NULLS LAST, p_value ASC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        signals = cur.fetchall()
        
        # Apply privacy protections
        sanitized_signals = []
        suppressed_count = 0
        
        for signal in signals:
            signal_dict = normalize_row(dict(signal))
            n_workers = signal_dict.get('n_workers', 0)
            n_events = signal_dict.get('n_events', 0)
            
            if not privacy_guard.check_cell_size(n_workers) or not privacy_guard.check_cell_size(n_events):
                signal_dict['suppressed'] = True
                signal_dict['estimate'] = None
                signal_dict['ci_lower'] = None
                signal_dict['ci_upper'] = None
                signal_dict['p_value'] = None
                suppressed_count += 1
            else:
                signal_dict['suppressed'] = False
            
            sanitized_signals.append(signal_dict)
        
        audit_logger.log(
            action=AuditAction.VIEW_OCCUPATIONAL,
            user_id=user.user_id,
            resource_type='occupational_signals',
            details={
                'industry_query': industry_query,
                'hazard_query': hazard_query,
                'location_id': location_id,
                'result_count': len(sanitized_signals)
            }
        )
        
        return IndustryRiskResponse(
            signals=sanitized_signals,
            total_count=total_count,
            suppressed_count=suppressed_count,
            privacy_note=f"Results with fewer than {MIN_CELL_SIZE} workers or events are suppressed for privacy protection."
        )
    
    finally:
        conn.close()


@router.get("/industries")
async def get_industries(
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, Any]:
    """Get list of industries with signal summaries (privacy-protected)."""
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                industry_code,
                industry_name,
                SUM(n_workers) as worker_count,
                COUNT(*) as signal_count,
                SUM(CASE WHEN flagged THEN 1 ELSE 0 END) as high_risk_hazards
            FROM occupational_risk_signals
            GROUP BY industry_code, industry_name
            ORDER BY signal_count DESC
        """)
        
        industries = []
        for row in cur.fetchall():
            row_dict = normalize_row(dict(row))
            # Suppress counts below threshold
            if not privacy_guard.check_cell_size(row_dict.get('worker_count', 0)):
                row_dict['worker_count'] = None
                row_dict['suppressed'] = True
            else:
                row_dict['suppressed'] = False
            industries.append(row_dict)
        
        audit_logger.log(
            action=AuditAction.VIEW_OCCUPATIONAL,
            user_id=user.user_id,
            resource_type='industries_list',
            details={'result_count': len(industries)}
        )
        
        return {"industries": industries}
    
    finally:
        conn.close()


@router.get("/hazards")
async def get_hazards(
    industry_code: Optional[str] = Query(None, description="Filter by industry"),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, Any]:
    """Get list of occupational hazards with summaries (privacy-protected)."""
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT 
                hazard_code,
                hazard_name,
                COUNT(DISTINCT industry_code) as affected_industries,
                SUM(n_workers) as total_workers,
                AVG(mean_exposure_years) as avg_exposure_years
            FROM occupational_risk_signals
            WHERE 1=1
        """
        params: List[Any] = []
        
        if industry_code:
            query += " AND industry_code = %s"
            params.append(industry_code)
        
        query += " GROUP BY hazard_code, hazard_name ORDER BY total_workers DESC"
        
        cur.execute(query, params)
        hazards = []
        for row in cur.fetchall():
            row_dict = normalize_row(dict(row))
            # Suppress counts below threshold
            if not privacy_guard.check_cell_size(row_dict.get('total_workers', 0)):
                row_dict['total_workers'] = None
                row_dict['avg_exposure_years'] = None
                row_dict['suppressed'] = True
            else:
                row_dict['suppressed'] = False
            hazards.append(row_dict)
        
        audit_logger.log(
            action=AuditAction.VIEW_OCCUPATIONAL,
            user_id=user.user_id,
            resource_type='hazards_list',
            details={'industry_code': industry_code, 'result_count': len(hazards)}
        )
        
        return {"hazards": hazards}
    
    finally:
        conn.close()


@router.get("/exposure-distribution")
async def get_exposure_distribution(
    industry_code: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, Any]:
    """Get aggregated exposure duration distribution."""
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT 
                CASE 
                    WHEN mean_exposure_years < 1 THEN '< 1 year'
                    WHEN mean_exposure_years < 5 THEN '1-5 years'
                    WHEN mean_exposure_years < 10 THEN '5-10 years'
                    WHEN mean_exposure_years < 20 THEN '10-20 years'
                    ELSE '20+ years'
                END as exposure_bracket,
                SUM(n_workers) as worker_count,
                AVG(estimate) as avg_risk_estimate
            FROM occupational_risk_signals
            WHERE mean_exposure_years IS NOT NULL
        """
        params: List[Any] = []
        
        if industry_code:
            query += " AND industry_code = %s"
            params.append(industry_code)
        
        if hazard_code:
            query += " AND hazard_code = %s"
            params.append(hazard_code)
        
        query += """
            GROUP BY CASE 
                WHEN mean_exposure_years < 1 THEN '< 1 year'
                WHEN mean_exposure_years < 5 THEN '1-5 years'
                WHEN mean_exposure_years < 10 THEN '5-10 years'
                WHEN mean_exposure_years < 20 THEN '10-20 years'
                ELSE '20+ years'
            END
            ORDER BY MIN(mean_exposure_years)
        """
        
        cur.execute(query, params)
        distribution = []
        
        for row in cur.fetchall():
            row_dict = dict(row)
            if not privacy_guard.check_cell_size(row_dict.get('worker_count', 0)):
                row_dict['worker_count'] = '<10'
                row_dict['avg_risk_estimate'] = None
            distribution.append(row_dict)
        
        return {"distribution": distribution}
    
    finally:
        conn.close()


@router.get("/locations")
async def get_locations_with_signals(
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, List[Dict[str, Any]]]:
    """Get locations with occupational signal counts."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                o.location_id as id,
                COALESCE(l.name, o.location_id) as name,
                l.city,
                l.state,
                COUNT(*) as signal_count
            FROM occupational_risk_signals o
            LEFT JOIN locations l ON o.location_id = l.id
            WHERE o.location_id IS NOT NULL
            GROUP BY o.location_id, l.name, l.city, l.state
            ORDER BY signal_count DESC
        """)
        
        return {"locations": [dict(row) for row in cur.fetchall()]}
    
    finally:
        conn.close()
