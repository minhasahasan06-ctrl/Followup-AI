"""
Pharmaco-Epidemiology API Router
=================================
FastAPI endpoints for drug safety signal detection and analysis.
All endpoints return aggregated, privacy-protected data only.
"""

import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Header
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
import json
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

router = APIRouter(prefix="/api/v1/pharmaco", tags=["Pharmaco-Epidemiology"])

DB_URL = os.environ.get('DATABASE_URL')


class SignalQueryParams(BaseModel):
    drug_query: Optional[str] = None
    outcome_query: Optional[str] = None
    patient_location_id: Optional[str] = None
    scope: str = "all"
    model_type: Optional[str] = None
    flagged_only: bool = False
    limit: int = 100
    offset: int = 0


class SignalResponse(BaseModel):
    signals: List[Dict[str, Any]]
    total_count: int
    suppressed_count: int
    privacy_note: str


class SummaryResponse(BaseModel):
    summaries: List[Dict[str, Any]]
    total_count: int
    suppressed_count: int


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DB_URL)


@router.get("/signals", response_model=SignalResponse)
async def get_drug_signals(
    drug_query: Optional[str] = Query(None, description="Drug name or code to search"),
    outcome_query: Optional[str] = Query(None, description="Outcome/adverse event to search"),
    patient_location_id: Optional[str] = Query(None, description="Filter by patient location"),
    scope: str = Query("all", description="Analysis scope: all, my_patients, research_cohort"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    flagged_only: bool = Query(False, description="Only return flagged signals"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Search drug safety signals with privacy protection.
    
    Returns aggregated, de-identified signal data only.
    All results are subject to minimum cell-size suppression.
    """
    allowed_scopes = RoleBasedAccessControl.get_allowed_scopes(user.role.value)
    if scope not in allowed_scopes:
        raise HTTPException(status_code=403, detail=f"Scope '{scope}' not allowed for your role")
    
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    audit_logger = EpidemiologyAuditLogger(DB_URL)
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT 
                id, drug_code, drug_name, outcome_code, outcome_name,
                patient_location_id, analysis_scope, model_type, effect_measure,
                estimate, ci_lower, ci_upper, p_value, signal_strength,
                n_patients, n_events, multiple_testing_adjusted, flagged,
                details_json, updated_at
            FROM drug_outcome_signals
            WHERE analysis_scope = %s
        """
        params = [scope]
        
        if drug_query:
            query += " AND (drug_name ILIKE %s OR drug_code ILIKE %s)"
            params.extend([f"%{drug_query}%", f"%{drug_query}%"])
        
        if outcome_query:
            query += " AND (outcome_name ILIKE %s OR outcome_code ILIKE %s)"
            params.extend([f"%{outcome_query}%", f"%{outcome_query}%"])
        
        if patient_location_id:
            query += " AND patient_location_id = %s"
            params.append(patient_location_id)
        
        if model_type:
            query += " AND model_type = %s"
            params.append(model_type)
        
        if flagged_only:
            query += " AND flagged = TRUE"
        
        count_query = f"SELECT COUNT(*) FROM ({query}) subq"
        cur.execute(count_query, params)
        total_count = cur.fetchone()['count']
        
        query += " ORDER BY signal_strength DESC, updated_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        raw_signals = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        conn.close()
        
        sanitized_signals = []
        suppressed_count = 0
        
        for signal in raw_signals:
            if signal.get('details_json') and isinstance(signal['details_json'], str):
                signal['details_json'] = json.loads(signal['details_json'])
            if signal.get('updated_at'):
                signal['updated_at'] = signal['updated_at'].isoformat()
            
            sanitized = privacy_guard.sanitize_signal(signal)
            if sanitized.get('suppressed'):
                suppressed_count += 1
            sanitized_signals.append(sanitized)
        
        audit_logger.log_signal_access(
            user_id=user.user_id,
            drug_code=drug_query,
            outcome_code=outcome_query,
            location_id=patient_location_id,
            scope=scope,
            result_count=len(sanitized_signals)
        )
        
        return SignalResponse(
            signals=sanitized_signals,
            total_count=total_count,
            suppressed_count=suppressed_count,
            privacy_note=f"Results subject to minimum cell size of {MIN_CELL_SIZE}. Some values may be suppressed."
        )
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/summaries", response_model=SummaryResponse)
async def get_drug_summaries(
    drug_query: Optional[str] = Query(None),
    outcome_query: Optional[str] = Query(None),
    patient_location_id: Optional[str] = Query(None),
    scope: str = Query("all"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get aggregated drug-outcome summaries with privacy protection.
    
    Returns descriptive statistics only, no effect estimates.
    """
    allowed_scopes = RoleBasedAccessControl.get_allowed_scopes(user.role.value)
    if scope not in allowed_scopes:
        raise HTTPException(status_code=403, detail=f"Scope '{scope}' not allowed for your role")
    
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    audit_logger = EpidemiologyAuditLogger(DB_URL)
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT *
            FROM drug_outcome_summaries
            WHERE analysis_scope = %s
        """
        params = [scope]
        
        if drug_query:
            query += " AND (drug_name ILIKE %s OR drug_code ILIKE %s)"
            params.extend([f"%{drug_query}%", f"%{drug_query}%"])
        
        if outcome_query:
            query += " AND (outcome_name ILIKE %s OR outcome_code ILIKE %s)"
            params.extend([f"%{outcome_query}%", f"%{outcome_query}%"])
        
        if patient_location_id:
            query += " AND patient_location_id = %s"
            params.append(patient_location_id)
        
        count_query = f"SELECT COUNT(*) FROM ({query}) subq"
        cur.execute(count_query, params)
        total_count = cur.fetchone()['count']
        
        query += " ORDER BY n_patients DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        raw_summaries = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        conn.close()
        
        sanitized_summaries = []
        suppressed_count = 0
        
        for summary in raw_summaries:
            if summary.get('updated_at'):
                summary['updated_at'] = summary['updated_at'].isoformat()
            
            sanitized = privacy_guard.sanitize_summary(summary)
            if sanitized.get('suppressed'):
                suppressed_count += 1
            sanitized_summaries.append(sanitized)
        
        audit_logger.log(
            action=AuditAction.VIEW_SUMMARIES,
            user_id=user.user_id,
            resource_type='drug_outcome_summaries',
            details={
                'drug_query': drug_query,
                'outcome_query': outcome_query,
                'location_id': patient_location_id,
                'result_count': len(sanitized_summaries)
            }
        )
        
        return SummaryResponse(
            summaries=sanitized_summaries,
            total_count=total_count,
            suppressed_count=suppressed_count
        )
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/drugs")
async def list_drugs(
    query: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """List available drugs with signal counts"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        sql = """
            SELECT DISTINCT drug_code, drug_name, COUNT(*) as signal_count
            FROM drug_outcome_signals
        """
        params = []
        
        if query:
            sql += " WHERE drug_name ILIKE %s OR drug_code ILIKE %s"
            params.extend([f"%{query}%", f"%{query}%"])
        
        sql += " GROUP BY drug_code, drug_name ORDER BY signal_count DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(sql, params)
        drugs = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        conn.close()
        return {"drugs": drugs}
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/outcomes")
async def list_outcomes(
    query: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """List available outcomes/adverse events with signal counts"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        sql = """
            SELECT DISTINCT outcome_code, outcome_name, COUNT(*) as signal_count
            FROM drug_outcome_signals
        """
        params = []
        
        if query:
            sql += " WHERE outcome_name ILIKE %s OR outcome_code ILIKE %s"
            params.extend([f"%{query}%", f"%{query}%"])
        
        sql += " GROUP BY outcome_code, outcome_name ORDER BY signal_count DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(sql, params)
        outcomes = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        conn.close()
        return {"outcomes": outcomes}
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/locations")
async def list_locations_with_signals(
    limit: int = Query(50, ge=1, le=200),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """List locations that have drug safety signals"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT l.id, l.name, l.city, l.state, COUNT(s.id) as signal_count
            FROM locations l
            JOIN drug_outcome_signals s ON s.patient_location_id = l.id
            GROUP BY l.id, l.name, l.city, l.state
            ORDER BY signal_count DESC
            LIMIT %s
        """, (limit,))
        
        locations = [normalize_row(dict(row)) for row in cur.fetchall()]
        conn.close()
        
        return {"locations": locations}
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/signal/{signal_id}")
async def get_signal_detail(
    signal_id: int,
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """Get detailed information for a specific signal"""
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT s.*, l.name as location_name, l.city, l.state
            FROM drug_outcome_signals s
            LEFT JOIN locations l ON s.patient_location_id = l.id
            WHERE s.id = %s
        """, (signal_id,))
        
        signal = cur.fetchone()
        conn.close()
        
        if not signal:
            raise HTTPException(status_code=404, detail="Signal not found")
        
        signal = normalize_row(dict(signal))
        if signal.get('details_json') and isinstance(signal['details_json'], str):
            signal['details_json'] = json.loads(signal['details_json'])
        if signal.get('updated_at'):
            signal['updated_at'] = signal['updated_at'].isoformat()
        
        sanitized = privacy_guard.sanitize_signal(signal)
        
        return {"signal": sanitized}
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/run-scan")
async def trigger_drug_scan(
    drug_codes: Optional[List[str]] = None,
    outcome_codes: Optional[List[str]] = None,
    location_ids: Optional[List[str]] = None,
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Manually trigger a drug safety scan (admin only).
    
    By default runs a full scan. Optionally filter by specific drugs, outcomes, or locations.
    """
    if user.get('role') not in ['admin', 'researcher']:
        raise HTTPException(status_code=403, detail="Admin or researcher role required")
    
    from .drug_scanner import DrugSafetyScanner
    
    audit_logger = EpidemiologyAuditLogger(DB_URL)
    
    try:
        scanner = DrugSafetyScanner(DB_URL)
        result = scanner.run_scan(
            drug_codes=drug_codes,
            outcome_codes=outcome_codes,
            location_ids=location_ids
        )
        
        audit_logger.log_background_scan(
            scan_type='manual',
            drugs_scanned=result.get('drugs_scanned', 0),
            outcomes_scanned=result.get('outcomes_scanned', 0),
            signals_generated=result.get('signals_generated', 0),
            alerts_created=result.get('alerts_created', 0),
            duration_seconds=result.get('duration_seconds', 0)
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")
