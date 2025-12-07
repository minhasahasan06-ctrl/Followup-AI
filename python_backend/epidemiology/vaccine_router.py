"""
Vaccine Epidemiology API Router
================================
Endpoints for vaccine coverage, effectiveness, and adverse event monitoring.
"""

import os
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, Header
from pydantic import BaseModel
import psycopg2
import psycopg2.extras
import json
import math
from decimal import Decimal

from .privacy import PrivacyGuard, PrivacyConfig, MIN_CELL_SIZE
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

router = APIRouter(prefix="/api/v1/vaccine", tags=["Vaccine Epidemiology"])

DB_URL = os.environ.get('DATABASE_URL')


def get_db_connection():
    return psycopg2.connect(DB_URL)


class CoverageResponse(BaseModel):
    vaccine_code: str
    vaccine_name: str
    location_id: Optional[str]
    location_name: Optional[str]
    total_population: int
    vaccinated_count: int
    coverage_rate: float
    by_dose: Dict[int, int]
    suppressed: bool = False


class EffectivenessResponse(BaseModel):
    vaccine_code: str
    vaccine_name: str
    outcome_code: str
    outcome_name: str
    location_id: Optional[str]
    effectiveness: float
    ci_lower: float
    ci_upper: float
    n_vaccinated: int
    n_unvaccinated: int
    events_vaccinated: int
    events_unvaccinated: int
    model_type: str
    suppressed: bool = False


class VaccineAdverseEventSummary(BaseModel):
    vaccine_code: str
    vaccine_name: str
    event_code: str
    event_name: str
    event_count: int
    rate_per_1000: float
    seriousness_breakdown: Dict[str, int]
    suppressed: bool = False


class VaccineSummary(BaseModel):
    vaccine_code: str
    vaccine_name: str
    total_doses: int
    unique_recipients: int
    adverse_event_count: int
    adverse_event_rate: float


@router.get("/coverage", response_model=CoverageResponse)
async def get_vaccine_coverage(
    vaccine_code: str = Query(..., description="Vaccine code"),
    location_id: Optional[str] = Query(None),
    dose_number: Optional[int] = Query(None, description="Filter by dose number"),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get vaccine coverage rate for a location.
    
    Returns proportion of population vaccinated with privacy protection.
    """
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        location_filter = "AND pl.location_id = %s" if location_id else ""
        params = [vaccine_code]
        if location_id:
            params.append(location_id)
        
        cur.execute(f"""
            SELECT COUNT(DISTINCT i.patient_id) as vaccinated_count,
                   i.dose_number,
                   i.vaccine_name
            FROM epi_immunizations i
            JOIN patient_locations pl ON i.patient_id = pl.patient_id
            WHERE i.vaccine_code = %s
            {location_filter}
            GROUP BY i.dose_number, i.vaccine_name
        """, params)
        
        dose_rows = cur.fetchall()
        
        pop_query = """
            SELECT COUNT(DISTINCT patient_id) as total_population
            FROM patient_locations
        """
        if location_id:
            pop_query += " WHERE location_id = %s"
            cur.execute(pop_query, (location_id,))
        else:
            cur.execute(pop_query)
        
        pop_row = cur.fetchone()
        total_population = pop_row['total_population'] if pop_row else 0
        
        location_name = None
        if location_id:
            cur.execute("SELECT name FROM locations WHERE id = %s", (location_id,))
            loc_row = cur.fetchone()
            location_name = loc_row['name'] if loc_row else None
        
        conn.close()
        
        if not privacy_guard.check_cell_size(total_population):
            return CoverageResponse(
                vaccine_code=vaccine_code,
                vaccine_name=dose_rows[0]['vaccine_name'] if dose_rows else vaccine_code,
                location_id=location_id,
                location_name=location_name,
                total_population=0,
                vaccinated_count=0,
                coverage_rate=0,
                by_dose={},
                suppressed=True
            )
        
        by_dose = {}
        total_vaccinated = 0
        vaccine_name = vaccine_code
        
        for row in dose_rows:
            dose_num = row['dose_number']
            count = row['vaccinated_count']
            vaccine_name = row['vaccine_name']
            
            if privacy_guard.check_cell_size(count):
                by_dose[dose_num] = count
                if dose_num == 1:
                    total_vaccinated = count
        
        if total_vaccinated == 0 and by_dose:
            total_vaccinated = max(by_dose.values())
        
        coverage_rate = (total_vaccinated / total_population * 100) if total_population > 0 else 0
        
        return CoverageResponse(
            vaccine_code=vaccine_code,
            vaccine_name=vaccine_name,
            location_id=location_id,
            location_name=location_name,
            total_population=total_population,
            vaccinated_count=total_vaccinated,
            coverage_rate=round(coverage_rate, 2),
            by_dose=by_dose,
            suppressed=False
        )
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/effectiveness", response_model=EffectivenessResponse)
async def get_vaccine_effectiveness(
    vaccine_code: str = Query(...),
    outcome_code: str = Query(..., description="Outcome to measure against (e.g., infection, hospitalization)"),
    location_id: Optional[str] = Query(None),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Estimate vaccine effectiveness against a specific outcome.
    
    Uses test-negative design or cohort comparison with privacy protection.
    """
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        location_filter = "AND pl.location_id = %s" if location_id else ""
        params_base = [vaccine_code]
        if location_id:
            params_base.append(location_id)
        
        cur.execute(f"""
            WITH vaccinated AS (
                SELECT DISTINCT i.patient_id
                FROM epi_immunizations i
                JOIN patient_locations pl ON i.patient_id = pl.patient_id
                WHERE i.vaccine_code = %s
                {location_filter}
            ),
            outcomes AS (
                SELECT DISTINCT patient_id
                FROM infectious_events
                WHERE pathogen_code = %s
            ),
            all_patients AS (
                SELECT DISTINCT patient_id
                FROM patient_locations
                {'WHERE location_id = %s' if location_id else ''}
            )
            SELECT
                (SELECT COUNT(*) FROM vaccinated) as n_vaccinated,
                (SELECT COUNT(*) FROM all_patients) - (SELECT COUNT(*) FROM vaccinated) as n_unvaccinated,
                (SELECT COUNT(*) FROM vaccinated v WHERE v.patient_id IN (SELECT patient_id FROM outcomes)) as events_vaccinated,
                (SELECT COUNT(*) FROM outcomes o 
                 WHERE o.patient_id IN (SELECT patient_id FROM all_patients)
                 AND o.patient_id NOT IN (SELECT patient_id FROM vaccinated)) as events_unvaccinated
        """, params_base + [outcome_code] + ([location_id] if location_id else []))
        
        row = cur.fetchone()
        
        cur.execute("""
            SELECT vaccine_name FROM epi_immunizations 
            WHERE vaccine_code = %s LIMIT 1
        """, (vaccine_code,))
        name_row = cur.fetchone()
        vaccine_name = name_row['vaccine_name'] if name_row else vaccine_code
        
        cur.execute("""
            SELECT pathogen_name FROM infectious_events 
            WHERE pathogen_code = %s LIMIT 1
        """, (outcome_code,))
        outcome_row = cur.fetchone()
        outcome_name = outcome_row['pathogen_name'] if outcome_row else outcome_code
        
        conn.close()
        
        if not row:
            raise HTTPException(status_code=404, detail="No data found")
        
        n_vaccinated = row['n_vaccinated'] or 0
        n_unvaccinated = row['n_unvaccinated'] or 0
        events_vaccinated = row['events_vaccinated'] or 0
        events_unvaccinated = row['events_unvaccinated'] or 0
        
        if not privacy_guard.check_cell_size(n_vaccinated) or not privacy_guard.check_cell_size(n_unvaccinated):
            return EffectivenessResponse(
                vaccine_code=vaccine_code,
                vaccine_name=vaccine_name,
                outcome_code=outcome_code,
                outcome_name=outcome_name,
                location_id=location_id,
                effectiveness=0,
                ci_lower=0,
                ci_upper=0,
                n_vaccinated=0,
                n_unvaccinated=0,
                events_vaccinated=0,
                events_unvaccinated=0,
                model_type="suppressed",
                suppressed=True
            )
        
        rate_vaccinated = events_vaccinated / n_vaccinated if n_vaccinated > 0 else 0
        rate_unvaccinated = events_unvaccinated / n_unvaccinated if n_unvaccinated > 0 else 0
        
        if rate_unvaccinated > 0:
            rr = rate_vaccinated / rate_unvaccinated
            ve = (1 - rr) * 100
            
            log_rr = math.log(rr) if rr > 0 else 0
            se = math.sqrt(
                (1 - rate_vaccinated) / (events_vaccinated + 0.5) +
                (1 - rate_unvaccinated) / (events_unvaccinated + 0.5)
            )
            ci_lower = (1 - math.exp(log_rr + 1.96 * se)) * 100
            ci_upper = (1 - math.exp(log_rr - 1.96 * se)) * 100
        else:
            ve = 100 if rate_vaccinated == 0 else 0
            ci_lower = 0
            ci_upper = 100
        
        return EffectivenessResponse(
            vaccine_code=vaccine_code,
            vaccine_name=vaccine_name,
            outcome_code=outcome_code,
            outcome_name=outcome_name,
            location_id=location_id,
            effectiveness=round(ve, 2),
            ci_lower=round(ci_lower, 2),
            ci_upper=round(ci_upper, 2),
            n_vaccinated=n_vaccinated,
            n_unvaccinated=n_unvaccinated,
            events_vaccinated=events_vaccinated,
            events_unvaccinated=events_unvaccinated,
            model_type="cohort_comparison",
            suppressed=False
        )
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/adverse-events", response_model=List[VaccineAdverseEventSummary])
async def get_vaccine_adverse_events(
    vaccine_code: Optional[str] = Query(None),
    location_id: Optional[str] = Query(None),
    seriousness: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get aggregated vaccine adverse event summary.
    
    Returns event counts and rates with privacy protection.
    """
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT 
                vae.vaccine_code,
                i.vaccine_name,
                vae.event_code,
                vae.event_name,
                COUNT(*) as event_count,
                vae.seriousness
            FROM vaccine_adverse_events vae
            JOIN epi_immunizations i ON vae.immunization_id = i.id
            WHERE 1=1
        """
        params = []
        
        if vaccine_code:
            query += " AND vae.vaccine_code = %s"
            params.append(vaccine_code)
        
        if seriousness:
            query += " AND vae.seriousness = %s"
            params.append(seriousness)
        
        query += " GROUP BY vae.vaccine_code, i.vaccine_name, vae.event_code, vae.event_name, vae.seriousness"
        query += " ORDER BY event_count DESC LIMIT %s"
        params.append(limit)
        
        cur.execute(query, params)
        rows = cur.fetchall()
        
        cur.execute("""
            SELECT vaccine_code, COUNT(*) as total_doses
            FROM epi_immunizations
            GROUP BY vaccine_code
        """)
        dose_counts = {row['vaccine_code']: row['total_doses'] for row in cur.fetchall()}
        
        conn.close()
        
        aggregated = {}
        for row in rows:
            key = (row['vaccine_code'], row['event_code'])
            if key not in aggregated:
                aggregated[key] = {
                    'vaccine_code': row['vaccine_code'],
                    'vaccine_name': row['vaccine_name'],
                    'event_code': row['event_code'],
                    'event_name': row['event_name'],
                    'event_count': 0,
                    'seriousness_breakdown': {}
                }
            aggregated[key]['event_count'] += row['event_count']
            aggregated[key]['seriousness_breakdown'][row['seriousness']] = row['event_count']
        
        results = []
        for data in aggregated.values():
            if not privacy_guard.check_cell_size(data['event_count']):
                continue
            
            total_doses = dose_counts.get(data['vaccine_code'], 1)
            rate = (data['event_count'] / total_doses) * 1000
            
            results.append(VaccineAdverseEventSummary(
                vaccine_code=data['vaccine_code'],
                vaccine_name=data['vaccine_name'],
                event_code=data['event_code'],
                event_name=data['event_name'],
                event_count=data['event_count'],
                rate_per_1000=round(rate, 4),
                seriousness_breakdown=data['seriousness_breakdown'],
                suppressed=False
            ))
        
        return results
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.get("/vaccines", response_model=List[VaccineSummary])
async def list_vaccines(
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """List available vaccines with summary statistics"""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                i.vaccine_code,
                i.vaccine_name,
                COUNT(*) as total_doses,
                COUNT(DISTINCT i.patient_id) as unique_recipients,
                COALESCE(ae.adverse_count, 0) as adverse_event_count
            FROM epi_immunizations i
            LEFT JOIN (
                SELECT vaccine_code, COUNT(*) as adverse_count
                FROM vaccine_adverse_events
                GROUP BY vaccine_code
            ) ae ON i.vaccine_code = ae.vaccine_code
            GROUP BY i.vaccine_code, i.vaccine_name, ae.adverse_count
            ORDER BY total_doses DESC
        """)
        
        rows = cur.fetchall()
        conn.close()
        
        return [
            VaccineSummary(
                vaccine_code=row['vaccine_code'],
                vaccine_name=row['vaccine_name'],
                total_doses=row['total_doses'],
                unique_recipients=row['unique_recipients'],
                adverse_event_count=row['adverse_event_count'],
                adverse_event_rate=round((row['adverse_event_count'] / row['total_doses']) * 1000, 4) if row['total_doses'] > 0 else 0
            )
            for row in rows
        ]
        
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
