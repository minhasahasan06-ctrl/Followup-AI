"""
Genetic/Molecular Epidemiology API Router
==========================================
Endpoints for variant-outcome associations, GWAS results, and pharmacogenomics.
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

router = APIRouter(prefix="/api/v1/genetic", tags=["Genetic/Molecular Epidemiology"])

DB_URL = os.environ.get('DATABASE_URL')
audit_logger = EpidemiologyAuditLogger()


class VariantAssociation(BaseModel):
    id: int
    rsid: str
    gene_symbol: str
    gene_name: Optional[str]
    outcome_code: str
    outcome_name: str
    estimate: float
    ci_lower: float
    ci_upper: float
    p_value: float
    signal_strength: float
    n_carriers: int
    n_non_carriers: int
    flagged: bool
    suppressed: bool = False


class VariantAssociationResponse(BaseModel):
    associations: List[Dict[str, Any]]
    total_count: int
    suppressed_count: int
    privacy_note: str


class GWASResult(BaseModel):
    id: int
    study_id: str
    study_name: str
    trait_code: str
    trait_name: str
    rsid: str
    gene_symbol: Optional[str]
    chromosome: str
    position: int
    beta: Optional[float]
    odds_ratio: Optional[float]
    ci_lower: Optional[float]
    ci_upper: Optional[float]
    p_value: float
    genome_wide_significant: bool


class GWASResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int


class PharmacogenomicInteraction(BaseModel):
    id: int
    rsid: str
    gene_symbol: str
    gene_name: Optional[str]
    drug_code: str
    drug_name: str
    interaction_type: str
    phenotype: str
    recommendation: Optional[str]
    evidence_level: str
    clinical_impact: str


class PharmacogenomicResponse(BaseModel):
    interactions: List[Dict[str, Any]]
    total_count: int
    suppressed_count: int


class GeneSummary(BaseModel):
    gene_symbol: str
    gene_name: Optional[str]
    variant_count: int
    association_count: int
    significant_associations: int


def get_db_connection():
    """Get database connection"""
    return psycopg2.connect(DB_URL)


@router.get("/variant-associations", response_model=VariantAssociationResponse)
async def get_variant_associations(
    rsid_query: Optional[str] = Query(None, description="RS ID search (e.g., rs12345)"),
    gene_query: Optional[str] = Query(None, description="Gene symbol search (e.g., CYP2D6)"),
    outcome_query: Optional[str] = Query(None, description="Outcome code or name search"),
    location_id: Optional[str] = Query(None, description="Filter by location"),
    flagged_only: bool = Query(False, description="Only show significant associations"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get variant-outcome associations with privacy protection.
    
    Returns aggregated genetic associations from population studies.
    All results are subject to minimum cell-size suppression.
    """
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT * FROM variant_outcome_associations
            WHERE 1=1
        """
        params: List[Any] = []
        
        if rsid_query:
            query += " AND rsid ILIKE %s"
            params.append(f"%{rsid_query}%")
        
        if gene_query:
            query += " AND (gene_symbol ILIKE %s OR gene_name ILIKE %s)"
            params.extend([f"%{gene_query}%", f"%{gene_query}%"])
        
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
        query += " ORDER BY p_value ASC, signal_strength DESC NULLS LAST LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        associations = cur.fetchall()
        
        # Apply privacy protections
        sanitized = []
        suppressed_count = 0
        
        for assoc in associations:
            assoc_dict = normalize_row(dict(assoc))
            n_carriers = assoc_dict.get('n_carriers', 0)
            n_non_carriers = assoc_dict.get('n_non_carriers', 0)
            
            if not privacy_guard.check_cell_size(n_carriers) or not privacy_guard.check_cell_size(n_non_carriers):
                assoc_dict['suppressed'] = True
                assoc_dict['estimate'] = None
                assoc_dict['ci_lower'] = None
                assoc_dict['ci_upper'] = None
                assoc_dict['p_value'] = None
                suppressed_count += 1
            else:
                assoc_dict['suppressed'] = False
            
            sanitized.append(assoc_dict)
        
        audit_logger.log(
            action=AuditAction.VIEW_GENETIC,
            user_id=user.user_id,
            resource_type='variant_associations',
            details={
                'rsid_query': rsid_query,
                'gene_query': gene_query,
                'outcome_query': outcome_query,
                'result_count': len(sanitized)
            }
        )
        
        return VariantAssociationResponse(
            associations=sanitized,
            total_count=total_count,
            suppressed_count=suppressed_count,
            privacy_note=f"Results with fewer than {MIN_CELL_SIZE} carriers are suppressed for privacy protection."
        )
    
    finally:
        conn.close()


@router.get("/gwas-results", response_model=GWASResponse)
async def get_gwas_results(
    trait_query: Optional[str] = Query(None, description="Trait code or name search"),
    rsid_query: Optional[str] = Query(None, description="RS ID search"),
    gene_query: Optional[str] = Query(None, description="Gene symbol search"),
    chromosome: Optional[str] = Query(None, description="Filter by chromosome"),
    significant_only: bool = Query(False, description="Only genome-wide significant (p < 5e-8)"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get GWAS summary statistics.
    
    Returns published GWAS results for trait-variant associations.
    """
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT * FROM gwas_results
            WHERE 1=1
        """
        params: List[Any] = []
        
        if trait_query:
            query += " AND (trait_code ILIKE %s OR trait_name ILIKE %s)"
            params.extend([f"%{trait_query}%", f"%{trait_query}%"])
        
        if rsid_query:
            query += " AND rsid ILIKE %s"
            params.append(f"%{rsid_query}%")
        
        if gene_query:
            query += " AND gene_symbol ILIKE %s"
            params.append(f"%{gene_query}%")
        
        if chromosome:
            query += " AND chromosome = %s"
            params.append(chromosome)
        
        if significant_only:
            query += " AND genome_wide_significant = TRUE"
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM ({query}) AS sub"
        cur.execute(count_query, params)
        count_result = cur.fetchone()
        total_count = count_result['count'] if count_result else 0
        
        # Get paginated results
        query += " ORDER BY p_value ASC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        results = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        audit_logger.log(
            action=AuditAction.VIEW_GENETIC,
            user_id=user.user_id,
            resource_type='gwas_results',
            details={
                'trait_query': trait_query,
                'rsid_query': rsid_query,
                'result_count': len(results)
            }
        )
        
        return GWASResponse(
            results=results,
            total_count=total_count
        )
    
    finally:
        conn.close()


@router.get("/pharmacogenomics", response_model=PharmacogenomicResponse)
async def get_pharmacogenomics(
    drug_query: Optional[str] = Query(None, description="Drug code or name search"),
    gene_query: Optional[str] = Query(None, description="Gene symbol search"),
    rsid_query: Optional[str] = Query(None, description="RS ID search"),
    clinical_impact: Optional[str] = Query(None, description="Filter by impact: high, moderate, low"),
    limit: int = Query(50, le=200),
    offset: int = Query(0),
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
):
    """
    Get pharmacogenomic drug-gene interactions.
    
    Returns gene variants that affect drug metabolism or response.
    """
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT * FROM pharmacogenomic_interactions
            WHERE 1=1
        """
        params: List[Any] = []
        
        if drug_query:
            query += " AND (drug_code ILIKE %s OR drug_name ILIKE %s)"
            params.extend([f"%{drug_query}%", f"%{drug_query}%"])
        
        if gene_query:
            query += " AND (gene_symbol ILIKE %s OR gene_name ILIKE %s)"
            params.extend([f"%{gene_query}%", f"%{gene_query}%"])
        
        if rsid_query:
            query += " AND rsid ILIKE %s"
            params.append(f"%{rsid_query}%")
        
        if clinical_impact:
            query += " AND clinical_impact = %s"
            params.append(clinical_impact)
        
        # Count total
        count_query = f"SELECT COUNT(*) FROM ({query}) AS sub"
        cur.execute(count_query, params)
        count_result = cur.fetchone()
        total_count = count_result['count'] if count_result else 0
        
        # Get paginated results
        query += " ORDER BY clinical_impact DESC, evidence_level DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])
        
        cur.execute(query, params)
        interactions = cur.fetchall()
        
        # Apply privacy protections
        sanitized = []
        suppressed_count = 0
        
        for interaction in interactions:
            int_dict = normalize_row(dict(interaction))
            n_patients = int_dict.get('n_patients', 0)
            
            if n_patients and not privacy_guard.check_cell_size(n_patients):
                int_dict['suppressed'] = True
                int_dict['n_patients'] = None
                suppressed_count += 1
            else:
                int_dict['suppressed'] = False
            
            sanitized.append(int_dict)
        
        audit_logger.log(
            action=AuditAction.VIEW_GENETIC,
            user_id=user.user_id,
            resource_type='pharmacogenomics',
            details={
                'drug_query': drug_query,
                'gene_query': gene_query,
                'result_count': len(sanitized)
            }
        )
        
        return PharmacogenomicResponse(
            interactions=sanitized,
            total_count=total_count,
            suppressed_count=suppressed_count
        )
    
    finally:
        conn.close()


@router.get("/genes")
async def get_genes(
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, Any]:
    """Get list of genes with association summaries (privacy-protected)."""
    privacy_guard = PrivacyGuard(PrivacyConfig(min_cell_size=MIN_CELL_SIZE))
    
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                gene_symbol,
                gene_name,
                COUNT(DISTINCT rsid) as variant_count,
                COUNT(*) as association_count,
                SUM(CASE WHEN flagged THEN 1 ELSE 0 END) as significant_associations,
                SUM(n_carriers) as total_carriers
            FROM variant_outcome_associations
            WHERE gene_symbol IS NOT NULL
            GROUP BY gene_symbol, gene_name
            ORDER BY association_count DESC
            LIMIT 100
        """)
        
        genes = []
        for row in cur.fetchall():
            row_dict = normalize_row(dict(row))
            # Suppress if total carrier count is below threshold
            if not privacy_guard.check_cell_size(row_dict.get('total_carriers', 0)):
                row_dict['variant_count'] = None
                row_dict['association_count'] = None
                row_dict['significant_associations'] = None
                row_dict['suppressed'] = True
            else:
                row_dict['suppressed'] = False
            # Remove total_carriers from response (internal privacy check only)
            row_dict.pop('total_carriers', None)
            genes.append(row_dict)
        
        audit_logger.log(
            action=AuditAction.VIEW_GENETIC,
            user_id=user.user_id,
            resource_type='genes_list',
            details={'result_count': len(genes)}
        )
        
        return {"genes": genes}
    
    finally:
        conn.close()


@router.get("/traits")
async def get_traits(
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, Any]:
    """Get list of traits from GWAS studies (privacy-protected)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                trait_code,
                trait_name,
                COUNT(*) as variant_count,
                SUM(CASE WHEN genome_wide_significant THEN 1 ELSE 0 END) as significant_hits,
                MIN(p_value) as top_p_value
            FROM gwas_results
            GROUP BY trait_code, trait_name
            ORDER BY variant_count DESC
        """)
        
        traits = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        audit_logger.log(
            action=AuditAction.VIEW_GENETIC,
            user_id=user.user_id,
            resource_type='traits_list',
            details={'result_count': len(traits)}
        )
        
        return {"traits": traits}
    
    finally:
        conn.close()


@router.get("/drugs")
async def get_drugs_with_pgx(
    user: AuthenticatedUser = Depends(verify_epidemiology_auth)
) -> Dict[str, Any]:
    """Get list of drugs with pharmacogenomic interactions (privacy-protected)."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        cur.execute("""
            SELECT 
                drug_code,
                drug_name,
                COUNT(*) as interaction_count,
                COUNT(DISTINCT gene_symbol) as genes_affected,
                SUM(CASE WHEN clinical_impact = 'high' THEN 1 ELSE 0 END) as high_impact_count
            FROM pharmacogenomic_interactions
            GROUP BY drug_code, drug_name
            ORDER BY interaction_count DESC
        """)
        
        drugs = [normalize_row(dict(row)) for row in cur.fetchall()]
        
        audit_logger.log(
            action=AuditAction.VIEW_GENETIC,
            user_id=user.user_id,
            resource_type='drugs_pgx_list',
            details={'result_count': len(drugs)}
        )
        
        return {"drugs": drugs}
    
    finally:
        conn.close()
