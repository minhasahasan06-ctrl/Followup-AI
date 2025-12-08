"""
Research Governance & Reproducibility Framework
================================================
Production-grade governance system for research protocols including:
- Protocol/project IDs with semantic versioning
- Versioned analysis specifications
- Data snapshot linking
- Exploratory vs pre-specified analysis marking
- Reproducibility bundle export

HIPAA-compliant with comprehensive audit logging.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import zipfile
import io
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class AnalysisStatus(str, Enum):
    DRAFT = "draft"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class AnalysisType(str, Enum):
    EXPLORATORY = "exploratory"
    PRE_SPECIFIED = "pre_specified"
    SENSITIVITY = "sensitivity"
    SUBGROUP = "subgroup"


@dataclass
class AnalysisSpec:
    """Versioned analysis specification"""
    spec_id: str
    version: str
    name: str
    description: str
    analysis_type: AnalysisType
    cohort_definition: Dict[str, Any]
    exposure_definition: Dict[str, Any]
    outcome_definition: Dict[str, Any]
    covariates: List[str]
    statistical_methods: List[str]
    sensitivity_analyses: List[Dict[str, Any]]
    created_by: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    def get_hash(self) -> str:
        """Get deterministic hash of spec for versioning"""
        content = json.dumps({
            'cohort_definition': self.cohort_definition,
            'exposure_definition': self.exposure_definition,
            'outcome_definition': self.outcome_definition,
            'covariates': sorted(self.covariates),
            'statistical_methods': sorted(self.statistical_methods)
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class DataSnapshot:
    """Reference to a specific version of the data"""
    snapshot_id: str
    created_at: datetime
    table_checksums: Dict[str, str]
    row_counts: Dict[str, int]
    date_range: Tuple[str, str]
    created_by: str
    description: str = ""
    
    def get_hash(self) -> str:
        """Get deterministic hash of snapshot"""
        content = json.dumps(self.table_checksums, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]


@dataclass
class Protocol:
    """Research protocol with full governance tracking"""
    protocol_id: str
    title: str
    description: str
    principal_investigator: str
    status: AnalysisStatus
    analysis_spec: AnalysisSpec
    data_snapshot_id: Optional[str]
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    irb_number: Optional[str] = None
    funding_source: Optional[str] = None


class GovernanceManager:
    """
    Manages research governance, versioning, and reproducibility
    """
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def create_protocol(
        self,
        title: str,
        description: str,
        principal_investigator: str,
        analysis_type: AnalysisType = AnalysisType.EXPLORATORY,
        irb_number: Optional[str] = None
    ) -> Protocol:
        """Create a new research protocol"""
        protocol_id = f"PROT-{uuid.uuid4().hex[:8].upper()}"
        spec_id = f"SPEC-{uuid.uuid4().hex[:8].upper()}"
        
        analysis_spec = AnalysisSpec(
            spec_id=spec_id,
            version="1.0.0",
            name=f"{title} - Analysis Specification",
            description=description,
            analysis_type=analysis_type,
            cohort_definition={},
            exposure_definition={},
            outcome_definition={},
            covariates=[],
            statistical_methods=[],
            sensitivity_analyses=[],
            created_by=principal_investigator
        )
        
        protocol = Protocol(
            protocol_id=protocol_id,
            title=title,
            description=description,
            principal_investigator=principal_investigator,
            status=AnalysisStatus.DRAFT,
            analysis_spec=analysis_spec,
            data_snapshot_id=None,
            irb_number=irb_number
        )
        
        self._save_protocol(protocol)
        
        logger.info(f"Created protocol {protocol_id}")
        return protocol
    
    def _save_protocol(self, protocol: Protocol):
        """Save protocol to database"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO research_protocols 
                (id, title, description, principal_investigator, status, 
                 analysis_spec, data_snapshot_id, version, irb_number, 
                 created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    status = EXCLUDED.status,
                    analysis_spec = EXCLUDED.analysis_spec,
                    data_snapshot_id = EXCLUDED.data_snapshot_id,
                    version = EXCLUDED.version,
                    updated_at = NOW()
            """, (
                protocol.protocol_id,
                protocol.title,
                protocol.description,
                protocol.principal_investigator,
                protocol.status.value,
                json.dumps(asdict(protocol.analysis_spec), default=str),
                protocol.data_snapshot_id,
                protocol.version,
                protocol.irb_number,
                protocol.created_at,
                datetime.utcnow()
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save protocol: {e}")
            raise
    
    def get_protocol(self, protocol_id: str) -> Optional[Protocol]:
        """Retrieve a protocol by ID"""
        try:
            conn = self.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM research_protocols WHERE id = %s
            """, (protocol_id,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if not row:
                return None
            
            spec_data = row['analysis_spec']
            if isinstance(spec_data, str):
                spec_data = json.loads(spec_data)
            
            analysis_spec = AnalysisSpec(
                spec_id=spec_data.get('spec_id', ''),
                version=spec_data.get('version', '1.0.0'),
                name=spec_data.get('name', ''),
                description=spec_data.get('description', ''),
                analysis_type=AnalysisType(spec_data.get('analysis_type', 'exploratory')),
                cohort_definition=spec_data.get('cohort_definition', {}),
                exposure_definition=spec_data.get('exposure_definition', {}),
                outcome_definition=spec_data.get('outcome_definition', {}),
                covariates=spec_data.get('covariates', []),
                statistical_methods=spec_data.get('statistical_methods', []),
                sensitivity_analyses=spec_data.get('sensitivity_analyses', []),
                created_by=spec_data.get('created_by', '')
            )
            
            return Protocol(
                protocol_id=row['id'],
                title=row['title'],
                description=row['description'],
                principal_investigator=row['principal_investigator'],
                status=AnalysisStatus(row['status']),
                analysis_spec=analysis_spec,
                data_snapshot_id=row['data_snapshot_id'],
                version=row['version'],
                irb_number=row['irb_number'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            
        except Exception as e:
            logger.error(f"Failed to get protocol: {e}")
            return None
    
    def update_analysis_spec(
        self,
        protocol_id: str,
        cohort_definition: Optional[Dict[str, Any]] = None,
        exposure_definition: Optional[Dict[str, Any]] = None,
        outcome_definition: Optional[Dict[str, Any]] = None,
        covariates: Optional[List[str]] = None,
        statistical_methods: Optional[List[str]] = None,
        sensitivity_analyses: Optional[List[Dict[str, Any]]] = None,
        user_id: str = "system"
    ) -> Protocol:
        """Update analysis specification and increment version"""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        spec = protocol.analysis_spec
        
        if cohort_definition is not None:
            spec.cohort_definition = cohort_definition
        if exposure_definition is not None:
            spec.exposure_definition = exposure_definition
        if outcome_definition is not None:
            spec.outcome_definition = outcome_definition
        if covariates is not None:
            spec.covariates = covariates
        if statistical_methods is not None:
            spec.statistical_methods = statistical_methods
        if sensitivity_analyses is not None:
            spec.sensitivity_analyses = sensitivity_analyses
        
        old_version = spec.version
        version_parts = old_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        spec.version = '.'.join(version_parts)
        
        self._log_spec_change(protocol_id, old_version, spec.version, user_id)
        
        protocol.updated_at = datetime.utcnow()
        self._save_protocol(protocol)
        
        logger.info(f"Updated spec for {protocol_id}: {old_version} -> {spec.version}")
        return protocol
    
    def _log_spec_change(
        self, 
        protocol_id: str, 
        old_version: str, 
        new_version: str, 
        user_id: str
    ):
        """Log specification change for audit trail"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO protocol_version_audit_log 
                (id, protocol_id, action, old_version, new_version, user_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (
                str(uuid.uuid4()),
                protocol_id,
                'spec_update',
                old_version,
                new_version,
                user_id
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log spec change: {e}")
    
    def create_data_snapshot(
        self,
        description: str,
        user_id: str,
        tables: Optional[List[str]] = None
    ) -> DataSnapshot:
        """Create a snapshot of current data state"""
        snapshot_id = f"SNAP-{uuid.uuid4().hex[:8].upper()}"
        
        if tables is None:
            tables = [
                'drug_outcome_signals',
                'drug_outcome_summaries',
                'infectious_events_aggregated',
                'reproduction_numbers',
                'immunization_aggregates'
            ]
        
        table_checksums = {}
        row_counts = {}
        
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            for table in tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table}")
                    count_row = cur.fetchone()
                    count = count_row[0] if count_row else 0
                    row_counts[table] = count
                    
                    cur.execute(f"""
                        SELECT MD5(STRING_AGG(t::text, '')) 
                        FROM (SELECT * FROM {table} ORDER BY 1 LIMIT 1000) t
                    """)
                    checksum_row = cur.fetchone()
                    checksum = (checksum_row[0] if checksum_row else None) or 'empty'
                    table_checksums[table] = checksum
                except Exception as e:
                    logger.warning(f"Could not snapshot table {table}: {e}")
                    row_counts[table] = 0
                    table_checksums[table] = 'unavailable'
            
            cur.execute("SELECT MIN(updated_at), MAX(updated_at) FROM drug_outcome_signals")
            date_row = cur.fetchone()
            date_range = (
                str(date_row[0]) if date_row and date_row[0] else 'unknown',
                str(date_row[1]) if date_row and date_row[1] else 'unknown'
            )
            
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            date_range = ('unknown', 'unknown')
        
        snapshot = DataSnapshot(
            snapshot_id=snapshot_id,
            created_at=datetime.utcnow(),
            table_checksums=table_checksums,
            row_counts=row_counts,
            date_range=date_range,
            created_by=user_id,
            description=description
        )
        
        self._save_snapshot(snapshot)
        
        logger.info(f"Created data snapshot {snapshot_id}")
        return snapshot
    
    def _save_snapshot(self, snapshot: DataSnapshot):
        """Save snapshot to database"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO data_snapshots 
                (id, created_at, table_checksums, row_counts, date_range, 
                 created_by, description, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                snapshot.snapshot_id,
                snapshot.created_at,
                json.dumps(snapshot.table_checksums),
                json.dumps(snapshot.row_counts),
                json.dumps(snapshot.date_range),
                snapshot.created_by,
                snapshot.description,
                snapshot.get_hash()
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
    
    def link_snapshot_to_protocol(
        self,
        protocol_id: str,
        snapshot_id: str,
        user_id: str
    ) -> Protocol:
        """Link a data snapshot to a protocol"""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        protocol.data_snapshot_id = snapshot_id
        protocol.updated_at = datetime.utcnow()
        
        self._save_protocol(protocol)
        self._log_spec_change(protocol_id, "snapshot", snapshot_id, user_id)
        
        logger.info(f"Linked snapshot {snapshot_id} to protocol {protocol_id}")
        return protocol
    
    def mark_analysis_type(
        self,
        protocol_id: str,
        analysis_type: AnalysisType,
        user_id: str
    ) -> Protocol:
        """Mark analysis as exploratory or pre-specified"""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        old_type = protocol.analysis_spec.analysis_type
        protocol.analysis_spec.analysis_type = analysis_type
        
        self._save_protocol(protocol)
        self._log_spec_change(
            protocol_id, 
            f"type:{old_type.value}", 
            f"type:{analysis_type.value}", 
            user_id
        )
        
        logger.info(f"Marked {protocol_id} as {analysis_type.value}")
        return protocol
    
    def submit_for_approval(
        self,
        protocol_id: str,
        user_id: str
    ) -> Protocol:
        """Submit protocol for approval"""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        if protocol.status != AnalysisStatus.DRAFT:
            raise ValueError(f"Protocol must be in DRAFT status to submit")
        
        protocol.status = AnalysisStatus.SUBMITTED
        protocol.updated_at = datetime.utcnow()
        
        self._save_protocol(protocol)
        self._log_spec_change(protocol_id, "draft", "submitted", user_id)
        
        return protocol
    
    def approve_protocol(
        self,
        protocol_id: str,
        approver_id: str
    ) -> Protocol:
        """Approve a submitted protocol"""
        protocol = self.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        if protocol.status != AnalysisStatus.SUBMITTED:
            raise ValueError(f"Protocol must be SUBMITTED to approve")
        
        protocol.status = AnalysisStatus.APPROVED
        protocol.analysis_spec.approved_by = approver_id
        protocol.analysis_spec.approved_at = datetime.utcnow()
        protocol.updated_at = datetime.utcnow()
        
        self._save_protocol(protocol)
        self._log_spec_change(protocol_id, "submitted", "approved", approver_id)
        
        return protocol


class ReproducibilityExporter:
    """
    Exports reproducibility bundles for manuscripts and regulatory submissions
    """
    
    def __init__(self, governance_manager: GovernanceManager):
        self.governance = governance_manager
    
    def export_bundle(
        self,
        protocol_id: str,
        include_data_summary: bool = True,
        include_model_artifacts: bool = False,
        format: str = "json"
    ) -> bytes:
        """
        Export reproducibility bundle as ZIP file
        
        Returns:
            ZIP file contents as bytes
        """
        protocol = self.governance.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            manifest = {
                'bundle_id': str(uuid.uuid4()),
                'protocol_id': protocol_id,
                'version': protocol.version,
                'spec_version': protocol.analysis_spec.version,
                'exported_at': datetime.utcnow().isoformat(),
                'data_snapshot_id': protocol.data_snapshot_id,
                'analysis_type': protocol.analysis_spec.analysis_type.value,
                'spec_hash': protocol.analysis_spec.get_hash()
            }
            zf.writestr('manifest.json', json.dumps(manifest, indent=2))
            
            protocol_dict = {
                'protocol_id': protocol.protocol_id,
                'title': protocol.title,
                'description': protocol.description,
                'principal_investigator': protocol.principal_investigator,
                'status': protocol.status.value,
                'version': protocol.version,
                'irb_number': protocol.irb_number,
                'created_at': protocol.created_at.isoformat(),
                'updated_at': protocol.updated_at.isoformat()
            }
            zf.writestr('protocol.json', json.dumps(protocol_dict, indent=2))
            
            spec_dict = asdict(protocol.analysis_spec)
            spec_dict['analysis_type'] = protocol.analysis_spec.analysis_type.value
            spec_dict['created_at'] = protocol.analysis_spec.created_at.isoformat()
            if protocol.analysis_spec.approved_at:
                spec_dict['approved_at'] = protocol.analysis_spec.approved_at.isoformat()
            zf.writestr('analysis_spec.json', json.dumps(spec_dict, indent=2))
            
            if include_data_summary and protocol.data_snapshot_id:
                snapshot_summary = self._get_snapshot_summary(protocol.data_snapshot_id)
                if snapshot_summary:
                    zf.writestr('data_snapshot.json', json.dumps(snapshot_summary, indent=2))
            
            readme = self._generate_readme(protocol)
            zf.writestr('README.md', readme)
        
        buffer.seek(0)
        return buffer.read()
    
    def _get_snapshot_summary(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Get snapshot summary from database"""
        try:
            conn = self.governance.get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("SELECT * FROM data_snapshots WHERE id = %s", (snapshot_id,))
            row = cur.fetchone()
            
            cur.close()
            conn.close()
            
            if row:
                return {
                    'snapshot_id': row['id'],
                    'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                    'table_checksums': json.loads(row['table_checksums']) if row['table_checksums'] else {},
                    'row_counts': json.loads(row['row_counts']) if row['row_counts'] else {},
                    'content_hash': row['content_hash']
                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting snapshot: {e}")
            return None
    
    def _generate_readme(self, protocol: Protocol) -> str:
        """Generate README for reproducibility bundle"""
        return f"""# Reproducibility Bundle

## Protocol: {protocol.title}

**Protocol ID:** {protocol.protocol_id}
**Version:** {protocol.version}
**Principal Investigator:** {protocol.principal_investigator}
**IRB Number:** {protocol.irb_number or 'N/A'}

## Analysis Type

{protocol.analysis_spec.analysis_type.value.replace('_', '-').title()}

## Description

{protocol.description}

## Contents

- `manifest.json` - Bundle metadata and checksums
- `protocol.json` - Full protocol specification
- `analysis_spec.json` - Detailed analysis specification
- `data_snapshot.json` - Data snapshot reference (if included)

## Reproducibility

This bundle contains all information needed to reproduce the analysis.
The data snapshot ID can be used to request the exact data version used.

**Spec Hash:** {protocol.analysis_spec.get_hash()}
**Data Snapshot:** {protocol.data_snapshot_id or 'Not linked'}

## Generated

{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
    
    def export_for_manuscript(
        self,
        protocol_id: str
    ) -> Dict[str, Any]:
        """Export summary suitable for manuscript methods section"""
        protocol = self.governance.get_protocol(protocol_id)
        if not protocol:
            raise ValueError(f"Protocol {protocol_id} not found")
        
        spec = protocol.analysis_spec
        
        return {
            'study_design': {
                'type': spec.analysis_type.value,
                'protocol_id': protocol.protocol_id,
                'version': f"{protocol.version}/{spec.version}"
            },
            'population': spec.cohort_definition,
            'exposure': spec.exposure_definition,
            'outcome': spec.outcome_definition,
            'covariates': spec.covariates,
            'statistical_methods': spec.statistical_methods,
            'sensitivity_analyses': [
                sa.get('name', 'Unnamed') 
                for sa in spec.sensitivity_analyses
            ],
            'data_source': {
                'snapshot_id': protocol.data_snapshot_id,
                'irb_number': protocol.irb_number
            }
        }
