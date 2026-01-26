"""
Model Version Manager
======================
Production-grade model versioning with:
- Semantic versioning support
- Version comparison
- Deployment tracking
- Rollback support

HIPAA-compliant with audit logging.
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    """Model deployment status"""
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelVersion:
    """Represents a specific model version"""
    model_id: str
    model_name: str
    version: str
    major: int
    minor: int
    patch: int
    artifact_id: str
    deployment_status: DeploymentStatus
    is_active: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    deployed_at: Optional[datetime] = None
    created_by: str = "system"
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "artifact_id": self.artifact_id,
            "deployment_status": self.deployment_status.value,
            "is_active": self.is_active,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
            "created_by": self.created_by,
            "notes": self.notes
        }
    
    def __lt__(self, other: 'ModelVersion') -> bool:
        """Compare versions for sorting"""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModelVersion):
            return False
        return self.version == other.version and self.model_name == other.model_name


class ModelVersionManager:
    """
    Manages model versions with semantic versioning.
    
    Features:
    - Semantic version parsing and comparison
    - Active model tracking per deployment environment
    - Version history and rollback
    - Metrics comparison between versions
    """
    
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get('DATABASE_URL')
        self._ensure_tables()
    
    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def _ensure_tables(self):
        """Ensure database tables exist"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ml_model_versions (
                    model_id VARCHAR(50) PRIMARY KEY,
                    model_name VARCHAR(100) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    major_version INTEGER NOT NULL,
                    minor_version INTEGER NOT NULL,
                    patch_version INTEGER NOT NULL,
                    artifact_id VARCHAR(50),
                    deployment_status VARCHAR(20) DEFAULT 'staging',
                    is_active BOOLEAN DEFAULT FALSE,
                    metrics JSONB DEFAULT '{}',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    deployed_at TIMESTAMPTZ,
                    created_by VARCHAR(100) DEFAULT 'system',
                    notes TEXT DEFAULT '',
                    UNIQUE(model_name, version)
                );
                
                CREATE INDEX IF NOT EXISTS idx_model_versions_name 
                    ON ml_model_versions(model_name);
                CREATE INDEX IF NOT EXISTS idx_model_versions_active 
                    ON ml_model_versions(model_name, is_active) WHERE is_active = TRUE;
                CREATE INDEX IF NOT EXISTS idx_model_versions_status 
                    ON ml_model_versions(deployment_status);
                    
                -- Deployment history table
                CREATE TABLE IF NOT EXISTS ml_model_deployment_history (
                    id SERIAL PRIMARY KEY,
                    model_id VARCHAR(50) NOT NULL,
                    model_name VARCHAR(100) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    previous_status VARCHAR(20),
                    new_status VARCHAR(20),
                    deployed_by VARCHAR(100),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    notes TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_deployment_history_model 
                    ON ml_model_deployment_history(model_name, created_at DESC);
            """)
            
            conn.commit()
            cur.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error ensuring version tables: {e}")
    
    @staticmethod
    def parse_version(version: str) -> Tuple[int, int, int]:
        """Parse semantic version string"""
        parts = version.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version}")
        return int(parts[0]), int(parts[1]), int(parts[2])
    
    @staticmethod
    def compare_versions(v1: str, v2: str) -> int:
        """Compare two versions. Returns -1 if v1 < v2, 0 if equal, 1 if v1 > v2"""
        p1 = ModelVersionManager.parse_version(v1)
        p2 = ModelVersionManager.parse_version(v2)
        
        if p1 < p2:
            return -1
        elif p1 > p2:
            return 1
        return 0
    
    def register_version(
        self,
        model_name: str,
        version: str,
        artifact_id: str,
        metrics: Dict[str, Any],
        created_by: str = "system",
        notes: str = ""
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Args:
            model_name: Name of the model
            version: Semantic version string (e.g., "1.2.3")
            artifact_id: ID of the model artifact
            metrics: Training/evaluation metrics
            created_by: Creator ID
            notes: Optional notes
            
        Returns:
            Created ModelVersion
        """
        import uuid
        
        major, minor, patch = self.parse_version(version)
        model_id = str(uuid.uuid4())
        
        model_version = ModelVersion(
            model_id=model_id,
            model_name=model_name,
            version=version,
            major=major,
            minor=minor,
            patch=patch,
            artifact_id=artifact_id,
            deployment_status=DeploymentStatus.STAGING,
            is_active=False,
            metrics=metrics,
            created_by=created_by,
            notes=notes
        )
        
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                INSERT INTO ml_model_versions (
                    model_id, model_name, version, major_version, minor_version,
                    patch_version, artifact_id, deployment_status, is_active,
                    metrics, created_by, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                model_id, model_name, version, major, minor, patch,
                artifact_id, DeploymentStatus.STAGING.value, False,
                json.dumps(metrics), created_by, notes
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Registered model version {model_name} v{version}")
            
        except Exception as e:
            logger.error(f"Error registering version: {e}")
            raise
        
        return model_version
    
    def promote_to_production(
        self,
        model_name: str,
        version: str,
        promoted_by: str = "system"
    ) -> bool:
        """
        Promote a model version to production.
        Deactivates any currently active version.
        
        Args:
            model_name: Model name
            version: Version to promote
            promoted_by: User performing promotion
            
        Returns:
            True if successful
        """
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Deactivate current active version
            cur.execute("""
                UPDATE ml_model_versions
                SET is_active = FALSE, deployment_status = 'deprecated'
                WHERE model_name = %s AND is_active = TRUE
            """, (model_name,))
            
            # Activate new version
            cur.execute("""
                UPDATE ml_model_versions
                SET is_active = TRUE, 
                    deployment_status = 'production',
                    deployed_at = NOW()
                WHERE model_name = %s AND version = %s
                RETURNING model_id
            """, (model_name, version))
            
            row = cur.fetchone()
            if not row:
                logger.error(f"Version {version} not found for model {model_name}")
                conn.rollback()
                cur.close()
                conn.close()
                return False
            
            model_id = row[0]
            
            # Log deployment
            cur.execute("""
                INSERT INTO ml_model_deployment_history 
                (model_id, model_name, version, action, new_status, deployed_by)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (model_id, model_name, version, 'promote_to_production', 'production', promoted_by))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Promoted {model_name} v{version} to production by {promoted_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error promoting version: {e}")
            return False
    
    def rollback_to_version(
        self,
        model_name: str,
        target_version: str,
        rolled_back_by: str = "system"
    ) -> bool:
        """
        Rollback to a previous version.
        
        Args:
            model_name: Model name
            target_version: Version to rollback to
            rolled_back_by: User performing rollback
            
        Returns:
            True if successful
        """
        try:
            # Get current active version for logging
            current = self.get_active_version(model_name)
            
            conn = self._get_connection()
            cur = conn.cursor()
            
            # Deactivate current version
            if current:
                cur.execute("""
                    UPDATE ml_model_versions
                    SET is_active = FALSE
                    WHERE model_name = %s AND is_active = TRUE
                """, (model_name,))
            
            # Activate target version
            cur.execute("""
                UPDATE ml_model_versions
                SET is_active = TRUE, 
                    deployment_status = 'production',
                    deployed_at = NOW()
                WHERE model_name = %s AND version = %s
                RETURNING model_id
            """, (model_name, target_version))
            
            row = cur.fetchone()
            if not row:
                logger.error(f"Target version {target_version} not found")
                conn.rollback()
                cur.close()
                conn.close()
                return False
            
            model_id = row[0]
            
            # Log rollback
            cur.execute("""
                INSERT INTO ml_model_deployment_history 
                (model_id, model_name, version, action, previous_status, new_status, deployed_by, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                model_id, model_name, target_version, 'rollback',
                current.version if current else None, 'production',
                rolled_back_by, f"Rolled back from {current.version if current else 'none'}"
            ))
            
            conn.commit()
            cur.close()
            conn.close()
            
            logger.info(f"Rolled back {model_name} to v{target_version} by {rolled_back_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back version: {e}")
            return False
    
    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a model"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM ml_model_versions
                WHERE model_name = %s AND is_active = TRUE
                LIMIT 1
            """, (model_name,))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            if row:
                return self._row_to_version(dict(row))
            return None
            
        except Exception as e:
            logger.error(f"Error getting active version: {e}")
            return None
    
    def get_all_model_names(self) -> List[str]:
        """Get list of all distinct model names in the registry"""
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT DISTINCT model_name FROM ml_model_versions
                ORDER BY model_name
            """)
            
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            return [row[0] for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting model names: {e}")
            return []
    
    def get_version_history(
        self,
        model_name: str,
        limit: int = 20
    ) -> List[ModelVersion]:
        """Get version history for a model"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM ml_model_versions
                WHERE model_name = %s
                ORDER BY major_version DESC, minor_version DESC, patch_version DESC
                LIMIT %s
            """, (model_name, limit))
            
            rows = cur.fetchall()
            cur.close()
            conn.close()
            
            return [self._row_to_version(dict(row)) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting version history: {e}")
            return []
    
    def compare_versions_metrics(
        self,
        model_name: str,
        version1: str,
        version2: str
    ) -> Dict[str, Any]:
        """Compare metrics between two versions"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT version, metrics FROM ml_model_versions
                WHERE model_name = %s AND version IN (%s, %s)
            """, (model_name, version1, version2))
            
            rows = {row['version']: row['metrics'] for row in cur.fetchall()}
            cur.close()
            conn.close()
            
            if version1 not in rows or version2 not in rows:
                return {"error": "One or both versions not found"}
            
            metrics1 = rows[version1] or {}
            metrics2 = rows[version2] or {}
            
            comparison = {}
            all_keys = set(metrics1.keys()) | set(metrics2.keys())
            
            for key in all_keys:
                v1 = metrics1.get(key)
                v2 = metrics2.get(key)
                
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    diff = v2 - v1
                    pct_change = (diff / v1 * 100) if v1 != 0 else 0
                    comparison[key] = {
                        version1: v1,
                        version2: v2,
                        "difference": round(diff, 4),
                        "percent_change": round(pct_change, 2)
                    }
                else:
                    comparison[key] = {
                        version1: v1,
                        version2: v2
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}
    
    def _row_to_version(self, row: Dict[str, Any]) -> ModelVersion:
        """Convert database row to ModelVersion"""
        return ModelVersion(
            model_id=row['model_id'],
            model_name=row['model_name'],
            version=row['version'],
            major=row['major_version'],
            minor=row['minor_version'],
            patch=row['patch_version'],
            artifact_id=row['artifact_id'] or '',
            deployment_status=DeploymentStatus(row['deployment_status']) if row['deployment_status'] else DeploymentStatus.STAGING,
            is_active=row['is_active'],
            metrics=row['metrics'] or {},
            created_at=row['created_at'],
            deployed_at=row['deployed_at'],
            created_by=row['created_by'] or 'system',
            notes=row['notes'] or ''
        )
    
    def get_deployment_history(
        self,
        model_name: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get deployment history for a model"""
        try:
            conn = self._get_connection()
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT * FROM ml_model_deployment_history
                WHERE model_name = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (model_name, limit))
            
            rows = [dict(row) for row in cur.fetchall()]
            cur.close()
            conn.close()
            
            # Convert datetime for JSON serialization
            for row in rows:
                if row.get('created_at'):
                    row['created_at'] = row['created_at'].isoformat()
            
            return rows
            
        except Exception as e:
            logger.error(f"Error getting deployment history: {e}")
            return []
