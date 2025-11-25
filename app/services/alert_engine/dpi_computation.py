"""
Composite Deterioration Index (DPI) Computation Service.

DPI Formula:
DPI_raw = Σ (W_j * O_j) where O_j are normalized organ scores and W_j are organ weights
DPI = (DPI_raw - DPI_min) / (DPI_max - DPI_min) * 100

Color Buckets:
- Green: DPI < 25
- Yellow: 25 ≤ DPI < 50
- Orange: 50 ≤ DPI < 75
- Red: DPI ≥ 75

Includes:
- DPI computation with explainability (which organ contributed how much)
- Trend tracking (previous DPI, bucket changes)
- Jump detection (>X points in 24h)
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

from .config_service import AlertConfigService
from .organ_scoring import OrganScoringResult, OrganScore


@dataclass
class DPIComponent:
    """Individual organ contribution to DPI"""
    organ_name: str
    organ_score: float
    weight: float
    contribution: float  # weighted contribution to raw DPI
    percentage: float  # percentage of total DPI


@dataclass
class DPIResult:
    """Complete DPI computation result"""
    patient_id: str
    dpi_raw: float
    dpi_normalized: float  # 0-100 scale
    dpi_bucket: str  # green, yellow, orange, red
    components: List[DPIComponent]
    previous_dpi: Optional[float]
    previous_bucket: Optional[str]
    bucket_changed: bool
    dpi_delta_24h: Optional[float]
    jump_detected: bool
    computed_at: datetime


class DPIComputationService:
    """Service for computing Composite Deterioration Index"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
    
    def compute_dpi(
        self,
        patient_id: str,
        organ_result: OrganScoringResult
    ) -> DPIResult:
        """
        Compute DPI from organ scores.
        
        DPI_raw = Σ (W_j * O_j) normalized by total weights
        DPI_normalized = mapped to 0-100 scale
        """
        config = self.config_service.config
        
        # Calculate weighted sum of organ scores
        weighted_sum = 0.0
        total_weight = 0.0
        components = []
        
        for group_name, organ_score in organ_result.organ_scores.items():
            weight = config.get_organ_weight(group_name)
            contribution = organ_score.normalized_score * weight
            weighted_sum += contribution
            total_weight += weight
            
            components.append(DPIComponent(
                organ_name=organ_score.organ_name,
                organ_score=organ_score.normalized_score,
                weight=weight,
                contribution=round(contribution, 2),
                percentage=0  # Will be calculated after total
            ))
        
        # Compute raw DPI
        if total_weight > 0:
            dpi_raw = weighted_sum / total_weight
        else:
            dpi_raw = 50.0  # Default neutral
        
        # Normalize to 0-100 scale
        dpi_normalized = round(min(max(dpi_raw, 0), 100), 2)
        
        # Calculate contribution percentages
        for component in components:
            if dpi_raw > 0:
                component.percentage = round((component.contribution / weighted_sum) * 100, 1)
            else:
                component.percentage = 100.0 / len(components) if components else 0
        
        # Sort components by contribution
        components.sort(key=lambda x: x.contribution, reverse=True)
        
        # Determine bucket
        dpi_bucket = config.get_dpi_bucket(dpi_normalized)
        
        # Get previous DPI for comparison
        previous_dpi, previous_bucket = self._get_previous_dpi(patient_id)
        
        # Check for bucket change
        bucket_changed = previous_bucket is not None and previous_bucket != dpi_bucket
        
        # Get 24h delta
        dpi_24h_ago = self._get_dpi_at_time(patient_id, hours_ago=24)
        dpi_delta_24h = None
        jump_detected = False
        
        if dpi_24h_ago is not None:
            dpi_delta_24h = round(dpi_normalized - dpi_24h_ago, 2)
            jump_detected = abs(dpi_delta_24h) >= config.dpi_jump_threshold_24h
        
        return DPIResult(
            patient_id=patient_id,
            dpi_raw=round(dpi_raw, 4),
            dpi_normalized=dpi_normalized,
            dpi_bucket=dpi_bucket,
            components=components,
            previous_dpi=previous_dpi,
            previous_bucket=previous_bucket,
            bucket_changed=bucket_changed,
            dpi_delta_24h=dpi_delta_24h,
            jump_detected=jump_detected,
            computed_at=datetime.utcnow()
        )
    
    def _get_previous_dpi(self, patient_id: str) -> Tuple[Optional[float], Optional[str]]:
        """Get most recent previous DPI and bucket"""
        query = text("""
            SELECT dpi_normalized, dpi_bucket
            FROM dpi_history
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC
            LIMIT 1
        """)
        
        try:
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            if result:
                return float(result[0]) if result[0] else None, result[1]
            return None, None
        except Exception as e:
            logger.warning(f"Error getting previous DPI: {e}")
            return None, None
    
    def _get_dpi_at_time(self, patient_id: str, hours_ago: int) -> Optional[float]:
        """Get DPI from approximately X hours ago"""
        target_time = datetime.utcnow() - timedelta(hours=hours_ago)
        window_start = target_time - timedelta(hours=2)
        window_end = target_time + timedelta(hours=2)
        
        query = text("""
            SELECT dpi_normalized
            FROM dpi_history
            WHERE patient_id = :patient_id
            AND computed_at BETWEEN :window_start AND :window_end
            ORDER BY computed_at DESC
            LIMIT 1
        """)
        
        try:
            result = self.db.execute(query, {
                "patient_id": patient_id,
                "window_start": window_start,
                "window_end": window_end
            }).fetchone()
            
            if result:
                return float(result[0]) if result[0] else None
            return None
        except Exception as e:
            logger.warning(f"Error getting historical DPI: {e}")
            return None
    
    async def store_dpi(self, result: DPIResult) -> bool:
        """Persist DPI result to database"""
        try:
            insert_query = text("""
                INSERT INTO dpi_history (
                    id, patient_id, dpi_raw, dpi_normalized, dpi_bucket,
                    components, previous_dpi, previous_bucket,
                    bucket_changed, dpi_delta_24h, jump_detected, computed_at
                ) VALUES (
                    gen_random_uuid(), :patient_id, :dpi_raw, :dpi_normalized, :dpi_bucket,
                    :components::jsonb, :previous_dpi, :previous_bucket,
                    :bucket_changed, :dpi_delta_24h, :jump_detected, NOW()
                )
            """)
            
            components_json = [
                {
                    "organ_name": c.organ_name,
                    "organ_score": c.organ_score,
                    "weight": c.weight,
                    "contribution": c.contribution,
                    "percentage": c.percentage
                }
                for c in result.components
            ]
            
            self.db.execute(insert_query, {
                "patient_id": result.patient_id,
                "dpi_raw": result.dpi_raw,
                "dpi_normalized": result.dpi_normalized,
                "dpi_bucket": result.dpi_bucket,
                "components": json.dumps(components_json),
                "previous_dpi": result.previous_dpi,
                "previous_bucket": result.previous_bucket,
                "bucket_changed": result.bucket_changed,
                "dpi_delta_24h": result.dpi_delta_24h,
                "jump_detected": result.jump_detected
            })
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing DPI: {e}")
            self.db.rollback()
            return False
    
    async def get_dpi_history(
        self,
        patient_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get DPI history for a patient"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT id, dpi_raw, dpi_normalized, dpi_bucket, components,
                   previous_dpi, previous_bucket, bucket_changed,
                   dpi_delta_24h, jump_detected, computed_at
            FROM dpi_history
            WHERE patient_id = :patient_id
            AND computed_at >= :cutoff
            ORDER BY computed_at DESC
        """)
        
        try:
            results = self.db.execute(query, {
                "patient_id": patient_id,
                "cutoff": cutoff
            }).fetchall()
            
            return [
                {
                    "id": str(row[0]),
                    "dpi_raw": float(row[1]) if row[1] else 0,
                    "dpi_normalized": float(row[2]) if row[2] else 0,
                    "dpi_bucket": row[3] or "green",
                    "components": row[4],
                    "previous_dpi": float(row[5]) if row[5] else None,
                    "previous_bucket": row[6],
                    "bucket_changed": row[7] or False,
                    "dpi_delta_24h": float(row[8]) if row[8] else None,
                    "jump_detected": row[9] or False,
                    "computed_at": row[10].isoformat() if row[10] else None
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error fetching DPI history: {e}")
            return []
    
    async def get_current_dpi(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get current DPI for a patient"""
        query = text("""
            SELECT id, dpi_raw, dpi_normalized, dpi_bucket, components,
                   previous_dpi, previous_bucket, bucket_changed,
                   dpi_delta_24h, jump_detected, computed_at
            FROM dpi_history
            WHERE patient_id = :patient_id
            ORDER BY computed_at DESC
            LIMIT 1
        """)
        
        try:
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            
            if result:
                return {
                    "id": str(result[0]),
                    "dpi_raw": float(result[1]) if result[1] else 0,
                    "dpi_normalized": float(result[2]) if result[2] else 0,
                    "dpi_bucket": result[3] or "green",
                    "components": result[4],
                    "previous_dpi": float(result[5]) if result[5] else None,
                    "previous_bucket": result[6],
                    "bucket_changed": result[7] or False,
                    "dpi_delta_24h": float(result[8]) if result[8] else None,
                    "jump_detected": result[9] or False,
                    "computed_at": result[10].isoformat() if result[10] else None
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching current DPI: {e}")
            return None
    
    def get_bucket_color(self, bucket: str) -> str:
        """Get hex color for DPI bucket"""
        colors = {
            "green": "#22c55e",
            "yellow": "#eab308",
            "orange": "#f97316",
            "red": "#ef4444"
        }
        return colors.get(bucket, "#6b7280")
    
    def get_bucket_description(self, bucket: str) -> str:
        """Get human-readable description for DPI bucket"""
        descriptions = {
            "green": "Stable - Metrics within normal patterns",
            "yellow": "Elevated - Some metrics showing deviation from baseline",
            "orange": "Concerning - Multiple elevated indicators",
            "red": "Critical - Significant deterioration patterns detected"
        }
        return descriptions.get(bucket, "Unknown status")
