"""
Organ-Level Scoring Service - 5-organ system scoring with weighted z-sums.

Organ Groups:
1. Respiratory - RR, thoracoabdominal async, cough, wheeze, audio RR
2. Cardio/Fluid - Edema, weight delta, perfusion, facial puffiness
3. Hepatic/Hematologic - Sclera jaundice, palmar pallor, conjunctiva pallor
4. Mobility - Gait speed, step count, gait variability, tremor
5. Cognitive/Behavioral - Mood, reaction time, checkin compliance, anxiety
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

from .config_service import AlertConfigService, OrganGroupConfig


@dataclass
class MetricZScore:
    """Individual metric z-score with metadata"""
    metric_name: str
    raw_value: float
    z_score: float
    confidence: float
    organ_group: str
    timestamp: datetime


@dataclass
class OrganScore:
    """Score for a single organ system"""
    organ_name: str
    raw_score: float
    normalized_score: float  # 0-100 scale
    contributing_metrics: List[MetricZScore]
    num_metrics: int
    severity: str  # normal, yellow, red
    timestamp: datetime


@dataclass
class OrganScoringResult:
    """Complete organ scoring result"""
    patient_id: str
    organ_scores: Dict[str, OrganScore]
    total_metrics: int
    computed_at: datetime


class OrganScoringService:
    """Service for computing organ-level health scores"""
    
    def __init__(self, db: Session):
        self.db = db
        self.config_service = AlertConfigService()
    
    def compute_organ_scores(
        self,
        patient_id: str,
        metric_z_scores: List[MetricZScore]
    ) -> OrganScoringResult:
        """
        Compute organ-level scores from individual metric z-scores.
        
        Formula per organ:
        organ_score = Σ (w_i * z_i) where z_i are metric z-scores in that organ group
        normalized = organ_score / sqrt(N) mapped to 0-100 scale
        """
        config = self.config_service.config
        
        # Group metrics by organ system
        organ_metrics: Dict[str, List[MetricZScore]] = defaultdict(list)
        
        for metric in metric_z_scores:
            organ_group = config.get_metric_organ_group(metric.metric_name)
            if organ_group:
                organ_metrics[organ_group].append(metric)
            else:
                # Unclassified metrics go to cognitive/behavioral by default
                organ_metrics["cognitive_behavioral"].append(metric)
        
        # Compute scores for each organ group
        organ_scores: Dict[str, OrganScore] = {}
        
        for group_name, group_config in config.organ_groups.items():
            metrics = organ_metrics.get(group_name, [])
            
            if not metrics:
                # No data for this organ - create placeholder
                organ_scores[group_name] = OrganScore(
                    organ_name=group_config.name,
                    raw_score=0.0,
                    normalized_score=50.0,  # Neutral score
                    contributing_metrics=[],
                    num_metrics=0,
                    severity="normal",
                    timestamp=datetime.utcnow()
                )
                continue
            
            # Compute weighted z-sum
            total_weight = 0.0
            weighted_sum = 0.0
            
            for metric in metrics:
                weight = 1.0  # Could be metric-specific weight
                weighted_sum += weight * abs(metric.z_score)
                total_weight += weight
            
            if total_weight > 0:
                raw_score = weighted_sum / np.sqrt(len(metrics))
            else:
                raw_score = 0.0
            
            # Normalize to 0-100 scale
            # Map z-sum to percentage (z=0 -> 0%, z=3 -> 75%, z=4+ -> 100%)
            normalized_score = min(raw_score / 4.0 * 100, 100.0)
            
            # Determine severity
            if normalized_score >= group_config.red_threshold:
                severity = "red"
            elif normalized_score >= group_config.yellow_threshold:
                severity = "yellow"
            else:
                severity = "normal"
            
            organ_scores[group_name] = OrganScore(
                organ_name=group_config.name,
                raw_score=round(raw_score, 4),
                normalized_score=round(normalized_score, 2),
                contributing_metrics=metrics,
                num_metrics=len(metrics),
                severity=severity,
                timestamp=datetime.utcnow()
            )
        
        return OrganScoringResult(
            patient_id=patient_id,
            organ_scores=organ_scores,
            total_metrics=len(metric_z_scores),
            computed_at=datetime.utcnow()
        )
    
    async def compute_from_recent_data(
        self,
        patient_id: str,
        hours: int = 24
    ) -> OrganScoringResult:
        """
        Compute organ scores from recent metric data in the database.
        Queries trend metrics and ingested metrics to build z-scores.
        """
        config = self.config_service.config
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        metric_z_scores = []
        
        # Get recent trend metrics with z-scores
        trend_query = text("""
            SELECT DISTINCT ON (metric_name)
                metric_name, raw_value, z_score, metric_category, recorded_at
            FROM ai_trend_metrics
            WHERE patient_id = :patient_id
            AND recorded_at >= :cutoff
            AND z_score IS NOT NULL
            ORDER BY metric_name, recorded_at DESC
        """)
        
        trend_results = self.db.execute(trend_query, {
            "patient_id": patient_id,
            "cutoff": cutoff
        }).fetchall()
        
        for row in trend_results:
            organ_group = config.get_metric_organ_group(row[0])
            metric_z_scores.append(MetricZScore(
                metric_name=row[0],
                raw_value=float(row[1]) if row[1] else 0,
                z_score=float(row[2]) if row[2] else 0,
                confidence=1.0,
                organ_group=organ_group or "unclassified",
                timestamp=row[4]
            ))
        
        # Get recent ingested metrics
        ingest_query = text("""
            SELECT DISTINCT ON (metric_name)
                metric_name, metric_value, confidence, organ_group, timestamp
            FROM metric_ingest_log
            WHERE patient_id = :patient_id
            AND timestamp >= :cutoff
            ORDER BY metric_name, timestamp DESC
        """)
        
        try:
            ingest_results = self.db.execute(ingest_query, {
                "patient_id": patient_id,
                "cutoff": cutoff
            }).fetchall()
            
            for row in ingest_results:
                # Skip if we already have this metric from trend_metrics
                if any(m.metric_name == row[0] for m in metric_z_scores):
                    continue
                
                # Compute basic z-score from baseline
                z_score = await self._compute_baseline_z_score(
                    patient_id, row[0], float(row[1]) if row[1] else 0
                )
                
                metric_z_scores.append(MetricZScore(
                    metric_name=row[0],
                    raw_value=float(row[1]) if row[1] else 0,
                    z_score=z_score,
                    confidence=float(row[2]) if row[2] else 1.0,
                    organ_group=row[3] or "unclassified",
                    timestamp=row[4]
                ))
        except Exception as e:
            logger.warning(f"Error fetching ingest metrics: {e}")
        
        return self.compute_organ_scores(patient_id, metric_z_scores)
    
    async def _compute_baseline_z_score(
        self,
        patient_id: str,
        metric_name: str,
        current_value: float
    ) -> float:
        """Compute z-score against 14-day baseline"""
        config = self.config_service.config
        cutoff = datetime.utcnow() - timedelta(days=config.baseline_window_days)
        
        # Get baseline values
        baseline_query = text("""
            SELECT metric_value
            FROM metric_ingest_log
            WHERE patient_id = :patient_id
            AND metric_name = :metric_name
            AND timestamp >= :cutoff
            ORDER BY timestamp ASC
        """)
        
        try:
            results = self.db.execute(baseline_query, {
                "patient_id": patient_id,
                "metric_name": metric_name,
                "cutoff": cutoff
            }).fetchall()
            
            if len(results) < config.min_baseline_days:
                return 0.0  # Insufficient data
            
            values = [float(row[0]) for row in results if row[0] is not None]
            if len(values) < 3:
                return 0.0
            
            mean = np.mean(values)
            std = np.std(values)
            
            if std < config.epsilon:
                return 0.0
            
            z_score = (current_value - mean) / std
            return round(z_score, 3)
            
        except Exception as e:
            logger.warning(f"Error computing baseline z-score: {e}")
            return 0.0
    
    async def store_organ_scores(
        self,
        result: OrganScoringResult
    ) -> bool:
        """Persist organ scores to database"""
        try:
            insert_query = text("""
                INSERT INTO organ_scores (
                    id, patient_id, respiratory_score, respiratory_severity,
                    cardio_fluid_score, cardio_fluid_severity,
                    hepatic_score, hepatic_severity,
                    mobility_score, mobility_severity,
                    cognitive_score, cognitive_severity,
                    total_metrics, score_components, computed_at
                ) VALUES (
                    gen_random_uuid(), :patient_id,
                    :resp_score, :resp_severity,
                    :cardio_score, :cardio_severity,
                    :hepatic_score, :hepatic_severity,
                    :mobility_score, :mobility_severity,
                    :cognitive_score, :cognitive_severity,
                    :total_metrics, :components::jsonb, NOW()
                )
            """)
            
            # Build components JSON
            components = {}
            for group_name, score in result.organ_scores.items():
                components[group_name] = {
                    "normalized_score": score.normalized_score,
                    "raw_score": score.raw_score,
                    "num_metrics": score.num_metrics,
                    "severity": score.severity,
                    "contributing_metrics": [
                        {
                            "name": m.metric_name,
                            "z_score": m.z_score,
                            "confidence": m.confidence
                        }
                        for m in score.contributing_metrics
                    ]
                }
            
            # Get scores with defaults
            resp = result.organ_scores.get("respiratory", OrganScore("Respiratory", 0, 50, [], 0, "normal", datetime.utcnow()))
            cardio = result.organ_scores.get("cardio_fluid", OrganScore("Cardio", 0, 50, [], 0, "normal", datetime.utcnow()))
            hepatic = result.organ_scores.get("hepatic_hematologic", OrganScore("Hepatic", 0, 50, [], 0, "normal", datetime.utcnow()))
            mobility = result.organ_scores.get("mobility", OrganScore("Mobility", 0, 50, [], 0, "normal", datetime.utcnow()))
            cognitive = result.organ_scores.get("cognitive_behavioral", OrganScore("Cognitive", 0, 50, [], 0, "normal", datetime.utcnow()))
            
            self.db.execute(insert_query, {
                "patient_id": result.patient_id,
                "resp_score": resp.normalized_score,
                "resp_severity": resp.severity,
                "cardio_score": cardio.normalized_score,
                "cardio_severity": cardio.severity,
                "hepatic_score": hepatic.normalized_score,
                "hepatic_severity": hepatic.severity,
                "mobility_score": mobility.normalized_score,
                "mobility_severity": mobility.severity,
                "cognitive_score": cognitive.normalized_score,
                "cognitive_severity": cognitive.severity,
                "total_metrics": result.total_metrics,
                "components": json.dumps(components)
            })
            self.db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing organ scores: {e}")
            self.db.rollback()
            return False
    
    async def get_organ_scores_history(
        self,
        patient_id: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical organ scores for a patient"""
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        query = text("""
            SELECT id, respiratory_score, respiratory_severity,
                   cardio_fluid_score, cardio_fluid_severity,
                   hepatic_score, hepatic_severity,
                   mobility_score, mobility_severity,
                   cognitive_score, cognitive_severity,
                   total_metrics, score_components, computed_at
            FROM organ_scores
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
                    "respiratory": {"score": float(row[1]) if row[1] else 50, "severity": row[2] or "normal"},
                    "cardio_fluid": {"score": float(row[3]) if row[3] else 50, "severity": row[4] or "normal"},
                    "hepatic": {"score": float(row[5]) if row[5] else 50, "severity": row[6] or "normal"},
                    "mobility": {"score": float(row[7]) if row[7] else 50, "severity": row[8] or "normal"},
                    "cognitive": {"score": float(row[9]) if row[9] else 50, "severity": row[10] or "normal"},
                    "total_metrics": row[11] or 0,
                    "components": row[12],
                    "computed_at": row[13].isoformat() if row[13] else None
                }
                for row in results
            ]
        except Exception as e:
            logger.error(f"Error fetching organ scores history: {e}")
            return []
