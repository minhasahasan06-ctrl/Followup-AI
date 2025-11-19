"""
Edema Baseline Management Service
==================================

Manages patient baselines for edema detection and tracks historical comparisons.
HIPAA-compliant service for storing and retrieving baseline segmentation masks.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import numpy as np

from app.models.video_ai_models import EdemaSegmentationMetrics

logger = logging.getLogger(__name__)


class EdemaBaselineService:
    """
    Service for managing edema detection baselines
    
    Features:
    - Automatic baseline selection (first healthy measurement)
    - Manual baseline override by clinician
    - Historical baseline tracking
    - Baseline drift detection
    - Multi-baseline comparison (seasonal, condition-specific)
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_latest_baseline(
        self,
        patient_id: str,
        condition: Optional[str] = None
    ) -> Optional[EdemaSegmentationMetrics]:
        """
        Get patient's most recent baseline measurement
        
        Args:
            patient_id: Patient identifier
            condition: Optional condition filter (e.g., 'heart_failure')
        
        Returns:
            Latest baseline EdemaSegmentationMetrics record
        """
        query = self.db.query(EdemaSegmentationMetrics).filter(
            and_(
                EdemaSegmentationMetrics.patient_id == patient_id,
                EdemaSegmentationMetrics.has_baseline == False,  # Is a baseline record
                EdemaSegmentationMetrics.swelling_detected == False  # No swelling
            )
        )
        
        if condition:
            # Filter by condition if specified
            query = query.filter(
                EdemaSegmentationMetrics.patient_conditions.contains([condition])
            )
        
        baseline = query.order_by(
            desc(EdemaSegmentationMetrics.analyzed_at)
        ).first()
        
        if baseline:
            logger.info(f"Found baseline for patient {patient_id}: {baseline.id}")
        else:
            logger.warning(f"No baseline found for patient {patient_id}")
        
        return baseline
    
    def create_baseline_from_analysis(
        self,
        patient_id: str,
        analysis_metrics: EdemaSegmentationMetrics,
        s3_client,
        segmentation_mask: np.ndarray
    ) -> EdemaSegmentationMetrics:
        """
        Create a new baseline from current analysis
        
        Args:
            patient_id: Patient identifier
            analysis_metrics: Current analysis metrics
            s3_client: Boto3 S3 client
            segmentation_mask: Segmentation mask to save as baseline
        
        Returns:
            New baseline record
        """
        from app.services.edema_segmentation_service import EdemaSegmentationService
        
        edema_service = EdemaSegmentationService()
        
        # Save mask to S3
        mask_s3_uri = edema_service.save_baseline_mask(
            patient_id=patient_id,
            mask=segmentation_mask,
            s3_client=s3_client
        )
        
        # Create baseline record
        baseline = EdemaSegmentationMetrics(
            patient_id=patient_id,
            analyzed_at=datetime.utcnow(),
            # Copy regional measurements from analysis
            total_body_area_px=analysis_metrics.total_body_area_px,
            face_upper_body_area_px=analysis_metrics.face_upper_body_area_px,
            torso_hands_area_px=analysis_metrics.torso_hands_area_px,
            legs_feet_area_px=analysis_metrics.legs_feet_area_px,
            left_lower_limb_area_px=analysis_metrics.left_lower_limb_area_px,
            right_lower_limb_area_px=analysis_metrics.right_lower_limb_area_px,
            lower_leg_left_area_px=analysis_metrics.lower_leg_left_area_px,
            lower_leg_right_area_px=analysis_metrics.lower_leg_right_area_px,
            periorbital_area_px=analysis_metrics.periorbital_area_px,
            # Baseline metadata
            has_baseline=False,  # This IS the baseline
            swelling_detected=False,
            swelling_severity='none',
            segmentation_mask_s3_uri=mask_s3_uri,
            # Model info
            model_type=analysis_metrics.model_type,
            model_version=analysis_metrics.model_version,
            segmentation_confidence=analysis_metrics.segmentation_confidence,
            processing_time_ms=analysis_metrics.processing_time_ms,
            patient_conditions=analysis_metrics.patient_conditions
        )
        
        self.db.add(baseline)
        self.db.commit()
        self.db.refresh(baseline)
        
        logger.info(f"✅ Created baseline for patient {patient_id}: {baseline.id}")
        
        return baseline
    
    def compare_to_baseline(
        self,
        current_metrics: Dict[str, Any],
        baseline_id: int
    ) -> Dict[str, Any]:
        """
        Compare current analysis to baseline
        
        Args:
            current_metrics: Current edema metrics dict
            baseline_id: Database ID of baseline record
        
        Returns:
            Comparison metrics with expansion percentages
        """
        baseline = self.db.query(EdemaSegmentationMetrics).filter(
            EdemaSegmentationMetrics.id == baseline_id
        ).first()
        
        if not baseline:
            logger.error(f"Baseline {baseline_id} not found")
            return {}
        
        comparison = {
            "baseline_id": baseline_id,
            "baseline_date": baseline.analyzed_at.isoformat(),
            "has_baseline": True,
            "regional_comparisons": {}
        }
        
        # Compare each region
        regions = [
            ("total_body", "total_body_area_px"),
            ("face_upper_body", "face_upper_body_area_px"),
            ("torso_hands", "torso_hands_area_px"),
            ("legs_feet", "legs_feet_area_px"),
            ("left_lower_limb", "left_lower_limb_area_px"),
            ("right_lower_limb", "right_lower_limb_area_px"),
            ("lower_leg_left", "lower_leg_left_area_px"),
            ("lower_leg_right", "lower_leg_right_area_px"),
            ("periorbital", "periorbital_area_px")
        ]
        
        for region_name, field_name in regions:
            current_area = current_metrics.get(field_name)
            baseline_area = getattr(baseline, field_name, None)
            
            if current_area and baseline_area and baseline_area > 0:
                expansion_pct = ((current_area - baseline_area) / baseline_area) * 100
                
                comparison["regional_comparisons"][region_name] = {
                    "current_area_px": current_area,
                    "baseline_area_px": baseline_area,
                    "expansion_percent": float(expansion_pct),
                    "swelling_detected": expansion_pct > 5.0
                }
        
        # Calculate asymmetry
        if ("left_lower_limb" in comparison["regional_comparisons"] and
            "right_lower_limb" in comparison["regional_comparisons"]):
            
            left_exp = comparison["regional_comparisons"]["left_lower_limb"]["expansion_percent"]
            right_exp = comparison["regional_comparisons"]["right_lower_limb"]["expansion_percent"]
            
            asymmetry_diff = abs(left_exp - right_exp)
            comparison["asymmetry_detected"] = asymmetry_diff > 3.0
            comparison["asymmetry_difference_percent"] = float(asymmetry_diff)
        
        return comparison
    
    def get_baseline_history(
        self,
        patient_id: str,
        days_back: int = 90
    ) -> List[EdemaSegmentationMetrics]:
        """
        Get historical baselines for patient
        
        Args:
            patient_id: Patient identifier
            days_back: How many days of history to fetch
        
        Returns:
            List of baseline records
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        baselines = self.db.query(EdemaSegmentationMetrics).filter(
            and_(
                EdemaSegmentationMetrics.patient_id == patient_id,
                EdemaSegmentationMetrics.has_baseline == False,
                EdemaSegmentationMetrics.analyzed_at >= cutoff_date
            )
        ).order_by(
            desc(EdemaSegmentationMetrics.analyzed_at)
        ).all()
        
        logger.info(f"Found {len(baselines)} baseline records for patient {patient_id}")
        
        return baselines
    
    def detect_baseline_drift(
        self,
        patient_id: str,
        threshold_percent: float = 10.0
    ) -> Dict[str, Any]:
        """
        Detect if patient's baseline has drifted over time
        
        Useful for detecting gradual condition changes that require
        baseline recalibration.
        
        Args:
            patient_id: Patient identifier
            threshold_percent: Drift threshold percentage
        
        Returns:
            Drift analysis results
        """
        baselines = self.get_baseline_history(patient_id, days_back=180)
        
        if len(baselines) < 2:
            return {
                "drift_detected": False,
                "reason": "Insufficient baseline history"
            }
        
        # Compare oldest vs newest
        oldest = baselines[-1]
        newest = baselines[0]
        
        drift_detected = False
        regional_drift = {}
        
        regions = [
            ("total_body", "total_body_area_px"),
            ("legs_feet", "legs_feet_area_px"),
            ("left_lower_limb", "left_lower_limb_area_px"),
            ("right_lower_limb", "right_lower_limb_area_px")
        ]
        
        for region_name, field_name in regions:
            old_area = getattr(oldest, field_name, None)
            new_area = getattr(newest, field_name, None)
            
            if old_area and new_area and old_area > 0:
                drift_pct = ((new_area - old_area) / old_area) * 100
                
                if abs(drift_pct) > threshold_percent:
                    drift_detected = True
                
                regional_drift[region_name] = {
                    "drift_percent": float(drift_pct),
                    "exceeds_threshold": abs(drift_pct) > threshold_percent
                }
        
        return {
            "drift_detected": drift_detected,
            "oldest_baseline_date": oldest.analyzed_at.isoformat(),
            "newest_baseline_date": newest.analyzed_at.isoformat(),
            "regional_drift": regional_drift,
            "recommendation": "Recalibrate baseline" if drift_detected else "Baseline stable"
        }
