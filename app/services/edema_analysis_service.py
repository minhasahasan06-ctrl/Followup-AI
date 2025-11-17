"""
Edema Analysis Service
AI-powered video analysis of peripheral edema with pitting test grading
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc
import logging

from app.models import EdemaMetric, EdemaBaseline

logger = logging.getLogger(__name__)


class EdemaAnalysisService:
    """
    Analyzes video for peripheral edema (swelling) with pitting test support
    """
    
    # Pitting edema grading scale (clinical standard)
    PITTING_GRADES = {
        1: {
            'rebound_time': (0, 2),  # Immediate rebound (0-2 seconds)
            'pit_depth_mm': (0, 2),  # 2mm pit
            'severity': 'trace',
            'description': 'Immediate rebound with 2mm pit - minimal fluid retention'
        },
        2: {
            'rebound_time': (2, 15),  # Less than 15 seconds
            'pit_depth_mm': (3, 4),  # 3-4mm pit
            'severity': 'mild',
            'description': 'Rebound in 2-15 seconds with 3-4mm pit - mild fluid buildup'
        },
        3: {
            'rebound_time': (15, 60),  # 15-60 seconds
            'pit_depth_mm': (5, 6),  # 5-6mm pit
            'severity': 'moderate',
            'description': 'Rebound in 15-60 seconds with 5-6mm pit - moderate edema'
        },
        4: {
            'rebound_time': (60, 180),  # 2-3 minutes (120-180 sec)
            'pit_depth_mm': (8, float('inf')),  # 8mm+ pit
            'severity': 'severe',
            'description': 'Rebound in 2-3 minutes with 8mm+ pit - severe edema'
        }
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    def analyze_pitting_test(
        self,
        patient_id: str,
        session_id: str,
        location: str,
        side: str,
        video_frames: List[np.ndarray],
        press_start_frame: int,
        press_end_frame: int,
        fps: float
    ) -> EdemaMetric:
        """
        Analyze pitting edema test from video
        
        Args:
            patient_id: Patient identifier
            session_id: Video exam session ID
            location: Body location ('legs', 'ankles', 'feet', 'hands')
            side: 'left', 'right', or 'bilateral'
            video_frames: Video frames as numpy arrays
            press_start_frame: Frame when pressure applied
            press_end_frame: Frame when pressure released
            fps: Video frame rate
        
        Returns:
            EdemaMetric with pitting test results
        """
        # 1. Detect pit formation
        pit_depth_mm, pit_detected = self._measure_pit_depth(
            video_frames, press_end_frame, location
        )
        
        # 2. Measure rebound time
        rebound_time_sec = self._measure_rebound_time(
            video_frames, press_end_frame, fps
        )
        
        # 3. Grade pitting edema
        pitting_grade = self._grade_pitting_edema(rebound_time_sec, pit_depth_mm)
        
        # 4. Assess skin tightness
        skin_tightness = self._assess_skin_tightness(video_frames, location)
        
        # 5. Detect color changes
        color_change = self._detect_color_change(video_frames, location)
        
        # Create metric
        metric = EdemaMetric(
            patient_id=patient_id,
            session_id=session_id,
            location=location,
            side=side,
            pitting_detected=pit_detected,
            pitting_grade=pitting_grade if pit_detected else None,
            rebound_time_seconds=rebound_time_sec if pit_detected else None,
            pit_depth_mm=pit_depth_mm if pit_detected else None,
            skin_tightness_score=skin_tightness,
            color_change_detected=color_change,
            detection_confidence=0.8,  # Placeholder - would come from CV model
            analysis_method='pitting_test',
            metadata={
                'press_start_frame': press_start_frame,
                'press_end_frame': press_end_frame,
                'fps': fps,
                'grade_description': self.PITTING_GRADES.get(pitting_grade, {}).get('description', '')
            }
        )
        
        self.db.add(metric)
        self.db.commit()
        self.db.refresh(metric)
        
        logger.info(f"Pitting test analyzed: Grade {pitting_grade}, Rebound {rebound_time_sec:.1f}s, Depth {pit_depth_mm:.1f}mm")
        
        return metric
    
    def analyze_limb_volume(
        self,
        patient_id: str,
        session_id: str,
        location: str,
        video_frames: List[np.ndarray],
        fps: float
    ) -> Dict[str, EdemaMetric]:
        """
        Analyze bilateral limb volumes for edema comparison
        
        Returns:
            Dict with 'left' and 'right' EdemaMetric entries
        """
        # 1. Segment left and right limbs
        left_mask, right_mask = self._segment_bilateral_limbs(video_frames, location)
        
        # 2. Estimate volumes
        left_volume = self._estimate_limb_volume(left_mask, location)
        right_volume = self._estimate_limb_volume(right_mask, location)
        
        # 3. Get baselines
        left_baseline = self._get_or_create_baseline(patient_id, location, 'left')
        right_baseline = self._get_or_create_baseline(patient_id, location, 'right')
        
        # 4. Calculate peripheral edema index (% change from baseline)
        left_pei = self._calculate_peripheral_edema_index(
            left_volume, left_baseline.baseline_volume_ml if left_baseline else left_volume
        )
        right_pei = self._calculate_peripheral_edema_index(
            right_volume, right_baseline.baseline_volume_ml if right_baseline else right_volume
        )
        
        # 5. Calculate asymmetry
        asymmetry_ratio = abs(left_volume - right_volume) / max(left_volume, right_volume)
        bilateral_swelling = left_pei > 10 and right_pei > 10  # Both >10% increase
        
        # Create metrics for each side
        left_metric = EdemaMetric(
            patient_id=patient_id,
            session_id=session_id,
            location=location,
            side='left',
            peripheral_edema_index=left_pei,
            volume_ml_estimate=left_volume,
            baseline_volume_ml=left_baseline.baseline_volume_ml if left_baseline else None,
            bilateral_swelling=bilateral_swelling,
            left_volume_ml=left_volume,
            right_volume_ml=right_volume,
            asymmetry_ratio=asymmetry_ratio,
            detection_confidence=0.75,
            analysis_method='video_segmentation',
            metadata={'frames_analyzed': len(video_frames), 'fps': fps}
        )
        
        right_metric = EdemaMetric(
            patient_id=patient_id,
            session_id=session_id,
            location=location,
            side='right',
            peripheral_edema_index=right_pei,
            volume_ml_estimate=right_volume,
            baseline_volume_ml=right_baseline.baseline_volume_ml if right_baseline else None,
            bilateral_swelling=bilateral_swelling,
            left_volume_ml=left_volume,
            right_volume_ml=right_volume,
            asymmetry_ratio=asymmetry_ratio,
            detection_confidence=0.75,
            analysis_method='video_segmentation',
            metadata={'frames_analyzed': len(video_frames), 'fps': fps}
        )
        
        self.db.add(left_metric)
        self.db.add(right_metric)
        self.db.commit()
        
        # Update baselines
        self._update_baseline(patient_id, location, 'left', left_volume)
        self._update_baseline(patient_id, location, 'right', right_volume)
        
        logger.info(f"Limb volume analyzed: Left {left_volume:.1f}ml ({left_pei:.1f}% PEI), Right {right_volume:.1f}ml ({right_pei:.1f}% PEI), Asymmetry {asymmetry_ratio:.2f}")
        
        return {'left': left_metric, 'right': right_metric}
    
    def _measure_pit_depth(
        self,
        frames: List[np.ndarray],
        release_frame: int,
        location: str
    ) -> Tuple[float, bool]:
        """
        Measure pit depth from video after pressure release
        Uses depth estimation from frame analysis
        
        Returns:
            (pit_depth_mm, pit_detected)
        """
        # Simplified placeholder - in production would use:
        # - Depth map estimation
        # - Surface contour analysis
        # - Shadow/lighting analysis
        
        if release_frame >= len(frames):
            return 0.0, False
        
        release_frame_img = frames[release_frame]
        
        # Placeholder: Analyze surface depression
        # Would use computer vision to detect indentation
        # For now, return simulated values based on visual analysis
        
        # Simulate pit detection (in production, use CV)
        pit_detected = True  # Would come from actual CV analysis
        pit_depth_mm = 3.5  # Placeholder - would measure from depth map
        
        return pit_depth_mm, pit_detected
    
    def _measure_rebound_time(
        self,
        frames: List[np.ndarray],
        release_frame: int,
        fps: float
    ) -> float:
        """
        Measure time for pit to disappear after pressure release
        Tracks surface recovery frame-by-frame
        """
        if release_frame >= len(frames) - 1:
            return 0.0
        
        # Track surface recovery
        for i, frame in enumerate(frames[release_frame:]):
            if i == 0:
                continue
            
            # Placeholder: Check if pit has rebounded
            # Would compare surface contours frame-by-frame
            # For now, simulate based on clinical expectations
            
            # Simulate rebound detection (in production, use CV)
            if i > 10:  # Assume rebound after ~10 frames at 30fps
                rebound_time = i / fps
                return rebound_time
        
        # If never rebounded in video, return max time observed
        return (len(frames) - release_frame) / fps
    
    def _grade_pitting_edema(
        self,
        rebound_time: float,
        pit_depth: float
    ) -> int:
        """
        Grade pitting edema (1-4) based on rebound time and pit depth
        """
        # Check each grade criteria
        for grade in [4, 3, 2, 1]:  # Check highest first
            grade_info = self.PITTING_GRADES[grade]
            time_min, time_max = grade_info['rebound_time']
            depth_min, depth_max = grade_info['pit_depth_mm']
            
            if (time_min <= rebound_time < time_max and 
                depth_min <= pit_depth <= depth_max):
                return grade
        
        # Default to grade 1 if no match
        return 1
    
    def _segment_bilateral_limbs(
        self,
        frames: List[np.ndarray],
        location: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment left and right limbs from video frames
        Returns binary masks for each side
        """
        # Placeholder - in production would use:
        # - MediaPipe pose detection
        # - Semantic segmentation model
        # - Depth estimation
        
        # Return placeholder masks
        h, w = frames[0].shape[:2]
        left_mask = np.zeros((h, w), dtype=np.uint8)
        right_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Simulate segmentation (left = left half, right = right half)
        left_mask[:, :w//2] = 1
        right_mask[:, w//2:] = 1
        
        return left_mask, right_mask
    
    def _estimate_limb_volume(
        self,
        mask: np.ndarray,
        location: str
    ) -> float:
        """
        Estimate limb volume in ml from segmentation mask
        Uses depth estimation and geometric approximation
        """
        # Placeholder - in production would use:
        # - Multi-view geometry
        # - Depth estimation
        # - Cylinder/ellipsoid volume models
        
        # Count pixels in mask
        pixel_count = np.sum(mask)
        
        # Convert to volume estimate (very rough approximation)
        # Would use calibrated depth-to-volume mapping in production
        volume_ml = pixel_count * 0.01  # Placeholder scaling factor
        
        return float(volume_ml)
    
    def _calculate_peripheral_edema_index(
        self,
        current_volume: float,
        baseline_volume: float
    ) -> float:
        """
        Calculate PEI as % change from baseline
        PEI = ((current - baseline) / baseline) * 100
        """
        if baseline_volume == 0:
            return 0.0
        
        pei = ((current_volume - baseline_volume) / baseline_volume) * 100
        return float(pei)
    
    def _assess_skin_tightness(
        self,
        frames: List[np.ndarray],
        location: str
    ) -> float:
        """
        Assess skin tightness/tautness from visual appearance
        Returns score 0-1 (0=normal, 1=very tight/shiny)
        """
        # Placeholder - would analyze:
        # - Skin shininess (specular highlights)
        # - Surface smoothness
        # - Lack of natural wrinkles/texture
        
        return 0.3  # Placeholder
    
    def _detect_color_change(
        self,
        frames: List[np.ndarray],
        location: str
    ) -> bool:
        """
        Detect redness, discoloration, or color changes
        """
        # Placeholder - would analyze:
        # - HSV color analysis
        # - Compare to normal skin tones
        # - Detect inflammatory redness
        
        return False  # Placeholder
    
    def _get_or_create_baseline(
        self,
        patient_id: str,
        location: str,
        side: str
    ) -> Optional[EdemaBaseline]:
        """Get existing baseline or return None"""
        return self.db.query(EdemaBaseline).filter(
            and_(
                EdemaBaseline.patient_id == patient_id,
                EdemaBaseline.location == location,
                EdemaBaseline.side == side
            )
        ).first()
    
    def _update_baseline(
        self,
        patient_id: str,
        location: str,
        side: str,
        new_volume: float
    ):
        """Update baseline volume using EMA"""
        baseline = self._get_or_create_baseline(patient_id, location, side)
        
        if not baseline:
            baseline = EdemaBaseline(
                patient_id=patient_id,
                location=location,
                side=side,
                baseline_volume_ml=new_volume,
                sample_size=1,
                confidence=0.5,
                source='auto'
            )
            self.db.add(baseline)
        else:
            # Exponential moving average
            alpha = 0.2 if baseline.sample_size >= 5 else 1.0 / (baseline.sample_size + 1)
            baseline.baseline_volume_ml = baseline.baseline_volume_ml * (1 - alpha) + new_volume * alpha
            baseline.sample_size += 1
            baseline.confidence = min(0.95, baseline.confidence + 0.05)
        
        self.db.commit()
    
    def get_edema_summary(self, patient_id: str) -> Dict[str, Any]:
        """Get comprehensive edema summary for patient"""
        # Get recent metrics (last 7 days)
        cutoff = datetime.utcnow() - timedelta(days=7)
        recent_metrics = self.db.query(EdemaMetric).filter(
            and_(
                EdemaMetric.patient_id == patient_id,
                EdemaMetric.recorded_at >= cutoff
            )
        ).order_by(EdemaMetric.recorded_at).all()
        
        # Group by location
        by_location = {}
        for metric in recent_metrics:
            loc = metric.location
            if loc not in by_location:
                by_location[loc] = []
            by_location[loc].append(metric)
        
        # Summarize
        summary = {
            'patient_id': patient_id,
            'locations_monitored': list(by_location.keys()),
            'recent_count': len(recent_metrics),
            'by_location': {}
        }
        
        for loc, metrics in by_location.items():
            latest = metrics[-1]
            summary['by_location'][loc] = {
                'latest_pei': latest.peripheral_edema_index,
                'pitting_grade': latest.pitting_grade,
                'bilateral': latest.bilateral_swelling,
                'asymmetry': latest.asymmetry_ratio,
                'trend': 'worsening' if len(metrics) > 1 and metrics[-1].peripheral_edema_index > metrics[0].peripheral_edema_index else 'stable'
            }
        
        return summary
