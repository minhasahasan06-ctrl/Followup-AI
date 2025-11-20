"""
Cognitive Test Service
=====================

Manages weekly cognitive micro-tests and detects cognitive drift.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_
import numpy as np

logger = logging.getLogger(__name__)


class CognitiveTestService:
    """
    Service for administering and analyzing cognitive tests
    
    Test Types:
    - Reaction time (tap speed)
    - Tapping speed test
    - Memory recall
    - Pattern recognition
    - Instruction following
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def score_test_results(
        self,
        patient_id: str,
        test_type: str,
        raw_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Score cognitive test and detect baseline deviation
        
        Args:
            patient_id: Patient identifier
            test_type: Type of test
            raw_results: Raw test results from frontend
        
        Returns:
            Scored results with deviation detection
        """
        
        # Extract performance metrics based on test type
        if test_type == 'reaction_time':
            metrics = self._score_reaction_time(raw_results)
        elif test_type == 'tapping':
            metrics = self._score_tapping(raw_results)
        elif test_type == 'memory':
            metrics = self._score_memory(raw_results)
        elif test_type == 'pattern_recall':
            metrics = self._score_pattern_recall(raw_results)
        elif test_type == 'instruction_follow':
            metrics = self._score_instruction_follow(raw_results)
        else:
            logger.warning(f"Unknown test type: {test_type}")
            metrics = {}
        
        # Calculate baseline deviation
        baseline_stats = self._get_patient_baseline(patient_id, test_type)
        deviation = self._calculate_deviation(metrics, baseline_stats)
        
        # Anomaly detection (Z-score > 2.0)
        anomaly_detected = abs(deviation) > 2.0
        
        return {
            **metrics,
            'baseline_deviation': deviation,
            'anomaly_detected': anomaly_detected,
            'raw_results': raw_results
        }
    
    def _score_reaction_time(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Score reaction time test"""
        
        reaction_times = raw_results.get('reaction_times_ms', [])
        
        if not reaction_times:
            return {'reaction_time_ms': None, 'error_rate': 1.0}
        
        # Remove outliers (>3 seconds likely errors)
        valid_times = [rt for rt in reaction_times if rt < 3000]
        
        avg_reaction_time = int(np.mean(valid_times)) if valid_times else 0
        error_rate = 1.0 - (len(valid_times) / max(len(reaction_times), 1))
        
        return {
            'reaction_time_ms': avg_reaction_time,
            'error_rate': float(error_rate)
        }
    
    def _score_tapping(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Score tapping speed test"""
        
        tap_count = raw_results.get('tap_count', 0)
        duration_seconds = raw_results.get('duration_seconds', 10)
        
        tapping_speed = tap_count / max(duration_seconds, 1)
        
        return {
            'tapping_speed': float(tapping_speed),
            'error_rate': 0.0  # Tapping test has no errors
        }
    
    def _score_memory(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Score memory recall test"""
        
        correct_recalls = raw_results.get('correct_recalls', 0)
        total_items = raw_results.get('total_items', 1)
        
        memory_score = correct_recalls / max(total_items, 1)
        error_rate = 1.0 - memory_score
        
        return {
            'memory_score': float(memory_score),
            'error_rate': float(error_rate)
        }
    
    def _score_pattern_recall(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Score pattern recognition test"""
        
        correct_patterns = raw_results.get('correct_patterns', 0)
        total_patterns = raw_results.get('total_patterns', 1)
        
        pattern_accuracy = correct_patterns / max(total_patterns, 1)
        error_rate = 1.0 - pattern_accuracy
        
        return {
            'pattern_recall_accuracy': float(pattern_accuracy),
            'error_rate': float(error_rate)
        }
    
    def _score_instruction_follow(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """Score instruction following test"""
        
        correct_steps = raw_results.get('correct_steps', 0)
        total_steps = raw_results.get('total_steps', 1)
        
        instruction_accuracy = correct_steps / max(total_steps, 1)
        error_rate = 1.0 - instruction_accuracy
        
        return {
            'instruction_accuracy': float(instruction_accuracy),
            'error_rate': float(error_rate)
        }
    
    def _get_patient_baseline(
        self,
        patient_id: str,
        test_type: str,
        days: int = 90
    ) -> Dict[str, float]:
        """Get patient's baseline performance for test type"""
        from app.models.behavior_models import CognitiveTest
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        past_tests = self.db.query(CognitiveTest).filter(
            and_(
                CognitiveTest.patient_id == patient_id,
                CognitiveTest.test_type == test_type,
                CognitiveTest.started_at >= cutoff_date
            )
        ).all()
        
        if not past_tests:
            return {'mean': 0.0, 'std': 1.0}
        
        # Extract primary metric based on test type
        if test_type == 'reaction_time':
            values = [t.reaction_time_ms for t in past_tests if t.reaction_time_ms]
        elif test_type == 'tapping':
            values = [float(t.tapping_speed) for t in past_tests if t.tapping_speed]
        elif test_type == 'memory':
            values = [float(t.memory_score) for t in past_tests if t.memory_score]
        elif test_type == 'pattern_recall':
            values = [float(t.pattern_recall_accuracy) for t in past_tests if t.pattern_recall_accuracy]
        elif test_type == 'instruction_follow':
            values = [float(t.instruction_accuracy) for t in past_tests if t.instruction_accuracy]
        else:
            values = []
        
        if not values:
            return {'mean': 0.0, 'std': 1.0}
        
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)) if len(values) > 1 else 1.0
        }
    
    def _calculate_deviation(
        self,
        current_metrics: Dict[str, Any],
        baseline_stats: Dict[str, float]
    ) -> float:
        """
        Calculate Z-score deviation from baseline
        
        Args:
            current_metrics: Current test metrics
            baseline_stats: Baseline mean and std
        
        Returns:
            Z-score deviation
        """
        # Get primary metric value
        metric_value = None
        for key in ['reaction_time_ms', 'tapping_speed', 'memory_score', 
                    'pattern_recall_accuracy', 'instruction_accuracy']:
            if key in current_metrics and current_metrics[key] is not None:
                metric_value = current_metrics[key]
                break
        
        if metric_value is None:
            return 0.0
        
        mean = baseline_stats.get('mean', 0.0)
        std = baseline_stats.get('std', 1.0)
        
        if std == 0:
            std = 1.0
        
        z_score = (metric_value - mean) / std
        
        return float(z_score)
    
    def calculate_weekly_drift_score(
        self,
        patient_id: str
    ) -> float:
        """
        Calculate cognitive drift score over past week
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            Drift score (0-1, higher = more drift)
        """
        from app.models.behavior_models import CognitiveTest
        
        cutoff_date = datetime.utcnow() - timedelta(days=7)
        
        recent_tests = self.db.query(CognitiveTest).filter(
            and_(
                CognitiveTest.patient_id == patient_id,
                CognitiveTest.started_at >= cutoff_date
            )
        ).all()
        
        if not recent_tests:
            return 0.0
        
        # Count anomalies
        anomaly_count = sum(1 for test in recent_tests if test.anomaly_detected)
        
        # Drift = proportion of anomalous tests
        drift_score = anomaly_count / max(len(recent_tests), 1)
        
        return float(drift_score)
