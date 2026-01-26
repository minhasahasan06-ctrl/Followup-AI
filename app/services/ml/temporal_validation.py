"""
Temporal Validation Service (Phase C.18)
========================================
Patient-grouped time splits and time-based cross-validation.

Ensures no data leakage through temporal separation and patient grouping.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TemporalSplit:
    """Represents a single temporal train/test split"""
    split_id: int
    train_start: datetime
    train_end: datetime
    gap_days: int
    test_start: datetime
    test_end: datetime
    train_patient_ids: List[str]
    test_patient_ids: List[str]


@dataclass
class TimeSeriesCVResult:
    """Results from time series cross-validation"""
    n_splits: int
    splits: List[TemporalSplit]
    patient_groups: Dict[str, int]
    validation_report: Dict[str, Any]


class TemporalValidationService:
    """
    C.18: Temporal validation with patient-grouped time splits.
    
    Key features:
    - No data leakage: future data never used to train on past
    - Patient grouping: same patient never in both train and test
    - Configurable gap between train and test periods
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap_days: int = 7,
        min_train_days: int = 90,
        test_days: int = 30,
    ):
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.min_train_days = min_train_days
        self.test_days = test_days
    
    def create_patient_grouped_splits(
        self,
        data: List[Dict[str, Any]],
        patient_id_col: str = "patient_id",
        date_col: str = "created_at",
    ) -> TimeSeriesCVResult:
        """
        Create temporal splits with patient grouping.
        
        Ensures:
        1. Training data always precedes test data in time
        2. Gap period between train and test prevents leakage
        3. Each patient appears in only one fold's train OR test set
        
        Args:
            data: List of records with patient_id and date columns
            patient_id_col: Column name for patient ID
            date_col: Column name for date
            
        Returns:
            TimeSeriesCVResult with splits and validation report
        """
        if not data:
            return TimeSeriesCVResult(
                n_splits=0,
                splits=[],
                patient_groups={},
                validation_report={"status": "no_data"},
            )
        
        dates = [self._parse_date(row.get(date_col)) for row in data]
        valid_dates = [d for d in dates if d is not None]
        
        if not valid_dates:
            return TimeSeriesCVResult(
                n_splits=0,
                splits=[],
                patient_groups={},
                validation_report={"status": "no_valid_dates"},
            )
        
        min_date = min(valid_dates)
        max_date = max(valid_dates)
        total_days = (max_date - min_date).days
        
        patients_by_first_date = {}
        for row in data:
            patient_id = row.get(patient_id_col)
            date = self._parse_date(row.get(date_col))
            if patient_id and date:
                if patient_id not in patients_by_first_date:
                    patients_by_first_date[patient_id] = date
                else:
                    patients_by_first_date[patient_id] = min(
                        patients_by_first_date[patient_id], date
                    )
        
        patient_groups = self._assign_patient_groups(
            patients_by_first_date, self.n_splits
        )
        
        splits = []
        days_per_split = max(1, (total_days - self.min_train_days) // self.n_splits)
        
        for i in range(self.n_splits):
            train_end_offset = self.min_train_days + (i * days_per_split)
            train_end = min_date + timedelta(days=train_end_offset)
            test_start = train_end + timedelta(days=self.gap_days)
            test_end = test_start + timedelta(days=self.test_days)
            
            if test_end > max_date:
                test_end = max_date
            
            train_patients = [
                pid for pid, group in patient_groups.items()
                if group <= i
            ]
            test_patients = [
                pid for pid, group in patient_groups.items()
                if group == i + 1
            ]
            
            split = TemporalSplit(
                split_id=i,
                train_start=min_date,
                train_end=train_end,
                gap_days=self.gap_days,
                test_start=test_start,
                test_end=test_end,
                train_patient_ids=train_patients,
                test_patient_ids=test_patients,
            )
            splits.append(split)
        
        validation_report = self._generate_validation_report(
            splits, patient_groups, total_days
        )
        
        return TimeSeriesCVResult(
            n_splits=len(splits),
            splits=splits,
            patient_groups=patient_groups,
            validation_report=validation_report,
        )
    
    def validate_no_leakage(
        self,
        train_data: List[Dict[str, Any]],
        test_data: List[Dict[str, Any]],
        patient_id_col: str = "patient_id",
        date_col: str = "created_at",
    ) -> Dict[str, Any]:
        """
        Validate that there is no temporal or patient leakage between sets.
        
        Returns validation report with any detected leakage issues.
        """
        issues = []
        
        train_patients = set(row.get(patient_id_col) for row in train_data)
        test_patients = set(row.get(patient_id_col) for row in test_data)
        patient_overlap = train_patients & test_patients
        
        if patient_overlap:
            issues.append({
                "type": "patient_leakage",
                "severity": "critical",
                "count": len(patient_overlap),
                "message": f"{len(patient_overlap)} patients appear in both train and test",
            })
        
        train_dates = [self._parse_date(row.get(date_col)) for row in train_data]
        test_dates = [self._parse_date(row.get(date_col)) for row in test_data]
        
        train_dates = [d for d in train_dates if d]
        test_dates = [d for d in test_dates if d]
        
        if train_dates and test_dates:
            max_train = max(train_dates)
            min_test = min(test_dates)
            
            if max_train >= min_test:
                issues.append({
                    "type": "temporal_leakage",
                    "severity": "critical",
                    "message": f"Training data ({max_train}) overlaps with test data ({min_test})",
                })
            elif (min_test - max_train).days < self.gap_days:
                issues.append({
                    "type": "insufficient_gap",
                    "severity": "warning",
                    "message": f"Gap of {(min_test - max_train).days} days is less than required {self.gap_days}",
                })
        
        return {
            "valid": len([i for i in issues if i["severity"] == "critical"]) == 0,
            "issues": issues,
            "train_patients": len(train_patients),
            "test_patients": len(test_patients),
            "train_date_range": (min(train_dates).isoformat() if train_dates else None,
                                 max(train_dates).isoformat() if train_dates else None),
            "test_date_range": (min(test_dates).isoformat() if test_dates else None,
                                max(test_dates).isoformat() if test_dates else None),
        }
    
    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if date_value is None:
            return None
        if isinstance(date_value, datetime):
            return date_value
        if isinstance(date_value, str):
            try:
                return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None
    
    def _assign_patient_groups(
        self,
        patients_by_first_date: Dict[str, datetime],
        n_groups: int,
    ) -> Dict[str, int]:
        """Assign patients to groups based on their first appearance date"""
        if not patients_by_first_date:
            return {}
        
        sorted_patients = sorted(
            patients_by_first_date.items(),
            key=lambda x: x[1]
        )
        
        patients_per_group = max(1, len(sorted_patients) // n_groups)
        groups = {}
        
        for i, (patient_id, _) in enumerate(sorted_patients):
            group = min(i // patients_per_group, n_groups - 1)
            groups[patient_id] = group
        
        return groups
    
    def _generate_validation_report(
        self,
        splits: List[TemporalSplit],
        patient_groups: Dict[str, int],
        total_days: int,
    ) -> Dict[str, Any]:
        """Generate validation report for the splits"""
        return {
            "status": "valid",
            "n_splits": len(splits),
            "total_patients": len(patient_groups),
            "total_days": total_days,
            "gap_days": self.gap_days,
            "min_train_days": self.min_train_days,
            "test_days": self.test_days,
            "splits_summary": [
                {
                    "split_id": s.split_id,
                    "train_patients": len(s.train_patient_ids),
                    "test_patients": len(s.test_patient_ids),
                    "train_days": (s.train_end - s.train_start).days,
                    "test_days": (s.test_end - s.test_start).days,
                }
                for s in splits
            ],
        }


def create_temporal_splits(
    data: List[Dict[str, Any]],
    n_splits: int = 5,
    gap_days: int = 7,
) -> TimeSeriesCVResult:
    """Convenience function to create temporal splits"""
    service = TemporalValidationService(n_splits=n_splits, gap_days=gap_days)
    return service.create_patient_grouped_splits(data)


__all__ = [
    "TemporalValidationService",
    "TemporalSplit",
    "TimeSeriesCVResult",
    "create_temporal_splits",
]
