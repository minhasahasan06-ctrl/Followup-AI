"""
Leakage Detection Service (Phase C.19)
======================================
Scan for data leakage using timestamps and feature analysis.

Detects common leakage patterns that can invalidate ML models.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import re

logger = logging.getLogger(__name__)


@dataclass
class LeakageIssue:
    """Represents a detected leakage issue"""
    issue_type: str
    severity: str
    column: Optional[str]
    description: str
    recommendation: str


@dataclass
class LeakageReport:
    """Complete leakage scan report"""
    scan_id: str
    timestamp: str
    issues: List[LeakageIssue]
    columns_scanned: int
    records_sampled: int
    is_clean: bool
    risk_score: float


class LeakageDetectionService:
    """
    C.19: Leakage detection service.
    
    Scans for:
    - Temporal leakage (future data in features)
    - Target leakage (target info in features)
    - Train-test leakage (data overlap)
    - Identifier leakage (IDs correlating with target)
    """
    
    SUSPICIOUS_COLUMN_PATTERNS = [
        (r".*_outcome.*", "target_leakage", "Column may contain outcome information"),
        (r".*_result.*", "target_leakage", "Column may contain result information"),
        (r".*_label.*", "target_leakage", "Column may contain label information"),
        (r".*_future.*", "temporal_leakage", "Column may contain future data"),
        (r".*_next_.*", "temporal_leakage", "Column may reference future values"),
        (r".*_prediction.*", "target_leakage", "Column may contain predictions"),
        (r".*_score$", "potential_leakage", "Column may be derived from target"),
    ]
    
    IDENTIFIER_PATTERNS = [
        r".*_id$",
        r"^id$",
        r".*_uuid$",
        r".*_key$",
    ]
    
    def __init__(
        self,
        target_column: Optional[str] = None,
        date_column: Optional[str] = None,
        prediction_date_column: Optional[str] = None,
    ):
        self.target_column = target_column
        self.date_column = date_column
        self.prediction_date_column = prediction_date_column
    
    def scan_for_leakage(
        self,
        data: List[Dict[str, Any]],
        feature_columns: Optional[List[str]] = None,
    ) -> LeakageReport:
        """
        Scan dataset for potential leakage issues.
        
        Args:
            data: List of records to scan
            feature_columns: Columns to use as features (if None, infer from data)
            
        Returns:
            LeakageReport with all detected issues
        """
        from uuid import uuid4
        
        if not data:
            return LeakageReport(
                scan_id=str(uuid4()),
                timestamp=datetime.utcnow().isoformat(),
                issues=[],
                columns_scanned=0,
                records_sampled=0,
                is_clean=True,
                risk_score=0.0,
            )
        
        columns = feature_columns or list(data[0].keys())
        issues = []
        
        issues.extend(self._check_column_names(columns))
        
        issues.extend(self._check_temporal_leakage(data, columns))
        
        issues.extend(self._check_target_leakage(data, columns))
        
        issues.extend(self._check_identifier_leakage(data, columns))
        
        issues.extend(self._check_high_cardinality(data, columns))
        
        risk_score = self._calculate_risk_score(issues)
        
        return LeakageReport(
            scan_id=str(uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            issues=issues,
            columns_scanned=len(columns),
            records_sampled=len(data),
            is_clean=len([i for i in issues if i.severity == "critical"]) == 0,
            risk_score=risk_score,
        )
    
    def _check_column_names(self, columns: List[str]) -> List[LeakageIssue]:
        """Check column names for suspicious patterns"""
        issues = []
        
        for col in columns:
            col_lower = col.lower()
            for pattern, issue_type, description in self.SUSPICIOUS_COLUMN_PATTERNS:
                if re.match(pattern, col_lower):
                    issues.append(LeakageIssue(
                        issue_type=issue_type,
                        severity="warning" if issue_type == "potential_leakage" else "high",
                        column=col,
                        description=description,
                        recommendation=f"Review column '{col}' to ensure it doesn't leak future/target information",
                    ))
                    break
        
        return issues
    
    def _check_temporal_leakage(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> List[LeakageIssue]:
        """Check for temporal leakage in date columns"""
        issues = []
        
        if not self.date_column or not self.prediction_date_column:
            return issues
        
        for row in data[:1000]:
            record_date = self._parse_date(row.get(self.date_column))
            pred_date = self._parse_date(row.get(self.prediction_date_column))
            
            if record_date and pred_date and record_date > pred_date:
                issues.append(LeakageIssue(
                    issue_type="temporal_leakage",
                    severity="critical",
                    column=self.date_column,
                    description="Records found with dates after prediction date",
                    recommendation="Filter records to only include data available at prediction time",
                ))
                break
        
        for col in columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ["future", "next", "forward", "later"]):
                if col != self.date_column:
                    issues.append(LeakageIssue(
                        issue_type="temporal_leakage",
                        severity="high",
                        column=col,
                        description=f"Column '{col}' may contain forward-looking information",
                        recommendation="Remove or replace with historical-only version",
                    ))
        
        return issues
    
    def _check_target_leakage(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> List[LeakageIssue]:
        """Check for target leakage in features"""
        issues = []
        
        if not self.target_column:
            return issues
        
        target_values = [row.get(self.target_column) for row in data[:1000] if row.get(self.target_column) is not None]
        
        if not target_values:
            return issues
        
        for col in columns:
            if col == self.target_column:
                continue
            
            col_values = [row.get(col) for row in data[:1000] if row.get(col) is not None]
            
            if not col_values:
                continue
            
            if self._check_perfect_correlation(col_values, target_values[:len(col_values)]):
                issues.append(LeakageIssue(
                    issue_type="target_leakage",
                    severity="critical",
                    column=col,
                    description=f"Column '{col}' appears perfectly correlated with target",
                    recommendation="Remove this column or investigate the relationship",
                ))
        
        return issues
    
    def _check_identifier_leakage(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> List[LeakageIssue]:
        """Check for identifier columns that might leak information"""
        issues = []
        
        for col in columns:
            col_lower = col.lower()
            for pattern in self.IDENTIFIER_PATTERNS:
                if re.match(pattern, col_lower):
                    unique_values = len(set(row.get(col) for row in data if row.get(col) is not None))
                    total_records = len(data)
                    
                    if unique_values > 0.9 * total_records:
                        issues.append(LeakageIssue(
                            issue_type="identifier_leakage",
                            severity="warning",
                            column=col,
                            description=f"Identifier column '{col}' has high cardinality and may cause overfitting",
                            recommendation="Remove identifier columns from features",
                        ))
                    break
        
        return issues
    
    def _check_high_cardinality(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
    ) -> List[LeakageIssue]:
        """Check for high cardinality categorical columns"""
        issues = []
        
        for col in columns:
            values = [row.get(col) for row in data if row.get(col) is not None]
            
            if not values:
                continue
            
            if all(isinstance(v, str) for v in values[:100]):
                unique_count = len(set(values))
                total_count = len(values)
                
                if unique_count > 0.5 * total_count and unique_count > 100:
                    issues.append(LeakageIssue(
                        issue_type="high_cardinality",
                        severity="warning",
                        column=col,
                        description=f"Column '{col}' has very high cardinality ({unique_count} unique values)",
                        recommendation="Consider encoding or grouping this column",
                    ))
        
        return issues
    
    def _check_perfect_correlation(
        self,
        values1: List[Any],
        values2: List[Any],
    ) -> bool:
        """Check if two value lists are perfectly correlated"""
        if len(values1) != len(values2) or len(values1) < 10:
            return False
        
        if all(v1 == v2 for v1, v2 in zip(values1, values2)):
            return True
        
        if all(isinstance(v1, (int, float)) and isinstance(v2, (int, float)) 
               for v1, v2 in zip(values1[:10], values2[:10])):
            try:
                diff = [float(v1) - float(v2) for v1, v2 in zip(values1, values2)]
                if len(set(diff)) == 1:
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
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
    
    def _calculate_risk_score(self, issues: List[LeakageIssue]) -> float:
        """Calculate overall risk score from issues"""
        if not issues:
            return 0.0
        
        severity_weights = {
            "critical": 1.0,
            "high": 0.5,
            "warning": 0.2,
            "info": 0.05,
        }
        
        total_weight = sum(
            severity_weights.get(issue.severity, 0.1)
            for issue in issues
        )
        
        return min(1.0, total_weight)


def scan_for_leakage(
    data: List[Dict[str, Any]],
    target_column: Optional[str] = None,
    date_column: Optional[str] = None,
) -> LeakageReport:
    """Convenience function to scan for leakage"""
    service = LeakageDetectionService(
        target_column=target_column,
        date_column=date_column,
    )
    return service.scan_for_leakage(data)


__all__ = [
    "LeakageDetectionService",
    "LeakageIssue",
    "LeakageReport",
    "scan_for_leakage",
]
