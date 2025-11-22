"""
Medication Adherence Service
============================

Calculates medication adherence metrics including:
- 7-day adherence trend
- Regimen risk analysis
- Missed dose escalation detection
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta, timezone, date
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc, text
from decimal import Decimal

logger = logging.getLogger(__name__)


class MedicationAdherenceService:
    """Service for medication adherence analytics"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_adherence_analytics(self, patient_id: str) -> Dict[str, Any]:
        """
        Get comprehensive medication adherence analytics
        
        Returns:
            Dict with currentAdherenceRate, sevenDayTrend, regimenRisk, missedDoseEscalation
        """
        try:
            # Get current adherence rate from behavioral metrics
            current_rate = self._get_current_adherence_rate(patient_id)
            
            # Get 7-day adherence trend from behavioral metrics
            seven_day_trend = self._get_seven_day_trend(patient_id)
            
            # Calculate regimen risk based on multiple factors
            regimen_risk = self._calculate_regimen_risk(patient_id)
            
            # Get missed dose escalation data
            missed_dose_escalation = self._get_missed_dose_escalation(patient_id)
            
            return {
                "currentAdherenceRate": current_rate,
                "sevenDayTrend": seven_day_trend,
                "regimenRisk": regimen_risk,
                "missedDoseEscalation": missed_dose_escalation
            }
        
        except Exception as e:
            logger.error(f"Error getting adherence analytics for patient {patient_id}: {str(e)}")
            return {
                "currentAdherenceRate": None,
                "sevenDayTrend": [],
                "regimenRisk": {"level": "unknown", "rationale": "Error calculating risk"},
                "missedDoseEscalation": {"count": 0, "severity": "none"}
            }
    
    def _get_current_adherence_rate(self, patient_id: str) -> Optional[float]:
        """Get current adherence rate from behavioral metrics"""
        try:
            # Query latest behavioral metric using raw SQL
            query = text("""
                SELECT medication_adherence_rate
                FROM behavior_metrics
                WHERE patient_id = :patient_id
                ORDER BY date DESC
                LIMIT 1
            """)
            
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            
            if result and result[0] is not None:
                # Convert Decimal to float
                return float(result[0])
            
            # Fallback: calculate from medication_adherence table (last 7 days)
            return self._calculate_adherence_from_timeline(patient_id, days=7)
        
        except Exception as e:
            logger.error(f"Error getting current adherence rate: {str(e)}")
            return None
    
    def _get_seven_day_trend(self, patient_id: str) -> List[Dict[str, Any]]:
        """Get 7-day adherence trend from behavioral metrics table"""
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=6)  # 7 days including today
            
            # Query behavioral metrics for last 7 days using raw SQL
            query = text("""
                SELECT date, medication_adherence_rate
                FROM behavior_metrics
                WHERE patient_id = :patient_id
                AND date >= :start_date
                AND date <= :end_date
                ORDER BY date ASC
            """)
            
            results = self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchall()
            
            # Create date map from results
            date_map = {}
            for row in results:
                date_str = row[0].strftime('%Y-%m-%d') if hasattr(row[0], 'strftime') else str(row[0])[:10]
                adherence = float(row[1]) if row[1] is not None else 0.0
                date_map[date_str] = adherence
            
            # Build 7-day trend with gaps filled
            trend = []
            for i in range(7):
                date = start_date + timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
                
                # Use actual data if available, otherwise fallback
                if date_str in date_map:
                    adherence = date_map[date_str]
                else:
                    # Fallback: calculate from medication_adherence for this specific date
                    adherence = self._calculate_adherence_for_date(patient_id, date)
                
                trend.append({
                    "date": date_str,
                    "adherenceRate": adherence
                })
            
            return trend
        
        except Exception as e:
            logger.error(f"Error getting 7-day trend: {str(e)}")
            return []
    
    def _calculate_adherence_for_date(self, patient_id: str, date_param: date) -> float:
        """Calculate adherence percentage for a specific date from medication_adherence table"""
        try:
            # Query medication_adherence for the specific date
            query = text("""
                SELECT 
                    COUNT(*) as total_scheduled,
                    COUNT(CASE WHEN status = 'taken' THEN 1 END) as taken_count
                FROM medication_adherence
                WHERE patient_id = :patient_id
                AND DATE(scheduled_time) = :date
            """)
            
            result = self.db.execute(query, {
                "patient_id": patient_id,
                "date": date_param
            }).fetchone()
            
            if result and result[0] > 0:  # total_scheduled > 0
                total = result[0]
                taken = result[1] or 0
                return taken / total
            
            # No data for this date
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating adherence for date {date_param}: {str(e)}")
            return 0.0
    
    def _calculate_adherence_from_timeline(self, patient_id: str, days: int = 7) -> Optional[float]:
        """Calculate overall adherence from medication_adherence table over last N days"""
        try:
            # Calculate date range
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            # Query medication_adherence for date range
            query = text("""
                SELECT 
                    COUNT(*) as total_scheduled,
                    COUNT(CASE WHEN status = 'taken' THEN 1 END) as taken_count
                FROM medication_adherence
                WHERE patient_id = :patient_id
                AND scheduled_time >= :start_date
                AND scheduled_time <= :end_date
            """)
            
            result = self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()
            
            if result and result[0] > 0:  # total_scheduled > 0
                total = result[0]
                taken = result[1] or 0
                return taken / total
            
            # No adherence data
            return None
        
        except Exception as e:
            logger.error(f"Error calculating adherence from timeline: {str(e)}")
            return None
    
    def _calculate_regimen_risk(self, patient_id: str) -> Dict[str, str]:
        """
        Calculate regimen risk based on:
        - Active medication count (from medications table)
        - Missed doses count (last 30 days from medication_adherence table)
        
        NOTE: Drug interactions are not included because the 'drugs' reference table
        does not exist in the current schema. Risk is calculated from medication
        count and adherence data only.
        """
        try:
            active_med_count = self._get_active_medication_count(patient_id)
            missed_doses_count = self._get_total_missed_doses(patient_id)
            
            # Calculate risk level based on thresholds
            risk_factors = []
            
            # Medication count risk
            if active_med_count > 10:
                risk_factors.append(f"{active_med_count} active medications (high complexity)")
            elif active_med_count > 5:
                risk_factors.append(f"{active_med_count} active medications")
            else:
                risk_factors.append(f"{active_med_count} active medications (manageable)")
            
            # Missed doses risk
            if missed_doses_count > 10:
                risk_factors.append(f"{missed_doses_count} missed doses in last 30 days (critical)")
            elif missed_doses_count > 5:
                risk_factors.append(f"{missed_doses_count} missed doses in last 30 days (warning)")
            elif missed_doses_count > 0:
                risk_factors.append(f"{missed_doses_count} missed doses in last 30 days")
            
            # Determine overall risk level (based on medication count and missed doses only)
            if active_med_count > 10 or missed_doses_count > 10:
                level = "high"
            elif active_med_count > 5 or missed_doses_count > 5:
                level = "moderate"
            else:
                level = "low"
            
            rationale = ", ".join(risk_factors) if risk_factors else "No significant risk factors"
            
            return {
                "level": level,
                "rationale": rationale
            }
        
        except Exception as e:
            logger.error(f"Error calculating regimen risk: {str(e)}")
            return {"level": "unknown", "rationale": "Error calculating risk"}
    
    def _get_active_medication_count(self, patient_id: str) -> int:
        """Get count of active medications from medications table"""
        try:
            query = text("""
                SELECT COUNT(*)
                FROM medications
                WHERE patient_id = :patient_id
                AND active = true
                AND (end_date IS NULL OR end_date >= CURRENT_DATE)
            """)
            
            result = self.db.execute(query, {"patient_id": patient_id}).fetchone()
            return result[0] if result else 0
        
        except Exception as e:
            logger.error(f"Error getting active medication count: {str(e)}")
            return 0
    
    def _get_total_missed_doses(self, patient_id: str) -> int:
        """Get total missed doses count from medication_adherence (last 30 days)"""
        try:
            # Calculate date range (last 30 days)
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)
            
            query = text("""
                SELECT COUNT(*)
                FROM medication_adherence
                WHERE patient_id = :patient_id
                AND status = 'missed'
                AND scheduled_time >= :start_date
                AND scheduled_time <= :end_date
            """)
            
            result = self.db.execute(query, {
                "patient_id": patient_id,
                "start_date": start_date,
                "end_date": end_date
            }).fetchone()
            
            return result[0] if result else 0
        
        except Exception as e:
            logger.error(f"Error getting missed doses count: {str(e)}")
            return 0
    
    def _get_drug_interaction_count(self, patient_id: str) -> int:
        """
        Get count of potential drug interactions for patient's active medications
        
        NOTE: Currently returns 0 because the 'drugs' reference table does not exist
        in the schema. The drug_interactions table requires drug IDs, but medications
        table only stores drug names without references to a drugs table.
        
        To enable this feature:
        1. Create a 'drugs' reference table with standardized drug names and IDs
        2. Link medications.name to drugs.id (or drugs.name)
        3. Then query drug_interactions for patient's active drug combinations
        """
        # Cannot query drug interactions without a drugs reference table
        # Returning 0 to maintain data integrity (no fabricated interaction counts)
        return 0
    
    def _get_missed_dose_escalation(self, patient_id: str) -> Dict[str, Any]:
        """Get missed dose escalation data"""
        try:
            missed_count = self._get_total_missed_doses(patient_id)
            
            # Determine severity based on missed count
            if missed_count > 10:
                severity = "critical"
            elif missed_count > 5:
                severity = "warning"
            else:
                severity = "none"
            
            return {
                "count": missed_count,
                "severity": severity
            }
        
        except Exception as e:
            logger.error(f"Error getting missed dose escalation: {str(e)}")
            return {"count": 0, "severity": "none"}
