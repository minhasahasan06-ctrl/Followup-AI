"""
Medication-Symptom Correlation Engine
AI-powered analysis of temporal patterns between medication changes and symptom onset
REGULATORY COMPLIANCE: Uses "change detection" and "monitoring" language, NOT diagnosis
HIPAA-compliant: Operates on patient data with proper authorization
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from openai import OpenAI

from app.models.medication_side_effects import (
    MedicationTimeline,
    DosageChange,
    SymptomLog,
    SideEffectCorrelation,
    MedicationEffectsSummary,
    ChangeReason,
    CorrelationStrength
)


class MedicationCorrelationEngine:
    """
    Analyzes temporal correlations between medication changes and symptom onset
    Uses AI to detect patterns and calculate likelihood scores
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def analyze_patient_correlations(
        self,
        patient_id: str,
        days_back: int = 90,
        min_correlation_score: float = 0.3
    ) -> List[SideEffectCorrelation]:
        """
        Analyze all medication-symptom correlations for a patient
        Returns list of detected correlations above threshold
        
        Args:
            patient_id: Patient ID to analyze
            days_back: How many days of history to analyze
            min_correlation_score: Minimum correlation score to include (0.0-1.0)
        
        Returns:
            List of SideEffectCorrelation objects
        """
        # Get patient's medications (FIX: include long-running meds started before cutoff)
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        medications = self.db.query(MedicationTimeline).filter(
            and_(
                MedicationTimeline.patient_id == patient_id,
                or_(
                    # Started within window
                    MedicationTimeline.started_at >= cutoff_date,
                    # Started before but still active during window
                    and_(
                        MedicationTimeline.started_at < cutoff_date,
                        or_(
                            MedicationTimeline.stopped_at == None,
                            MedicationTimeline.stopped_at >= cutoff_date
                        )
                    )
                )
            )
        ).all()
        
        # Get patient's symptoms in same time period
        symptoms = self.db.query(SymptomLog).filter(
            and_(
                SymptomLog.patient_id == patient_id,
                SymptomLog.reported_at >= cutoff_date
            )
        ).all()
        
        if not medications or not symptoms:
            return []
        
        correlations = []
        
        # Analyze each medication against symptoms
        for medication in medications:
            # Get dosage changes for this medication
            dosage_changes = self.db.query(DosageChange).filter(
                DosageChange.medication_timeline_id == medication.id
            ).order_by(DosageChange.change_date).all()
            
            # Analyze temporal correlation for each symptom type
            symptom_groups = self._group_symptoms_by_name(symptoms)
            
            for symptom_name, symptom_list in symptom_groups.items():
                correlation_list = self._analyze_single_correlation(
                    medication,
                    dosage_changes,
                    symptom_name,
                    symptom_list
                )
                
                # Filter by confidence score threshold
                for correlation in correlation_list:
                    if correlation.confidence_score >= min_correlation_score:
                        correlations.append(correlation)
        
        return correlations
    
    def _group_symptoms_by_name(self, symptoms: List[SymptomLog]) -> Dict[str, List[SymptomLog]]:
        """Group symptoms by normalized name for pattern analysis"""
        groups = {}
        for symptom in symptoms:
            name = symptom.symptom_name.lower().strip()
            if name not in groups:
                groups[name] = []
            groups[name].append(symptom)
        return groups
    
    def _analyze_single_correlation(
        self,
        medication: MedicationTimeline,
        dosage_changes: List[DosageChange],
        symptom_name: str,
        symptom_occurrences: List[SymptomLog]
    ) -> List[SideEffectCorrelation]:
        """
        Analyze correlation between medication and symptom occurrences
        Creates individual correlation records for each symptom-medication pair
        Uses temporal analysis and AI pattern matching
        """
        correlations = []
        
        for symptom in symptom_occurrences:
            # Find the most recent dosage change before this symptom
            relevant_change = None
            for change in sorted(dosage_changes, key=lambda x: x.change_date):
                if change.change_date <= symptom.reported_at:
                    relevant_change = change
                else:
                    break
            
            if not relevant_change:
                continue
            
            # Calculate time-to-onset
            time_diff = symptom.reported_at - relevant_change.change_date
            onset_hours = int(time_diff.total_seconds() / 3600)
            
            # Get AI analysis for this specific occurrence
            ai_analysis = self._ai_pattern_analysis(
                medication,
                symptom_name,
                [{"hours": onset_hours, "symptom": symptom, "change": relevant_change}],
                dosage_changes
            )
            
            # Create correlation record with ALL required fields (including analysis_date)
            correlation = SideEffectCorrelation(
                # Required IDs
                patient_id=medication.patient_id,
                medication_timeline_id=medication.id,
                symptom_log_id=symptom.id,
                dosage_change_id=relevant_change.id if relevant_change else None,
                
                # Required denormalized data
                medication_name=medication.medication_name,
                dosage_at_onset=relevant_change.new_dosage if relevant_change else medication.dosage,
                symptom_name=symptom_name,
                
                # Timing analysis
                time_to_onset_hours=onset_hours,
                symptom_onset_date=symptom.reported_at,
                medication_change_date=relevant_change.change_date if relevant_change else medication.started_at,
                
                # AI correlation analysis
                correlation_strength=ai_analysis["strength"],
                confidence_score=ai_analysis["confidence_score"],
                analysis_date=datetime.utcnow(),  # FIX: Required field!
                
                # Analysis details
                supporting_evidence=[
                    {
                        "type": "temporal_proximity",
                        "hours_after_change": onset_hours,
                        "change_reason": relevant_change.change_reason.value if relevant_change else "initial_dose"
                    }
                ],
                temporal_pattern=ai_analysis["temporal_pattern"],
                ai_reasoning=ai_analysis["ai_reasoning"],
                known_side_effect=False,  # TODO: Cross-reference with drug database
                similar_cases=[],  # TODO: Literature search
                
                # Clinical relevance
                patient_impact=self._assess_patient_impact(symptom.severity),
                action_recommended="Monitor and discuss with healthcare provider"
            )
            
            # Save to database
            self.db.add(correlation)
            correlations.append(correlation)
        
        if correlations:
            self.db.commit()
            for corr in correlations:
                self.db.refresh(corr)
        
        return correlations
    
    def _ai_pattern_analysis(
        self,
        medication: MedicationTimeline,
        symptom_name: str,
        onset_data: List[Dict],
        dosage_changes: List[DosageChange]
    ) -> Dict[str, Any]:
        """
        Use OpenAI to analyze medication-symptom pattern
        Returns correlation score, strength, and explanations
        """
        # Build context for AI
        onset_hours = [x["hours"] for x in onset_data]
        avg_onset = sum(onset_hours) / len(onset_hours)
        
        # Prepare prompt for AI analysis
        prompt = f"""Analyze this medication-symptom temporal pattern for change detection monitoring (NOT diagnosis):

Medication: {medication.medication_name}
Generic Name: {medication.generic_name or 'N/A'}
Drug Class: {medication.drug_class or 'N/A'}
Current Dosage: {medication.dosage}
Started: {medication.started_at.strftime('%Y-%m-%d')}

Symptom: {symptom_name}
Occurrences: {len(onset_data)}
Average onset after medication change: {avg_onset:.1f} hours
Onset times: {[round(h, 1) for h in onset_hours]} hours

Dosage Changes: {len(dosage_changes)} changes
Change reasons: {[change.change_reason.value for change in dosage_changes]}

Task: Assess the temporal correlation between this medication/dosage changes and symptom onset.
This is for CHANGE DETECTION and MONITORING purposes only, not diagnosis.

Provide analysis in this exact JSON format:
{{
  "confidence_score": <float 0.0-1.0>,
  "strength": "<STRONG|LIKELY|POSSIBLE|UNLIKELY>",
  "temporal_pattern": "<concise description of time-to-onset pattern>",
  "ai_reasoning": "<explanation of why this assessment was made, focusing on temporal evidence>"
}}

Guidelines:
- confidence_score 0.8-1.0 + strength "STRONG" = clear temporal correlation
- confidence_score 0.6-0.8 + strength "LIKELY" = probable correlation
- confidence_score 0.4-0.6 + strength "POSSIBLE" = weak correlation
- confidence_score 0.0-0.4 + strength "UNLIKELY" = no clear correlation
- Use MONITORING language, not diagnostic language
- Focus on temporal consistency and pattern strength
"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a medical data analyst specializing in temporal pattern detection for medication monitoring. Provide objective, data-driven assessments using change detection language, not diagnosis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent analysis
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Validate and normalize
            confidence_score = max(0.0, min(1.0, float(analysis.get("confidence_score", 0.5))))
            strength_map = {
                "STRONG": CorrelationStrength.STRONG,
                "LIKELY": CorrelationStrength.LIKELY,
                "POSSIBLE": CorrelationStrength.POSSIBLE,
                "UNLIKELY": CorrelationStrength.UNLIKELY
            }
            strength = strength_map.get(analysis.get("strength", "POSSIBLE"), CorrelationStrength.POSSIBLE)
            
            return {
                "confidence_score": confidence_score,
                "strength": strength,
                "temporal_pattern": analysis.get("temporal_pattern", "Temporal pattern analysis pending"),
                "ai_reasoning": analysis.get("ai_reasoning", "Analysis based on temporal consistency")
            }
            
        except Exception as e:
            # Fallback to rule-based scoring if AI fails
            return self._fallback_correlation_scoring(onset_data)
    
    def _fallback_correlation_scoring(self, onset_data: List[Dict]) -> Dict[str, Any]:
        """
        Fallback rule-based correlation scoring when AI is unavailable
        Uses simple statistical measures
        """
        if not onset_data:
            return {
                "confidence_score": 0.0,
                "strength": CorrelationStrength.UNLIKELY,
                "temporal_pattern": "Insufficient data for correlation analysis",
                "ai_reasoning": "No temporal pattern detected"
            }
        
        onset_hours = [x["hours"] for x in onset_data]
        avg_onset = sum(onset_hours) / len(onset_hours)
        
        # Calculate variance (how consistent is the onset time?)
        variance = sum((h - avg_onset) ** 2 for h in onset_hours) / len(onset_hours)
        std_dev = variance ** 0.5
        
        # Calculate confidence score based on consistency
        # Lower variance = higher confidence
        if std_dev < 6:  # Very consistent (within 6 hours)
            score = 0.85
            strength = CorrelationStrength.STRONG
        elif std_dev < 12:  # Moderately consistent
            score = 0.65
            strength = CorrelationStrength.LIKELY
        elif std_dev < 24:  # Somewhat consistent
            score = 0.45
            strength = CorrelationStrength.POSSIBLE
        else:  # Not consistent
            score = 0.25
            strength = CorrelationStrength.UNLIKELY
        
        # Adjust for number of occurrences (more data = more confidence)
        if len(onset_data) >= 5:
            score += 0.1
        elif len(onset_data) <= 2:
            score -= 0.1
        
        score = max(0.0, min(1.0, score))
        
        return {
            "confidence_score": score,
            "strength": strength,
            "temporal_pattern": f"Symptom appears {avg_onset:.1f} hours after medication changes on average (±{std_dev:.1f}h)",
            "ai_reasoning": f"Pattern consistency: {len(onset_data)} occurrences with temporal clustering"
        }
    
    def _assess_patient_impact(self, severity: Optional[int]) -> str:
        """
        Assess patient impact based on symptom severity
        Returns: "mild", "moderate", or "severe"
        """
        if not severity:
            return "mild"
        
        if severity <= 3:
            return "mild"
        elif severity <= 6:
            return "moderate"
        else:
            return "severe"
    
    def generate_patient_summary(
        self,
        patient_id: str,
        days_back: int = 90
    ) -> MedicationEffectsSummary:
        """
        Generate comprehensive medication effects summary for a patient
        Includes all correlations and analysis metadata
        """
        # Run correlation analysis
        correlations = self.analyze_patient_correlations(patient_id, days_back)
        
        # Group by medication
        medication_summaries = {}
        for correlation in correlations:
            med_id = correlation.medication_timeline_id
            if med_id not in medication_summaries:
                medication = self.db.query(MedicationTimeline).filter(
                    MedicationTimeline.id == med_id
                ).first()
                medication_summaries[med_id] = {
                    "medication_name": medication.medication_name,
                    "dosage": medication.dosage,
                    "correlations": []
                }
            
            medication_summaries[med_id]["correlations"].append({
                "symptom": correlation.symptom_name,
                "strength": correlation.correlation_strength.value,
                "score": correlation.confidence_score,
                "onset_hours": correlation.time_to_onset_hours
            })
        
        # Count strong correlations
        strong_correlations = sum(1 for c in correlations if c.correlation_strength == CorrelationStrength.STRONG)
        
        # Calculate average confidence for each medication
        medication_avg_confidence = {}
        for med_id, med_data in medication_summaries.items():
            if med_data["correlations"]:
                avg_conf = sum(c["score"] for c in med_data["correlations"]) / len(med_data["correlations"])
                medication_avg_confidence[med_id] = avg_conf
        
        # Create summary
        summary = MedicationEffectsSummary(
            patient_id=patient_id,
            analysis_period_days=days_back,
            total_correlations_found=len(correlations),
            strong_correlations_count=strong_correlations,
            summary_data={
                "medications_analyzed": len(medication_summaries),
                "medication_summaries": medication_summaries,
                "analysis_timestamp": datetime.utcnow().isoformat()
            },
            recommendations="Review correlations with healthcare provider for personalized guidance"
        )
        
        # Save to database
        self.db.add(summary)
        self.db.commit()
        self.db.refresh(summary)
        
        return summary
    
    def update_correlation_on_new_symptom(
        self,
        symptom_log: SymptomLog
    ) -> Optional[SideEffectCorrelation]:
        """
        Automatically update correlations when a new symptom is logged
        Checks if symptom relates to any active medications
        """
        # Get patient's active medications
        active_meds = self.db.query(MedicationTimeline).filter(
            and_(
                MedicationTimeline.patient_id == symptom_log.patient_id,
                MedicationTimeline.is_active == True
            )
        ).all()
        
        if not active_meds:
            return None
        
        # Check each medication for potential correlation
        for medication in active_meds:
            # Get recent dosage changes (last 30 days)
            cutoff = datetime.utcnow() - timedelta(days=30)
            recent_changes = self.db.query(DosageChange).filter(
                and_(
                    DosageChange.medication_timeline_id == medication.id,
                    DosageChange.change_date >= cutoff
                )
            ).all()
            
            if not recent_changes:
                continue
            
            # Find the most recent change before symptom
            relevant_change = None
            for change in sorted(recent_changes, key=lambda x: x.change_date):
                if change.change_date <= symptom_log.reported_at:
                    relevant_change = change
            
            if relevant_change:
                time_diff = symptom_log.reported_at - relevant_change.change_date
                hours_since_change = time_diff.total_seconds() / 3600
                
                # If symptom appears within 72 hours of medication change, flag for analysis
                if hours_since_change <= 72:
                    # Get all similar symptoms for this patient
                    similar_symptoms = self.db.query(SymptomLog).filter(
                        and_(
                            SymptomLog.patient_id == symptom_log.patient_id,
                            SymptomLog.symptom_name == symptom_log.symptom_name
                        )
                    ).all()
                    
                    # Re-analyze correlation
                    dosage_changes = self.db.query(DosageChange).filter(
                        DosageChange.medication_timeline_id == medication.id
                    ).all()
                    
                    correlation_list = self._analyze_single_correlation(
                        medication,
                        dosage_changes,
                        symptom_log.symptom_name,
                        similar_symptoms
                    )
                    
                    # Return the most recent correlation
                    return correlation_list[-1] if correlation_list else None
        
        return None
