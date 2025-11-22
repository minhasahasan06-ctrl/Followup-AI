"""
Symptom Check-in Service - Daily symptom tracking with conversational extraction.

Handles structured daily check-ins, conversational symptom extraction from Agent Clona,
and trend analysis for clinician review. All outputs are observational and non-diagnostic.

HIPAA Compliance:
- All symptom data is patient-owned and encrypted
- OpenAI GPT-4o extraction requires valid BAA
- Comprehensive audit logging for all operations
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func

from app.config import get_openai_client, check_openai_baa_compliance
from app.services.audit_logger import AuditLogger


class SymptomExtractionService:
    """Service for extracting symptoms from Agent Clona conversations using GPT-4o"""
    
    SYMPTOM_EXTRACTION_PROMPT = """You are a medical symptom extraction assistant for a HIPAA-compliant health monitoring platform.

Analyze the patient's conversational message and extract structured symptom information. Focus on:
- Body locations mentioned (e.g., "head", "chest", "abdomen")
- Symptom types (e.g., "pain", "nausea", "fatigue", "shortness of breath")
- Intensity mentions (e.g., "mild", "severe", "unbearable", numerical scales)
- Temporal information (e.g., "started yesterday", "for 3 days", "since morning")
- Aggravating factors (e.g., "worse when standing", "after eating")
- Relieving factors (e.g., "better with rest", "improves with medication")

Return ONLY valid JSON in this exact format:
{{
  "locations": ["location1", "location2"],
  "symptomTypes": ["symptom1", "symptom2"],
  "intensityMentions": ["descriptor1", "descriptor2"],
  "temporalInfo": "when/duration description",
  "aggravatingFactors": ["factor1", "factor2"],
  "relievingFactors": ["factor1", "factor2"]
}}

If no symptoms are mentioned, return empty arrays but keep the structure.

Patient message: {message}

Important: Return ONLY the JSON object, no additional text."""
    
    @staticmethod
    def extract_from_conversation(
        patient_id: str,
        message_text: str,
        session_id: str,
        message_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract symptom data from Agent Clona conversation message.
        
        Args:
            patient_id: User ID of patient
            message_text: Conversational message text
            session_id: Chat session ID
            message_id: Optional chat message ID
            
        Returns:
            Dictionary with extraction results and confidence score
        """
        try:
            # HIPAA compliance check
            check_openai_baa_compliance()
            
            # Call OpenAI for extraction (synchronous)
            client = get_openai_client()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise medical symptom extraction assistant. Always return valid JSON."
                    },
                    {
                        "role": "user",
                        "content": SymptomExtractionService.SYMPTOM_EXTRACTION_PROMPT.format(
                            message=message_text
                        )
                    }
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content
            if content:
                content = content.strip()
            else:
                content = "{}"
            
            # Clean up markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            extracted_json = json.loads(content)
            
            # Calculate confidence based on completeness
            confidence = SymptomExtractionService._calculate_confidence(extracted_json)
            
            return {
                "success": True,
                "extracted_json": extracted_json,
                "confidence": confidence,
                "model": "gpt-4o"
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": "Failed to parse extraction response",
                "confidence": 0.0
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "confidence": 0.0
            }
    
    @staticmethod
    def _calculate_confidence(extracted_json: Dict[str, Any]) -> float:
        """
        Calculate confidence score based on extraction completeness.
        
        Returns:
            Float between 0.0 and 1.0
        """
        score = 0.0
        max_score = 6.0
        
        # Check each field for content
        if extracted_json.get("symptomTypes") and len(extracted_json["symptomTypes"]) > 0:
            score += 2.0  # Symptom types are most important
        if extracted_json.get("locations") and len(extracted_json["locations"]) > 0:
            score += 1.0
        if extracted_json.get("intensityMentions") and len(extracted_json["intensityMentions"]) > 0:
            score += 1.0
        if extracted_json.get("temporalInfo"):
            score += 1.0
        if extracted_json.get("aggravatingFactors") and len(extracted_json["aggravatingFactors"]) > 0:
            score += 0.5
        if extracted_json.get("relievingFactors") and len(extracted_json["relievingFactors"]) > 0:
            score += 0.5
        
        return round(score / max_score, 2)


class SymptomTrendService:
    """Service for ML-based symptom trend analysis and anomaly detection"""
    
    @staticmethod
    def calculate_aggregated_metrics(
        symptom_checkins: List[Dict[str, Any]],
        passive_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate aggregated metrics from symptom check-ins and passive device data.
        
        Args:
            symptom_checkins: List of symptom check-in records
            passive_metrics: List of passive metric records (wearables, phone sensors)
            
        Returns:
            Dictionary with aggregated averages and trends
        """
        metrics = {
            "avgPainLevel": None,
            "avgFatigueLevel": None,
            "avgBreathlessness": None,
            "avgSleepQuality": None,
            "avgMobilityScore": None,
            "avgSteps": None,
            "avgHrMean": None,
            "avgHrv": None,
            "avgSleepMinutes": None,
            "avgSpo2": None,
            "topSymptoms": [],
            "topTriggers": []
        }
        
        # Process symptom check-ins
        if symptom_checkins:
            pain_values = [c["painLevel"] for c in symptom_checkins if c.get("painLevel") is not None]
            fatigue_values = [c["fatigueLevel"] for c in symptom_checkins if c.get("fatigueLevel") is not None]
            breathlessness_values = [c["breathlessnessLevel"] for c in symptom_checkins if c.get("breathlessnessLevel") is not None]
            sleep_values = [c["sleepQuality"] for c in symptom_checkins if c.get("sleepQuality") is not None]
            mobility_values = [c["mobilityScore"] for c in symptom_checkins if c.get("mobilityScore") is not None]
            
            if pain_values:
                metrics["avgPainLevel"] = round(sum(pain_values) / len(pain_values), 1)
            if fatigue_values:
                metrics["avgFatigueLevel"] = round(sum(fatigue_values) / len(fatigue_values), 1)
            if breathlessness_values:
                metrics["avgBreathlessness"] = round(sum(breathlessness_values) / len(breathlessness_values), 1)
            if sleep_values:
                metrics["avgSleepQuality"] = round(sum(sleep_values) / len(sleep_values), 1)
            if mobility_values:
                metrics["avgMobilityScore"] = round(sum(mobility_values) / len(mobility_values), 1)
            
            # Aggregate symptoms
            symptom_counts = {}
            trigger_counts = {}
            
            for checkin in symptom_checkins:
                if checkin.get("symptoms"):
                    for symptom in checkin["symptoms"]:
                        symptom_counts[symptom] = symptom_counts.get(symptom, 0) + 1
                if checkin.get("triggers"):
                    for trigger in checkin["triggers"]:
                        trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
            
            metrics["topSymptoms"] = [
                {"symptom": k, "frequency": v}
                for k, v in sorted(symptom_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            metrics["topTriggers"] = [
                {"trigger": k, "frequency": v}
                for k, v in sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        
        # Process passive metrics
        if passive_metrics:
            steps_values = [m["steps"] for m in passive_metrics if m.get("steps") is not None]
            hr_values = [m["hrMean"] for m in passive_metrics if m.get("hrMean") is not None]
            hrv_values = [m["hrv"] for m in passive_metrics if m.get("hrv") is not None]
            sleep_mins = [m["sleepMinutes"] for m in passive_metrics if m.get("sleepMinutes") is not None]
            spo2_values = [m["spo2Mean"] for m in passive_metrics if m.get("spo2Mean") is not None]
            
            if steps_values:
                metrics["avgSteps"] = int(sum(steps_values) / len(steps_values))
            if hr_values:
                metrics["avgHrMean"] = int(sum(hr_values) / len(hr_values))
            if hrv_values:
                metrics["avgHrv"] = int(sum(hrv_values) / len(hrv_values))
            if sleep_mins:
                metrics["avgSleepMinutes"] = int(sum(sleep_mins) / len(sleep_mins))
            if spo2_values:
                metrics["avgSpo2"] = int(sum(spo2_values) / len(spo2_values))
        
        return metrics
    
    @staticmethod
    def detect_anomalies(
        symptom_checkins: List[Dict[str, Any]],
        historical_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalies using simple statistical methods (Z-score).
        
        Args:
            symptom_checkins: Recent check-ins to analyze
            historical_data: Historical baseline data (30+ days recommended)
            
        Returns:
            List of detected anomalies with observational severity
        """
        anomalies = []
        
        # Metrics to analyze
        metrics_to_check = [
            ("painLevel", "Pain Level"),
            ("fatigueLevel", "Fatigue Level"),
            ("breathlessnessLevel", "Breathlessness"),
            ("sleepQuality", "Sleep Quality"),
            ("mobilityScore", "Mobility Score")
        ]
        
        for metric_key, metric_name in metrics_to_check:
            # Get historical values for baseline
            historical_values = [
                h[metric_key] for h in historical_data 
                if h.get(metric_key) is not None
            ]
            
            if len(historical_values) < 3:  # Need at least 3 data points for stats
                continue
            
            # Calculate baseline statistics
            mean = sum(historical_values) / len(historical_values)
            variance = sum((x - mean) ** 2 for x in historical_values) / len(historical_values)
            std_dev = variance ** 0.5
            
            if std_dev == 0:  # No variation in historical data
                continue
            
            # Check recent values for anomalies
            for checkin in symptom_checkins:
                value = checkin.get(metric_key)
                if value is None:
                    continue
                
                # Calculate Z-score
                z_score = (value - mean) / std_dev
                
                # Determine if anomalous (|Z| > 2 is commonly used threshold)
                if abs(z_score) > 2:
                    # Determine observational severity
                    if abs(z_score) > 3:
                        severity = "significant"
                    elif abs(z_score) > 2.5:
                        severity = "moderate"
                    else:
                        severity = "mild"
                    
                    # Create patient-friendly description
                    direction = "higher" if z_score > 0 else "lower"
                    description = f"{metric_name} was {direction} than usual ({value} vs typical {round(mean, 1)})"
                    
                    anomalies.append({
                        "metricName": metric_key,
                        "date": checkin["timestamp"],
                        "value": value,
                        "expectedRange": {
                            "min": round(mean - 2 * std_dev, 1),
                            "max": round(mean + 2 * std_dev, 1)
                        },
                        "severity": severity,
                        "description": description
                    })
        
        return anomalies
    
    @staticmethod
    def generate_clinician_summary(
        aggregated_metrics: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
        correlations: List[Dict[str, Any]],
        period_days: int
    ) -> str:
        """
        Generate plain-English clinician summary with observational language.
        
        Args:
            aggregated_metrics: Aggregated metric values
            anomalies: Detected anomalies
            correlations: Correlation insights
            period_days: Number of days in analysis period
            
        Returns:
            Human-readable summary string
        """
        summary_parts = []
        
        # Header
        summary_parts.append(f"**{period_days}-Day Symptom Observational Summary**")
        summary_parts.append("")
        summary_parts.append("*This is observational data collected from patient self-reports and does not constitute a diagnosis.*")
        summary_parts.append("")
        
        # Self-reported metrics
        summary_parts.append("**Self-Reported Symptom Averages:**")
        metrics_reported = []
        
        if aggregated_metrics.get("avgPainLevel") is not None:
            metrics_reported.append(f"- Pain: {aggregated_metrics['avgPainLevel']}/10")
        if aggregated_metrics.get("avgFatigueLevel") is not None:
            metrics_reported.append(f"- Fatigue: {aggregated_metrics['avgFatigueLevel']}/10")
        if aggregated_metrics.get("avgBreathlessness") is not None:
            metrics_reported.append(f"- Breathlessness: {aggregated_metrics['avgBreathlessness']}/10")
        if aggregated_metrics.get("avgSleepQuality") is not None:
            metrics_reported.append(f"- Sleep Quality: {aggregated_metrics['avgSleepQuality']}/10")
        if aggregated_metrics.get("avgMobilityScore") is not None:
            metrics_reported.append(f"- Mobility: {aggregated_metrics['avgMobilityScore']}/10")
        
        if metrics_reported:
            summary_parts.extend(metrics_reported)
        else:
            summary_parts.append("- No self-reported metrics in this period")
        summary_parts.append("")
        
        # Device metrics
        if aggregated_metrics.get("avgSteps") or aggregated_metrics.get("avgSleepMinutes"):
            summary_parts.append("**Device-Collected Data:**")
            if aggregated_metrics.get("avgSteps"):
                summary_parts.append(f"- Average Daily Steps: {aggregated_metrics['avgSteps']:,}")
            if aggregated_metrics.get("avgSleepMinutes"):
                hours = aggregated_metrics['avgSleepMinutes'] / 60
                summary_parts.append(f"- Average Sleep Duration: {hours:.1f} hours")
            if aggregated_metrics.get("avgHrMean"):
                summary_parts.append(f"- Average Heart Rate: {aggregated_metrics['avgHrMean']} bpm")
            if aggregated_metrics.get("avgHrv"):
                summary_parts.append(f"- Average HRV: {aggregated_metrics['avgHrv']} ms")
            summary_parts.append("")
        
        # Top symptoms
        if aggregated_metrics.get("topSymptoms"):
            summary_parts.append("**Most Frequently Reported Symptoms:**")
            for symptom_data in aggregated_metrics["topSymptoms"]:
                summary_parts.append(f"- {symptom_data['symptom']} ({symptom_data['frequency']}x)")
            summary_parts.append("")
        
        # Anomalies
        if anomalies:
            summary_parts.append("**Observational Pattern Changes:**")
            for anomaly in anomalies[:5]:  # Limit to top 5
                summary_parts.append(f"- {anomaly['description']} (observational {anomaly['severity']} deviation)")
            summary_parts.append("")
        
        # Correlations
        if correlations:
            summary_parts.append("**Observational Associations:**")
            for corr in correlations[:3]:  # Limit to top 3
                summary_parts.append(f"- {corr['observationalNote']}")
            summary_parts.append("")
        
        # Footer disclaimer
        summary_parts.append("*This summary is for informational and tracking purposes only. Clinical interpretation and decision-making remain with the healthcare provider.*")
        
        return "\n".join(summary_parts)
