"""
Mental Health Questionnaire Service

Provides comprehensive mental health screening with:
- Standardized questionnaire templates (PHQ-9, GAD-7, PSS-10)
- Validated scoring algorithms
- Crisis detection and intervention
- LLM-based pattern recognition and symptom clustering
- Non-diagnostic neutral summaries
- Temporal trend analysis

CRITICAL: All summaries use neutral, non-diagnostic language.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc
import openai
import os

from app.models.mental_health_models import MentalHealthResponse, MentalHealthPatternAnalysis
from app.models.user import User

logger = logging.getLogger(__name__)


class MentalHealthService:
    """Service for mental health questionnaire management and analysis"""
    
    def __init__(self, db: Session = None):
        self.db = db
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # ==================== Questionnaire Templates ====================
    
    def get_all_questionnaire_templates(self) -> List[Dict[str, Any]]:
        """Return all available questionnaire templates"""
        return [
            self._get_phq9_template(),
            self._get_gad7_template(),
            self._get_pss10_template()
        ]
    
    def get_questionnaire_template(self, questionnaire_type: str) -> Dict[str, Any]:
        """Get a specific questionnaire template"""
        templates = {
            "PHQ9": self._get_phq9_template(),
            "GAD7": self._get_gad7_template(),
            "PSS10": self._get_pss10_template()
        }
        return templates.get(questionnaire_type)
    
    def _get_phq9_template(self) -> Dict[str, Any]:
        """
        Patient Health Questionnaire-9 (PHQ-9)
        Public domain depression screening tool
        """
        return {
            "type": "PHQ9",
            "full_name": "Patient Health Questionnaire-9",
            "description": "A 9-item self-reported measure used to screen for depression and monitor symptom severity.",
            "source": "Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9.",
            "public_domain": True,
            "timeframe": "Over the last 2 weeks",
            "instructions": "Over the last 2 weeks, how often have you been bothered by any of the following problems?",
            "response_options": [
                {"value": 0, "label": "Not at all"},
                {"value": 1, "label": "Several days"},
                {"value": 2, "label": "More than half the days"},
                {"value": 3, "label": "Nearly every day"}
            ],
            "questions": [
                {
                    "id": "PHQ9_1",
                    "text": "Little interest or pleasure in doing things",
                    "cluster": "anhedonia"
                },
                {
                    "id": "PHQ9_2",
                    "text": "Feeling down, depressed, or hopeless",
                    "cluster": "mood"
                },
                {
                    "id": "PHQ9_3",
                    "text": "Trouble falling or staying asleep, or sleeping too much",
                    "cluster": "sleep"
                },
                {
                    "id": "PHQ9_4",
                    "text": "Feeling tired or having little energy",
                    "cluster": "energy"
                },
                {
                    "id": "PHQ9_5",
                    "text": "Poor appetite or overeating",
                    "cluster": "appetite"
                },
                {
                    "id": "PHQ9_6",
                    "text": "Feeling bad about yourself - or that you are a failure or have let yourself or your family down",
                    "cluster": "self_worth"
                },
                {
                    "id": "PHQ9_7",
                    "text": "Trouble concentrating on things, such as reading the newspaper or watching television",
                    "cluster": "concentration"
                },
                {
                    "id": "PHQ9_8",
                    "text": "Moving or speaking so slowly that other people could have noticed. Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual",
                    "cluster": "psychomotor"
                },
                {
                    "id": "PHQ9_9",
                    "text": "Thoughts that you would be better off dead, or of hurting yourself in some way",
                    "cluster": "self_harm",
                    "crisis_flag": True
                }
            ],
            "scoring": {
                "max_score": 27,
                "severity_levels": [
                    {"range": [0, 4], "level": "minimal", "description": "Minimal symptoms"},
                    {"range": [5, 9], "level": "mild", "description": "Mild symptoms"},
                    {"range": [10, 14], "level": "moderate", "description": "Moderate symptoms"},
                    {"range": [15, 19], "level": "moderately_severe", "description": "Moderately severe symptoms"},
                    {"range": [20, 27], "level": "severe", "description": "Severe symptoms"}
                ]
            },
            "clusters": {
                "mood": {"label": "Low Mood", "items": ["PHQ9_1", "PHQ9_2"]},
                "sleep": {"label": "Sleep Disruption", "items": ["PHQ9_3"]},
                "energy": {"label": "Energy Level", "items": ["PHQ9_4"]},
                "appetite": {"label": "Appetite Changes", "items": ["PHQ9_5"]},
                "self_worth": {"label": "Self-Perception", "items": ["PHQ9_6"]},
                "concentration": {"label": "Focus & Attention", "items": ["PHQ9_7"]},
                "psychomotor": {"label": "Movement & Activity", "items": ["PHQ9_8"]}
            }
        }
    
    def _get_gad7_template(self) -> Dict[str, Any]:
        """
        Generalized Anxiety Disorder-7 (GAD-7)
        Public domain anxiety screening tool
        """
        return {
            "type": "GAD7",
            "full_name": "Generalized Anxiety Disorder-7",
            "description": "A 7-item self-reported measure used to screen for anxiety and monitor symptom severity.",
            "source": "Spitzer, R. L., Kroenke, K., Williams, J. B., & Löwe, B. (2006). The GAD-7.",
            "public_domain": True,
            "timeframe": "Over the last 2 weeks",
            "instructions": "Over the last 2 weeks, how often have you been bothered by the following problems?",
            "response_options": [
                {"value": 0, "label": "Not at all"},
                {"value": 1, "label": "Several days"},
                {"value": 2, "label": "More than half the days"},
                {"value": 3, "label": "Nearly every day"}
            ],
            "questions": [
                {
                    "id": "GAD7_1",
                    "text": "Feeling nervous, anxious, or on edge",
                    "cluster": "nervousness"
                },
                {
                    "id": "GAD7_2",
                    "text": "Not being able to stop or control worrying",
                    "cluster": "worry"
                },
                {
                    "id": "GAD7_3",
                    "text": "Worrying too much about different things",
                    "cluster": "worry"
                },
                {
                    "id": "GAD7_4",
                    "text": "Trouble relaxing",
                    "cluster": "relaxation"
                },
                {
                    "id": "GAD7_5",
                    "text": "Being so restless that it is hard to sit still",
                    "cluster": "restlessness"
                },
                {
                    "id": "GAD7_6",
                    "text": "Becoming easily annoyed or irritable",
                    "cluster": "irritability"
                },
                {
                    "id": "GAD7_7",
                    "text": "Feeling afraid, as if something awful might happen",
                    "cluster": "fear"
                }
            ],
            "scoring": {
                "max_score": 21,
                "severity_levels": [
                    {"range": [0, 4], "level": "minimal", "description": "Minimal symptoms"},
                    {"range": [5, 9], "level": "mild", "description": "Mild symptoms"},
                    {"range": [10, 14], "level": "moderate", "description": "Moderate symptoms"},
                    {"range": [15, 21], "level": "severe", "description": "Severe symptoms"}
                ]
            },
            "clusters": {
                "nervousness": {"label": "Nervousness", "items": ["GAD7_1"]},
                "worry": {"label": "Worry & Rumination", "items": ["GAD7_2", "GAD7_3"]},
                "relaxation": {"label": "Difficulty Relaxing", "items": ["GAD7_4"]},
                "restlessness": {"label": "Physical Restlessness", "items": ["GAD7_5"]},
                "irritability": {"label": "Irritability", "items": ["GAD7_6"]},
                "fear": {"label": "Anticipatory Fear", "items": ["GAD7_7"]}
            }
        }
    
    def _get_pss10_template(self) -> Dict[str, Any]:
        """
        Perceived Stress Scale-10 (PSS-10)
        Public domain stress assessment tool
        """
        return {
            "type": "PSS10",
            "full_name": "Perceived Stress Scale-10",
            "description": "A 10-item self-reported measure of perceived stress over the past month.",
            "source": "Cohen, S., Kamarck, T., & Mermelstein, R. (1983). The PSS-10.",
            "public_domain": True,
            "timeframe": "In the last month",
            "instructions": "In the last month, how often have you felt or experienced the following?",
            "response_options": [
                {"value": 0, "label": "Never"},
                {"value": 1, "label": "Almost never"},
                {"value": 2, "label": "Sometimes"},
                {"value": 3, "label": "Fairly often"},
                {"value": 4, "label": "Very often"}
            ],
            "questions": [
                {
                    "id": "PSS10_1",
                    "text": "Been upset because of something that happened unexpectedly",
                    "cluster": "perceived_stress",
                    "reverse_scored": False
                },
                {
                    "id": "PSS10_2",
                    "text": "Felt that you were unable to control the important things in your life",
                    "cluster": "lack_of_control",
                    "reverse_scored": False
                },
                {
                    "id": "PSS10_3",
                    "text": "Felt nervous and stressed",
                    "cluster": "perceived_stress",
                    "reverse_scored": False
                },
                {
                    "id": "PSS10_4",
                    "text": "Felt confident about your ability to handle your personal problems",
                    "cluster": "coping",
                    "reverse_scored": True
                },
                {
                    "id": "PSS10_5",
                    "text": "Felt that things were going your way",
                    "cluster": "coping",
                    "reverse_scored": True
                },
                {
                    "id": "PSS10_6",
                    "text": "Found that you could not cope with all the things that you had to do",
                    "cluster": "lack_of_control",
                    "reverse_scored": False
                },
                {
                    "id": "PSS10_7",
                    "text": "Been able to control irritations in your life",
                    "cluster": "coping",
                    "reverse_scored": True
                },
                {
                    "id": "PSS10_8",
                    "text": "Felt that you were on top of things",
                    "cluster": "coping",
                    "reverse_scored": True
                },
                {
                    "id": "PSS10_9",
                    "text": "Been angered because of things that were outside of your control",
                    "cluster": "perceived_stress",
                    "reverse_scored": False
                },
                {
                    "id": "PSS10_10",
                    "text": "Felt difficulties were piling up so high that you could not overcome them",
                    "cluster": "lack_of_control",
                    "reverse_scored": False
                }
            ],
            "scoring": {
                "max_score": 40,
                "severity_levels": [
                    {"range": [0, 13], "level": "low", "description": "Low perceived stress"},
                    {"range": [14, 26], "level": "moderate", "description": "Moderate perceived stress"},
                    {"range": [27, 40], "level": "high", "description": "High perceived stress"}
                ]
            },
            "clusters": {
                "perceived_stress": {"label": "Stress Reactions", "items": ["PSS10_1", "PSS10_3", "PSS10_9"]},
                "lack_of_control": {"label": "Feeling Overwhelmed", "items": ["PSS10_2", "PSS10_6", "PSS10_10"]},
                "coping": {"label": "Coping Ability", "items": ["PSS10_4", "PSS10_5", "PSS10_7", "PSS10_8"]}
            }
        }
    
    # ==================== Scoring Algorithms ====================
    
    def score_questionnaire(self, questionnaire_type: str, responses: List[Any]) -> Dict[str, Any]:
        """
        Score questionnaire using validated algorithms.
        Returns: {total_score, max_score, severity_level, severity_description, cluster_scores, neutral_summary, key_observations}
        """
        template = self.get_questionnaire_template(questionnaire_type)
        if not template:
            raise ValueError(f"Unknown questionnaire type: {questionnaire_type}")
        
        # Calculate total score
        total_score = 0
        response_dict = {r.question_id: r for r in responses}
        
        # Handle reverse scoring for PSS-10
        for question in template['questions']:
            q_id = question['id']
            if q_id in response_dict:
                response_val = response_dict[q_id].response
                
                # Apply reverse scoring if applicable
                if question.get('reverse_scored', False):
                    max_val = max([opt['value'] for opt in template['response_options']])
                    response_val = max_val - response_val
                
                total_score += response_val
        
        max_score = template['scoring']['max_score']
        
        # Determine severity level
        severity_level = "unknown"
        severity_description = ""
        for level_info in template['scoring']['severity_levels']:
            if level_info['range'][0] <= total_score <= level_info['range'][1]:
                severity_level = level_info['level']
                severity_description = level_info['description']
                break
        
        # Calculate cluster scores
        cluster_scores = {}
        if 'clusters' in template:
            for cluster_name, cluster_info in template['clusters'].items():
                cluster_total = 0
                cluster_max = 0
                for item_id in cluster_info['items']:
                    if item_id in response_dict:
                        cluster_total += response_dict[item_id].response
                        cluster_max += max([opt['value'] for opt in template['response_options']])
                
                cluster_scores[cluster_name] = {
                    "score": cluster_total,
                    "maxScore": cluster_max,
                    "label": cluster_info['label'],
                    "items": cluster_info['items']
                }
        
        # Generate neutral summary
        neutral_summary = self._generate_neutral_summary(
            questionnaire_type, total_score, max_score, severity_level, cluster_scores, responses
        )
        
        # Generate key observations
        key_observations = self._generate_key_observations(
            questionnaire_type, cluster_scores, responses
        )
        
        return {
            "total_score": total_score,
            "max_score": max_score,
            "severity_level": severity_level,
            "severity_description": severity_description,
            "cluster_scores": cluster_scores,
            "neutral_summary": neutral_summary,
            "key_observations": key_observations
        }
    
    def _generate_neutral_summary(
        self, 
        questionnaire_type: str, 
        total_score: int, 
        max_score: int, 
        severity_level: str,
        cluster_scores: Dict[str, Any],
        responses: List[Any]
    ) -> str:
        """Generate neutral, non-diagnostic summary"""
        
        # Find high-scoring clusters
        high_clusters = []
        for cluster_name, cluster_data in cluster_scores.items():
            if cluster_data['score'] >= (cluster_data['maxScore'] * 0.6):
                high_clusters.append(cluster_data['label'])
        
        summaries = {
            "PHQ9": {
                "minimal": "You reported minimal symptoms across the assessed areas.",
                "mild": f"You reported experiencing some symptoms, particularly in areas like {', '.join(high_clusters[:2])}." if high_clusters else "You reported experiencing some symptoms in various areas.",
                "moderate": f"You reported moderate levels of symptoms, especially related to {', '.join(high_clusters[:3])}." if high_clusters else "You reported moderate levels of symptoms across various areas.",
                "moderately_severe": f"You reported frequent symptoms across multiple areas, notably {', '.join(high_clusters[:3])}." if high_clusters else "You reported frequent symptoms across multiple areas.",
                "severe": "You reported experiencing symptoms very frequently across multiple areas."
            },
            "GAD7": {
                "minimal": "You reported minimal symptoms related to worry or nervousness.",
                "mild": f"You reported some symptoms, particularly in areas like {', '.join(high_clusters[:2])}." if high_clusters else "You reported some symptoms related to worry or nervousness.",
                "moderate": f"You reported moderate levels of symptoms, especially {', '.join(high_clusters[:2])}." if high_clusters else "You reported moderate levels of symptoms.",
                "severe": "You reported frequent symptoms across multiple areas related to worry and nervousness."
            },
            "PSS10": {
                "low": "You reported low levels of perceived stress over the past month.",
                "moderate": f"You reported moderate stress levels, particularly in areas like {', '.join(high_clusters[:2])}." if high_clusters else "You reported moderate stress levels.",
                "high": f"You reported high levels of perceived stress, especially related to {', '.join(high_clusters[:3])}." if high_clusters else "You reported high levels of perceived stress."
            }
        }
        
        return summaries.get(questionnaire_type, {}).get(severity_level, "Summary unavailable")
    
    def _generate_key_observations(
        self,
        questionnaire_type: str,
        cluster_scores: Dict[str, Any],
        responses: List[Any]
    ) -> List[str]:
        """Generate key non-diagnostic observations"""
        observations = []
        
        # Identify significant clusters (> 60% of max)
        for cluster_name, cluster_data in cluster_scores.items():
            percentage = (cluster_data['score'] / cluster_data['maxScore']) * 100 if cluster_data['maxScore'] > 0 else 0
            
            if percentage >= 60:
                observations.append(f"Higher reported frequency in {cluster_data['label']}")
        
        # Add frequency observations
        high_frequency_count = sum(1 for r in responses if isinstance(r.response, int) and r.response >= 2)
        if high_frequency_count >= len(responses) * 0.5:
            observations.append("Symptoms reported as occurring more than half the time or more frequently")
        
        return observations[:5]  # Limit to top 5 observations
    
    def get_severity_description(self, questionnaire_type: str, severity_level: str) -> str:
        """Get human-readable severity description"""
        template = self.get_questionnaire_template(questionnaire_type)
        if not template:
            return severity_level
        
        for level_info in template['scoring']['severity_levels']:
            if level_info['level'] == severity_level:
                return level_info['description']
        return severity_level
    
    # ==================== Crisis Detection ====================
    
    def detect_crisis(
        self, 
        questionnaire_type: str, 
        responses: List[Any],
        total_score: int
    ) -> Dict[str, Any]:
        """
        Detect crisis indicators requiring immediate intervention.
        Returns: {crisis_detected, severity, message, next_steps, question_ids, responses}
        """
        template = self.get_questionnaire_template(questionnaire_type)
        crisis_flags = []
        crisis_responses = []
        
        # Check for crisis flag questions
        for question in template['questions']:
            if question.get('crisis_flag', False):
                for response in responses:
                    if response.question_id == question['id']:
                        # Any non-zero response to crisis question is flagged
                        if isinstance(response.response, int) and response.response > 0:
                            crisis_flags.append(question['id'])
                            crisis_responses.append({
                                "questionId": response.question_id,
                                "questionText": response.question_text,
                                "response": response.response
                            })
        
        # Determine crisis severity
        crisis_detected = len(crisis_flags) > 0
        crisis_severity = "none"
        
        if crisis_detected:
            # Check response value for PHQ9-9 (self-harm question)
            for cr in crisis_responses:
                response_val = cr['response']
                if response_val == 3:  # Nearly every day
                    crisis_severity = "severe"
                elif response_val == 2:  # More than half the days
                    crisis_severity = "high"
                else:
                    crisis_severity = "moderate"
        
        # Generate intervention message
        intervention_messages = {
            "severe": (
                "Your responses indicate you've been having thoughts of self-harm nearly every day. "
                "This is serious and you should seek help immediately. Please contact a mental health "
                "professional, call 988 (Suicide & Crisis Lifeline), or go to your nearest emergency room."
            ),
            "high": (
                "Your responses indicate you've been having thoughts of self-harm more than half the days. "
                "It's important to talk to a mental health professional soon. Consider calling 988 "
                "(Suicide & Crisis Lifeline) or contacting your healthcare provider."
            ),
            "moderate": (
                "Your responses indicate you've had some thoughts of self-harm. Please consider reaching out "
                "to a mental health professional or calling 988 (Suicide & Crisis Lifeline) to talk through "
                "these feelings."
            )
        }
        
        next_steps = {
            "severe": [
                "Call 988 (Suicide & Crisis Lifeline) immediately for 24/7 support",
                "Go to your nearest emergency room if you feel unsafe",
                "Reach out to a trusted friend or family member",
                "Contact your healthcare provider or therapist urgently"
            ],
            "high": [
                "Call 988 (Suicide & Crisis Lifeline) to speak with a trained counselor",
                "Schedule an urgent appointment with a mental health professional",
                "Reach out to supportive friends or family members",
                "Remove access to means of self-harm if possible"
            ],
            "moderate": [
                "Consider calling 988 (Suicide & Crisis Lifeline) to talk through your feelings",
                "Schedule an appointment with a mental health professional",
                "Talk to someone you trust about how you're feeling",
                "Engage in activities that have helped you feel better in the past"
            ]
        }
        
        return {
            "crisis_detected": crisis_detected,
            "severity": crisis_severity,
            "message": intervention_messages.get(crisis_severity, ""),
            "next_steps": next_steps.get(crisis_severity, []),
            "question_ids": crisis_flags,
            "crisis_responses": crisis_responses
        }
    
    # ==================== Database Operations ====================
    
    def save_questionnaire_response(
        self,
        patient_id: str,
        questionnaire_type: str,
        responses: List[Dict[str, Any]],
        score_result: Dict[str, Any],
        crisis_result: Dict[str, Any],
        duration_seconds: Optional[int],
        allow_clinical_sharing: bool
    ) -> MentalHealthResponse:
        """Save questionnaire response to database"""
        
        response_record = MentalHealthResponse(
            patient_id=patient_id,
            questionnaire_type=questionnaire_type,
            responses=responses,
            total_score=score_result['total_score'],
            max_score=score_result['max_score'],
            severity_level=score_result['severity_level'],
            cluster_scores=score_result['cluster_scores'],
            crisis_detected=crisis_result['crisis_detected'],
            crisis_question_ids=crisis_result['question_ids'],
            crisis_responses=crisis_result['crisis_responses'],
            duration_seconds=duration_seconds,
            allow_storage=True,
            allow_clinical_sharing=allow_clinical_sharing
        )
        
        self.db.add(response_record)
        self.db.commit()
        self.db.refresh(response_record)
        
        logger.info(f"[MH-SERVICE] Saved response {response_record.id} for patient {patient_id}")
        
        return response_record
    
    # ==================== LLM Pattern Analysis ====================
    
    async def generate_pattern_analysis(
        self,
        patient_id: str,
        response_id: str,
        questionnaire_type: str,
        responses: List[Any],
        score_result: Dict[str, Any]
    ) -> Optional[MentalHealthPatternAnalysis]:
        """
        Generate LLM-based pattern analysis with symptom clustering.
        CRITICAL: Uses strictly non-diagnostic language.
        """
        logger.info(f"[MH-SERVICE] Generating pattern analysis for response {response_id}")
        
        try:
            # Get historical responses for temporal analysis
            historical_responses = self.db.query(MentalHealthResponse).filter(
                MentalHealthResponse.patient_id == patient_id,
                MentalHealthResponse.questionnaire_type == questionnaire_type
            ).order_by(desc(MentalHealthResponse.completed_at)).limit(5).all()
            
            # Prepare context for LLM
            current_responses_text = "\n".join([
                f"- {r.question_text}: {r.response_text or r.response}" 
                for r in responses
            ])
            
            historical_scores = [
                {
                    "date": hr.completed_at.isoformat(),
                    "score": hr.total_score,
                    "severity": hr.severity_level
                }
                for hr in historical_responses
            ]
            
            # Build LLM prompt - STRICTLY NON-DIAGNOSTIC
            prompt = f"""Analyze the following mental health questionnaire responses and provide a neutral, non-diagnostic summary.

CRITICAL RULES:
1. Use ONLY neutral, observational language
2. NEVER use diagnostic terms (depression, anxiety disorder, PTSD, etc.)
3. Describe REPORTED symptoms, not conditions
4. Focus on patterns, clusters, and trends
5. Suggest general wellness actions, not treatment

Questionnaire Type: {questionnaire_type}
Current Responses:
{current_responses_text}

Score: {score_result['total_score']}/{score_result['max_score']}
Cluster Scores: {score_result['cluster_scores']}

Historical Scores (if any): {historical_scores}

Provide analysis in the following JSON format:
{{
  "symptom_clusters": {{
    "cluster_name": {{
      "symptoms": ["list", "of", "symptoms"],
      "frequency": "occasional/frequent/very_frequent",
      "severity": "mild/moderate/high",
      "neutralDescription": "Neutral description of this symptom group"
    }}
  }},
  "temporal_trends": [
    {{
      "metric": "overall_score",
      "direction": "improving/worsening/stable/fluctuating",
      "magnitude": "slight/moderate/significant",
      "timeframe": "description"
    }}
  ],
  "neutral_summary": "Comprehensive neutral summary of reported symptoms WITHOUT diagnostic language",
  "key_observations": ["observation1", "observation2"],
  "suggested_actions": [
    {{
      "category": "sleep/stress/social/physical/professional",
      "action": "Suggested wellness action",
      "priority": "low/medium/high"
    }}
  ]
}}

Remember: Describe what was REPORTED, not what someone "has" or "suffers from"."""
            
            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mental health data analyst who provides strictly neutral, non-diagnostic summaries of self-reported symptoms. Never use clinical diagnostic language."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            # Parse response
            import json
            analysis_data = json.loads(response.choices[0].message.content)
            tokens_used = response.usage.total_tokens
            
            # Save to database
            analysis_record = MentalHealthPatternAnalysis(
                patient_id=patient_id,
                response_id=response_id,
                analysis_type="llm_insights",
                patterns=[],  # Could be extended
                symptom_clusters=analysis_data.get("symptom_clusters", {}),
                temporal_trends=analysis_data.get("temporal_trends", []),
                neutral_summary=analysis_data.get("neutral_summary", ""),
                key_observations=analysis_data.get("key_observations", []),
                suggested_actions=analysis_data.get("suggested_actions", []),
                llm_model="gpt-4o",
                llm_tokens_used=tokens_used
            )
            
            self.db.add(analysis_record)
            self.db.commit()
            self.db.refresh(analysis_record)
            
            logger.info(f"[MH-SERVICE] Pattern analysis completed: {analysis_record.id}")
            
            return analysis_record
            
        except Exception as e:
            logger.error(f"[MH-SERVICE] Pattern analysis failed: {str(e)}", exc_info=True)
            return None
    
    # ==================== Temporal Trends ====================
    
    def calculate_temporal_trends(self, responses: List[MentalHealthResponse]) -> Dict[str, Any]:
        """Calculate trends across multiple questionnaire responses"""
        if len(responses) < 2:
            return None
        
        # Sort by date (oldest to newest)
        sorted_responses = sorted(responses, key=lambda r: r.completed_at)
        
        # Calculate overall trend
        first_score = sorted_responses[0].total_score
        last_score = sorted_responses[-1].total_score
        score_change = last_score - first_score
        
        direction = "stable"
        if score_change > 3:
            direction = "worsening"
        elif score_change < -3:
            direction = "improving"
        elif abs(score_change) <= 3:
            direction = "stable"
        
        # Calculate variability
        scores = [r.total_score for r in sorted_responses]
        avg_score = sum(scores) / len(scores)
        variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
        
        return {
            "overall_trend": {
                "direction": direction,
                "score_change": score_change,
                "first_score": first_score,
                "last_score": last_score,
                "timeframe_days": (sorted_responses[-1].completed_at - sorted_responses[0].completed_at).days
            },
            "variability": {
                "average_score": round(avg_score, 1),
                "variance": round(variance, 2),
                "pattern": "stable" if variance < 10 else "fluctuating"
            },
            "data_points": [
                {
                    "date": r.completed_at.isoformat(),
                    "score": r.total_score,
                    "severity": r.severity_level
                }
                for r in sorted_responses
            ]
        }
