"""
Condition Personalization Service
Provides disease-specific respiratory monitoring emphasis and wellness guidance
"""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from app.models import RespiratoryConditionProfile
import logging

logger = logging.getLogger(__name__)


class ConditionPersonalizationService:
    """
    Provides personalized respiratory monitoring based on patient's chronic conditions
    IMPORTANT: Wellness monitoring only - not medical diagnosis
    """
    
    # Disease-specific monitoring profiles
    CONDITION_PROFILES = {
        'asthma': {
            'display_name': 'Asthma',
            'emphasis': {
                'variability_index': 'high',  # Most important for asthma
                'accessory_muscles': 'high',
                'gasping': 'medium',
                'chest_asymmetry': 'low',
                'synchrony': 'medium',
                'baseline_offset': 0  # Normal baseline
            },
            'position_preference': 'sitting',  # Better airway clearance
            'monitoring_focus': 'Track breathing variability and effort. Variable patterns may indicate changes.',
            'rvi_thresholds': {
                'mild': 15.0,  # Lower threshold - asthma varies more
                'critical': 30.0
            },
            'rr_range': {'min': 12, 'max': 25},  # May be elevated during flares
            'wellness_guidance': 'Monitor for increased use of neck muscles and irregular breathing patterns. Consider noting environmental triggers.'
        },
        
        'copd': {
            'display_name': 'COPD',
            'emphasis': {
                'variability_index': 'medium',
                'accessory_muscles': 'high',  # Key indicator
                'gasping': 'medium',
                'chest_asymmetry': 'high',  # Barrel chest common
                'synchrony': 'medium',
                'baseline_offset': 3  # Often higher baseline RR
            },
            'position_preference': 'sitting',  # Tripod position helpful
            'monitoring_focus': 'Track accessory muscle use and chest shape changes. Prolonged effort may indicate progression.',
            'rvi_thresholds': {
                'mild': 20.0,
                'critical': 35.0
            },
            'rr_range': {'min': 12, 'max': 28},
            'wellness_guidance': 'Watch for increased neck muscle use and barrel-shaped chest changes. Regular monitoring helps track stability.'
        },
        
        'heart_failure': {
            'display_name': 'Heart Failure',
            'emphasis': {
                'variability_index': 'medium',
                'accessory_muscles': 'medium',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'high',  # Important for HF
                'baseline_offset': 2  # Often slightly elevated
            },
            'position_preference': 'sitting',  # Reduce fluid burden
            'monitoring_focus': 'Track breathing coordination and rate trends. Gradual increases may indicate fluid changes.',
            'rvi_thresholds': {
                'mild': 20.0,
                'critical': 40.0
            },
            'rr_range': {'min': 12, 'max': 26},
            'wellness_guidance': 'Monitor breathing rate trends over days. Gradual increases combined with reduced coordination may suggest fluid retention.'
        },
        
        'pulmonary_embolism': {
            'display_name': 'Pulmonary Embolism',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'medium',
                'gasping': 'high',  # Sudden breathlessness
                'chest_asymmetry': 'low',
                'synchrony': 'medium',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track for sudden breathing rate changes. Rapid increases require immediate medical attention.',
            'rvi_thresholds': {
                'mild': 20.0,
                'critical': 40.0
            },
            'rr_range': {'min': 12, 'max': 30},
            'sudden_change_threshold': 6.0,  # Alert if RR changes >6 bpm in 30 min
            'wellness_guidance': 'Monitor for sudden increases in breathing rate or gasping patterns. Sudden changes warrant immediate medical evaluation.'
        },
        
        'pneumonia': {
            'display_name': 'Pneumonia',
            'emphasis': {
                'variability_index': 'medium',
                'accessory_muscles': 'high',
                'gasping': 'medium',
                'chest_asymmetry': 'medium',
                'synchrony': 'medium',
                'baseline_offset': 4  # Elevated during infection
            },
            'position_preference': 'sitting',  # Easier breathing
            'monitoring_focus': 'Track sustained elevation in breathing rate and effort. Improvement should occur with recovery.',
            'rvi_thresholds': {
                'mild': 20.0,
                'critical': 40.0
            },
            'rr_range': {'min': 14, 'max': 30},
            'wellness_guidance': 'Monitor breathing rate and effort trends. Sustained elevation or worsening may indicate need for medical review.'
        },
        
        'pulmonary_tb': {
            'display_name': 'Pulmonary TB',
            'emphasis': {
                'variability_index': 'medium',
                'accessory_muscles': 'medium',
                'gasping': 'low',
                'chest_asymmetry': 'medium',
                'synchrony': 'medium',
                'baseline_offset': 2
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track chronic breathing patterns and gradual changes. Long-term trends important.',
            'rvi_thresholds': {
                'mild': 20.0,
                'critical': 40.0
            },
            'rr_range': {'min': 12, 'max': 28},
            'wellness_guidance': 'Monitor breathing patterns during treatment. Gradual improvement expected with effective therapy.'
        },
        
        'bronchiectasis': {
            'display_name': 'Bronchiectasis',
            'emphasis': {
                'variability_index': 'medium',
                'accessory_muscles': 'high',
                'gasping': 'medium',
                'chest_asymmetry': 'medium',
                'synchrony': 'medium',
                'baseline_offset': 2
            },
            'position_preference': 'sitting',  # Better drainage
            'monitoring_focus': 'Track breathing effort and variability. Changes may indicate exacerbation.',
            'rvi_thresholds': {
                'mild': 20.0,
                'critical': 40.0
            },
            'rr_range': {'min': 12, 'max': 28},
            'wellness_guidance': 'Monitor for increased breathing effort and irregular patterns, especially during productive cough periods.'
        },
        
        'allergic_reaction': {
            'display_name': 'Allergic Reactions',
            'emphasis': {
                'variability_index': 'high',
                'accessory_muscles': 'high',
                'gasping': 'high',  # Critical for severe reactions
                'chest_asymmetry': 'low',
                'synchrony': 'low',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track for sudden onset breathing changes. Rapid worsening requires emergency care.',
            'rvi_thresholds': {
                'mild': 15.0,
                'critical': 30.0
            },
            'rr_range': {'min': 12, 'max': 30},
            'sudden_change_threshold': 8.0,  # Very sensitive
            'wellness_guidance': 'Monitor for sudden breathing difficulty, especially with known allergen exposure. Severe reactions are medical emergencies.'
        }
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_patient_conditions(self, patient_id: str) -> List[str]:
        """Get active respiratory conditions for patient"""
        profiles = self.db.query(RespiratoryConditionProfile).filter(
            RespiratoryConditionProfile.patient_id == patient_id,
            RespiratoryConditionProfile.active == True
        ).all()
        
        return [p.condition for p in profiles]
    
    def get_personalized_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized respiratory monitoring configuration
        Combines multiple conditions if patient has several
        """
        conditions = self.get_patient_conditions(patient_id)
        
        if not conditions:
            return self._get_default_config()
        
        # Start with first condition's profile
        primary_condition = conditions[0]
        config = self.CONDITION_PROFILES.get(primary_condition, self._get_default_config())
        
        # Merge additional conditions (take max emphasis)
        if len(conditions) > 1:
            config = self._merge_condition_profiles([
                self.CONDITION_PROFILES.get(c, {}) for c in conditions
            ])
        
        config['conditions'] = conditions
        return config
    
    def get_examination_instructions(self, patient_id: str) -> Dict[str, Any]:
        """
        Get position and setup instructions for examination
        """
        config = self.get_personalized_config(patient_id)
        position = config.get('position_preference', 'sitting')
        
        return {
            'position': position,
            'instructions': self._get_position_instructions(position, config.get('conditions', [])),
            'duration': '60-90 seconds',
            'focus_areas': self._get_focus_areas(config),
            'wellness_guidance': config.get('wellness_guidance', '')
        }
    
    def _get_position_instructions(self, position: str, conditions: List[str]) -> str:
        """Generate position-specific instructions"""
        base_instructions = {
            'sitting': """
**Sitting Position (Recommended):**
1. Sit upright in a chair with back support
2. Feet flat on the floor
3. Hands resting on thighs or armrests
4. Relax shoulders - avoid tensing
5. Breathe naturally for 60-90 seconds
6. Camera positioned at chest level

**Why sitting:** Provides most accurate respiratory rate measurement with clear chest movement visibility.
            """,
            'lying': """
**Lying Down Position:**
1. Lie flat on your back on a firm surface
2. Arms relaxed at your sides
3. Head on a thin pillow (optional)
4. Relax body completely
5. Breathe naturally for 60-90 seconds
6. Camera positioned above chest

**Note:** Lying down may reduce some breathing effort signals but can show chest expansion clearly.
            """
        }
        
        instructions = base_instructions.get(position, base_instructions['sitting'])
        
        # Add condition-specific notes
        if 'copd' in conditions or 'heart_failure' in conditions:
            instructions += "\n**For your condition:** Sitting position helps reduce breathing effort and provides clearer measurements."
        
        if 'asthma' in conditions:
            instructions += "\n**For asthma monitoring:** Sitting upright allows better airway clearance and more accurate effort detection."
        
        return instructions.strip()
    
    def _get_focus_areas(self, config: Dict[str, Any]) -> List[str]:
        """Get ordered list of monitoring priorities"""
        emphasis = config.get('emphasis', {})
        
        # Map to user-friendly descriptions
        focus_map = {
            'variability_index': 'Breathing rhythm stability',
            'accessory_muscles': 'Neck and shoulder muscle use',
            'gasping': 'Irregular breathing patterns',
            'chest_asymmetry': 'Chest shape and expansion',
            'synchrony': 'Breathing coordination'
        }
        
        # Sort by emphasis level
        emphasis_order = {'high': 3, 'medium': 2, 'low': 1}
        sorted_areas = sorted(
            emphasis.items(),
            key=lambda x: emphasis_order.get(x[1], 0),
            reverse=True
        )
        
        return [focus_map.get(area, area) for area, level in sorted_areas if level in ['high', 'medium']]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration for patients without specific conditions"""
        return {
            'display_name': 'General Respiratory Wellness',
            'emphasis': {
                'variability_index': 'medium',
                'accessory_muscles': 'medium',
                'gasping': 'medium',
                'chest_asymmetry': 'medium',
                'synchrony': 'medium',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track general breathing patterns and trends over time.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 20},
            'wellness_guidance': 'Regular monitoring helps establish your normal breathing baseline.',
            'conditions': []
        }
    
    def _merge_condition_profiles(self, profiles: List[Dict]) -> Dict[str, Any]:
        """Merge multiple condition profiles (take max emphasis)"""
        merged = profiles[0].copy()
        
        for profile in profiles[1:]:
            # Merge emphasis (take higher level)
            emphasis_order = {'low': 1, 'medium': 2, 'high': 3}
            for key, value in profile.get('emphasis', {}).items():
                current = merged['emphasis'].get(key, 'low')
                if emphasis_order.get(value, 0) > emphasis_order.get(current, 0):
                    merged['emphasis'][key] = value
            
            # Take lower RVI thresholds (more sensitive)
            if 'rvi_thresholds' in profile:
                merged['rvi_thresholds']['mild'] = min(
                    merged['rvi_thresholds']['mild'],
                    profile['rvi_thresholds']['mild']
                )
        
        merged['display_name'] = 'Multiple Conditions'
        merged['monitoring_focus'] = 'Comprehensive monitoring for multiple respiratory conditions.'
        
        return merged
    
    def get_alert_message(
        self,
        patient_id: str,
        metric_type: str,
        value: float,
        severity: str
    ) -> str:
        """
        Generate wellness-compliant alert message
        IMPORTANT: No diagnosis - only wellness trend observations
        """
        config = self.get_personalized_config(patient_id)
        conditions = config.get('conditions', [])
        
        # Base templates (wellness language only)
        templates = {
            'rr_elevated': {
                'mild': "Your breathing rate is slightly above your usual range. Consider noting any activities or symptoms.",
                'critical': "Your breathing rate is notably above your usual range. Consider contacting your healthcare provider if this persists."
            },
            'rvi_high': {
                'mild': "Your breathing rhythm shows more variation than usual. This may be normal, but monitor for changes.",
                'critical': "Your breathing pattern shows significant variation. Consider discussing this with your care team."
            },
            'accessory_muscles': {
                'mild': "Increased neck muscle use detected during breathing. Note if you're feeling any breathing effort.",
                'critical': "Significant neck muscle use detected. This may indicate breathing effort - consider medical evaluation if concerning."
            },
            'gasping': {
                'mild': "Irregular breathing pattern detected. This may be temporary, but monitor how you feel.",
                'critical': "Significant irregular breathing detected. If you feel short of breath, seek immediate medical attention."
            }
        }
        
        message = templates.get(metric_type, {}).get(severity, "Breathing pattern change detected.")
        
        # Add condition-specific context (wellness language only)
        if 'pulmonary_embolism' in conditions or 'allergic_reaction' in conditions:
            if severity == 'critical':
                message += " Given your medical history, sudden breathing changes warrant prompt medical evaluation."
        
        if 'copd' in conditions or 'asthma' in conditions:
            if metric_type == 'accessory_muscles':
                message += " Increased breathing effort may indicate a flare. Consider your action plan."
        
        return message
