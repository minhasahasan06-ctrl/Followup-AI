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
    
    # Disease-specific monitoring profiles (respiratory + edema)
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
            'wellness_guidance': 'Monitor for increased use of neck muscles and irregular breathing patterns. Consider noting environmental triggers.',
            'edema': {
                'priority': 'low',  # Not primary concern for asthma
                'expected_pattern': None,
                'focus_locations': [],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling not typically associated with asthma alone.'
            }
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
            'wellness_guidance': 'Watch for increased neck muscle use and barrel-shaped chest changes. Regular monitoring helps track stability.',
            'edema': {
                'priority': 'medium',  # Can occur with advanced COPD
                'expected_pattern': 'bilateral',
                'focus_locations': ['legs', 'ankles'],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Leg swelling in COPD may indicate heart strain. Track changes and discuss with healthcare provider.'
            }
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
            'wellness_guidance': 'Monitor breathing rate trends over days. Gradual increases combined with reduced coordination may suggest fluid retention.',
            'edema': {
                'priority': 'critical',  # VERY important for heart failure
                'expected_pattern': 'bilateral',  # Both legs typically
                'focus_locations': ['legs', 'ankles', 'feet'],
                'pei_thresholds': {'mild': 8.0, 'critical': 20.0},  # More sensitive
                'pitting_watchpoints': [2, 3, 4],  # Any pitting is significant
                'wellness_guidance': 'Bilateral leg swelling is an important wellness indicator for heart health. Increasing swelling may suggest fluid retention. Track daily and discuss trends with your care team.'
            }
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
            'wellness_guidance': 'Monitor for sudden increases in breathing rate or gasping patterns. Sudden changes warrant immediate medical evaluation.',
            'edema': {
                'priority': 'high',  # Unilateral leg swelling is risk factor
                'expected_pattern': 'unilateral',  # Often ONE leg
                'focus_locations': ['legs'],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'asymmetry_alert_threshold': 0.30,  # Alert if >30% difference
                'wellness_guidance': 'Unilateral (one-sided) leg swelling may be a concern, especially if sudden. Combined with breathing changes, seek immediate medical evaluation.'
            }
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
            'wellness_guidance': 'Monitor breathing rate and effort trends. Sustained elevation or worsening may indicate need for medical review.',
            'edema': {
                'priority': 'low',  # Not primary concern
                'expected_pattern': None,
                'focus_locations': [],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling not typically associated with pneumonia alone.'
            }
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
            'wellness_guidance': 'Monitor breathing patterns during treatment. Gradual improvement expected with effective therapy.',
            'edema': {
                'priority': 'low',
                'expected_pattern': None,
                'focus_locations': [],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling not typically associated with pulmonary TB.'
            }
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
            'wellness_guidance': 'Monitor for increased breathing effort and irregular patterns, especially during productive cough periods.',
            'edema': {
                'priority': 'low',
                'expected_pattern': None,
                'focus_locations': [],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling not typically associated with bronchiectasis.'
            }
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
            'wellness_guidance': 'Monitor for sudden breathing difficulty, especially with known allergen exposure. Severe reactions are medical emergencies.',
            'edema': {
                'priority': 'critical',  # Facial swelling is URGENT
                'expected_pattern': 'facial',  # Face, lips, tongue
                'focus_locations': ['face'],
                'pei_thresholds': {'mild': 5.0, 'critical': 15.0},  # Very sensitive
                'pitting_watchpoints': [1, 2, 3, 4],
                'sudden_change_alert': True,  # Any sudden face swelling
                'wellness_guidance': 'Sudden facial swelling, especially lips/tongue, can be a medical emergency. Seek immediate care if rapid or with breathing difficulty.'
            }
        },
        
        # Additional conditions with significant edema profiles
        'kidney_disease': {
            'display_name': 'Kidney Disease',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'low',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'medium',
                'baseline_offset': 1
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track for fluid-related breathing changes.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 24},
            'wellness_guidance': 'Breathing changes may indicate fluid retention. Monitor regularly.',
            'edema': {
                'priority': 'critical',  # Primary manifestation
                'expected_pattern': 'bilateral + facial',  # Both legs AND face
                'focus_locations': ['face', 'legs', 'ankles', 'feet'],  # Multiple sites
                'pei_thresholds': {'mild': 8.0, 'critical': 20.0},
                'pitting_watchpoints': [2, 3, 4],
                'wellness_guidance': 'Swelling in face (especially around eyes) and legs is important to track with kidney wellness. Morning facial puffiness and leg swelling by evening are key patterns. Track daily weight and discuss trends with your care team.'
            }
        },
        
        'liver_disease': {
            'display_name': 'Liver Disease',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'low',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'medium',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Monitor breathing stability.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 24},
            'wellness_guidance': 'Track breathing patterns as part of overall wellness monitoring.',
            'edema': {
                'priority': 'critical',
                'expected_pattern': 'bilateral + ascites',  # Legs + abdominal
                'focus_locations': ['legs', 'ankles', 'feet'],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'pitting_watchpoints': [2, 3, 4],
                'wellness_guidance': 'Leg swelling and abdominal fullness may indicate fluid retention. Monitor for progressive swelling and discuss with your hepatologist.'
            }
        },
        
        'thyroid_disorder': {
            'display_name': 'Thyroid Disorder',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'low',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'low',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track baseline respiratory patterns.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 24},
            'wellness_guidance': 'Monitor breathing as part of overall thyroid wellness tracking.',
            'edema': {
                'priority': 'high',
                'expected_pattern': 'facial',  # Periorbital (around eyes)
                'focus_locations': ['face'],  # Primarily facial
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'pitting_watchpoints': [1, 2],  # Usually mild pitting
                'wellness_guidance': 'Facial puffiness (especially around eyes) may relate to thyroid function. Morning swelling that improves during the day is a common pattern. Track and discuss with your endocrinologist.'
            }
        },
        
        'lymphedema': {
            'display_name': 'Lymphedema',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'low',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'low',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Breathing monitoring not primary concern.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 20},
            'wellness_guidance': 'Focus on limb wellness monitoring.',
            'edema': {
                'priority': 'critical',  # Primary condition
                'expected_pattern': 'unilateral',  # Usually one limb
                'focus_locations': ['legs', 'feet', 'hands'],
                'pei_thresholds': {'mild': 5.0, 'critical': 15.0},  # Very sensitive
                'pitting_watchpoints': [1, 2],  # Often non-pitting or mild
                'asymmetry_alert_threshold': 0.20,  # Alert if >20% difference
                'wellness_guidance': 'Unilateral (one-sided) limb swelling with tight, shiny skin is characteristic. Track circumference changes, skin texture, and any hardening. Early detection of changes helps with management. Discuss with lymphedema specialist.'
            }
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
    
    # === EDEMA-SPECIFIC METHODS ===
    
    def get_edema_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized edema monitoring configuration
        Combines edema sections from multiple conditions
        """
        conditions = self.get_patient_conditions(patient_id)
        
        if not conditions:
            return self._get_default_edema_config()
        
        # Collect edema profiles
        edema_profiles = []
        for condition in conditions:
            profile = self.CONDITION_PROFILES.get(condition, {})
            if 'edema' in profile:
                edema_profiles.append(profile['edema'])
        
        if not edema_profiles:
            return self._get_default_edema_config()
        
        # Merge profiles if multiple conditions
        if len(edema_profiles) == 1:
            config = edema_profiles[0].copy()
        else:
            config = self._merge_edema_profiles(edema_profiles)
        
        config['conditions'] = conditions
        return config
    
    def get_edema_examination_focus(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized edema examination instructions and focus areas
        """
        edema_config = self.get_edema_config(patient_id)
        conditions = edema_config.get('conditions', [])
        
        # Determine priority
        priority = edema_config.get('priority', 'low')
        
        # Build focus message
        if priority == 'critical':
            importance_msg = "🔴 CRITICAL: Swelling monitoring is very important for your wellness tracking."
        elif priority == 'high':
            importance_msg = "🟠 HIGH: Swelling monitoring is an important part of your wellness tracking."
        elif priority == 'medium':
            importance_msg = "🟡 MODERATE: Swelling monitoring may provide useful wellness information."
        else:
            importance_msg = "🟢 LOW: Swelling monitoring included for completeness."
        
        # Get focus locations
        focus_locations = edema_config.get('focus_locations', ['legs', 'ankles'])
        expected_pattern = edema_config.get('expected_pattern', None)
        
        # Build examination instructions
        instructions = []
        
        if 'face' in focus_locations:
            instructions.append("📸 FACE: Front view + side views (30 sec)")
        
        if any(loc in focus_locations for loc in ['legs', 'ankles', 'feet']):
            instructions.append("📸 LEGS/ANKLES: Show both sides for symmetry comparison (30 sec)")
        
        if 'hands' in focus_locations:
            instructions.append("📸 HANDS: Show both hands (20 sec)")
        
        # Add pitting test if high priority
        if priority in ['critical', 'high']:
            instructions.append("👆 PITTING TEST (Recommended): Press swollen area 5-15 sec, observe rebound")
        else:
            instructions.append("👆 PITTING TEST (Optional): Press swollen area 5-15 sec if swelling present")
        
        # Expected pattern guidance
        pattern_guidance = ""
        if expected_pattern == 'bilateral':
            pattern_guidance = "⚠️ Pay special attention to both sides (bilateral swelling)"
        elif expected_pattern == 'unilateral':
            pattern_guidance = "⚠️ Pay special attention to one-sided swelling (unilateral)"
        elif expected_pattern == 'facial':
            pattern_guidance = "⚠️ Pay special attention to facial swelling, especially around eyes"
        elif expected_pattern == 'bilateral + facial':
            pattern_guidance = "⚠️ Pay special attention to face AND leg swelling"
        
        return {
            'priority': priority,
            'importance_message': importance_msg,
            'focus_locations': focus_locations,
            'expected_pattern': expected_pattern,
            'examination_instructions': instructions,
            'pattern_guidance': pattern_guidance,
            'wellness_guidance': edema_config.get('wellness_guidance', ''),
            'pei_thresholds': edema_config.get('pei_thresholds', {'mild': 10.0, 'critical': 25.0})
        }
    
    def _get_default_edema_config(self) -> Dict[str, Any]:
        """Default edema configuration for patients without specific conditions"""
        return {
            'priority': 'low',
            'expected_pattern': None,
            'focus_locations': ['legs', 'ankles'],
            'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
            'pitting_watchpoints': [],
            'wellness_guidance': 'General swelling monitoring for wellness tracking.',
            'conditions': []
        }
    
    def _merge_edema_profiles(self, profiles: List[Dict]) -> Dict[str, Any]:
        """
        Merge multiple edema profiles
        Takes highest priority, most sensitive thresholds, combined locations
        """
        # Priority levels
        priority_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        
        # Start with highest priority profile
        sorted_profiles = sorted(
            profiles,
            key=lambda p: priority_order.get(p.get('priority', 'low'), 0),
            reverse=True
        )
        
        merged = sorted_profiles[0].copy()
        
        # Merge locations (combine all unique locations)
        all_locations = set(merged.get('focus_locations', []))
        for profile in sorted_profiles[1:]:
            all_locations.update(profile.get('focus_locations', []))
        merged['focus_locations'] = list(all_locations)
        
        # Take most sensitive PEI thresholds (lowest values)
        for profile in sorted_profiles[1:]:
            if 'pei_thresholds' in profile:
                merged['pei_thresholds']['mild'] = min(
                    merged['pei_thresholds'].get('mild', 10.0),
                    profile['pei_thresholds'].get('mild', 10.0)
                )
                merged['pei_thresholds']['critical'] = min(
                    merged['pei_thresholds'].get('critical', 25.0),
                    profile['pei_thresholds'].get('critical', 25.0)
                )
        
        # Merge pitting watchpoints
        all_watchpoints = set(merged.get('pitting_watchpoints', []))
        for profile in sorted_profiles[1:]:
            all_watchpoints.update(profile.get('pitting_watchpoints', []))
        merged['pitting_watchpoints'] = sorted(list(all_watchpoints))
        
        # Combine wellness guidance
        guidance_parts = [merged.get('wellness_guidance', '')]
        for profile in sorted_profiles[1:]:
            guidance = profile.get('wellness_guidance', '')
            if guidance and guidance not in guidance_parts:
                guidance_parts.append(guidance)
        merged['wellness_guidance'] = ' '.join(guidance_parts)
        
        return merged
