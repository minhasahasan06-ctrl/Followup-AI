"""
Condition Personalization Service
Provides disease-specific respiratory monitoring emphasis and wellness guidance
"""

from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
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
            },
            'skin_analysis': {
                'priority': 'high',
                'key_indicators': ['pallor', 'perfusion_index', 'capillary_refill'],
                'perfusion_thresholds': {
                    'facial': {'mild': 42.0, 'moderate': 32.0, 'severe': 22.0},  # Reduced in poor cardiac output
                    'palmar': {'mild': 38.0, 'moderate': 28.0, 'severe': 18.0},
                    'nailbed': {'mild': 35.0, 'moderate': 25.0, 'severe': 15.0}
                },
                'pallor_regions_priority': ['palmar', 'nailbed', 'facial'],
                'capillary_refill': {
                    'threshold_sec': 2.0,
                    'concern_sec': 3.5,  # Prolonged in reduced cardiac output
                    'monitoring': 'high'
                },
                'wellness_guidance': 'Pale skin and prolonged capillary refill may indicate reduced circulation from heart function changes. Cool extremities combined with pallor warrant discussion with your cardiology team.'
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
            },
            'skin_analysis': {
                'priority': 'high',
                'key_indicators': ['pallor', 'hydration', 'texture', 'uremic_frost'],
                'perfusion_thresholds': {
                    'facial': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0},  # Pallor from anemia
                    'palmar': {'mild': 38.0, 'moderate': 28.0, 'severe': 18.0}
                },
                'pallor_regions_priority': ['palmar', 'facial', 'nailbed'],
                'hydration_status_monitoring': 'critical',  # Dry skin common
                'texture_monitoring': 'high',  # Rough, dry texture
                'uremic_frost_detection': True,  # White crystals on skin (advanced)
                'wellness_guidance': 'Pale, dry skin may relate to kidney function and associated anemia. Very dry, rough skin texture is common. In advanced stages, white powdery deposits (uremic frost) may appear. Track skin changes and discuss with your nephrologist.'
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
            },
            'skin_analysis': {
                'priority': 'critical',  # Jaundice is KEY indicator
                'key_indicators': ['jaundice', 'scleral_yellowing', 'perfusion_index'],
                'jaundice_monitoring': {
                    'priority': 'critical',
                    'b_channel_threshold': 20.0,  # High b* = yellow
                    'severity_levels': {'mild': 25.0, 'moderate': 35.0, 'severe': 45.0},
                    'regions': ['facial', 'sclera']  # Sclera most sensitive
                },
                'perfusion_thresholds': {
                    'facial': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0},
                    'palmar': {'mild': 38.0, 'moderate': 28.0, 'severe': 18.0}
                },
                'wellness_guidance': 'Yellowish discoloration of skin and especially the whites of eyes (sclera) may indicate changes in liver function. Progressive yellowing warrants discussion with your hepatologist. This is a key wellness indicator for liver health.'
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
            },
            'skin_analysis': {
                'priority': 'medium',
                'key_indicators': ['hydration', 'texture', 'temperature_proxy'],
                'perfusion_thresholds': {
                    'facial': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0}
                },
                'hydration_status_monitoring': 'high',  # Hypothyroid: dry; Hyperthyroid: moist
                'texture_monitoring': 'high',  # Rough (hypo) vs smooth (hyper)
                'temperature_proxy_monitoring': 'high',  # Cool (hypo) vs warm (hyper)
                'wellness_guidance': 'Skin texture, moisture, and temperature may reflect thyroid function. Hypothyroidism: dry, rough, cool skin. Hyperthyroidism: moist, smooth, warm skin. Track patterns and discuss with your endocrinologist.'
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
        },
        
        # NEW: Skin-focused conditions
        'anemia': {
            'display_name': 'Anemia',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'low',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'low',
                'baseline_offset': 1  # May be slightly elevated
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Monitor for signs of reduced oxygen delivery.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 24},
            'wellness_guidance': 'Track breathing rate and skin color changes.',
            'edema': {
                'priority': 'low',
                'expected_pattern': None,
                'focus_locations': [],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling not typically associated with anemia.'
            },
            'skin_analysis': {
                'priority': 'critical',  # PRIMARY diagnostic indicator
                'key_indicators': ['pallor', 'perfusion_index', 'nailbed_color'],
                'perfusion_thresholds': {
                    'facial': {'mild': 45.0, 'moderate': 35.0, 'severe': 25.0},  # Low perfusion
                    'palmar': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0},  # Gold standard
                    'nailbed': {'mild': 35.0, 'moderate': 25.0, 'severe': 15.0}
                },
                'pallor_regions_priority': ['palmar', 'nailbed', 'facial'],  # Palm most reliable
                'capillary_refill': {
                    'threshold_sec': 2.0,  # Normal <2s
                    'concern_sec': 3.0,  # Prolonged in severe anemia
                    'monitoring': 'important'
                },
                'wellness_guidance': 'Pale skin (especially palms and nail beds) and reduced skin color may indicate low iron levels. Palmar pallor is a key indicator. Track changes and discuss with your healthcare provider, especially if accompanied by fatigue.'
            }
        },
        
        'sepsis': {
            'display_name': 'Sepsis Risk',
            'emphasis': {
                'variability_index': 'high',
                'accessory_muscles': 'high',
                'gasping': 'high',
                'chest_asymmetry': 'low',
                'synchrony': 'medium',
                'baseline_offset': 5  # Significantly elevated
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Track for rapid deterioration. Urgent medical evaluation needed.',
            'rvi_thresholds': {'mild': 15.0, 'critical': 25.0},  # Very sensitive
            'rr_range': {'min': 16, 'max': 35},  # Often >20 bpm
            'sudden_change_threshold': 5.0,
            'wellness_guidance': 'Sepsis requires immediate medical attention. This wellness tool is not a substitute for emergency care.',
            'edema': {
                'priority': 'medium',
                'expected_pattern': 'variable',
                'focus_locations': ['legs', 'ankles'],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Fluid shifts can occur rapidly. Medical monitoring required.'
            },
            'skin_analysis': {
                'priority': 'critical',
                'key_indicators': ['perfusion_pattern', 'temperature_proxy', 'mottling'],
                'perfusion_thresholds': {
                    'facial': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0},  # Mottled, poor perfusion
                    'palmar': {'mild': 35.0, 'moderate': 25.0, 'severe': 15.0}
                },
                'mottling_detection': True,  # Patchy perfusion pattern
                'perfusion_variability_alert': 15.0,  # High std dev = mottling
                'temperature_proxy_monitoring': 'critical',  # Cool extremities
                'capillary_refill': {
                    'threshold_sec': 2.0,
                    'concern_sec': 3.0,
                    'monitoring': 'critical'  # Prolonged refill = shock
                },
                'wellness_guidance': 'Mottled (patchy) skin coloration, cool extremities, and prolonged capillary refill are urgent warning signs. This is a medical emergency - seek immediate care. This wellness tool cannot diagnose sepsis.'
            }
        },
        
        'raynauds': {
            'display_name': "Raynaud's Phenomenon",
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
            'wellness_guidance': 'Focus on extremity color and temperature monitoring.',
            'edema': {
                'priority': 'low',
                'expected_pattern': None,
                'focus_locations': [],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling not typically associated with Raynaud\'s.'
            },
            'skin_analysis': {
                'priority': 'critical',  # PRIMARY manifestation
                'key_indicators': ['cyanosis', 'perfusion_pattern', 'temperature_proxy'],
                'cyanosis_monitoring': {
                    'priority': 'critical',
                    'regions': ['nailbed', 'fingertips', 'facial'],  # Extremities first
                    'severity_thresholds': {'mild': 0.3, 'moderate': 0.5, 'severe': 0.7}  # Severity score
                },
                'perfusion_thresholds': {
                    'nailbed': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0},  # Reduced during attack
                    'facial': {'mild': 45.0, 'moderate': 35.0, 'severe': 25.0}
                },
                'temperature_proxy_monitoring': 'critical',  # Cool extremities during attack
                'triphasic_pattern_detection': True,  # White→Blue→Red phases
                'wellness_guidance': 'Bluish discoloration (cyanosis) of fingertips and nails, especially in cold environments, is characteristic. Track color changes through warming/cooling cycles. Episodes typically resolve with warmth. Discuss patterns with your rheumatologist.'
            }
        },
        
        'diabetes': {
            'display_name': 'Diabetes',
            'emphasis': {
                'variability_index': 'low',
                'accessory_muscles': 'low',
                'gasping': 'low',
                'chest_asymmetry': 'low',
                'synchrony': 'low',
                'baseline_offset': 0
            },
            'position_preference': 'sitting',
            'monitoring_focus': 'Monitor for overall wellness stability.',
            'rvi_thresholds': {'mild': 20.0, 'critical': 40.0},
            'rr_range': {'min': 12, 'max': 22},
            'wellness_guidance': 'Regular monitoring helps track overall health status.',
            'edema': {
                'priority': 'medium',
                'expected_pattern': 'bilateral',  # Legs if neuropathy/kidney involvement
                'focus_locations': ['legs', 'ankles', 'feet'],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Leg swelling may indicate kidney or circulation changes. Track and discuss with your diabetes care team.'
            },
            'skin_analysis': {
                'priority': 'high',
                'key_indicators': ['capillary_refill', 'perfusion_index', 'ulcer_detection', 'hydration'],
                'perfusion_thresholds': {
                    'facial': {'mild': 40.0, 'moderate': 30.0, 'severe': 20.0},
                    'palmar': {'mild': 38.0, 'moderate': 28.0, 'severe': 18.0},
                    'nailbed': {'mild': 35.0, 'moderate': 25.0, 'severe': 15.0}  # Reduced in PVD
                },
                'capillary_refill': {
                    'threshold_sec': 2.5,  # Slightly prolonged acceptable
                    'concern_sec': 4.0,  # Significant peripheral vascular disease
                    'monitoring': 'critical'  # Key indicator of microvascular health
                },
                'ulcer_monitoring': True,  # Skin breakdown detection
                'hydration_status_monitoring': 'high',  # Dry skin common
                'texture_monitoring': 'high',  # Rough, dry texture
                'wellness_guidance': 'Prolonged capillary refill (>3 seconds) and reduced foot perfusion may indicate circulation changes. Dry, rough skin and any breaks in skin integrity (ulcers) require prompt attention. Perform regular foot checks and discuss changes with your diabetes care team.'
            }
        },
        
        'peripheral_vascular_disease': {
            'display_name': 'Peripheral Vascular Disease',
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
            'wellness_guidance': 'Focus on extremity perfusion monitoring.',
            'edema': {
                'priority': 'medium',
                'expected_pattern': 'bilateral',
                'focus_locations': ['legs', 'ankles', 'feet'],
                'pei_thresholds': {'mild': 10.0, 'critical': 25.0},
                'wellness_guidance': 'Swelling combined with poor circulation requires medical evaluation.'
            },
            'skin_analysis': {
                'priority': 'critical',  # PRIMARY diagnostic indicator
                'key_indicators': ['capillary_refill', 'perfusion_index', 'pallor', 'ulcer_detection'],
                'perfusion_thresholds': {
                    'facial': {'mild': 45.0, 'moderate': 35.0, 'severe': 25.0},  # Usually normal
                    'palmar': {'mild': 35.0, 'moderate': 25.0, 'severe': 15.0},  # Reduced
                    'nailbed': {'mild': 30.0, 'moderate': 20.0, 'severe': 10.0}  # VERY reduced
                },
                'capillary_refill': {
                    'threshold_sec': 3.0,  # Often prolonged
                    'concern_sec': 5.0,  # Severe disease
                    'monitoring': 'critical'  # Key diagnostic finding
                },
                'pallor_regions_priority': ['nailbed', 'palmar', 'facial'],  # Extremities most affected
                'temperature_proxy_monitoring': 'critical',  # Cool extremities
                'ulcer_monitoring': True,  # Non-healing wounds
                'texture_monitoring': 'high',  # Shiny, hairless skin
                'wellness_guidance': 'Prolonged capillary refill (>3 seconds), pale or bluish extremities, and cool skin temperature indicate reduced circulation. Any skin breaks or non-healing wounds require immediate medical attention. Regular monitoring helps detect changes early. Discuss with your vascular specialist.'
            }
        }
    }
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_patient_conditions(self, patient_id: str) -> List[str]:
        """
        Get active conditions for patient
        Uses database query if available, otherwise returns stub data
        
        FIX ISSUE C: Explicit logging when personalization cannot be determined
        """
        # Try database query first
        try:
            from app.models import RespiratoryConditionProfile  # type: ignore
            profiles = self.db.query(RespiratoryConditionProfile).filter(
                RespiratoryConditionProfile.patient_id == patient_id,
                RespiratoryConditionProfile.active == True
            ).all()
            if profiles:
                conditions = [p.condition for p in profiles]
                logger.info(
                    f"✓ PERSONALIZATION: Patient {patient_id} conditions loaded from "
                    f"RespiratoryConditionProfile: {conditions}"
                )
                return conditions
        except Exception as e:
            logger.warning(
                f"⚠ PERSONALIZATION: RespiratoryConditionProfile query failed for "
                f"patient {patient_id}: {e}"
            )
        
        # Fallback: Check patient table for conditions
        try:
            from app.models import Patient  # type: ignore
            patient = self.db.query(Patient).filter(Patient.id == patient_id).first()
            if patient and hasattr(patient, 'conditions'):
                # Return conditions if patient has them
                if isinstance(patient.conditions, list):
                    logger.info(
                        f"✓ PERSONALIZATION: Patient {patient_id} conditions loaded from "
                        f"Patient model (list): {patient.conditions}"
                    )
                    return patient.conditions
                elif isinstance(patient.conditions, str):
                    conditions = [c.strip() for c in patient.conditions.split(',')]
                    logger.info(
                        f"✓ PERSONALIZATION: Patient {patient_id} conditions loaded from "
                        f"Patient model (string): {conditions}"
                    )
                    return conditions
        except Exception as e:
            logger.warning(
                f"⚠ PERSONALIZATION: Patient model query failed for patient {patient_id}: {e}"
            )
        
        # FIX ISSUE C: Explicit alert when personalization cannot be determined
        logger.error(
            f"❌ PERSONALIZATION ALERT: Cannot determine conditions for patient {patient_id}. "
            f"Using default thresholds (general population). This may reduce clinical accuracy "
            f"for disease-specific monitoring. Action required: Ensure patient has conditions "
            f"recorded in RespiratoryConditionProfile or Patient.conditions field."
        )
        return []  # Empty for general population (will use default thresholds)
    
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
    
    def get_guided_exam_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized thresholds for guided video examination
        Returns hepatic and anemia monitoring configs based on patient conditions
        """
        conditions = self.get_patient_conditions(patient_id)
        
        # Get hepatic and anemia configs
        hepatic_config = self.get_hepatic_monitoring_config(patient_id)
        anemia_config = self.get_anemia_monitoring_config(patient_id)
        
        return {
            'hepatic': hepatic_config,
            'anemia': anemia_config,
            'conditions': conditions,
            'examination_stages': {
                'eyes': {
                    'purpose': 'Sclera examination for jaundice detection',
                    'priority': hepatic_config['priority'],
                    'key_metrics': ['scleral_chromaticity_index', 'sclera_b_channel', 'sclera_yellowness_ratio']
                },
                'palm': {
                    'purpose': 'Palm examination for anemia detection',
                    'priority': anemia_config['priority'],
                    'key_metrics': ['conjunctival_pallor_index', 'palmar_redness_a', 'palmar_l_channel']
                },
                'tongue': {
                    'purpose': 'Tongue examination for coating and color',
                    'priority': 'medium',
                    'key_metrics': ['tongue_color_index', 'tongue_coating_detected', 'tongue_coating_color']
                },
                'lips': {
                    'purpose': 'Lip examination for hydration and cyanosis',
                    'priority': 'medium',
                    'key_metrics': ['lip_hydration_score', 'lip_cyanosis_detected', 'lip_color_uniformity']
                }
            }
        }
    
    def get_hepatic_monitoring_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized hepatic (jaundice) monitoring thresholds
        Based on liver disease condition profile
        """
        conditions = self.get_patient_conditions(patient_id)
        
        # Check if patient has liver disease
        has_liver_disease = 'liver_disease' in conditions
        
        if has_liver_disease:
            profile = self.CONDITION_PROFILES.get('liver_disease', {})
            skin_analysis = profile.get('skin_analysis', {})
            jaundice_config = skin_analysis.get('jaundice_monitoring', {})
            
            return {
                'priority': 'critical',
                'conditions': ['liver_disease'],
                'scleral_chromaticity_thresholds': {
                    'normal': {'min': 0, 'max': 20},
                    'mild': {'min': 20, 'max': 35},
                    'moderate': {'min': 35, 'max': 50},
                    'severe': {'min': 50, 'max': 100}
                },
                'b_channel_thresholds': jaundice_config.get('severity_levels', {
                    'mild': 25.0,
                    'moderate': 35.0,
                    'severe': 45.0
                }),
                'yellowness_ratio_thresholds': {
                    'normal': {'max': 1.2},
                    'mild': {'min': 1.2, 'max': 1.5},
                    'moderate': {'min': 1.5, 'max': 2.0},
                    'severe': {'min': 2.0}
                },
                'monitoring_regions': ['sclera', 'facial'],
                'wellness_guidance': skin_analysis.get('wellness_guidance', 
                    'Yellowish discoloration of the whites of eyes may indicate changes in liver function.')
            }
        else:
            # Default config for patients without liver disease
            return {
                'priority': 'low',
                'conditions': [],
                'scleral_chromaticity_thresholds': {
                    'normal': {'min': 0, 'max': 25},
                    'mild': {'min': 25, 'max': 40},
                    'moderate': {'min': 40, 'max': 60},
                    'severe': {'min': 60, 'max': 100}
                },
                'b_channel_thresholds': {
                    'mild': 30.0,
                    'moderate': 45.0,
                    'severe': 60.0
                },
                'yellowness_ratio_thresholds': {
                    'normal': {'max': 1.3},
                    'mild': {'min': 1.3, 'max': 1.8},
                    'moderate': {'min': 1.8, 'max': 2.5},
                    'severe': {'min': 2.5}
                },
                'monitoring_regions': ['sclera'],
                'wellness_guidance': 'General jaundice screening for wellness monitoring.'
            }
    
    def get_anemia_monitoring_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized anemia (pallor) monitoring thresholds
        Based on anemia condition profile
        """
        conditions = self.get_patient_conditions(patient_id)
        
        # Check if patient has anemia or related conditions
        has_anemia = 'anemia' in conditions
        has_kidney_disease = 'kidney_disease' in conditions  # Often associated with anemia
        has_heart_failure = 'heart_failure' in conditions  # Can have reduced perfusion
        
        priority = 'critical' if has_anemia else ('high' if has_kidney_disease or has_heart_failure else 'medium')
        related_conditions = [c for c in [has_anemia and 'anemia', 
                                          has_kidney_disease and 'kidney_disease',
                                          has_heart_failure and 'heart_failure'] if c]
        
        if has_anemia:
            profile = self.CONDITION_PROFILES.get('anemia', {})
            skin_analysis = profile.get('skin_analysis', {})
            perfusion_thresholds = skin_analysis.get('perfusion_thresholds', {})
            
            return {
                'priority': priority,
                'conditions': related_conditions,
                'conjunctival_pallor_thresholds': {
                    'normal': {'min': 45, 'max': 100},
                    'mild': {'min': 35, 'max': 45},
                    'moderate': {'min': 25, 'max': 35},
                    'severe': {'min': 0, 'max': 25}
                },
                'palmar_perfusion_thresholds': perfusion_thresholds.get('palmar', {
                    'mild': 40.0,
                    'moderate': 30.0,
                    'severe': 20.0
                }),
                'nailbed_perfusion_thresholds': perfusion_thresholds.get('nailbed', {
                    'mild': 35.0,
                    'moderate': 25.0,
                    'severe': 15.0
                }),
                'a_channel_thresholds': {
                    'normal': {'min': 15},  # Higher a* = more red
                    'mild_pallor': {'min': 10, 'max': 15},
                    'moderate_pallor': {'min': 5, 'max': 10},
                    'severe_pallor': {'max': 5}
                },
                'l_channel_thresholds': {
                    'normal': {'min': 55, 'max': 75},  # Mid-range lightness
                    'pale': {'min': 75}  # Too bright = pale
                },
                'monitoring_regions_priority': skin_analysis.get('pallor_regions_priority', 
                    ['palmar', 'nailbed', 'facial']),
                'capillary_refill': skin_analysis.get('capillary_refill', {
                    'threshold_sec': 2.0,
                    'concern_sec': 3.0,
                    'monitoring': 'important'
                }),
                'wellness_guidance': skin_analysis.get('wellness_guidance',
                    'Pale skin, especially in palms and nail beds, may indicate low iron levels.')
            }
        else:
            # Default config
            return {
                'priority': priority,
                'conditions': related_conditions,
                'conjunctival_pallor_thresholds': {
                    'normal': {'min': 40, 'max': 100},
                    'mild': {'min': 30, 'max': 40},
                    'moderate': {'min': 20, 'max': 30},
                    'severe': {'min': 0, 'max': 20}
                },
                'palmar_perfusion_thresholds': {
                    'mild': 38.0,
                    'moderate': 28.0,
                    'severe': 18.0
                },
                'nailbed_perfusion_thresholds': {
                    'mild': 33.0,
                    'moderate': 23.0,
                    'severe': 13.0
                },
                'a_channel_thresholds': {
                    'normal': {'min': 12},
                    'mild_pallor': {'min': 8, 'max': 12},
                    'moderate_pallor': {'min': 4, 'max': 8},
                    'severe_pallor': {'max': 4}
                },
                'l_channel_thresholds': {
                    'normal': {'min': 50, 'max': 80},
                    'pale': {'min': 80}
                },
                'monitoring_regions_priority': ['palmar', 'nailbed', 'facial'],
                'capillary_refill': {
                    'threshold_sec': 2.0,
                    'concern_sec': 3.5,
                    'monitoring': 'standard'
                },
                'wellness_guidance': 'General pallor screening for wellness monitoring.'
            }
    
    def get_audio_examination_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized audio examination configuration
        Determines which audio stages to prioritize based on patient conditions
        
        Returns:
            - prioritized_stages: List of stages in priority order
            - stage_durations: Recommended recording duration per stage
            - skip_optional_stages: Whether non-critical stages can be skipped
        """
        conditions = self.get_patient_conditions(patient_id)
        
        # Respiratory conditions prioritize breathing + coughing stages
        respiratory_conditions = ['asthma', 'copd', 'heart_failure', 'pulmonary_embolism',
                                 'pneumonia', 'bronchiectasis', 'allergic_reactions']
        
        # Neurological conditions prioritize speaking + reading stages
        neuro_conditions = ['parkinsons', 'als', 'ms', 'stroke', 'dementia']
        
        has_respiratory = any(c in conditions for c in respiratory_conditions)
        has_neuro = any(c in conditions for c in neuro_conditions)
        
        config = {
            'conditions': conditions,
            'has_respiratory_emphasis': has_respiratory,
            'has_neuro_emphasis': has_neuro
        }
        
        if has_respiratory and has_neuro:
            # All stages critical
            config.update({
                'prioritized_stages': ['breathing', 'coughing', 'speaking', 'reading'],
                'stage_durations': {
                    'breathing': 30,  # seconds
                    'coughing': 15,
                    'speaking': 30,
                    'reading': 45
                },
                'skip_optional_stages': False,
                'critical_stages': ['breathing', 'coughing', 'speaking', 'reading'],
                'wellness_guidance': 'Complete all examination stages for comprehensive respiratory and neurological monitoring.'
            })
        elif has_respiratory:
            # Breathing first, then coughing (most important)
            config.update({
                'prioritized_stages': ['breathing', 'coughing', 'speaking', 'reading'],
                'stage_durations': {
                    'breathing': 30,
                    'coughing': 20,  # Longer for respiratory patients
                    'speaking': 20,
                    'reading': 30
                },
                'skip_optional_stages': True,
                'critical_stages': ['breathing', 'coughing'],
                'optional_stages': ['speaking', 'reading'],
                'wellness_guidance': 'Focus on breathing and coughing stages to monitor respiratory wellness patterns.'
            })
        elif has_neuro:
            # Speaking first, then reading (most important)
            config.update({
                'prioritized_stages': ['speaking', 'reading', 'breathing', 'coughing'],
                'stage_durations': {
                    'breathing': 20,
                    'coughing': 15,
                    'speaking': 40,  # Longer for neuro patients
                    'reading': 50  # Longest for neurological assessment
                },
                'skip_optional_stages': True,
                'critical_stages': ['speaking', 'reading'],
                'optional_stages': ['breathing', 'coughing'],
                'wellness_guidance': 'Focus on speaking and reading stages to monitor speech fluency and cognitive patterns.'
            })
        else:
            # General wellness - all stages equal priority
            config.update({
                'prioritized_stages': ['breathing', 'coughing', 'speaking', 'reading'],
                'stage_durations': {
                    'breathing': 25,
                    'coughing': 15,
                    'speaking': 30,
                    'reading': 40
                },
                'skip_optional_stages': False,
                'critical_stages': ['breathing', 'speaking'],
                'optional_stages': ['coughing', 'reading'],
                'wellness_guidance': 'Complete all stages for comprehensive wellness audio monitoring.'
            })
        
        return config
    
    def get_respiratory_audio_config(self, patient_id: str) -> Dict[str, Any]:
        """
        Get personalized respiratory audio monitoring thresholds
        Used for cough detection, wheeze analysis, breath sound classification
        
        Returns:
            - cough_monitoring: Thresholds and focus for cough detection
            - wheeze_monitoring: Thresholds for wheeze detection
            - breath_sound_focus: Which breath sounds to emphasize
        """
        conditions = self.get_patient_conditions(patient_id)
        
        # Initialize default config
        config = {
            'conditions': conditions,
            'cough_monitoring': {
                'priority': 'medium',
                'frequency_threshold_per_minute': 2.0,  # Alert if >2 coughs/min
                'intensity_threshold': 50.0,  # 0-100 scale
                'track_cough_type': True,  # dry vs wet
                'wellness_guidance': 'Monitor cough frequency and type for respiratory wellness patterns.'
            },
            'wheeze_monitoring': {
                'priority': 'medium',
                'detection_threshold': 0.3,  # Probability threshold
                'frequency_range_hz': [100, 1000],  # Typical wheeze frequencies
                'track_timing': True,  # Inspiratory vs expiratory
                'wellness_guidance': 'Track wheeze patterns as indicator of airway changes.'
            },
            'breath_sound_classification': {
                'priority': 'medium',
                'focus_sounds': ['wheeze', 'crackles', 'stridor'],
                'breath_rate_range': {'min': 12, 'max': 20},
                'wellness_guidance': 'Monitor breathing sounds for changes in respiratory wellness.'
            }
        }
        
        # Condition-specific overrides
        if 'asthma' in conditions:
            config['cough_monitoring']['priority'] = 'high'
            config['wheeze_monitoring']['priority'] = 'critical'
            config['wheeze_monitoring']['detection_threshold'] = 0.2  # More sensitive
            config['breath_sound_classification']['focus_sounds'] = ['wheeze', 'prolonged_expiration']
            config['wellness_guidance'] = 'For asthma monitoring, track wheeze frequency and cough patterns as wellness indicators.'
        
        elif 'copd' in conditions:
            config['cough_monitoring']['priority'] = 'critical'
            config['cough_monitoring']['track_cough_type'] = True
            config['cough_monitoring']['frequency_threshold_per_minute'] = 1.5  # More sensitive
            config['wheeze_monitoring']['priority'] = 'high'
            config['breath_sound_classification']['focus_sounds'] = ['wheeze', 'prolonged_expiration', 'crackles']
            config['breath_sound_classification']['breath_rate_range'] = {'min': 12, 'max': 28}
            config['wellness_guidance'] = 'For COPD monitoring, track cough frequency and type. Productive coughs may indicate mucus changes.'
        
        elif 'heart_failure' in conditions:
            config['cough_monitoring']['priority'] = 'high'
            config['cough_monitoring']['cough_type_focus'] = 'dry'  # Cardiac cough often dry
            config['breath_sound_classification']['focus_sounds'] = ['crackles', 'rales']
            config['breath_sound_classification']['priority'] = 'critical'
            config['wellness_guidance'] = 'For heart failure monitoring, dry cough and crackles may indicate fluid changes.'
        
        elif 'pneumonia' in conditions:
            config['cough_monitoring']['priority'] = 'critical'
            config['cough_monitoring']['cough_type_focus'] = 'productive'  # Wet cough common
            config['breath_sound_classification']['focus_sounds'] = ['crackles', 'bronchial_sounds']
            config['wellness_guidance'] = 'For pneumonia recovery, track productive cough and crackles as indicators of lung clearance.'
        
        elif 'allergic_reactions' in conditions:
            config['wheeze_monitoring']['priority'] = 'critical'
            config['wheeze_monitoring']['detection_threshold'] = 0.15  # Very sensitive
            config['wheeze_monitoring']['rapid_onset_detection'] = True
            config['wellness_guidance'] = 'For allergy monitoring, sudden wheeze onset may indicate allergic airway response.'
        
        return config

