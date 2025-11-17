# Disease-Specific Edema Emphasis System

## Overview
This document details the disease-specific edema monitoring personalization system integrated into the Followup AI platform. Each chronic condition has a tailored edema profile that guides the AI video examination workflow to prioritize relevant swelling patterns, locations, and severity thresholds.

**IMPORTANT**: All guidance is wellness-compliant—this is NOT medical diagnosis. The system helps patients track wellness indicators related to fluid retention and swelling.

---

## System Architecture

### Multi-Domain Condition Profiles
Each condition profile in `CONDITION_PROFILES` (condition_personalization_service.py) now contains:
1. **Respiratory emphasis section** (breathing metrics priority)
2. **Edema emphasis section** (swelling monitoring priority) ← NEW

### Shared Condition Keys
All 12 conditions use the same keys across both domains:
- asthma, copd, heart_failure, pulmonary_embolism, pneumonia, pulmonary_tb, bronchiectasis, allergic_reaction
- kidney_disease, liver_disease, thyroid_disorder, lymphedema (NEW)

---

## Edema Profile Structure

Each edema section contains:

```python
'edema': {
    'priority': 'critical|high|medium|low',       # Examination emphasis
    'expected_pattern': 'bilateral|unilateral|facial|bilateral + facial',
    'focus_locations': ['face', 'legs', 'ankles', 'feet', 'hands'],
    'pei_thresholds': {'mild': 8.0, 'critical': 20.0},  # Personalized thresholds
    'pitting_watchpoints': [2, 3, 4],            # Grade levels requiring attention
    'asymmetry_alert_threshold': 0.20,           # For unilateral patterns (20% diff)
    'sudden_change_alert': True,                 # For emergent conditions
    'wellness_guidance': 'Condition-specific guidance text'
}
```

---

## 12 Condition Profiles with Edema Emphasis

### 1. Heart Failure (CRITICAL Priority)
**Edema Pattern**: Bilateral leg swelling (gravity-dependent)
**Focus Locations**: Legs, ankles, feet
**PEI Thresholds**: Mild 8.0, Critical 20.0 (MORE sensitive)
**Pitting Watchpoints**: [2, 3, 4] — Any pitting is significant
**Wellness Guidance**: "Bilateral leg swelling is an important wellness indicator for heart health. Increasing swelling may suggest fluid retention. Track daily and discuss trends with your care team."

**Why This Matters**: Heart failure patients accumulate fluid in lower extremities due to reduced cardiac output. Early detection of worsening edema helps prevent hospitalizations.

---

### 2. Kidney Disease (CRITICAL Priority)
**Edema Pattern**: Bilateral + Facial (morning facial puffiness, evening leg swelling)
**Focus Locations**: Face, legs, ankles, feet
**PEI Thresholds**: Mild 8.0, Critical 20.0
**Pitting Watchpoints**: [2, 3, 4]
**Wellness Guidance**: "Swelling in face (especially around eyes) and legs is important to track with kidney wellness. Morning facial puffiness and leg swelling by evening are key patterns. Track daily weight and discuss trends with your care team."

**Why This Matters**: Impaired fluid regulation causes characteristic periorbital edema (around eyes) in morning and peripheral edema by evening.

---

### 3. Liver Disease (CRITICAL Priority)
**Edema Pattern**: Bilateral + Ascites (leg swelling + abdominal fluid)
**Focus Locations**: Legs, ankles, feet
**PEI Thresholds**: Mild 10.0, Critical 25.0
**Pitting Watchpoints**: [2, 3, 4]
**Wellness Guidance**: "Leg swelling and abdominal fullness may indicate fluid retention. Monitor for progressive swelling and discuss with your hepatologist."

**Why This Matters**: Reduced albumin production and portal hypertension cause fluid accumulation in legs and abdomen.

---

### 4. Allergic Reactions (CRITICAL Priority)
**Edema Pattern**: Facial (lips, tongue, face)
**Focus Locations**: Face
**PEI Thresholds**: Mild 5.0, Critical 15.0 (VERY sensitive)
**Pitting Watchpoints**: [1, 2, 3, 4] — ANY pitting is concerning
**Sudden Change Alert**: TRUE (any rapid face swelling)
**Wellness Guidance**: "Sudden facial swelling, especially lips/tongue, can be a medical emergency. Seek immediate care if rapid or with breathing difficulty."

**Why This Matters**: Angioedema can progress rapidly and compromise airways. Early detection is life-saving.

---

### 5. Lymphedema (CRITICAL Priority)
**Edema Pattern**: Unilateral (one limb typically)
**Focus Locations**: Legs, feet, hands
**PEI Thresholds**: Mild 5.0, Critical 15.0 (VERY sensitive)
**Pitting Watchpoints**: [1, 2] — Often non-pitting or mild
**Asymmetry Alert Threshold**: 0.20 (Alert if >20% difference between limbs)
**Wellness Guidance**: "Unilateral (one-sided) limb swelling with tight, shiny skin is characteristic. Track circumference changes, skin texture, and any hardening. Early detection of changes helps with management. Discuss with lymphedema specialist."

**Why This Matters**: Impaired lymphatic drainage causes progressive, often non-pitting swelling requiring specialized management.

---

### 6. Thyroid Disorder (HIGH Priority)
**Edema Pattern**: Facial (periorbital - around eyes)
**Focus Locations**: Face
**PEI Thresholds**: Mild 10.0, Critical 25.0
**Pitting Watchpoints**: [1, 2] — Usually mild pitting
**Wellness Guidance**: "Facial puffiness (especially around eyes) may relate to thyroid function. Morning swelling that improves during the day is a common pattern. Track and discuss with your endocrinologist."

**Why This Matters**: Hypothyroidism causes myxedema (protein-rich fluid accumulation) with characteristic periorbital puffiness.

---

### 7. Pulmonary Embolism (HIGH Priority)
**Edema Pattern**: Unilateral leg (DVT-related)
**Focus Locations**: Legs
**PEI Thresholds**: Mild 10.0, Critical 25.0
**Asymmetry Alert Threshold**: 0.30 (Alert if >30% difference)
**Wellness Guidance**: "Unilateral (one-sided) leg swelling may be a concern, especially if sudden. Combined with breathing changes, seek immediate medical evaluation."

**Why This Matters**: DVT (deep vein thrombosis) often presents with unilateral leg swelling before PE. Early detection is critical.

---

### 8. COPD (MEDIUM Priority)
**Edema Pattern**: Bilateral leg (from cor pulmonale - right heart strain)
**Focus Locations**: Legs, ankles
**PEI Thresholds**: Mild 10.0, Critical 25.0
**Wellness Guidance**: "Leg swelling in COPD may indicate heart strain. Track changes and discuss with healthcare provider."

**Why This Matters**: Advanced COPD can cause right-sided heart failure (cor pulmonale) leading to peripheral edema.

---

### 9. Asthma (LOW Priority)
**Edema Pattern**: None typically
**Wellness Guidance**: "Swelling not typically associated with asthma alone."

---

### 10. Pneumonia (LOW Priority)
**Edema Pattern**: None typically
**Wellness Guidance**: "Swelling not typically associated with pneumonia alone."

---

### 11. Pulmonary TB (LOW Priority)
**Edema Pattern**: None typically
**Wellness Guidance**: "Swelling not typically associated with pulmonary TB."

---

### 12. Bronchiectasis (LOW Priority)
**Edema Pattern**: None typically
**Wellness Guidance**: "Swelling not typically associated with bronchiectasis."

---

## Service Methods

### 1. `get_edema_config(patient_id: str) -> Dict[str, Any]`
Retrieves personalized edema configuration by:
1. Fetching patient's active conditions from database
2. Extracting edema sections from each condition profile
3. Merging profiles if multiple conditions exist (using `_merge_edema_profiles()`)
4. Returns combined configuration with all conditions listed

**Returns**:
```python
{
    'priority': 'critical',
    'expected_pattern': 'bilateral',
    'focus_locations': ['legs', 'ankles', 'feet'],
    'pei_thresholds': {'mild': 8.0, 'critical': 20.0},
    'pitting_watchpoints': [2, 3, 4],
    'wellness_guidance': 'Combined guidance from all conditions',
    'conditions': ['heart_failure', 'kidney_disease']  # All active conditions
}
```

---

### 2. `get_edema_examination_focus(patient_id: str) -> Dict[str, Any]`
Generates personalized examination instructions with:
- Priority-based importance message (🔴/🟠/🟡/🟢)
- Examination instructions for each body area
- Pitting test guidance (Recommended vs Optional)
- Expected pattern guidance (bilateral/unilateral/facial alerts)

**Returns**:
```python
{
    'priority': 'critical',
    'importance_message': '🔴 CRITICAL: Swelling monitoring is very important for your wellness tracking.',
    'focus_locations': ['legs', 'ankles', 'feet'],
    'expected_pattern': 'bilateral',
    'examination_instructions': [
        '📸 LEGS/ANKLES: Show both sides for symmetry comparison (30 sec)',
        '👆 PITTING TEST (Recommended): Press swollen area 5-15 sec, observe rebound'
    ],
    'pattern_guidance': '⚠️ Pay special attention to both sides (bilateral swelling)',
    'wellness_guidance': 'Bilateral leg swelling is important wellness indicator...',
    'pei_thresholds': {'mild': 8.0, 'critical': 20.0}
}
```

---

### 3. `_merge_edema_profiles(profiles: List[Dict]) -> Dict[str, Any]`
Merges multiple edema profiles for patients with multiple conditions:
- **Priority**: Takes HIGHEST priority (critical > high > medium > low)
- **PEI Thresholds**: Takes LOWEST thresholds (most sensitive)
- **Focus Locations**: COMBINES all unique locations
- **Pitting Watchpoints**: COMBINES all grade levels
- **Wellness Guidance**: CONCATENATES all guidance texts

**Example Merge**:
Patient with heart_failure + kidney_disease:
- Priority: critical (both critical → critical)
- Expected Pattern: "bilateral + facial" (combined)
- Focus Locations: ['face', 'legs', 'ankles', 'feet'] (combined)
- PEI Thresholds: mild=8.0 (min of 8.0, 8.0), critical=20.0 (min of 20.0, 20.0)

---

## Integration with Video AI Workflow

### Examination Flow (ExamPrepStep.tsx)
The edema examination focus is displayed during the preparation step:

```typescript
const edemaFocus = await fetch(`/api/v1/conditions/${patientId}/edema-focus`);
// Displays:
// - Priority indicator (🔴 CRITICAL / 🟠 HIGH / etc.)
// - Examination instructions (📸 body areas + 👆 pitting test)
// - Expected pattern guidance (⚠️ bilateral/unilateral/facial alerts)
// - Wellness guidance text
```

### Video Analysis (EdemaAnalysisService)
The personalized configuration guides AI extraction:
```python
config = personalization_service.get_edema_config(patient_id)

# Use personalized thresholds for PEI calculation
pei_thresholds = config['pei_thresholds']

# Focus detection on expected locations
focus_locations = config['focus_locations']

# Apply condition-specific pitting grade emphasis
pitting_watchpoints = config['pitting_watchpoints']
```

---

## Database Models (No Changes Required)

The system uses existing models:
- **RespiratoryConditionProfile**: Stores patient condition assignments (shared across both domains)
  - `patient_id`, `condition` (e.g., 'heart_failure'), `active` (boolean)
- **EdemaMetric**: Stores edema measurements (existing)
- **EdemaBaseline**: Stores patient baselines (existing)

**NOTE**: There is NO separate `EdemaConditionProfile` table—edema profiles are integrated into `CONDITION_PROFILES` dictionary alongside respiratory profiles.

---

## Critical Condition Prioritization Examples

### Example 1: Heart Failure Patient
```python
# Configuration
edema_config = {
    'priority': 'critical',
    'expected_pattern': 'bilateral',
    'focus_locations': ['legs', 'ankles', 'feet'],
    'pei_thresholds': {'mild': 8.0, 'critical': 20.0},  # Lower = more sensitive
    'pitting_watchpoints': [2, 3, 4]  # Grades 2-4 require attention
}

# Examination Instructions
"🔴 CRITICAL: Swelling monitoring is very important for your wellness tracking."
"📸 LEGS/ANKLES: Show both sides for symmetry comparison (30 sec)"
"👆 PITTING TEST (Recommended): Press swollen area 5-15 sec, observe rebound"
"⚠️ Pay special attention to both sides (bilateral swelling)"
```

### Example 2: Allergic Reaction Patient
```python
# Configuration
edema_config = {
    'priority': 'critical',
    'expected_pattern': 'facial',
    'focus_locations': ['face'],
    'pei_thresholds': {'mild': 5.0, 'critical': 15.0},  # VERY sensitive
    'pitting_watchpoints': [1, 2, 3, 4],  # ANY pitting is concerning
    'sudden_change_alert': True
}

# Examination Instructions
"🔴 CRITICAL: Swelling monitoring is very important for your wellness tracking."
"📸 FACE: Front view + side views (30 sec)"
"👆 PITTING TEST (Recommended): Press swollen area 5-15 sec, observe rebound"
"⚠️ Pay special attention to facial swelling, especially around eyes"
"⚠️ URGENT: Sudden facial swelling, especially lips/tongue, can be a medical emergency."
```

### Example 3: Lymphedema Patient
```python
# Configuration
edema_config = {
    'priority': 'critical',
    'expected_pattern': 'unilateral',
    'focus_locations': ['legs', 'feet', 'hands'],
    'pei_thresholds': {'mild': 5.0, 'critical': 15.0},
    'asymmetry_alert_threshold': 0.20  # Alert if >20% difference
}

# Asymmetry Detection Logic
if abs(left_pei - right_pei) / max(left_pei, right_pei) > 0.20:
    alert = "ASYMMETRY DETECTED: >20% difference between limbs"
```

---

## Multi-Condition Patients

### Example: Heart Failure + Kidney Disease
Both conditions have critical edema priority but different patterns:

**Merged Configuration**:
```python
{
    'priority': 'critical',  # Both critical → critical
    'expected_pattern': 'bilateral + facial',  # Combined patterns
    'focus_locations': ['face', 'legs', 'ankles', 'feet'],  # Combined locations
    'pei_thresholds': {
        'mild': 8.0,     # min(8.0, 8.0) = 8.0
        'critical': 20.0  # min(20.0, 20.0) = 20.0
    },
    'pitting_watchpoints': [2, 3, 4],  # Combined unique grades
    'wellness_guidance': 'Bilateral leg swelling is important wellness indicator... Swelling in face (especially around eyes)...'
}
```

**Examination Instructions**:
```
🔴 CRITICAL: Swelling monitoring is very important for your wellness tracking.
📸 FACE: Front view + side views (30 sec)
📸 LEGS/ANKLES: Show both sides for symmetry comparison (30 sec)
👆 PITTING TEST (Recommended): Press swollen area 5-15 sec, observe rebound
⚠️ Pay special attention to face AND leg swelling
```

---

## Wellness Compliance

All edema monitoring guidance uses wellness-compliant language:
- ✅ "Swelling is an important wellness indicator"
- ✅ "Track trends and discuss with your care team"
- ✅ "Monitor for progressive swelling"
- ❌ "This indicates heart failure decompensation" (DIAGNOSIS - PROHIBITED)
- ❌ "You have grade 3 pitting edema" (DIAGNOSIS - PROHIBITED)

---

## Next Steps

1. ✅ **Completed**: Extended all 12 condition profiles with edema sections
2. ✅ **Completed**: Added service methods (get_edema_config, get_edema_examination_focus)
3. ⏭️ **Next**: Create FastAPI endpoints (`/api/v1/conditions/{patient_id}/edema-config`, `/api/v1/conditions/{patient_id}/edema-focus`)
4. ⏭️ **Next**: Integrate with ExamPrepStep.tsx to display personalized edema instructions
5. ⏭️ **Next**: Integrate with EdemaAnalysisService to use personalized thresholds during video analysis

---

## Summary

The disease-specific edema emphasis system provides:
- **12 condition profiles** with tailored edema monitoring priorities
- **Multi-domain architecture** (respiratory + edema in single profiles)
- **Intelligent merging** for multi-condition patients
- **Personalized thresholds** (PEI, pitting grades, asymmetry)
- **Expected pattern guidance** (bilateral/unilateral/facial)
- **Wellness-compliant messaging** throughout

This system ensures each patient receives examination guidance specific to their chronic conditions, maximizing early detection of concerning fluid retention patterns while maintaining regulatory compliance.
