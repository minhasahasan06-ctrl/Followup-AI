# Facial Puffiness Score (FPS) - Complete Documentation

## Overview
The Facial Puffiness Score (FPS) is a comprehensive metric for tracking facial swelling and fluid retention using MediaPipe Face Mesh landmarks. This system provides quantitative measurement of facial contour expansion over time, critical for monitoring conditions like thyroid disorders, kidney disease, heart failure, liver disease, and allergic reactions.

**Clinical Significance**: Facial swelling is often an early indicator of fluid retention or metabolic changes. Periorbital puffiness (around eyes) is particularly sensitive for thyroid and kidney conditions.

---

## System Architecture

### MediaPipe Face Mesh Integration
The system uses **MediaPipe Face Mesh** with 468 3D facial landmarks to track:
- Periorbital region (around eyes)
- Cheek region
- Jawline region
- Forehead region
- Overall facial contour

**Tracking Mode**: Dynamic tracking (non-static) with landmark refinement enabled for maximum precision.

---

## Facial Regions Tracked

### 1. Periorbital Region (Around Eyes) - 30% Weight
**Critical for**: Thyroid disorders, kidney disease, allergic reactions

**Landmarks Used**:
- **Left Eye**:
  - Outer corner: Landmark 33
  - Inner corner: Landmark 133
  - Top: Landmark 159
  - Bottom: Landmark 145
  
- **Right Eye**:
  - Outer corner: Landmark 263
  - Inner corner: Landmark 362
  - Top: Landmark 386
  - Bottom: Landmark 374

**Measurements**:
- `left_eye_width` = Distance from outer to inner corner
- `left_eye_height` = Distance from top to bottom
- `left_eye_area` = width × height (periorbital puffiness indicator)
- Same calculations for right eye
- `avg_eye_area` = Average of both eyes

**Why This Matters**: The periorbital region is highly sensitive to fluid accumulation. Hypothyroidism causes myxedema (protein-rich fluid) around eyes. Kidney disease impairs fluid regulation, leading to morning facial puffiness. Allergic reactions cause rapid periorbital angioedema.

---

### 2. Cheek Region - 30% Weight
**Critical for**: General facial swelling, heart failure (cor pulmonale), liver disease

**Landmarks Used**:
- Left cheek: Landmark 234
- Right cheek: Landmark 454
- Nose bridge (reference): Landmark 168

**Measurements**:
- `cheek_width` = Distance from left to right cheek
- `left_cheek_projection` = Distance from left cheek to nose bridge
- `right_cheek_projection` = Distance from right cheek to nose bridge
- `avg_cheek_projection` = Average of both projections

**Why This Matters**: Cheek swelling indicates generalized facial fluid retention. Bilateral cheek swelling suggests systemic fluid overload (heart failure, kidney disease). Unilateral swelling may indicate localized issues.

---

### 3. Jawline Region - 20% Weight
**Critical for**: Lower face fluid retention, dental/jaw issues

**Landmarks Used**:
- Left jaw: Landmark 172
- Right jaw: Landmark 397
- Chin: Landmark 152

**Measurements**:
- `jawline_width` = Distance from left to right jaw
- `left_jaw_projection` = Distance from left jaw to chin
- `right_jaw_projection` = Distance from right jaw to chin

**Why This Matters**: Jawline expansion can indicate lower facial edema or lymphatic obstruction. Important for lymphedema patients.

---

### 4. Forehead Region - 10% Weight
**Critical for**: Upper facial swelling, generalized edema

**Landmarks Used**:
- Forehead center: Landmark 10
- Left temple: Landmark 108
- Right temple: Landmark 337

**Measurements**:
- `forehead_width` = Distance from left to right temple

**Why This Matters**: Forehead swelling is less common but can indicate severe generalized edema or specific conditions affecting upper face.

---

### 5. Overall Face Contour - 10% Weight
**Critical for**: Comprehensive facial expansion detection

**Landmarks Used**:
- Forehead: Landmark 10
- Left cheek: Landmark 234
- Chin: Landmark 152
- Right cheek: Landmark 454

**Measurements**:
- `face_perimeter` = Sum of distances between consecutive points (approximates oval perimeter)

**Why This Matters**: Provides overall facial expansion metric that captures generalized puffiness not specific to one region.

---

## FPS Calculation Methodology

### Step 1: Extract Landmarks from Video
For each frame with detected face:
1. MediaPipe Face Mesh extracts 468 3D landmarks
2. Key landmarks for each region are extracted (see regions above)
3. Distances calculated using Euclidean norm: `distance = ||point1 - point2||`

### Step 2: Aggregate Across Video
All measurements from all frames are averaged:
```python
avg_eye_area = mean(all_eye_areas_from_all_frames)
avg_cheek_width = mean(all_cheek_widths_from_all_frames)
avg_jawline_width = mean(all_jawline_widths_from_all_frames)
avg_forehead_width = mean(all_forehead_widths_from_all_frames)
avg_face_perimeter = mean(all_face_perimeters_from_all_frames)
```

### Step 3: Compare to Patient Baseline
Each measurement is compared to patient's personal baseline (from initial healthy state):

```python
# Periorbital FPS (around eyes)
baseline_eye_area = patient_baseline['baseline_eye_area']
periorbital_fps = (avg_eye_area - baseline_eye_area) / baseline_eye_area * 100

# Cheek FPS
baseline_cheek_width = patient_baseline['baseline_cheek_width']
cheek_fps = (avg_cheek_width - baseline_cheek_width) / baseline_cheek_width * 100

# Jawline FPS
baseline_jawline_width = patient_baseline['baseline_jawline_width']
jawline_fps = (avg_jawline_width - baseline_jawline_width) / baseline_jawline_width * 100

# Forehead FPS
baseline_forehead_width = patient_baseline['baseline_forehead_width']
forehead_fps = (avg_forehead_width - baseline_forehead_width) / baseline_forehead_width * 100

# Overall contour FPS
baseline_face_perimeter = patient_baseline['baseline_face_perimeter']
overall_fps = (avg_face_perimeter - baseline_face_perimeter) / baseline_face_perimeter * 100
```

**Result**: % expansion from baseline (0% = no change, 10% = 10% larger than baseline)

### Step 4: Calculate Composite FPS
Weighted average of all region scores:

```python
composite_fps = (
    periorbital_fps × 0.30 +    # 30% weight - critical for thyroid/kidney
    cheek_fps × 0.30 +           # 30% weight - general facial swelling
    jawline_fps × 0.20 +         # 20% weight - lower face
    forehead_fps × 0.10 +        # 10% weight - upper face
    overall_fps × 0.10           # 10% weight - overall expansion
)
```

**Final FPS**: Composite score representing overall facial puffiness percentage above baseline

---

## FPS Output Metrics

### Primary Metric
- **`facial_puffiness_score`** (float): Composite FPS (0-100+ scale)
  - 0-10: Normal/minimal puffiness
  - 10-25: Moderate puffiness
  - 25+: Significant puffiness

### Regional Scores
- **`fps_periorbital`** (float): Eye region puffiness %
- **`fps_cheek`** (float): Cheek region puffiness %
- **`fps_jawline`** (float): Jawline region puffiness %
- **`fps_forehead`** (float): Forehead region puffiness %
- **`fps_overall_contour`** (float): Overall contour expansion %

### Raw Measurements (for baseline calculation)
- **`raw_eye_area`** (float): Current average eye area
- **`raw_cheek_width`** (float): Current cheek width
- **`raw_cheek_projection`** (float): Current cheek projection
- **`raw_jawline_width`** (float): Current jawline width
- **`raw_forehead_width`** (float): Current forehead width
- **`raw_face_perimeter`** (float): Current face perimeter

### Asymmetry Detection
- **`facial_asymmetry_score`** (float): % difference between left and right eye areas
  - Detects unilateral swelling (important for lymphedema, allergic reactions)

### Risk Classification
- **`facial_puffiness_risk`** (string): "low" | "medium" | "high" | "unknown"
  - Low: FPS < 10
  - Medium: FPS 10-25
  - High: FPS > 25

---

## Disease-Specific Integration

### Integration with Condition Personalization System
The FPS system integrates with the disease-specific edema emphasis system (see `DISEASE_SPECIFIC_EDEMA_MONITORING.md`):

#### 1. Thyroid Disorder Patients
**Expected Pattern**: Periorbital puffiness (around eyes)
**FPS Emphasis**: `fps_periorbital` is primary metric
**Alert Threshold**: FPS > 15% (personalized from condition profile)
**Wellness Guidance**: "Facial puffiness, especially around eyes, may relate to thyroid function. Morning swelling that improves during the day is a common pattern."

#### 2. Kidney Disease Patients
**Expected Pattern**: Bilateral facial + periorbital puffiness
**FPS Emphasis**: `fps_periorbital` + `fps_cheek` combined
**Alert Threshold**: FPS > 12% (more sensitive)
**Temporal Pattern**: Morning facial puffiness (especially eyes), evening leg swelling
**Wellness Guidance**: "Facial puffiness around eyes is important to track with kidney wellness. Morning swelling is a key pattern."

#### 3. Heart Failure Patients
**Expected Pattern**: Generalized facial swelling (less common than leg edema)
**FPS Emphasis**: `fps_overall_contour`
**Alert Threshold**: FPS > 20%
**Wellness Guidance**: "Facial swelling may indicate fluid retention. Track with leg swelling patterns."

#### 4. Allergic Reaction Patients
**Expected Pattern**: Rapid facial/lip/tongue swelling (angioedema)
**FPS Emphasis**: `fps_periorbital` + `facial_asymmetry_score`
**Alert Threshold**: FPS > 10% (very sensitive) + any rapid change
**Sudden Change Alert**: TRUE (any rapid FPS increase triggers alert)
**Wellness Guidance**: "Sudden facial swelling, especially lips/tongue, can be a medical emergency. Seek immediate care if rapid or with breathing difficulty."

#### 5. Liver Disease Patients
**Expected Pattern**: Generalized facial swelling + ascites
**FPS Emphasis**: `fps_cheek` + `fps_jawline`
**Alert Threshold**: FPS > 15%
**Wellness Guidance**: "Facial swelling may accompany abdominal fullness. Monitor progressive changes."

#### 6. Lymphedema Patients
**Expected Pattern**: Unilateral facial swelling (rare, usually limbs)
**FPS Emphasis**: `facial_asymmetry_score`
**Alert Threshold**: Asymmetry > 20%
**Wellness Guidance**: "Facial lymphedema is less common. Track any one-sided swelling."

---

## Baseline Calculation & Tracking

### Initial Baseline Establishment
When patient first uses the system:
1. Patient records 3-5 videos in healthy/normal state
2. FPS calculated for each video
3. Baseline = average of raw measurements from all videos:
   ```python
   baseline_eye_area = mean(all_raw_eye_areas)
   baseline_cheek_width = mean(all_raw_cheek_widths)
   baseline_jawline_width = mean(all_raw_jawline_widths)
   baseline_forehead_width = mean(all_raw_forehead_widths)
   baseline_face_perimeter = mean(all_raw_face_perimeters)
   ```

### Baseline Update Strategy
**Option 1 - Fixed Baseline**: Baseline never changes (useful for tracking long-term progression)

**Option 2 - Exponential Moving Average** (recommended for dynamic tracking):
```python
# Update baseline with exponential moving average (α = 0.05)
new_baseline = (0.95 × old_baseline) + (0.05 × current_measurement)
```
Only update baseline when:
- FPS is in "low" risk category (FPS < 10)
- Patient confirms feeling normal
- No known acute changes

---

## Temporal Analytics

### Time-Series Tracking
Store FPS over time to detect:
1. **Acute Changes**: FPS spike >20% in 24 hours → Urgent alert
2. **Chronic Progression**: FPS gradually increasing over weeks → Trend alert
3. **Morning vs Evening Patterns**: Kidney patients show morning facial puffiness
4. **Medication Response**: FPS should decrease after diuretic adjustment

### Rolling Averages
- **24-hour rolling average**: Smooth out daily variations
- **7-day rolling average**: Detect weekly trends
- **30-day trend slope**: Linear regression to determine if FPS is increasing/decreasing/stable

---

## Clinical Use Cases

### Use Case 1: Thyroid Patient - Periorbital Puffiness Monitoring
**Scenario**: Patient with hypothyroidism notices morning eye puffiness

**FPS Workflow**:
1. Patient records 60-second facial video each morning
2. FPS extracted with emphasis on `fps_periorbital`
3. Baseline periorbital FPS = 5% (from initial healthy state)
4. Current periorbital FPS = 18% → **13% increase**
5. System generates alert: "Periorbital puffiness increased 13% from baseline. Discuss with endocrinologist."

**Outcome**: Early detection of thyroid function changes before clinical symptoms worsen

---

### Use Case 2: Kidney Patient - Morning Facial Puffiness Pattern
**Scenario**: Patient with chronic kidney disease experiences morning facial puffiness

**FPS Workflow**:
1. Patient records facial video morning + evening daily
2. Morning FPS = 22% (periorbital + cheek regions elevated)
3. Evening FPS = 8% (puffiness resolves during day)
4. **Temporal pattern detected**: Morning-predominant facial edema
5. System tracks trend: FPS morning average increasing from 15% → 22% over 2 weeks

**Outcome**: Objective data for nephrologist to adjust diuretic timing or dosage

---

### Use Case 3: Allergic Reaction - Rapid Angioedema Detection
**Scenario**: Patient with food allergy develops facial swelling after meal

**FPS Workflow**:
1. Baseline FPS = 3% (minimal puffiness)
2. 30 minutes after allergen exposure: FPS = 28% (rapid spike)
3. `fps_periorbital` = 45% (eyes swelling rapidly)
4. **Sudden change alert triggered**: "URGENT: Facial swelling increased 25% rapidly. Seek immediate medical attention if breathing difficulty."

**Outcome**: Early warning system for potentially life-threatening angioedema

---

## API Integration

### Video Analysis Endpoint
```python
# POST /api/v1/video-ai/analyze
{
  "video_path": "s3://bucket/patient-video.mp4",
  "patient_id": "patient_123",
  "patient_baseline": {
    "baseline_eye_area": 0.045,
    "baseline_cheek_width": 0.32,
    "baseline_jawline_width": 0.28,
    "baseline_forehead_width": 0.25,
    "baseline_face_perimeter": 1.15
  }
}

# Response includes:
{
  "facial_puffiness_score": 16.8,
  "fps_periorbital": 22.3,
  "fps_cheek": 15.1,
  "fps_jawline": 12.4,
  "fps_forehead": 8.2,
  "fps_overall_contour": 14.6,
  "facial_puffiness_risk": "medium",
  "facial_asymmetry_score": 5.2,
  "raw_eye_area": 0.055,
  "raw_cheek_width": 0.37,
  ...
}
```

---

## Advantages Over Traditional Methods

### Traditional Clinical Assessment
- **Subjective**: "Patient reports facial swelling" (no quantification)
- **Intermittent**: Only assessed during clinic visits (miss day-to-day changes)
- **Delayed Detection**: By the time patient notices swelling, it's already significant

### FPS System Advantages
- ✅ **Objective**: Quantitative % change from baseline
- ✅ **Continuous**: Daily or multiple-times-daily tracking at home
- ✅ **Early Detection**: Detects 5-10% changes before visually obvious
- ✅ **Regional Specificity**: Identifies which facial region is swelling (periorbital vs cheek vs jawline)
- ✅ **Temporal Patterns**: Detects morning vs evening patterns critical for diagnosis
- ✅ **Trend Analysis**: Identifies gradual worsening or improvement over time

---

## Limitations & Considerations

### Technical Limitations
1. **Lighting Dependency**: Consistent lighting needed for accurate landmark detection
2. **Camera Positioning**: Patient must maintain similar distance/angle from camera
3. **Facial Hair**: Beards/mustaches may affect jawline landmark accuracy
4. **Glasses**: May partially occlude periorbital landmarks
5. **Makeup**: Heavy makeup may affect skin-based measurements

### Clinical Limitations
1. **Not Diagnostic**: FPS is a wellness tracking metric, NOT a diagnostic tool
2. **Baseline Dependency**: Accuracy depends on quality of initial baseline
3. **Individual Variation**: Natural facial structure variations require personalized baselines
4. **Confounding Factors**: Weight changes, aging, dental work can affect measurements

### Mitigation Strategies
- **Standardized Recording Protocol**: Same time of day, lighting, camera position
- **Quality Scoring**: Reject videos with poor lighting or face detection
- **Baseline Recalibration**: Periodic baseline updates when appropriate
- **Multi-Metric Integration**: Combine FPS with other edema metrics (PEI, leg swelling)

---

## Future Enhancements

### Planned Improvements
1. **3D Volumetric FPS**: Use depth information for true facial volume calculation
2. **Skin Texture Analysis**: Detect skin tightness/shininess associated with edema
3. **Color Analysis**: Track skin color changes (pallor, redness) alongside puffiness
4. **Machine Learning Baseline**: AI-learned baselines accounting for time-of-day, posture, hydration
5. **Integration with Wearables**: Correlate FPS with fluid intake, blood pressure, heart rate

---

## Summary

The Facial Puffiness Score (FPS) provides:
- **Comprehensive facial contour tracking** using MediaPipe Face Mesh 468 landmarks
- **5 regional scores** (periorbital, cheek, jawline, forehead, overall) with weighted composite
- **Baseline-relative measurements** for personalized tracking (% expansion from baseline)
- **Disease-specific integration** with condition profiles (thyroid, kidney, heart, liver, allergic)
- **Temporal pattern detection** for morning/evening variations and acute vs chronic changes
- **Asymmetry detection** for unilateral swelling conditions (lymphedema, localized reactions)
- **Objective, quantitative wellness tracking** to replace subjective patient reports

**Clinical Impact**: Early detection of fluid retention changes enables proactive care management, potentially preventing hospitalizations and complications from decompensated chronic conditions.

**Wellness Compliance**: All guidance is wellness-compliant—FPS helps patients track wellness indicators, not diagnose medical conditions. Patients are always directed to healthcare providers for medical evaluation.
