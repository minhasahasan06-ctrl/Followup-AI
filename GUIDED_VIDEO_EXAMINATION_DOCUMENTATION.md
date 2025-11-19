# Guided Video Examination System - Complete Documentation

## Overview

The Guided Video Examination System is a production-grade, HIPAA-compliant feature that enables patients to perform self-examinations using their device camera with AI-powered clinical-grade analysis. The system guides patients through a 4-stage examination workflow to detect health changes through hepatic (jaundice) and anemia (pallor) color metrics using LAB color space analysis.

**Key Principle:** This is a wellness monitoring and change detection platform - NOT a diagnostic tool. All analysis is for tracking health changes over time to assist patients and healthcare providers.

## System Architecture

### Frontend (React/TypeScript)
- **Component:** `client/src/pages/GuidedVideoExam.tsx`
- **Route:** `/guided-exam` (Patient role only)
- **Features:**
  - 4-stage workflow with 30-second prep screens
  - Real-time camera capture with countdown timers
  - Stage progression tracking
  - Results display with clinical metrics

### Backend (Python/FastAPI)
- **Router:** `app/routers/guided_exam.py`
- **Service:** `app/services/video_ai_engine.py`
- **Models:** `app/models/video_ai_models.py`
- **Endpoints:** 5 RESTful APIs with JWT authentication

### Database (PostgreSQL)
- **Tables:** 
  - `video_exam_sessions` - Session tracking with stage completion
  - `video_metrics` - Comprehensive metrics with 31 hepatic/anemia fields
  - `respiratory_baselines` - Patient baseline data
  - `patients` - Patient demographics and conditions

## 4-Stage Examination Workflow

### Stage 1: Eyes (Sclera Examination)
**Purpose:** Detect jaundice through scleral yellowing  
**Prep Time:** 30 seconds  
**Instructions:**
1. Position face 12-18 inches from camera
2. Look directly at camera
3. Ensure good lighting on face
4. Remove glasses
5. AI analyzes sclera color for jaundice detection

**Capture:** 3-second countdown, single frame capture  
**Metrics Extracted:**
- `scleral_chromaticity_index` (0-100 scale)
- `sclera_b_channel` (LAB b* yellowness)
- `sclera_yellowness_ratio` (b*/a* ratio)
- `sclera_l_channel`, `sclera_a_channel`, `sclera_saturation`

### Stage 2: Palm (Conjunctival Pallor)
**Purpose:** Detect anemia through palmar pallor  
**Prep Time:** 30 seconds  
**Instructions:**
1. Hold palm flat facing camera
2. Position palm 8-12 inches from camera
3. Ensure even lighting across palm
4. Keep fingers together and straight
5. AI analyzes palm color for anemia detection

**Capture:** 3-second countdown, single frame capture  
**Metrics Extracted:**
- `conjunctival_pallor_index` (0-100 scale)
- `palmar_redness_a` (LAB a* redness)
- `palmar_l_channel` (LAB L* brightness)
- `palmar_b_channel`, `palmar_saturation`, `palmar_perfusion_index`

### Stage 3: Tongue Examination
**Purpose:** Detect tongue coating and color changes  
**Prep Time:** 30 seconds  
**Instructions:**
1. Open mouth wide and stick out tongue
2. Position face 8-12 inches from camera
3. Ensure tongue is fully extended
4. Avoid shadows on tongue surface
5. AI analyzes tongue color and coating

**Capture:** 3-second countdown, single frame capture  
**Metrics Extracted:**
- `tongue_color_index` (composite score)
- `tongue_coating_detected` (boolean)
- `tongue_coating_color` (white/yellow/brown/black)
- `tongue_coating_thickness` (0-100)
- `tongue_l_channel`, `tongue_a_channel`, `tongue_b_channel`

### Stage 4: Lip Examination
**Purpose:** Detect cyanosis and hydration status  
**Prep Time:** 30 seconds  
**Instructions:**
1. Position face 12-18 inches from camera
2. Relax lips naturally (closed)
3. Ensure good lighting on lips
4. Remove lipstick or lip gloss
5. AI analyzes lip color and hydration

**Capture:** 3-second countdown, single frame capture  
**Metrics Extracted:**
- `lip_hydration_score` (0-100 scale)
- `lip_cyanosis_detected` (boolean)
- `lip_color_uniformity` (0-100)
- `lip_l_channel`, `lip_a_channel`, `lip_b_channel`
- `lip_dryness_score`, `lip_cracking_detected`

## API Endpoints

### 1. Create Session
**Endpoint:** `POST /api/v1/guided-exam/sessions`  
**Auth:** JWT token required  
**Request Body:**
```json
{
  "patient_id": "current_user",
  "device_info": {
    "userAgent": "Mozilla/5.0...",
    "screenResolution": "1920x1080"
  }
}
```
**Response:**
```json
{
  "session_id": "uuid-v4",
  "status": "in_progress",
  "created_at": "2025-11-19T10:30:00Z"
}
```

### 2. Capture Frame
**Endpoint:** `POST /api/v1/guided-exam/sessions/{session_id}/capture`  
**Auth:** JWT token required  
**Request Body:**
```json
{
  "stage": "eyes",
  "frame_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```
**Response:**
```json
{
  "success": true,
  "stage": "eyes",
  "next_stage": "palm",
  "progress": 0.25
}
```

### 3. Complete Session
**Endpoint:** `POST /api/v1/guided-exam/sessions/{session_id}/complete`  
**Auth:** JWT token required  
**Request:** No body required  
**Response:**
```json
{
  "session_id": "uuid-v4",
  "status": "completed",
  "analysis_complete": true,
  "metrics_created": true
}
```

### 4. Get Session Details
**Endpoint:** `GET /api/v1/guided-exam/sessions/{session_id}`  
**Auth:** JWT token required  
**Response:**
```json
{
  "session_id": "uuid-v4",
  "patient_id": "patient-123",
  "status": "completed",
  "eyes_stage_completed": true,
  "palm_stage_completed": true,
  "tongue_stage_completed": true,
  "lips_stage_completed": true,
  "created_at": "2025-11-19T10:30:00Z",
  "completed_at": "2025-11-19T10:35:00Z"
}
```

### 5. Get Results
**Endpoint:** `GET /api/v1/guided-exam/sessions/{session_id}/results`  
**Auth:** JWT token required  
**Response:**
```json
{
  "session_id": "uuid-v4",
  "scleral_chromaticity_index": 15.3,
  "conjunctival_pallor_index": 42.7,
  "tongue_color_index": 55.2,
  "tongue_coating_detected": true,
  "tongue_coating_color": "white",
  "lip_hydration_score": 68.4,
  "lip_cyanosis_detected": false,
  "palmar_redness_a": 18.5,
  "sclera_b_channel": 12.3,
  ... (31 total metrics)
}
```

## Disease-Specific Personalization

### ConditionPersonalizationService Extension

The system now includes 3 new methods for guided examination personalization:

#### 1. `get_guided_exam_config(patient_id: str)`
Returns comprehensive examination configuration based on patient's chronic conditions.

**Response:**
```python
{
    'hepatic': {...},  # Liver disease thresholds
    'anemia': {...},   # Anemia thresholds
    'conditions': ['liver_disease', 'anemia'],
    'examination_stages': {
        'eyes': {
            'purpose': 'Sclera examination for jaundice detection',
            'priority': 'critical',
            'key_metrics': ['scleral_chromaticity_index', ...]
        },
        ...
    }
}
```

#### 2. `get_hepatic_monitoring_config(patient_id: str)`
Returns personalized jaundice detection thresholds.

**For Liver Disease Patients:**
- `priority`: "critical"
- `scleral_chromaticity_thresholds`: Normal (0-20), Mild (20-35), Moderate (35-50), Severe (50-100)
- `b_channel_thresholds`: Mild (25.0), Moderate (35.0), Severe (45.0)
- `yellowness_ratio_thresholds`: Normal (<1.2), Mild (1.2-1.5), Moderate (1.5-2.0), Severe (>2.0)

**For General Population:**
- `priority`: "low"
- More relaxed thresholds for general screening

#### 3. `get_anemia_monitoring_config(patient_id: str)`
Returns personalized pallor detection thresholds.

**For Anemia Patients:**
- `priority`: "critical"
- `conjunctival_pallor_thresholds`: Normal (45-100), Mild (35-45), Moderate (25-35), Severe (0-25)
- `palmar_perfusion_thresholds`: Mild (40.0), Moderate (30.0), Severe (20.0)
- `a_channel_thresholds`: Normal (>15), Mild Pallor (10-15), Moderate (5-10), Severe (<5)

**For Heart Failure/Kidney Disease:**
- `priority`: "high"
- Enhanced monitoring for associated conditions

## Clinical-Grade Color Analysis

### LAB Color Space
The system uses LAB color space (CIELAB) for clinical accuracy:
- **L* channel:** Lightness (0-100, black to white)
- **a* channel:** Red-Green axis (negative = green, positive = red)
- **b* channel:** Yellow-Blue axis (negative = blue, positive = yellow)

### Why LAB over RGB/HSV?
1. **Perceptually Uniform:** Equal changes in LAB values correspond to equal perceptual color differences
2. **Device Independent:** Not tied to specific display characteristics
3. **Clinical Standard:** Used in dermatology and medical imaging
4. **Illumination Robust:** L* separates brightness from color

### Metric Calculation Examples

#### Scleral Chromaticity Index (Jaundice)
```python
# Extract sclera ROI from eye frame
sclera_roi = detect_sclera(eye_frame)
lab_image = cv2.cvtColor(sclera_roi, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_image)

# Calculate chromaticity (yellowness)
b_mean = np.mean(b)  # Higher b* = more yellow
yellowness_ratio = b_mean / max(abs(a_mean), 1.0)

# Normalize to 0-100 scale
scleral_chromaticity_index = min(100, max(0, (b_mean - 128) * 2))
```

#### Conjunctival Pallor Index (Anemia)
```python
# Extract palm ROI
palm_roi = detect_palm(palm_frame)
lab_image = cv2.cvtColor(palm_roi, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab_image)

# Calculate pallor (reduced redness)
a_mean = np.mean(a)  # Lower a* = less red = pallor
l_mean = np.mean(l)  # Higher L* = paler

# Perfusion index (inverse of pallor)
perfusion_index = (a_mean - 128) * 0.8 + (1 - l_mean/100) * 20
conjunctival_pallor_index = max(0, min(100, perfusion_index))
```

## Database Schema

### video_exam_sessions Table
```sql
CREATE TABLE video_exam_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'in_progress',  -- in_progress, completed, failed
    eyes_stage_completed BOOLEAN DEFAULT FALSE,
    eyes_frame_s3_uri VARCHAR(512),
    palm_stage_completed BOOLEAN DEFAULT FALSE,
    palm_frame_s3_uri VARCHAR(512),
    tongue_stage_completed BOOLEAN DEFAULT FALSE,
    tongue_frame_s3_uri VARCHAR(512),
    lips_stage_completed BOOLEAN DEFAULT FALSE,
    lips_frame_s3_uri VARCHAR(512),
    quality_score_eyes FLOAT,
    quality_score_palm FLOAT,
    quality_score_tongue FLOAT,
    quality_score_lips FLOAT,
    device_info JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

### video_metrics Table (Extended)
**31 New Hepatic/Anemia Fields Added:**

**Scleral Metrics (6 fields):**
- `scleral_chromaticity_index` (FLOAT)
- `sclera_l_channel`, `sclera_a_channel`, `sclera_b_channel` (FLOAT)
- `sclera_yellowness_ratio`, `sclera_saturation` (FLOAT)

**Palmar Metrics (6 fields):**
- `conjunctival_pallor_index` (FLOAT)
- `palmar_redness_a`, `palmar_l_channel`, `palmar_b_channel` (FLOAT)
- `palmar_saturation`, `palmar_perfusion_index` (FLOAT)

**Tongue Metrics (9 fields):**
- `tongue_color_index` (FLOAT)
- `tongue_coating_detected` (BOOLEAN)
- `tongue_coating_color` (VARCHAR)
- `tongue_coating_thickness` (FLOAT)
- `tongue_l_channel`, `tongue_a_channel`, `tongue_b_channel` (FLOAT)
- `tongue_redness`, `tongue_moisture_level` (FLOAT)

**Lip Metrics (10 fields):**
- `lip_hydration_score` (FLOAT)
- `lip_cyanosis_detected` (BOOLEAN)
- `lip_color_uniformity` (FLOAT)
- `lip_l_channel`, `lip_a_channel`, `lip_b_channel` (FLOAT)
- `lip_blueness_b`, `lip_dryness_score` (FLOAT)
- `lip_cracking_detected` (BOOLEAN)
- `lip_texture_roughness` (FLOAT)

**Migration:** `alembic/versions/9aebc6a2f848_add_hepatic_anemia_color_metrics.py`

## Security & HIPAA Compliance

### Authentication
- **Method:** AWS Cognito JWT tokens
- **Dependency:** `get_current_user` from `app/dependencies.py`
- **Role-Based Access:** Patient role required for all endpoints
- **PHI Isolation:** Queries filtered by `patient_id` to prevent cross-patient data access

### Data Storage
- **Frames:** S3 encrypted storage with server-side encryption (AES-256)
- **Naming:** `guided-exam/{patient_id}/{session_id}/{stage}_frame.jpg`
- **Retention:** Images deleted after analysis (optional retention for audit)
- **Database:** All metrics encrypted at rest in PostgreSQL

### Audit Logging
Every API call logs:
- Patient ID
- Session ID
- Action performed
- Timestamp
- IP address
- User agent
- Success/failure status

### Privacy Notices
Frontend displays:
- "All images are encrypted and used only for health monitoring"
- "This is not a diagnostic tool. Consult your doctor for medical advice"
- "Privacy Notice: Images are processed by AI and deleted after analysis"

## Testing & Quality Assurance

### Regression Tests
**File:** `tests/test_guided_exam_api.py`

**Test Coverage:**
1. `test_patient_a_can_access_own_session` - Authorized access
2. `test_patient_cannot_access_other_patient_session` - PHI isolation (403)
3. `test_session_results_returns_correct_metrics` - Session-specific metrics
4. Stage completion validation
5. Error handling for missing stages

### Manual Testing Workflow
```bash
# 1. Start session
curl -X POST http://localhost:8000/api/v1/guided-exam/sessions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"patient_id": "current_user", "device_info": {}}'

# 2. Capture eyes frame
curl -X POST http://localhost:8000/api/v1/guided-exam/sessions/$SESSION_ID/capture \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"stage": "eyes", "frame_base64": "..."}'

# 3. Repeat for palm, tongue, lips

# 4. Complete session
curl -X POST http://localhost:8000/api/v1/guided-exam/sessions/$SESSION_ID/complete \
  -H "Authorization: Bearer $TOKEN"

# 5. Get results
curl http://localhost:8000/api/v1/guided-exam/sessions/$SESSION_ID/results \
  -H "Authorization: Bearer $TOKEN"
```

## User Experience Flow

### Idle State
- Display examination overview
- Show 4 stages with icons and descriptions
- Privacy notice with HIPAA compliance message
- "Start Examination" button

### Stage Preparation (30 seconds)
- Stage title and icon
- Progress bar showing completion (0%, 25%, 50%, 75%)
- 5-step numbered instructions
- Countdown timer (30 → 0 seconds)
- "I'm Ready - Skip Countdown" button
- "Cancel Exam" button

### Capture State (3 seconds)
- Live camera feed
- Capture instructions overlay
- Countdown timer (3 → 0 seconds)
- Automatic frame capture at 0

### Processing State
- Loading spinner
- "Processing..." or "Analyzing all captured frames with AI..."
- Automatically advances to next stage or completion

### Completed State
- Green checkmark with "Examination Complete"
- Results display with clinical metrics:
  - Scleral Chromaticity (Jaundice Detection)
  - Conjunctival Pallor (Anemia Detection)
  - Tongue Color Index with coating status
  - Lip Hydration Score with cyanosis detection
- Wellness disclaimers
- "Start New Examination" button

## Error Handling

### Camera Access Denied
```typescript
{
  workflowState: 'error',
  message: 'Failed to access camera. Please grant camera permissions.'
}
```

### Incomplete Session
```python
HTTPException(400, "Cannot complete session - missing stages: ['palm', 'lips']")
```

### ML Analysis Failure
```python
{
  status: 'failed',
  error_message: 'ML analysis failed: OpenCV error in sclera detection'
}
```

### PHI Access Violation
```python
HTTPException(403, "Not authorized to access this session")
```

## Performance Characteristics

### Timing Benchmarks
- **Prep Phase:** 30 seconds per stage (120 seconds total)
- **Capture:** 3 seconds per stage (12 seconds total)
- **Frame Upload:** ~1-2 seconds per frame (4-8 seconds total)
- **ML Analysis:** ~10-15 seconds for all 4 frames
- **Total Duration:** ~3-4 minutes end-to-end

### Resource Usage
- **S3 Storage:** ~200-400 KB per frame (800-1600 KB per session)
- **Database:** ~2 KB per session + ~5 KB per metrics row
- **ML Inference:** CPU-based (OpenCV + NumPy), ~2-4s per frame

## Future Enhancements

### Planned Features
1. **Real-time Quality Feedback:** Lighting, focus, positioning guidance during prep
2. **Historical Trending:** Compare current metrics to 7/30/90-day baselines
3. **Multi-language Support:** Spanish, Mandarin, Hindi instructions
4. **Offline Mode:** Capture frames offline, upload when connected
5. **Doctor Review Interface:** Clinician dashboard to review patient examinations
6. **Automated Alerts:** Trigger notifications on significant metric changes

### Research Opportunities
1. **Deep Learning Models:** Replace rule-based metrics with CNN-based detection
2. **Explainable AI:** Highlight specific ROIs contributing to scores
3. **Longitudinal Studies:** Validate metrics against clinical diagnoses
4. **Multi-modal Fusion:** Combine video metrics with wearable data

## Troubleshooting

### Common Issues

**Issue:** "Session stuck in 'in_progress' status"  
**Cause:** Missing stage completion or ML analysis failure  
**Solution:** Check `error_message` field in session, validate all 4 stages completed

**Issue:** "Metrics not displaying in results"  
**Cause:** Query using wrong `session_id` or `patient_id`  
**Solution:** Ensure GET /results uses correct `guided_exam_session_id`

**Issue:** "Camera not starting on mobile"  
**Cause:** HTTPS required for camera access  
**Solution:** Ensure frontend served over HTTPS

**Issue:** "High b* channel values not detecting jaundice"  
**Cause:** Calibration needed for patient's baseline skin tone  
**Solution:** Compare to patient-specific baseline, not absolute thresholds

## References

### Clinical Literature
1. CIE LAB Color Space - International Commission on Illumination
2. "Scleral Icterus Detection Using Image Analysis" - Journal of Medical Systems
3. "Conjunctival Pallor Assessment for Anemia Detection" - PLOS ONE
4. "Tongue Diagnosis in Traditional Chinese Medicine" - Evidence-Based Complementary Medicine

### Technical Standards
- HIPAA Security Rule (45 CFR § 164.312)
- FHIR R4 Observation Resource
- DICOM Supplement 145 (Color Palette)
- W3C Media Capture and Streams API

---

**Last Updated:** November 19, 2025  
**Version:** 1.0.0  
**Maintained By:** Followup AI Engineering Team
