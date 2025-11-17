# AI Deterioration Detection System - Testing Guide

## Overview

This guide explains how to test all 52 AI endpoints in the Followup AI deterioration detection system.

## Prerequisites

### 1. Start Python Backend

The Python FastAPI backend must be running on port 8000:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Environment Variables Required

Verify these environment variables are set:
- `AWS_REGION=ap-southeast-2` (NOT "Asia Pacific (Sydney) ap-southeast-2")
- `AWS_COGNITO_REGION=ap-southeast-2`
- `AWS_COGNITO_USER_POOL_ID`
- `AWS_COGNITO_CLIENT_ID`
- `AWS_S3_BUCKET_NAME`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `OPENAI_API_KEY`
- `DATABASE_URL`

### 3. Get Authentication Token

You need a valid JWT token from AWS Cognito. Two options:

#### Option A: Use Frontend Login
1. Navigate to `/login` in your browser
2. Login with your credentials
3. Open browser DevTools → Application → Cookies
4. Find the `auth_token` cookie value

#### Option B: Use Cognito Direct API
```bash
# Replace with your credentials
curl -X POST https://cognito-idp.ap-southeast-2.amazonaws.com/ \
  -H "Content-Type: application/x-amz-json-1.1" \
  -H "X-Amz-Target: AWSCognitoIdentityProviderService.InitiateAuth" \
  -d '{
    "AuthFlow": "USER_PASSWORD_AUTH",
    "ClientId": "YOUR_CLIENT_ID",
    "AuthParameters": {
      "USERNAME": "your-email@example.com",
      "PASSWORD": "your-password"
    }
  }'
```

Save the `IdToken` from the response.

---

## Testing AI Dashboards (Frontend)

### 1. Video AI Dashboard

Navigate to: `http://localhost:5000/ai-video`

**Features to Test:**
- ✅ Video upload (drag-and-drop or click)
- ✅ Video analysis progress indicator
- ✅ Metrics display:
  - Respiratory rate (breaths/min)
  - Skin pallor score (0-100)
  - Sclera yellowness (0-100)
  - Facial swelling (0-100)
  - Head tremor (0-100)
- ✅ Session history list
- ✅ Download analysis results

### 2. Audio AI Dashboard

Navigate to: `http://localhost:5000/ai-audio`

**Features to Test:**
- ✅ Audio upload (drag-and-drop or click)
- ✅ Audio analysis progress indicator
- ✅ Metrics display:
  - Breath cycles (per minute)
  - Speech pace (words/min)
  - Cough detected (boolean)
  - Wheeze detected (boolean)
  - Voice quality (0-100)
- ✅ Session history list
- ✅ Play audio samples

### 3. Health Alerts Dashboard

Navigate to: `http://localhost:5000/ai-alerts`

**Features to Test:**
- ✅ Create new alert rule
- ✅ Configure alert thresholds
- ✅ Set notification channels (dashboard, email, SMS)
- ✅ View active alerts
- ✅ Acknowledge alerts
- ✅ Alert history

---

## Testing AI Endpoints (Backend API)

All endpoints require the `Authorization` header with your JWT token.

### Video AI Endpoints (13 endpoints)

#### 1. Upload Video for Analysis
```bash
curl -X POST http://localhost:8000/api/v1/video-ai/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@test-video.mp4"
```

**Expected Response:**
```json
{
  "session_id": 123,
  "s3_key": "video-ai/patient-123/20251117_073000_abc123.mp4",
  "processing_status": "pending",
  "message": "Video uploaded successfully"
}
```

#### 2. Get Video Analysis Results
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/sessions/123 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 3. List All Video Sessions
```bash
curl -X GET "http://localhost:8000/api/v1/video-ai/sessions?limit=10&offset=0" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 4. Get Video Metrics by Session ID
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/sessions/123/metrics \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 5. Re-analyze Existing Video
```bash
curl -X POST http://localhost:8000/api/v1/video-ai/sessions/123/reanalyze \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Audio AI Endpoints (13 endpoints)

#### 1. Upload Audio for Analysis
```bash
curl -X POST http://localhost:8000/api/v1/audio-ai/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "file=@test-audio.wav"
```

#### 2. Get Audio Analysis Results
```bash
curl -X GET http://localhost:8000/api/v1/audio-ai/sessions/456 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 3. List All Audio Sessions
```bash
curl -X GET "http://localhost:8000/api/v1/audio-ai/sessions?limit=10" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Trend Prediction Endpoints (13 endpoints)

#### 1. Get Risk Assessment for Patient
```bash
curl -X GET http://localhost:8000/api/v1/trends/risk-assessment/patient-123 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Expected Response:**
```json
{
  "patient_id": "patient-123",
  "risk_score": 7.5,
  "risk_level": "moderate",
  "confidence": 0.85,
  "anomaly_count": 3,
  "contributing_factors": [
    {"metric": "respiratory_rate", "deviation": 2.3, "weight": 0.4}
  ],
  "wellness_recommendations": [
    "Consider discussing respiratory changes with healthcare provider"
  ]
}
```

#### 2. Get Baseline Data for Patient
```bash
curl -X GET http://localhost:8000/api/v1/trends/baseline/patient-123 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 3. Calculate New Baseline (Force Recalculation)
```bash
curl -X POST http://localhost:8000/api/v1/trends/baseline/patient-123/calculate \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 4. Get Trend Snapshots
```bash
curl -X GET "http://localhost:8000/api/v1/trends/snapshots/patient-123?days=30" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 5. Get Risk Events
```bash
curl -X GET "http://localhost:8000/api/v1/trends/risk-events/patient-123?days=7" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### Alert Orchestration Endpoints (13 endpoints)

#### 1. Create Alert Rule
```bash
curl -X POST http://localhost:8000/api/v1/alerts/rules \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "rule_name": "High Respiratory Rate Alert",
    "rule_type": "metric_deviation",
    "conditions": {
      "metric": "respiratory_rate",
      "threshold": 25,
      "operator": "greater_than"
    },
    "notification_channels": ["dashboard", "email"],
    "severity": "high"
  }'
```

#### 2. Get All Alert Rules for Doctor
```bash
curl -X GET http://localhost:8000/api/v1/alerts/rules \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 3. Get Active Alerts
```bash
curl -X GET "http://localhost:8000/api/v1/alerts/active?patient_id=patient-123" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

#### 4. Acknowledge Alert
```bash
curl -X PATCH http://localhost:8000/api/v1/alerts/789/acknowledge \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "notes": "Discussed with patient via phone call"
  }'
```

#### 5. Get Alert History
```bash
curl -X GET "http://localhost:8000/api/v1/alerts/history?patient_id=patient-123&days=30" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

---

## Testing Workflow

### Complete End-to-End Test

1. **Upload Video**
```bash
SESSION_ID=$(curl -X POST http://localhost:8000/api/v1/video-ai/upload \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "file=@test-video.mp4" \
  | jq -r '.session_id')

echo "Session ID: $SESSION_ID"
```

2. **Wait for Processing** (check status)
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/sessions/$SESSION_ID \
  -H "Authorization: Bearer $JWT_TOKEN"
```

3. **Get Analysis Results**
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/sessions/$SESSION_ID/metrics \
  -H "Authorization: Bearer $JWT_TOKEN"
```

4. **Check Risk Assessment** (triggers automatically)
```bash
curl -X GET http://localhost:8000/api/v1/trends/risk-assessment/patient-123 \
  -H "Authorization: Bearer $JWT_TOKEN"
```

5. **Check for Alerts** (generated if thresholds exceeded)
```bash
curl -X GET http://localhost:8000/api/v1/alerts/active?patient_id=patient-123 \
  -H "Authorization: Bearer $JWT_TOKEN"
```

---

## Common Issues & Troubleshooting

### 1. AWS_REGION Format Error
**Error:** `Provided region_name 'Asia Pacific (Sydney) ap-southeast-2' doesn't match a supported format`

**Fix:** Set `AWS_REGION=ap-southeast-2` (region code only, no description)

### 2. 401 Unauthorized
**Cause:** Invalid or expired JWT token

**Fix:** Get a fresh token from Cognito

### 3. 500 Internal Server Error
**Check:**
- Python backend logs for detailed error messages
- Database connection (verify DATABASE_URL)
- S3 bucket access (verify AWS credentials)

### 4. Video/Audio Processing Stuck
**Check:**
- ML models loaded successfully (check startup logs)
- Sufficient memory for TensorFlow/MediaPipe
- Valid video/audio format (MP4, WAV, MP3)

---

## Performance Expectations

**Video Analysis:**
- Upload: < 5 seconds for 10MB file
- Processing: 30-60 seconds for 1-minute video
- Metrics extraction: Real-time (< 100ms)

**Audio Analysis:**
- Upload: < 3 seconds for 5MB file
- Processing: 15-30 seconds for 1-minute audio
- Metrics extraction: Real-time (< 100ms)

**Trend Prediction:**
- Baseline calculation: 1-2 seconds for 30 days data
- Risk scoring: < 500ms
- Alert evaluation: < 200ms per rule

---

## Test Data

### Sample Test Files

**Video:** Upload any MP4 file with a person's face visible (for facial analysis)
**Audio:** Upload WAV/MP3 with clear speech or breathing sounds

### Sample Patient IDs

Replace `patient-123` with actual patient IDs from your database:
```sql
SELECT id FROM patients LIMIT 5;
```

---

## Success Criteria

✅ All 52 endpoints return valid responses (not 500 errors)
✅ Video upload → analysis → metrics displayed in dashboard
✅ Audio upload → analysis → metrics displayed in dashboard
✅ Risk assessment updates after new data
✅ Alerts trigger when thresholds exceeded
✅ Email/SMS notifications delivered (if configured)
✅ HIPAA audit logs created for all operations

---

## Next Steps After Testing

1. **Review audit logs** to verify HIPAA compliance
2. **Optimize ML inference** performance if needed
3. **Configure production alert rules** with clinical team
4. **Set up monitoring** for API response times
5. **Test with real patient data** (de-identified for HIPAA)

For complete API reference, see `AI_API_DOCUMENTATION.md`.
