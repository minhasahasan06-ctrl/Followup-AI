# Followup AI - Deterioration Detection API Documentation

## Overview

Complete API documentation for the AI Deterioration Detection System featuring 52 production endpoints across 4 AI engines:

- **Video AI Engine**: 13 endpoints for video-based health analysis
- **Audio AI Engine**: 13 endpoints for audio-based respiratory monitoring
- **Trend Prediction Engine**: 13 endpoints for health change detection
- **Alert Orchestration Engine**: 13 endpoints for multi-channel notifications

**Base URL**: `http://localhost:8000`  
**Authentication**: AWS Cognito JWT (Bearer token required)  
**Content-Type**: `application/json` (except file uploads: `multipart/form-data`)

---

## 🎥 Video AI Engine

### POST /api/v1/video-ai/upload
Upload video for AI analysis

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/video-ai/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "video_file=@/path/to/video.mp4"
```

**Response**:
```json
{
  "session_id": 42,
  "s3_key": "videos/patient_123/20240101_120000.mp4",
  "upload_url": null,
  "processing_status": "queued",
  "message": "Video uploaded successfully. Analysis will begin shortly."
}
```

### GET /api/v1/video-ai/sessions/me
Fetch all video sessions for authenticated patient

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/sessions/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
[
  {
    "session_id": 42,
    "patient_id": "patient_123",
    "s3_key": "videos/patient_123/20240101_120000.mp4",
    "upload_timestamp": "2024-01-01T12:00:00Z",
    "processing_status": "completed",
    "analysis_completed_at": "2024-01-01T12:02:15Z"
  }
]
```

### GET /api/v1/video-ai/metrics/latest/me
Get latest video metrics for authenticated patient

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/metrics/latest/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
{
  "session_id": 42,
  "respiratory_rate": 16.5,
  "skin_pallor_score": 0.15,
  "sclera_yellowness": 0.08,
  "facial_swelling_score": 0.12,
  "head_tremor_detected": false,
  "facial_asymmetry_score": 0.05,
  "eye_redness_score": 0.10,
  "lighting_quality": 0.85,
  "frame_stability": 0.92,
  "face_detection_confidence": 0.95,
  "quality_score": 0.88,
  "confidence": 0.91,
  "processing_time": 45.2,
  "timestamp": "2024-01-01T12:02:15Z",
  "recommendations": [
    "Respiratory rate within healthy range",
    "No significant facial changes detected",
    "Continue regular monitoring"
  ]
}
```

### GET /api/v1/video-ai/metrics/{session_id}
Get video metrics for specific session

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/video-ai/metrics/42 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### GET /api/v1/video-ai/history/me
Get video metrics history (last 30 days)

**Request**:
```bash
curl -X GET "http://localhost:8000/api/v1/video-ai/history/me?days=30" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**: Array of video metrics with timestamps

### POST /api/v1/video-ai/reanalyze/{session_id}
Trigger re-analysis of existing video session

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/video-ai/reanalyze/42 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
{
  "message": "Re-analysis queued successfully",
  "session_id": 42,
  "status": "processing"
}
```

### DELETE /api/v1/video-ai/session/{session_id}
Delete video session and associated data

**Request**:
```bash
curl -X DELETE http://localhost:8000/api/v1/video-ai/session/42 \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### GET /api/v1/video-ai/compare
Compare two video sessions

**Request**:
```bash
curl -X GET "http://localhost:8000/api/v1/video-ai/compare?session_id_1=40&session_id_2=42" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
{
  "session_1": { "session_id": 40, "timestamp": "2024-01-01T10:00:00Z", "metrics": {...} },
  "session_2": { "session_id": 42, "timestamp": "2024-01-01T12:00:00Z", "metrics": {...} },
  "differences": {
    "respiratory_rate": { "change": "+2.3", "percent": "+16.2%" },
    "skin_pallor_score": { "change": "+0.05", "percent": "+50%" }
  },
  "interpretation": "Respiratory rate increased moderately. Consider discussing with provider."
}
```

---

## 🎤 Audio AI Engine

### POST /api/v1/audio-ai/upload
Upload audio for AI analysis

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/audio-ai/upload \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -F "audio_file=@/path/to/audio.wav"
```

**Response**:
```json
{
  "session_id": 15,
  "s3_key": "audio/patient_123/20240101_120000.wav",
  "upload_url": null,
  "processing_status": "queued",
  "message": "Audio uploaded successfully. Analysis will begin shortly."
}
```

### GET /api/v1/audio-ai/sessions/me
Fetch all audio sessions for authenticated patient

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/audio-ai/sessions/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

### GET /api/v1/audio-ai/metrics/latest/me
Get latest audio metrics for authenticated patient

**Response**:
```json
{
  "session_id": 15,
  "breath_cycles_per_min": 14.2,
  "speech_pace_wpm": 125.5,
  "cough_detected": false,
  "wheeze_detected": false,
  "voice_hoarseness_score": 0.12,
  "average_loudness_db": -18.5,
  "speech_to_silence_ratio": 0.65,
  "audio_quality_snr": 22.5,
  "confidence": 0.89,
  "quality_score": 0.85,
  "processing_time": 32.1,
  "timestamp": "2024-01-01T12:05:30Z",
  "recommendations": [
    "Breath cycles within healthy range",
    "No respiratory distress detected",
    "Speech pace normal"
  ]
}
```

### GET /api/v1/audio-ai/metrics/{session_id}
Get audio metrics for specific session

### GET /api/v1/audio-ai/history/me
Get audio metrics history (last 30 days)

### POST /api/v1/audio-ai/reanalyze/{session_id}
Trigger re-analysis of existing audio session

### DELETE /api/v1/audio-ai/session/{session_id}
Delete audio session and associated data

### GET /api/v1/audio-ai/compare
Compare two audio sessions

---

## 📈 Trend Prediction Engine

### GET /api/v1/trends/risk-score/me
Calculate current risk score for authenticated patient

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/trends/risk-score/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
{
  "patient_id": "patient_123",
  "risk_score": 8.5,
  "risk_level": "monitoring",
  "confidence": 0.87,
  "calculated_at": "2024-01-01T12:10:00Z",
  "contributing_factors": [
    {
      "metric": "respiratory_rate",
      "z_score": 2.1,
      "severity": "moderate",
      "points": 3.0
    },
    {
      "metric": "skin_pallor_score",
      "z_score": 1.8,
      "severity": "moderate",
      "points": 2.5
    }
  ],
  "recommendations": [
    "Consider scheduling follow-up with healthcare provider",
    "Continue daily monitoring",
    "Track symptoms in symptom journal"
  ],
  "interpretation": "Your wellness metrics show moderate changes from baseline. This is not an emergency, but discussing with your provider is recommended."
}
```

### GET /api/v1/trends/history/me
Get risk score history (last 30 days)

**Request**:
```bash
curl -X GET "http://localhost:8000/api/v1/trends/history/me?days=30" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
[
  {
    "date": "2024-01-01",
    "risk_score": 8.5,
    "risk_level": "monitoring",
    "deviation_count": 5,
    "critical_count": 0,
    "moderate_count": 2
  },
  {
    "date": "2023-12-31",
    "risk_score": 6.2,
    "risk_level": "stable",
    "deviation_count": 3,
    "critical_count": 0,
    "moderate_count": 1
  }
]
```

### POST /api/v1/trends/snapshot
Create manual health snapshot

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/trends/snapshot \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "video_session_id": 42,
    "audio_session_id": 15,
    "manual_metrics": {
      "temperature": 98.6,
      "blood_pressure_systolic": 120,
      "blood_pressure_diastolic": 80
    }
  }'
```

### GET /api/v1/trends/snapshots/me
Get all trend snapshots for patient

### GET /api/v1/trends/baseline/me
Get current baseline for authenticated patient

**Response**:
```json
{
  "baseline_id": 5,
  "patient_id": "patient_123",
  "calculated_at": "2024-01-01T00:00:00Z",
  "data_points_count": 45,
  "baseline_quality": "good",
  "metrics": {
    "respiratory_rate": {
      "mean": 14.2,
      "std": 1.8,
      "min": 11.5,
      "max": 17.2
    },
    "skin_pallor_score": {
      "mean": 0.10,
      "std": 0.03,
      "min": 0.05,
      "max": 0.15
    }
  }
}
```

### POST /api/v1/trends/baseline/recalculate
Trigger baseline recalculation

### GET /api/v1/trends/deviations/me
Get recent deviations from baseline

**Response**:
```json
[
  {
    "deviation_id": 123,
    "metric_name": "respiratory_rate",
    "current_value": 18.5,
    "baseline_mean": 14.2,
    "z_score": 2.4,
    "percent_change": 30.3,
    "severity": "moderate",
    "detected_at": "2024-01-01T12:00:00Z",
    "alert_triggered": true
  }
]
```

### GET /api/v1/trends/anomalies/me
Detect anomalies using Bayesian analysis

### POST /api/v1/trends/predict
Predict future health trajectory

---

## 🔔 Alert Orchestration Engine

### GET /api/v1/alerts/pending
Get pending (unacknowledged) alerts

**Request**:
```bash
curl -X GET http://localhost:8000/api/v1/alerts/pending \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
[
  {
    "alert_id": 89,
    "patient_id": "patient_123",
    "alert_type": "deterioration_detected",
    "severity": "high",
    "message": "Respiratory rate increased significantly above baseline. Consider consulting your healthcare provider.",
    "triggered_at": "2024-01-01T12:15:00Z",
    "acknowledged": false,
    "delivery_status": {
      "dashboard": true,
      "email": true,
      "sms": false
    },
    "metadata": {
      "metric": "respiratory_rate",
      "current_value": 22.5,
      "baseline": 14.2,
      "z_score": 4.6
    }
  }
]
```

### GET /api/v1/alerts/history/me
Get alert history (last 30 days)

### POST /api/v1/alerts/{alert_id}/acknowledge
Acknowledge an alert

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/alerts/89/acknowledge \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

**Response**:
```json
{
  "message": "Alert acknowledged successfully",
  "alert_id": 89,
  "acknowledged_at": "2024-01-01T12:30:00Z"
}
```

### POST /api/v1/alerts/acknowledge-all
Acknowledge all pending alerts

### GET /api/v1/alerts/rules/me
Get configured alert rules for patient

**Response**:
```json
[
  {
    "rule_id": 5,
    "patient_id": "patient_123",
    "rule_name": "High Respiratory Rate Alert",
    "metric_name": "respiratory_rate",
    "threshold_type": "exceeds",
    "threshold_value": 20.0,
    "severity": "high",
    "enabled": true,
    "notification_channels": ["dashboard", "email", "sms"],
    "created_at": "2024-01-01T00:00:00Z"
  }
]
```

### POST /api/v1/alerts/rules
Create new alert rule

**Request**:
```bash
curl -X POST http://localhost:8000/api/v1/alerts/rules \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "rule_name": "Low Speech Pace Alert",
    "metric_name": "speech_pace_wpm",
    "threshold_type": "below",
    "threshold_value": 100.0,
    "severity": "medium",
    "notification_channels": ["dashboard", "email"]
  }'
```

### PATCH /api/v1/alerts/rules/{rule_id}
Update alert rule

### DELETE /api/v1/alerts/rules/{rule_id}
Delete alert rule

### GET /api/v1/alerts/stats/me
Get alert statistics

**Response**:
```json
{
  "total_alerts_30d": 12,
  "pending_alerts": 2,
  "acknowledged_alerts": 10,
  "alerts_by_severity": {
    "critical": 1,
    "high": 3,
    "medium": 5,
    "low": 3
  },
  "most_common_alert_type": "respiratory_change",
  "avg_acknowledgment_time_minutes": 45.2
}
```

---

## 🔐 Authentication

All endpoints require AWS Cognito JWT authentication. Include the token in the Authorization header:

```bash
Authorization: Bearer eyJraWQiOiJxxx...
```

### Getting a JWT Token

1. Login via AWS Cognito
2. Receive JWT access token
3. Use token for all API requests
4. Token expires after 1 hour (renew as needed)

---

## 📊 Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request (invalid input) |
| 401 | Unauthorized (missing/invalid JWT) |
| 403 | Forbidden (not your data) |
| 404 | Not Found |
| 422 | Validation Error |
| 500 | Internal Server Error |

---

## 🧪 Testing Examples

### Example 1: Complete Video Analysis Workflow

```bash
# 1. Upload video
SESSION_ID=$(curl -X POST http://localhost:8000/api/v1/video-ai/upload \
  -H "Authorization: Bearer $JWT_TOKEN" \
  -F "video_file=@selfie_video.mp4" | jq -r '.session_id')

# 2. Wait for processing (poll status)
sleep 60

# 3. Get results
curl -X GET "http://localhost:8000/api/v1/video-ai/metrics/$SESSION_ID" \
  -H "Authorization: Bearer $JWT_TOKEN"

# 4. Check if alerts were triggered
curl -X GET http://localhost:8000/api/v1/alerts/pending \
  -H "Authorization: Bearer $JWT_TOKEN"
```

### Example 2: Trend Analysis Workflow

```bash
# 1. Get current risk score
curl -X GET http://localhost:8000/api/v1/trends/risk-score/me \
  -H "Authorization: Bearer $JWT_TOKEN"

# 2. Get 30-day history
curl -X GET "http://localhost:8000/api/v1/trends/history/me?days=30" \
  -H "Authorization: Bearer $JWT_TOKEN"

# 3. Check baseline
curl -X GET http://localhost:8000/api/v1/trends/baseline/me \
  -H "Authorization: Bearer $JWT_TOKEN"

# 4. Get deviations
curl -X GET http://localhost:8000/api/v1/trends/deviations/me \
  -H "Authorization: Bearer $JWT_TOKEN"
```

---

## 💡 Best Practices

1. **Upload Frequency**: Upload video/audio 1-2 times daily for best trend detection
2. **Video Quality**: Use good lighting, stable camera, clear face view (30-60 seconds)
3. **Audio Quality**: Record in quiet environment, speak naturally (30-60 seconds)
4. **Baseline Building**: Requires 7+ days of consistent data for accurate baselines
5. **Alert Management**: Review and acknowledge alerts promptly
6. **Privacy**: All media is encrypted at rest (S3 SSE-KMS) and in transit (TLS)

---

## 🎯 Regulatory & Legal

- **Classification**: General Wellness Product (NOT a medical device)
- **Purpose**: Wellness monitoring and change detection
- **Not For**: Medical diagnosis, treatment decisions, emergency detection
- **Always**: Discuss health concerns with qualified healthcare providers
- **HIPAA**: All endpoints are HIPAA-compliant with audit logging

---

## 📞 Support

- **Technical Issues**: Check logs at `/tmp/python_backend.log`
- **API Questions**: Reference this documentation
- **Healthcare Questions**: Consult your provider (NOT us!)

**Version**: 1.0.0  
**Last Updated**: January 2024
