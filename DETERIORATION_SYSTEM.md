# Deterioration Prediction System - Complete Implementation

## Overview
The Deterioration Prediction System is a comprehensive wellness monitoring platform that detects health pattern changes BEFORE deterioration occurs. This system is strategically positioned as a **General Wellness Product** to avoid FDA/CE medical device approval requirements.

## System Architecture

### 1. Baseline Calculation Engine (`app/services/baseline_service.py`)
**Purpose:** Establish personalized health baselines for each patient

**Features:**
- 7-day rolling window baseline calculation
- Tracks 4 core metrics:
  - Pain (facial analysis)
  - Pain (self-reported)
  - Respiratory rate
  - Symptom severity
- Calculates: mean, std, min, max for each metric
- Baseline quality classification: "excellent" (7+ days), "good" (5-6 days), "fair" (3-4 days), "poor" (<3 days)

**Key Fix:** Single-sample statistics bug resolved (now requires ≥1 sample for mean/min/max, ≥2 for std)

**API Endpoints:**
- `POST /api/v1/baseline/calculate` - Calculate new baseline
- `GET /api/v1/baseline/current/me` - Get current patient baseline
- `GET /api/v1/baseline/current/patient/{id}` - Get patient baseline (doctor only)

### 2. Deviation Detection Service (`app/services/deviation_service.py`)
**Purpose:** Detect significant changes from personal baseline

**Features:**
- Z-score analysis (measures standard deviations from baseline)
- Trend detection using linear regression (3-day and 7-day slopes)
- Severity classification:
  - **Critical High:** z > 3.0
  - **Moderate High:** z > 2.0
  - **Critical Low:** z < -2.5
  - **Moderate Low:** z < -1.5
- Automatic alert triggering for critical deviations

**Security Fix:** Added patient ownership verification to prevent enumeration attacks

**API Endpoints:**
- `POST /api/v1/deviation/detect` - Detect deviations
- `GET /api/v1/deviation/me` - Get patient's deviations
- `GET /api/v1/deviation/patient/{id}` - Get patient deviations (doctor only)

### 3. Risk Scoring Engine (`app/services/risk_scoring_service.py`)
**Purpose:** Generate composite wellness risk score (0-15 scale)

**Features:**
- Weighted scoring system:
  - Respiratory changes: +3 to +5 points (critical health indicator)
  - Pain increases: +3 to +4 points
  - Symptom severity: +2 to +3 points
  - Activity decline: +1 to +2 points
- Risk level classification:
  - **Stable (0-2):** No concerning changes
  - **Monitoring Needed (3-5):** Some changes detected
  - **Urgent Attention (6-15):** Multiple critical changes
- Wellness recommendations with action items
- Historical risk tracking for trend analysis

**Key Fix:** Increased maximum score from 10 to 15 to allow full range

**API Endpoints:**
- `GET /api/v1/risk/score/me` - Get current risk score (patient)
- `GET /api/v1/risk/history/me` - Get 7-day risk history (patient)
- `GET /api/v1/risk/score/patient/{id}` - Get patient risk score (doctor)

### 4. Deterioration Dashboard (`client/src/pages/DeteriorationDashboard.tsx`)
**Purpose:** Patient-facing visual wellness monitoring interface

**Features:**
- **Risk Score Display:** Large, color-coded score (0-15) with progress bar
- **7-Day Trend Chart:** Line graph showing risk score evolution
- **Pattern Changes:** Detailed breakdown of detected deviations
  - Metric name, z-score, severity level
  - Trend indicators (up/down arrows)
  - Percentage change from baseline
- **Wellness Recommendations:** AI-generated action items
- **Baseline Status:** Data quality indicators

**Regulatory Compliance:**
- Legal disclaimer on every page
- "Wellness monitoring" language (not "diagnosis")
- Clear guidance to consult healthcare providers
- Positioned as informational tool only

**Access:** Patient-only route at `/deterioration`

## Regulatory Positioning Strategy

### Why This Avoids FDA/CE Approval

**General Wellness Product Classification:**
1. ✅ **No diagnostic claims** - Monitors wellness trends, doesn't diagnose disease
2. ✅ **No treatment claims** - Provides information for discussion with doctors
3. ✅ **Wellness focus** - Tracks personal health patterns, not clinical outcomes
4. ✅ **User responsibility** - Explicitly states users make own health decisions
5. ✅ **Informational only** - All data "for informational purposes only"

**Language Compliance:**
- ❌ NEVER say: "diagnose", "treat", "prevent disease", "clinical decision", "medical device"
- ✅ ALWAYS say: "wellness monitoring", "change detection", "personal trends", "discuss with doctor"

**Legal Protection:**
- Terms of Service: clarifies wellness monitoring purpose
- Privacy Policy: explains data usage for personal tracking
- Disclaimer component: shown on every health monitoring page
- Agent Clona: programmed to avoid medical advice

## Technical Implementation Details

### Database Models
- `HealthBaseline` - Stores 7-day rolling baselines per patient
- `BaselineDeviation` - Records all detected deviations with z-scores
- Foreign key relationships ensure data integrity

### Security & HIPAA Compliance
- ✅ Role-based access control (patient vs doctor endpoints)
- ✅ Patient ownership verification on all queries
- ✅ 404 returns for unauthorized access (prevents enumeration)
- ✅ Audit logging for all PHI access
- ✅ Encrypted data transmission

### Performance Optimizations
- Baseline calculated once daily (not real-time)
- Deviations detected on new measurements only
- Risk score cached for 24-hour window
- Database indexes on patient_id, measurement_date

## Integration with Existing Features

The deterioration prediction system enhances:
1. **Pain Detection Camera:** Facial pain scores feed baseline calculation
2. **Symptom Journal:** Respiratory rate and symptom severity tracked
3. **Medication Side Effects:** Deviations can indicate medication changes
4. **Agent Clona:** Risk scores inform AI wellness recommendations
5. **Doctor Portal:** Doctors view patient risk via PatientReview page

## Startup Instructions

**IMPORTANT:** This system requires TWO backend servers:

1. **Frontend (Node.js):** Port 5000 - Already auto-starts via Replit workflow
2. **Backend (Python FastAPI):** Port 8000 - **Must start manually**

**To start Python backend:**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**OR use unified startup script:**
```bash
bash start-servers.sh
```

**Verify both servers:**
```bash
curl http://localhost:5000/  # Frontend
curl http://localhost:8000/health  # Python backend
```

## Testing Checklist

- [x] Baseline calculation with 1 sample (mean/min/max)
- [x] Baseline calculation with 2+ samples (includes std)
- [x] Deviation detection with z-score analysis
- [x] Risk score calculation (0-15 range)
- [x] Patient ownership verification on all endpoints
- [x] 404 returns for unauthorized access
- [x] Dashboard displays risk score correctly
- [x] 7-day trend chart renders
- [x] Wellness recommendations shown
- [x] Legal disclaimer on dashboard
- [x] Doctor access restricted from patient-only endpoints

## Known Limitations

1. **Startup Complexity:** Requires manual Python backend start
   - Solution: Use `start-servers.sh` script
   - Future: Could create unified workflow

2. **Baseline Requirement:** Needs 7+ days of data for accurate baselines
   - Expected behavior for new patients
   - Quality improves over time

3. **Real-Time Updates:** Risk scores update on page refresh only
   - Could add WebSocket for live updates
   - Current implementation sufficient for MVP

## Future Enhancements

1. **Machine Learning Models:** Replace rule-based scoring with ML predictions
2. **Wearable Integration:** Add heart rate, SpO2, temperature to baseline
3. **Predictive Alerts:** Push notifications when risk score increases
4. **Doctor Alerts:** Automatic alerts to assigned doctors for urgent cases
5. **Historical Comparison:** Compare current patterns to previous episodes
6. **Multi-Patient Dashboard:** Doctor view of all patients' risk scores

## Documentation References

- Baseline API: `app/routers/baseline.py`
- Deviation API: `app/routers/deviation.py`
- Risk Scoring API: `app/routers/risk_score.py`
- Dashboard UI: `client/src/pages/DeteriorationDashboard.tsx`
- Startup Guide: `README_STARTUP.md`
- Design Guidelines: `design_guidelines.md`
- Project Overview: `replit.md`

---

**Status:** ✅ Complete and ready for use
**Version:** 1.0.0
**Last Updated:** November 16, 2025
