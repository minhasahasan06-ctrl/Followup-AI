# Advanced Respiratory Metrics System - Implementation Guide

## Overview
This document describes the comprehensive respiratory monitoring system with baseline tracking, temporal analytics, and deterioration detection for the Followup AI platform.

## Clinical Metrics Implemented

### 1. Core Respiratory Metrics
- **Respiratory Rate (RR)**: Breaths per minute detected from chest movement using FFT analysis
- **RR Confidence**: Detection confidence score (0-1) based on signal quality
- **Chest Movement Amplitude**: Standard deviation of chest expansion movements

### 2. Baseline & Anomaly Detection
- **Baseline RR**: Patient's normal respiratory rate (auto-calculated from first 5+ sessions)
- **Baseline RR Std**: Standard deviation for Z-score computation
- **Z-Score vs Baseline**: Anomaly score = (current_RR - baseline_mean) / baseline_std
  - |Z| < 1: Normal variation
  - 1 < |Z| < 2: Mild anomaly  
  - |Z| > 2: Significant anomaly (potential deterioration)

### 3. Temporal Analytics
- **Rolling Daily Average**: Mean RR over last 24 hours
- **Rolling 3-Day Slope**: Linear regression trend over daily averages
  - Positive slope: RR increasing (potential worsening)
  - Negative slope: RR decreasing (potential improvement)
  - Near-zero slope: Stable condition

### 4. Respiratory Variability Index (RVI)
- **Definition**: Coefficient of variation of breath intervals over 1-5 minutes
- **Formula**: RVI = (std_dev_breath_intervals / mean_breath_interval) * 100
- **Interpretation**:
  - Low RVI (<15%): Stable, regular breathing
  - Medium RVI (15-30%): Moderate variability
  - High RVI (>30%): Irregular breathing, potential respiratory distress

### 5. Advanced Pattern Detection
- **Accessory Muscle Score**: Neck/shoulder elevation during breathing (0-100)
  - Indicator of labored breathing
  - High scores suggest respiratory effort
  
- **Gasping Detection**: Boolean flag for irregular deep breaths
  - Detects sudden amplitude spikes (>2 std dev from mean)
  - >20% of breaths being outliers = gasping detected
  
- **Chest Shape Asymmetry**: Coefficient of variation in chest width
  - Detects barrel chest or asymmetric expansion
  - May indicate COPD or structural issues
  
- **Thoracoabdominal Synchrony**: Score 0-1 measuring breathing coordination
  - 1.0 = perfectly synchronous (healthy)
  - <0.5 = asynchronous (potential respiratory fatigue)

## Database Schema

### RespiratoryBaseline Table
```sql
CREATE TABLE respiratory_baselines (
    patient_id VARCHAR PRIMARY KEY,
    baseline_rr_bpm FLOAT NOT NULL,
    baseline_rr_std FLOAT NOT NULL,
    sample_size INT DEFAULT 0,
    confidence FLOAT DEFAULT 0.0,
    source VARCHAR DEFAULT 'auto',  -- 'auto' or 'manual'
    updated_at TIMESTAMP,
    created_at TIMESTAMP
);
```

### RespiratoryMetric Table
```sql
CREATE TABLE respiratory_metrics (
    id VARCHAR PRIMARY KEY,
    patient_id VARCHAR NOT NULL,
    session_id VARCHAR,
    recorded_at TIMESTAMP NOT NULL,
    
    -- Core metrics
    rr_bpm FLOAT,
    rr_confidence FLOAT,
    
    -- Variability
    breath_interval_std FLOAT,
    variability_index FLOAT,
    
    -- Advanced detection
    accessory_muscle_score FLOAT,
    chest_expansion_amplitude FLOAT,
    gasping_detected BOOLEAN,
    chest_shape_asymmetry FLOAT,
    thoracoabdominal_synchrony FLOAT,
    
    -- Temporal analytics
    z_score_vs_baseline FLOAT,
    rolling_daily_avg FLOAT,
    rolling_three_day_slope FLOAT,
    
    -- Raw data
    metadata JSON,
    created_at TIMESTAMP,
    
    INDEX idx_respiratory_patient_time (patient_id, recorded_at)
);
```

## Service Architecture

### respiratory_metrics_service.py
Handles all temporal analytics and database persistence:

**Key Methods:**
```python
ingest_session(patient_id, session_id, rr_bpm, chest_movements, fps)
    → Computes all metrics
    → Stores in RespiratoryMetric table
    → Updates patient baseline
    → Returns stored record

_compute_variability_index(chest_movements, fps)
    → Detects breath cycles using peak detection
    → Computes inter-breath intervals
    → Returns RVI and breath_interval_std

_detect_gasping(chest_movements)
    → Analyzes amplitude distribution
    → Detects outlier breaths
    → Returns boolean gasping flag

_update_baseline(patient_id, new_rr, confidence)
    → Uses exponential moving average
    → Gradually refines baseline over time
    → Updates confidence score

_compute_rolling_daily_avg(patient_id, timestamp)
    → Queries last 24 hours of metrics
    → Returns mean RR

_compute_rolling_three_day_slope(patient_id, timestamp)
    → Groups metrics by day
    → Performs linear regression on daily averages
    → Returns slope (trend direction)
```

### video_ai_engine.py Integration
Enhanced detection methods:

**_detect_chest_movement()** now returns:
```python
{
    'chest_movement': float,  # Vertical displacement
    'accessory_muscle_activity': float,  # Neck elevation score
    'chest_width_proxy': float  # For asymmetry analysis
}
```

**After video processing**, calls respiratory metrics service:
```python
respiratory_service = RespiratoryMetricsService(db_session)
metric = respiratory_service.ingest_session(
    patient_id=patient_id,
    session_id=session_id,
    rr_bpm=metrics['respiratory_rate_bpm'],
    chest_movements=chest_movements_list,
    accessory_muscle_scores=accessory_muscle_list,
    chest_widths=chest_widths_list,
    fps=fps
)
```

## API Endpoints (Planned)

### POST /api/v1/respiratory/sessions
Ingest respiratory data from video analysis
```json
Request:
{
  "patient_id": "uuid",
  "session_id": "uuid",
  "rr_bpm": 16.5,
  "rr_confidence": 0.85,
  "chest_movements": [array],
  "fps": 30
}

Response:
{
  "id": "metric_uuid",
  "rr_bpm": 16.5,
  "z_score_vs_baseline": -0.3,
  "rolling_daily_avg": 17.2,
  "rolling_three_day_slope": 0.1,
  "variability_index": 12.5,
  "gasping_detected": false,
  "recorded_at": "2025-11-17T15:30:00Z"
}
```

### GET /api/v1/respiratory/metrics/{patient_id}
Retrieve respiratory history
```json
Query params:
- start_date (optional)
- end_date (optional)
- limit (default 100)

Response:
{
  "patient_id": "uuid",
  "metrics": [{...}, {...}],
  "count": 45
}
```

### GET /api/v1/respiratory/baselines/{patient_id}
Get patient baseline
```json
Response:
{
  "patient_id": "uuid",
  "baseline_rr_bpm": 16.0,
  "baseline_rr_std": 2.1,
  "sample_size": 12,
  "confidence": 0.90,
  "updated_at": "2025-11-17T10:00:00Z"
}
```

### PUT /api/v1/respiratory/baselines/{patient_id}
Clinician override of baseline (manual calibration)
```json
Request:
{
  "baseline_rr_bpm": 15.0,
  "baseline_rr_std": 1.5,
  "source": "manual"
}
```

### GET /api/v1/respiratory/summary/{patient_id}
Comprehensive respiratory dashboard data
```json
Response:
{
  "baseline": {...},
  "latest": {
    "rr_bpm": 18.5,
    "z_score": 1.2,
    "recorded_at": "2025-11-17T15:30:00Z",
    "variability_index": 18.3
  },
  "trend": "stable",
  "recent_count": 15,
  "alerts": [
    {
      "type": "elevated_rr",
      "severity": "medium",
      "message": "Respiratory rate elevated above baseline"
    }
  ]
}
```

## Clinical Interpretation Guide

### Normal Ranges (Adult)
- **RR**: 12-20 breaths/min
- **RVI**: <20% (stable breathing)
- **Accessory Muscle Score**: <20 (minimal effort)
- **Thoracoabdominal Synchrony**: >0.7 (good coordination)

### Alert Thresholds
| Metric | Warning | Critical |
|--------|---------|----------|
| RR | >24 or <10 | >28 or <8 |
| Z-Score | ±1.5 | ±2.5 |
| RVI | >25% | >40% |
| 3-Day Slope | >1.0 | >2.0 |
| Accessory Muscles | >40 | >60 |

### Deterioration Patterns
1. **Gradual Worsening**:
   - Positive 3-day slope
   - Increasing Z-scores
   - Rising RR over days

2. **Acute Event**:
   - Sudden Z-score spike (>2)
   - Gasping detected
   - High accessory muscle use

3. **Respiratory Fatigue**:
   - High RVI (irregular breathing)
   - Low synchrony (<0.5)
   - Increasing accessory muscle scores

## Implementation Status

### ✅ Completed
- [x] Database models (RespiratoryBaseline, RespiratoryMetric)
- [x] Enhanced video_ai_engine.py with chest movement, accessory muscle, chest width detection
- [x] Comprehensive respiratory analysis logic documented

### 🚧 In Progress
- [ ] respiratory_metrics_service.py implementation
- [ ] Integration with video_ai_engine
- [ ] FastAPI router endpoints

### 📋 Pending
- [ ] Database migration/schema sync
- [ ] End-to-end testing
- [ ] Frontend UI for respiratory trends
- [ ] Alert system integration

## Testing Plan

### Unit Tests
1. RVI calculation with synthetic breathing data
2. Gasping detection with irregular patterns
3. Baseline update algorithm with EMA
4. Z-score computation accuracy
5. Rolling average/slope calculations

### Integration Tests
1. Video analysis → metric storage workflow
2. Baseline creation and updates
3. Historical data retrieval
4. API endpoint responses

### Clinical Validation
1. Compare RVI against medical-grade equipment
2. Validate Z-score thresholds with clinical data
3. Test accessory muscle detection accuracy
4. Verify gasping detection sensitivity/specificity

## Next Steps

1. **Complete Service Implementation** (respiratory_metrics_service.py)
   - Implement all metric computation methods
   - Add database session management
   - Include error handling and logging

2. **Video AI Integration**
   - Modify video_ai_engine to call metrics service
   - Pass database session via dependency injection
   - Store session metadata properly

3. **API Endpoints**
   - Create respiratory.py router
   - Implement CRUD operations
   - Add authentication/authorization

4. **Testing & Validation**
   - End-to-end workflow tests
   - Clinical accuracy validation
   - Performance optimization

5. **Frontend Dashboard**
   - Respiratory trends visualization
   - Alert notifications
   - Historical comparison charts

---
**Document Version:** 1.0  
**Last Updated:** 2025-11-17  
**Status:** Implementation in progress
