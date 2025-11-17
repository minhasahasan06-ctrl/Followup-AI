# Respiratory Metrics - Implementation Status

## ✅ COMPLETED METRICS

### Core Detection (All Implemented)
| Metric | Field Name | Description | Status |
|--------|-----------|-------------|---------|
| Chest Expansion | `chest_expansion_amplitude` | Standard deviation of chest movement | ✅ |
| Accessory Muscles | `accessory_muscle_score` | Neck muscle usage (0-100) | ✅ |
| Gasping Detection | `gasping_detected` | Irregular deep breaths (boolean) | ✅ |
| Chest Shape | `chest_shape_asymmetry` | Barrel chest/asymmetry score | ✅ |
| Respiratory Rate | `rr_bpm` | Breaths per minute | ✅ |

### Baseline & Temporal Analytics (All Implemented)
| Metric | Field Name | Description | Status |
|--------|-----------|-------------|---------|
| Baseline RR | `baseline_rr_bpm` | Patient's normal RR (auto-calculated) | ✅ |
| Rolling Daily Avg | `rolling_daily_avg` | Mean RR last 24 hours | ✅ |
| Rolling 3-Day Slope | `rolling_three_day_slope` | Linear trend over 3 days | ✅ |
| **RR Variability Index** | `variability_index` | Stability: (std/mean)*100 over 1-5 min | ✅ |
| Z-Score Anomaly | `z_score_vs_baseline` | (RR - baseline)/std | ✅ |
| Thoracoabdominal Synchrony | `thoracoabdominal_synchrony` | Breathing coordination (0-1) | ✅ |

## Implementation Details

### Database Models ✅
**File:** `app/models.py`
- `RespiratoryBaseline`: Patient baseline tracking
- `RespiratoryMetric`: Time-series metrics storage
- `RespiratoryConditionProfile`: Disease-specific profiles (added)
- `RespiratoryConditionThreshold`: Condition-specific thresholds (added)

### Service Layer ✅
**File:** `app/services/respiratory_metrics_service.py`

**Main Method:**
```python
ingest_session(
    patient_id, session_id, rr_bpm, rr_confidence,
    chest_movements, accessory_muscle_scores, chest_widths, fps
) → RespiratoryMetric
```

**Computation Methods:**
- `_compute_variability_index()` - RVI with scipy peak detection
- `_detect_gasping()` - Outlier amplitude detection
- `_compute_chest_asymmetry()` - Coefficient of variation
- `_estimate_synchrony()` - Regularity-based synchrony
- `_update_baseline()` - Exponential moving average
- `_compute_z_score()` - Anomaly scoring
- `_compute_rolling_daily_avg()` - 24-hour mean
- `_compute_rolling_three_day_slope()` - Linear regression
- `get_patient_summary()` - Dashboard summary

**Bug Fixes Applied:**
1. ✅ Baseline updated BEFORE Z-score computation
2. ✅ baseline_rr_std initialized to 2.0 (prevents divide-by-zero)
3. ✅ Rolling stats computed AFTER metric insert (includes current session)

## 🔧 PENDING WORK

### Integration Points
- [ ] Integrate with `video_ai_engine.py` to call `ingest_session()` after analysis
- [ ] Create FastAPI router endpoints:
  - POST `/api/v1/respiratory/sessions` - Ingest metrics
  - GET `/api/v1/respiratory/metrics/{patient_id}` - Retrieve history
  - GET `/api/v1/respiratory/baselines/{patient_id}` - Get baseline
  - GET `/api/v1/respiratory/summary/{patient_id}` - Dashboard data
  
### Disease-Specific Personalization
- [ ] Complete `ConditionPersonalizationService` with threshold logic for:
  - Asthma (emphasize RVI, accessory muscles)
  - COPD (barrel chest, accessory muscles, elevated baseline)
  - Heart Failure (synchrony, edema proxy)
  - Pulmonary Embolism (sudden RR changes)
  - Pneumonia/TB (sustained elevation, cough)
  - Allergic Reactions (spike detection)
  - Bronchiectasis (chronic patterns)
- [ ] Integrate personalization with `RespiratoryMetricsService`
- [ ] Create wellness-compliant alert templates

### Testing & Validation
- [ ] Database migration to create tables
- [ ] Unit tests for each metric computation
- [ ] End-to-end test: video → metrics → storage → API
- [ ] Validate clinical accuracy of RVI, Z-scores, slopes

### Frontend
- [ ] Respiratory metrics dashboard UI
- [ ] Trend visualization (line charts for RR, RVI over time)
- [ ] Alert notifications for anomalies
- [ ] Condition profile management UI

## How Metrics Are Computed

### 1. Respiratory Variability Index (RVI)
```
1. Detect breath cycles using peak detection on chest movements
2. Calculate inter-breath intervals (in seconds)
3. RVI = (std_dev / mean) * 100

Interpretation:
- <15%: Stable breathing
- 15-30%: Moderate variability
- >30%: Irregular/distressed breathing
```

### 2. Baseline Tracking
```
First session: baseline = current RR, std = 2.0
Subsequent sessions (EMA update):
  alpha = 0.2 (after 5 sessions) or 1/(n+1) (first 5)
  new_baseline = old * (1-alpha) + new_rr * alpha
  new_std = old_std * (1-alpha) + deviation * alpha
```

### 3. Z-Score Anomaly
```
Z = (current_RR - baseline_mean) / baseline_std

Interpretation:
- |Z| < 1: Normal variation
- 1 < |Z| < 2: Mild anomaly
- |Z| > 2: Significant anomaly (alert!)
```

### 4. Rolling 3-Day Slope
```
1. Group metrics by day
2. Calculate daily mean RR
3. Linear regression: RR = slope * day + intercept
4. Positive slope = worsening, negative = improving
```

### 5. Gasping Detection
```
1. Divide chest movements into windows
2. Calculate amplitude per window
3. Detect outliers (>2 std dev from mean)
4. Gasping = >20% of windows are outliers
```

## Next Immediate Steps

1. **Run database migration** to create respiratory tables
2. **Test metric computation** with sample chest movement data
3. **Create API router** for respiratory endpoints
4. **Integrate with video_ai_engine** for automatic ingestion
5. **Add frontend dashboard** for visualization

---
**Status:** Core metrics fully implemented ✅  
**Next Phase:** Integration + API + Testing  
**Last Updated:** 2025-11-17
