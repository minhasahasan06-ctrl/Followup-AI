# ML Inference System - HIPAA-Compliant Self-Hosted Infrastructure

## Overview

This directory contains training scripts and utilities for the self-hosted ML inference system. All models are deployed on-premise to maintain HIPAA compliance and keep patient data within the secure AWS/Neon infrastructure.

## Architecture

### 🏗️ System Components

1. **Model Registry** (`app/services/ml_inference.py`)
   - Global registry for loaded ML models
   - Thread pool executor for async CPU-bound inference
   - Redis caching layer (10-100x speedup on repeated predictions)
   - HIPAA-compliant audit logging for all predictions

2. **API Layer** (`app/routers/ml_inference.py`)
   - RESTful endpoints for predictions
   - Batch processing support
   - Performance monitoring
   - Model management

3. **Database Schema** (`app/models/ml_models.py`)
   - `ml_models`: Model versions and metadata
   - `ml_predictions`: All predictions (audit trail)
   - `ml_performance_logs`: Performance metrics over time
   - `ml_batch_jobs`: Batch prediction jobs

4. **Training Scripts** (this directory)
   - `train_deterioration_model.py`: LSTM for deterioration prediction
   - `convert_to_onnx.py`: PyTorch → ONNX optimization

## Quick Start

### 1. Install Dependencies

```bash
# ML packages are in pyproject.toml
# They install automatically when workflow restarts
```

### 2. Run Database Migration

```bash
cd /home/runner/workspace
alembic upgrade head
```

### 3. Train Custom Model (Optional)

```bash
# Train LSTM deterioration prediction model
python ml_scripts/train_deterioration_model.py
```

### 4. Convert to ONNX (Optional - for 4-10x speedup)

```bash
# Convert PyTorch model to ONNX
python ml_scripts/convert_to_onnx.py
```

### 5. Start Backend

```bash
# Python backend with ML models
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Available Models

### 1. Clinical-BERT NER
- **Model**: `samrawal/bert-base-uncased_clinical-ner`
- **Task**: Named Entity Recognition for medical text
- **Use Case**: Extract symptoms, medications, conditions from patient notes
- **API**: `POST /api/v1/ml/predict/symptom-analysis`

**Example Request:**
```json
{
  "text": "Patient reports severe headache and nausea for 3 days",
  "analysis_type": "symptom_ner"
}
```

**Example Response:**
```json
{
  "entities": {
    "SYMPTOM": [
      {"text": "headache", "score": 0.98},
      {"text": "nausea", "score": 0.95}
    ]
  },
  "metadata": {
    "inference_time_ms": 45.2,
    "cache_hit": false
  }
}
```

### 2. LSTM Deterioration Predictor (Custom)
- **Model**: `deterioration_lstm.pt` (trained on patient baselines)
- **Task**: Binary classification (stable vs. deteriorating)
- **Use Case**: Predict patient health decline 3 days in advance
- **Status**: ⚠️ Requires training on production data

**How to Train:**
```bash
# Ensure DATABASE_URL is set
export DATABASE_URL="your_database_url"

# Train model (requires ≥100 patient sequences)
python ml_scripts/train_deterioration_model.py

# Convert to ONNX for production
python ml_scripts/convert_to_onnx.py
```

## API Endpoints

### Prediction Endpoints

#### Generic Prediction
```http
POST /api/v1/ml/predict
Content-Type: application/json

{
  "model_name": "clinical_ner",
  "input_data": {"text": "Patient has fever and cough"},
  "use_cache": true
}
```

#### Symptom Analysis
```http
POST /api/v1/ml/predict/symptom-analysis
Content-Type: application/json

{
  "text": "Severe pain in lower back, difficulty walking",
  "analysis_type": "symptom_ner"
}
```

### Management Endpoints

#### List Models
```http
GET /api/v1/ml/models
```

#### Get Model Performance
```http
GET /api/v1/ml/models/{model_name}/performance?hours=24
```

#### Get Prediction History
```http
GET /api/v1/ml/predictions/history?limit=50
```

#### Get System Stats
```http
GET /api/v1/ml/stats
```

## Performance Optimization

### Redis Caching

All predictions are cached in Redis with 1-hour TTL:
- **Cache Key**: MD5 hash of `model_name` + `input_data`
- **Speedup**: 10-100x for repeated predictions
- **Configuration**: `REDIS_HOST`, `REDIS_PORT` in environment

### ONNX Runtime

Convert PyTorch models to ONNX for 4-10x inference speedup:

```python
from ml_scripts.convert_to_onnx import convert_pytorch_to_onnx

convert_pytorch_to_onnx(
    pytorch_model_path="./ml_models/deterioration_lstm.pt",
    onnx_output_path="./ml_models/deterioration_lstm.onnx",
    input_shape=(1, 7, 4)  # (batch, sequence_length, features)
)
```

### Async Thread Pool

CPU-bound inference runs in thread pool to avoid blocking:
- **Workers**: 4 concurrent threads
- **Executor**: `ThreadPoolExecutor` for PyTorch/sklearn models
- **Result**: Non-blocking async API

## Monitoring Dashboard

Access ML monitoring at: `http://localhost:5000/ml-monitoring`

Features:
- ✅ Real-time system stats (predictions, cache rate, latency)
- ✅ Model deployment status
- ✅ Performance charts (latency, accuracy over time)
- ✅ Recent prediction history
- ✅ HIPAA audit trail

## HIPAA Compliance

### Audit Logging

Every prediction is logged to `ml_predictions` table:
- Patient ID (AWS Cognito sub)
- Input data (anonymized if needed)
- Prediction result
- Confidence score
- Timestamp and IP address
- Cache hit status

### Data Security

- ✅ All ML inference happens on-premise (no external API calls)
- ✅ Patient data stays in Neon PostgreSQL + AWS S3
- ✅ OpenAI models require BAA verification
- ✅ Predictions logged for audit trail
- ✅ Role-based access control (RBAC) via AWS Cognito

### Regulatory Positioning

**Important**: This system is positioned as a **General Wellness Product**, NOT a medical device:
- ✅ Use language: "wellness monitoring", "change detection"
- ❌ Avoid: "diagnosis", "medical device", "treatment"
- ✅ Focus: Track trends, suggest discussion with healthcare providers
- ❌ Never: Make diagnostic claims or treatment recommendations

## Troubleshooting

### Model Loading Errors

```bash
# Check if models are loaded
curl http://localhost:8000/api/v1/ml/models

# View application logs
tail -f /tmp/logs/start_application_*.log
```

### Redis Connection Issues

```bash
# Check Redis status
redis-cli ping

# Start Redis (if not running)
redis-server &
```

### Missing Dependencies

```bash
# Reinstall ML packages
pip install torch transformers onnxruntime redis scikit-learn joblib
```

## Training Data Requirements

### LSTM Deterioration Model

Minimum requirements:
- **Patients**: 50+ patients with at least 14 days of data
- **Measurements**: Pain scores, respiratory rate, symptom severity
- **Sequence Length**: 7 days (configurable)
- **Prediction Horizon**: 3 days ahead (configurable)

Data format (SQL):
```sql
SELECT 
    patient_id,
    measurement_date,
    pain_score_facial,
    pain_score_self_reported,
    respiratory_rate,
    symptom_severity_score
FROM patient_measurements
WHERE measurement_date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY patient_id, measurement_date
```

## Future Enhancements

- [ ] GPU support for faster inference (PyTorch CUDA)
- [ ] Model A/B testing framework
- [ ] Automated model retraining pipeline
- [ ] Federated learning across hospitals
- [ ] Explainable AI (SHAP values for predictions)
- [ ] Real-time streaming predictions (WebSocket)

## License & Compliance

This ML system is part of Followup AI's HIPAA-compliant health platform. All models and data handling must comply with:
- HIPAA Privacy Rule
- HIPAA Security Rule
- 21 CFR Part 11 (if applicable)
- AWS BAA requirements

For questions, contact the engineering team.
