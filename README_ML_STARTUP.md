# ML Inference System - Quick Start Guide

## 🚀 Starting the Application

The Followup AI platform now includes a self-hosted ML inference system. You need to run **two servers**:

### 1. Frontend + Legacy Backend (Already Running)
```bash
npm run dev
```
This serves the React frontend and Express.js backend on **port 5000**.

### 2. Python FastAPI Backend (ML Inference)
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
This starts the ML inference server on **port 8000**.

## ✅ Verify ML System is Running

### Check ML API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "database": "connected"
}
```

### List Available ML Models
```bash
curl http://localhost:8000/api/v1/ml/models
```

### Get ML System Stats
```bash
curl http://localhost:8000/api/v1/ml/stats
```

## 📊 Access ML Monitoring Dashboard

Once both servers are running, access the ML monitoring dashboard at:

```
http://localhost:5000/ml-monitoring
```

Features:
- Real-time prediction statistics
- Cache hit rate monitoring
- Model performance tracking
- Recent prediction history
- System health status

## 🧪 Test ML Predictions

### Test Clinical-BERT Symptom Analysis
```bash
curl -X POST http://localhost:8000/api/v1/ml/predict/symptom-analysis \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "text": "Patient reports severe headache and nausea for 3 days",
    "analysis_type": "symptom_ner"
  }'
```

### Test Generic Prediction
```bash
curl -X POST http://localhost:8000/api/v1/ml/predict \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "model_name": "clinical_ner",
    "input_data": {"text": "Patient has fever and cough"}
  }'
```

## 🗄️ Database Migration

Run the ML tables migration:

```bash
cd /home/runner/workspace
alembic upgrade head
```

This creates the following tables:
- `ml_models` - Model versions and metadata
- `ml_predictions` - Prediction audit logs (HIPAA compliance)
- `ml_performance_logs` - Performance metrics over time
- `ml_batch_jobs` - Batch prediction job tracking

## 📦 Install ML Dependencies (If Needed)

The ML packages should install automatically. If you encounter issues:

```bash
# Check if packages are installed
python -c "import torch, transformers, onnxruntime, redis, sklearn; print('✅ All ML packages installed')"

# If any are missing, they're already in pyproject.toml
# Just restart the workflow or run:
pip install torch transformers onnxruntime redis scikit-learn joblib
```

## 🔧 Troubleshooting

### Issue: "Module not found: torch"
**Solution**: Restart the workflow to install packages from pyproject.toml

### Issue: "Redis connection error"
**Solution**: Redis is optional. The system will work without caching, but predictions will be slower.

To start Redis (optional):
```bash
redis-server &
```

### Issue: "Model not found: clinical_ner"
**Solution**: The model downloads automatically on first startup. Wait ~30 seconds for HuggingFace to download Clinical-BERT.

### Issue: "Database table doesn't exist"
**Solution**: Run Alembic migration:
```bash
alembic upgrade head
```

## 📁 Project Structure

```
app/
├── services/
│   ├── ml_inference.py       # Model registry, caching, inference
│   └── batch_processing.py   # Batch prediction jobs
├── routers/
│   └── ml_inference.py        # ML API endpoints
├── models/
│   └── ml_models.py           # Database schema
└── main.py                    # FastAPI app with ML lifespan

ml_scripts/
├── train_deterioration_model.py  # Train custom LSTM
├── convert_to_onnx.py             # PyTorch → ONNX optimization
└── README.md                      # ML system documentation

client/src/pages/
└── MLMonitoring.tsx           # React monitoring dashboard

alembic/versions/
└── add_ml_inference_tables.py # Database migration
```

## 🎯 Next Steps

1. **Train Custom Model** (Optional):
   ```bash
   python ml_scripts/train_deterioration_model.py
   ```

2. **Optimize for Production** (Optional):
   ```bash
   python ml_scripts/convert_to_onnx.py
   ```

3. **Set up Redis** (Optional - for 10-100x speedup):
   ```bash
   # Set environment variables
   export REDIS_HOST=localhost
   export REDIS_PORT=6379
   
   # Start Redis
   redis-server &
   ```

## 🔒 HIPAA Compliance Notes

- ✅ All predictions logged to database (`ml_predictions` table)
- ✅ Patient ID (AWS Cognito sub) tracked in audit trail
- ✅ No external API calls - all inference happens on-premise
- ✅ Redis caching respects 1-hour TTL for patient data
- ✅ Role-based access control via JWT tokens

## 📞 Support

For issues or questions:
- Check `ml_scripts/README.md` for detailed documentation
- View logs: `tail -f /tmp/logs/Start_application_*.log`
- Check ML service: `curl http://localhost:8000/api/v1/ml/stats`

---

**Built with**: FastAPI • PyTorch • HuggingFace Transformers • ONNX Runtime • Redis • PostgreSQL
