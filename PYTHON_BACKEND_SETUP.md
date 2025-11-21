# Python Backend Setup for Mental Health Features

## Quick Start

The mental health assessment page requires **BOTH** backends to run:
1. **Express Backend** (port 5000) - Already running via "Start application" workflow  
2. **Python Backend** (port 8000) - Must be started manually

### Option 1: Automated Script (Recommended)

Run both backends simultaneously:

```bash
bash start-both-backends.sh
```

This will:
- Start Python FastAPI on port 8000 (with zero startup warnings)
- Start Express backend on port 5000
- Log output to `python_backend.log` and `express_backend.log`

### Option 2: Manual Python Backend Only

If the Express backend is already running via the Replit workflow, just start Python:

```bash
TF_CPP_MIN_LOG_LEVEL=3 CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 \
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Startup Process

**Expected startup time**: 30-60 seconds  
**Reason**: Heavy ML models (TensorFlow, MediaPipe, YAMNet) load at startup

### Startup Sequence
1. **TensorFlow initialization** (~10-15s)
2. **MediaPipe Face Mesh & Pose loading** (~10-15s) 
3. **YAMNet audio classifier loading** (~5-10s)
4. **AI engine initialization** (~5-10s)
5. **Server ready** - Look for: `INFO: Application startup complete`

### Zero Warnings Guarantee
All startup warnings have been suppressed:
- ✅ No CUDA driver warnings
- ✅ No DeepLab model errors
- ✅ No noisereduce warnings
- ✅ Clean startup with INFO logs only

## Verifying Backend is Running

### Check Python Backend Status
```bash
curl http://localhost:8000/api/v1/mental-health/questionnaires
```

Expected response: JSON with 3 questionnaires (PHQ-9, GAD-7, PSS-10)

### Check Express Backend Status
```bash
curl http://localhost:5000/api/v1/mental-health/questionnaires
```

Expected: Same response (Express proxies to Python)

## Accessing Mental Health Page

1. Navigate to: `/mental-health` (or click "Mental Health" in sidebar)
2. You should see 3 assessment cards:
   - **PHQ-9**: Depression screening (9 questions)
   - **GAD-7**: Anxiety screening (7 questions)  
   - **PSS-10**: Perceived Stress Scale (10 questions)

## Troubleshooting

### "Nothing shows" on Mental Health page

**Symptom**: Blank page or loading spinner forever  
**Cause**: Python backend not running  
**Solution**: Start Python backend (see Option 1 or 2 above)

### Connection Error Alert

If you see: `"Unable to load mental health questionnaires. The mental health service may not be running."`

This means:
- Python backend (port 8000) is not accessible
- Check if it's running: `ps aux | grep uvicorn`
- Check logs: `tail -f python_backend.log`

### Port Already in Use

```
ERROR: [Errno 98] Address already in use
```

**Solution**: Kill existing process
```bash
pkill -f "uvicorn app.main"
# Then restart
```

## Architecture

```
User Browser
    ↓
Express Backend (5000)
    ↓ [Proxy /api/v1/mental-health/*]
    ↓ [Generates JWT token for auth]
Python Backend (8000)
    ↓
FastAPI → Mental Health Service
    ↓
PostgreSQL Database
```

## Production Deployment

For production, use process managers like:
- **systemd** (Linux)
- **supervisor**
- **PM2** (Node.js ecosystem)
- **Docker Compose** (containerized)

Example PM2 config:
```javascript
module.exports = {
  apps: [
    {
      name: "python-backend",
      script: "python",
      args: "-m uvicorn app.main:app --host 0.0.0.0 --port 8000",
      env: {
        TF_CPP_MIN_LOG_LEVEL: "3",
        CUDA_VISIBLE_DEVICES: "-1",
        TF_ENABLE_ONEDNN_OPTS: "0"
      }
    },
    {
      name: "express-backend",
      script: "npm",
      args: "run start"
    }
  ]
}
```

## Support

If you encounter issues:
1. Check logs: `tail -f python_backend.log`
2. Verify both ports are listening: `lsof -i :5000,8000`
3. Ensure database is accessible: `psql $DATABASE_URL -c "SELECT 1"`
