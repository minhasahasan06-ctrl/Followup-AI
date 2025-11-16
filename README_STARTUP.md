# Followup AI - Startup Instructions

## ⚠️ IMPORTANT: Dual Backend Architecture

This application uses **two separate backend servers**:
1. **Node.js/Express** (port 5000) - Frontend + legacy endpoints
2. **Python/FastAPI** (port 8000) - NEW deterioration prediction features

**The deterioration dashboard requires BOTH servers running.**

## Quick Start

### Step 1: Start Frontend (Automatic)
The Replit workflow automatically starts the Node.js frontend:
```
✅ Already running on port 5000
```

### Step 2: Start Python Backend (Manual - Required for Deterioration Dashboard)

Open a **new terminal** and run:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**OR** use the startup script to run both:

```bash
bash start-servers.sh
```

This will start:
- 🐍 Python FastAPI backend on http://localhost:8000
- ⚛️  Node.js frontend on http://localhost:5000

### Accessing the Application

- **Main Application**: http://localhost:5000
- **Python API Documentation**: http://localhost:8000/docs
- **Python API Health Check**: http://localhost:8000/health

### Deterioration Dashboard

The new Deterioration Prediction System dashboard is available at:
- **Patient Access**: http://localhost:5000/deterioration

This dashboard shows:
- ✅ Current wellness score (0-15 scale)
- 📊 7-day trend charts
- 🔍 Pattern changes and deviations
- 💡 Wellness recommendations

### Important Notes

1. **Both servers must be running** for the deterioration dashboard to work
2. The Python backend handles all `/api/v1/baseline`, `/api/v1/deviation`, and `/api/v1/risk` endpoints
3. The frontend automatically connects to both backends

### Troubleshooting

**If the deterioration dashboard shows network errors:**
1. ✅ Verify Python backend is running: `curl http://localhost:8000/health`
   - Should return: `{"status":"healthy","timestamp":"...","database":"connected"}`
2. ❌ If not running, start it: `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`
3. Check Python backend logs for errors
4. Ensure database is accessible (PostgreSQL)

**Common Issues:**

❌ **"Failed to fetch" or "Network error"**
→ Python backend not running. Start it manually (see Step 2 above)

❌ **Import errors when starting Python backend**
→ Fixed! All import paths corrected (`app.dependencies` vs `app.config`)

❌ **"No baseline data"**
→ Normal for new patients. Need 7+ days of health tracking data

**Quick Health Check:**
```bash
# Test both servers
curl http://localhost:5000/  # Should return frontend
curl http://localhost:8000/health  # Should return {"status":"healthy"}
```
