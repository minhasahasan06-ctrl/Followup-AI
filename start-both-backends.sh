#!/bin/bash
# ============================================================================
# Followup AI - Dual Backend Startup Script
# ============================================================================
# Starts both Python FastAPI (port 8000) and Node.js Express (port 5000)
# in parallel. Use this for local development.
#
# Usage: bash start-both-backends.sh
# ============================================================================

echo "🚀 Starting Followup AI dual backends..."
echo ""

# Kill any existing processes on ports 8000 and 5000
echo "🧹 Cleaning up existing processes..."
pkill -f "uvicorn app.main" 2>/dev/null || true
pkill -f "tsx server/index.ts" 2>/dev/null || true
sleep 2

# Set environment variables for Python backend (suppress warnings)
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0

echo "🐍 Starting Python FastAPI backend on port 8000..."
echo "   (ML models loading - this takes 30-60 seconds)"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > python_backend.log 2>&1 &
PYTHON_PID=$!
echo "   Python backend PID: $PYTHON_PID"

echo ""
echo "⏳ Waiting 5 seconds before starting Node.js backend..."
sleep 5

echo "📦 Starting Node.js Express backend on port 5000..."
npm run dev > express_backend.log 2>&1 &
EXPRESS_PID=$!
echo "   Express backend PID: $EXPRESS_PID"

echo ""
echo "=========================================="
echo "✅ Both backends starting!"
echo "=========================================="
echo ""
echo "📊 Status:"
echo "   - Python Backend: http://localhost:8000"
echo "   - Express Backend: http://localhost:5000"
echo ""
echo "📝 Logs:"
echo "   - Python: tail -f python_backend.log"
echo "   - Express: tail -f express_backend.log"
echo ""
echo "⏳ Python backend ML initialization: ~60 seconds"
echo "   Monitor: tail -f python_backend.log | grep 'Application startup'"
echo ""
echo "⚠️  To stop both backends: pkill -f 'uvicorn app.main' && pkill -f 'tsx server'"
echo ""

# Wait for both processes
wait $PYTHON_PID $EXPRESS_PID
