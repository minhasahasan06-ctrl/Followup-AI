#!/bin/bash
# Deployment startup script for Followup AI
# Runs both Express (Node.js) and FastAPI (Python) backends in parallel

set -e

echo "🚀 Starting Followup AI Deployment..."

# Start Python FastAPI backend on port 8000 (background)
echo "📡 Starting Python FastAPI backend (port 8000)..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 &
PYTHON_PID=$!

# Wait for Python backend to initialize
sleep 5

# Start Node.js Express backend on port 5000 (foreground)
echo "📡 Starting Node.js Express backend (port 5000)..."
NODE_ENV=production node dist/index.js &
NODE_PID=$!

# Function to handle shutdown
cleanup() {
    echo "🛑 Shutting down backends..."
    kill $PYTHON_PID 2>/dev/null || true
    kill $NODE_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGTERM SIGINT

# Wait for both processes
wait -n

# If one process exits, kill the other
echo "⚠️  One backend process exited, shutting down..."
cleanup
