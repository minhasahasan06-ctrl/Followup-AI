#!/bin/bash

echo "🐍 Starting Python FastAPI Backend on port 5000"
echo ""

# Kill any process on port 5000
echo "Checking for processes on port 5000..."
lsof -ti:5000 | xargs kill -9 2>/dev/null || echo "✅ Port 5000 is available"

echo ""
echo "Starting Python backend..."
python3 start_python_server.py
