#!/bin/bash
# Start both Node.js Express (port 5000) and Python FastAPI (port 8000) servers

echo "🚀 Starting Followup AI - Dual Server Mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Node.js Express  → Port 5000"
echo "🐍 Python FastAPI   → Port 8000"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start Node.js server in background
npm run dev &
NODE_PID=$!
echo "✅ Node.js server started (PID: $NODE_PID)"

# Wait for Node.js to be ready
sleep 3

# Start Python FastAPI server in background
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
PYTHON_PID=$!
echo "✅ Python server started (PID: $PYTHON_PID)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 All services running!"
echo "   Frontend: Check Webview tab"
echo "   Node API: http://localhost:5000"
echo "   Python API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for both processes
wait $NODE_PID $PYTHON_PID
