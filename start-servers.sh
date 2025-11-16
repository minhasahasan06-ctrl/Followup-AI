#!/bin/bash

# Start both Node.js frontend and Python FastAPI backend
# This script runs both servers concurrently

echo "🚀 Starting Followup AI servers..."

# Start Python FastAPI backend on port 8000
echo "🐍 Starting Python FastAPI backend on port 8000..."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
PYTHON_PID=$!

# Give Python backend time to start
sleep 3

# Start Node.js frontend on port 5000
echo "⚛️  Starting Node.js frontend on port 5000..."
npm run dev &
NODE_PID=$!

echo "✅ Both servers started!"
echo "   - Frontend (Node.js): http://localhost:5000"
echo "   - Backend (Python FastAPI): http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"

# Handle shutdown
trap "echo '🛑 Shutting down servers...'; kill $PYTHON_PID $NODE_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for both processes
wait
