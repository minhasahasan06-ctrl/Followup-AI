#!/bin/bash
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=-1
export TF_ENABLE_ONEDNN_OPTS=0

echo "🐍 Starting Python FastAPI backend on port 8000..."
echo "⏳ Loading ML models (30-60 seconds)..."

python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
