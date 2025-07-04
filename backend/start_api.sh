#!/bin/bash

# Enhanced AutoDub API Startup Script for Google Colab

echo "🚀 Starting Enhanced AutoDub API v4.0 with Cloudflare Tunnel"

# Set environment variables
export API_HOST="0.0.0.0"
export API_PORT="8000"

# Create necessary directories
mkdir -p /tmp/autodub
mkdir -p /tmp/autodub/output

# Install dependencies if needed
echo "📦 Installing dependencies..."
pip install fastapi uvicorn[standard] python-multipart

# Install cloudflared if not present
if ! command -v cloudflared &> /dev/null; then
    echo "🔧 Installing cloudflared..."
    python install_cloudflared.py
fi

# Start the API
echo "🎬 Starting Enhanced AutoDub API..."
python enhanced_autodub_api.py
