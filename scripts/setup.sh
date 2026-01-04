#!/bin/bash
# Reference-to-Rig Setup Script for macOS/Linux

echo "========================================"
echo "Reference-to-Rig Setup"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python not found. Please install Python 3.10+"
    exit 1
fi
echo "[OK] Python found"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found. Please install Node.js 18+"
    exit 1
fi
echo "[OK] Node.js found"

# Setup Python environment
echo
echo "Setting up Python engine..."
cd engine
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cd ..
echo "[OK] Python engine setup complete"

# Setup UI
echo
echo "Setting up UI..."
cd ui
npm install
cd ..
echo "[OK] UI setup complete"

# Build index
echo
echo "Building capture library index..."
cd engine
source venv/bin/activate
python -m scripts.build_index
cd ..
echo "[OK] Index built"

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "To start the application:"
echo "  1. Open a terminal and run: cd engine && source venv/bin/activate && uvicorn app.main:app --reload --port 8000"
echo "  2. Open another terminal and run: cd ui && npm run dev"
echo
echo "Then open http://localhost:1420 in your browser."
echo


