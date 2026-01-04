# Reference-to-Rig Makefile
# Run 'make help' for available commands

.PHONY: help setup setup-engine setup-ui index engine ui dev build clean test lint

# Default target
help:
	@echo "Reference-to-Rig - Guitar Tone Matching Engine"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  setup         Install all dependencies (engine + UI)"
	@echo "  setup-engine  Install Python engine dependencies"
	@echo "  setup-ui      Install Node.js UI dependencies"
	@echo ""
	@echo "Development:"
	@echo "  engine        Start FastAPI server (port 8000)"
	@echo "  ui            Start UI development server (port 1420)"
	@echo "  dev           Start both engine and UI (requires two terminals)"
	@echo "  index         Build capture library FAISS index"
	@echo ""
	@echo "Build:"
	@echo "  build         Build production UI bundle"
	@echo "  build-tauri   Build Tauri desktop application"
	@echo ""
	@echo "Quality:"
	@echo "  test          Run Python tests"
	@echo "  lint          Run linters (ruff + black check)"
	@echo "  format        Auto-format code"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean         Remove build artifacts and caches"

# Setup
setup: setup-engine setup-ui index
	@echo "Setup complete! Run 'make engine' then 'make ui' to start."

setup-engine:
	@echo "Setting up Python engine..."
	cd engine && python -m venv venv
	cd engine && venv/Scripts/pip install --upgrade pip
	cd engine && venv/Scripts/pip install -r requirements.txt
	@echo "Engine setup complete."

setup-ui:
	@echo "Setting up UI..."
	cd ui && npm install
	@echo "UI setup complete."

# Index
index:
	@echo "Building capture library index..."
	cd engine && venv/Scripts/python -m scripts.build_index
	@echo "Index built."

# Development servers
engine:
	@echo "Starting FastAPI engine on http://localhost:8000"
	cd engine && venv/Scripts/uvicorn app.main:app --reload --port 8000

ui:
	@echo "Starting UI development server on http://localhost:1420"
	cd ui && npm run dev

dev:
	@echo "To run both servers, open two terminals:"
	@echo "  Terminal 1: make engine"
	@echo "  Terminal 2: make ui"

# Build
build:
	@echo "Building UI for production..."
	cd ui && npm run build

build-tauri:
	@echo "Building Tauri desktop application..."
	cd ui && npm run tauri build

# Quality
test:
	@echo "Running tests..."
	cd engine && venv/Scripts/pytest -v

lint:
	@echo "Running linters..."
	cd engine && venv/Scripts/ruff check app tests scripts
	cd engine && venv/Scripts/black --check app tests scripts

format:
	@echo "Formatting code..."
	cd engine && venv/Scripts/ruff check --fix app tests scripts
	cd engine && venv/Scripts/black app tests scripts

# Clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf engine/__pycache__ engine/**/__pycache__
	rm -rf engine/.pytest_cache engine/.ruff_cache
	rm -rf engine/data/*.bin engine/data/*.db
	rm -rf ui/dist ui/node_modules/.cache
	rm -rf ui/src-tauri/target
	@echo "Clean complete."


