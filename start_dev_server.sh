#!/bin/bash

# Development server startup script for the modern 16S report system

echo "Starting 16S Analysis Modern Report System..."

# Check if we're in the right directory
if [ ! -f "api/main.py" ]; then
    echo "Error: Please run this script from the workflow_16s root directory"
    exit 1
fi

# Install Python dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install -r api/requirements.txt
fi

# Install Node.js dependencies if needed
if [ -d "frontend" ] && [ ! -d "frontend/node_modules" ]; then
    echo "Installing Node.js dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Start FastAPI server in the background
echo "Starting FastAPI server on http://localhost:8000..."
cd api
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000 &
API_PID=$!
cd ..

# Start React development server if frontend exists
if [ -d "frontend" ]; then
    echo "Starting React development server on http://localhost:3000..."
    cd frontend
    REACT_APP_API_BASE="http://localhost:8000" npm start &
    REACT_PID=$!
    cd ..
else
    echo "React frontend not found. Only API server is running."
    REACT_PID=""
fi

# Function to cleanup on exit
cleanup() {
    echo "Shutting down servers..."
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
    fi
    if [ ! -z "$REACT_PID" ]; then
        kill $REACT_PID 2>/dev/null
    fi
    exit 0
}

# Register cleanup function
trap cleanup EXIT INT TERM

# Wait for user to stop
echo "Development servers running. Press Ctrl+C to stop."
echo "API: http://localhost:8000"
if [ ! -z "$REACT_PID" ]; then
    echo "Frontend: http://localhost:3000"
fi
echo "API Documentation: http://localhost:8000/docs"

# Keep script running
wait