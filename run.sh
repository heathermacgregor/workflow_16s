#!/bin/bash

set -e  # Exit on error
set -o pipefail

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Set environment name
ENV_NAME="workflow_16s"
PYTHON_SCRIPT="$SCRIPT_DIR/src/run.py"

# Log function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed or not available in PATH"
    exit 1
fi

# Check for existing environments
echo "🔍 Checking for existing 16s workflow environments..."
EXACT_ENV_EXISTS=$(conda env list | awk '{print $1}' | grep -x "$ENV_NAME")
ALT_ENV_NAME=$(conda env list | awk '/^[^#]/ {print $1}' | grep -E 'workflow_16s$' | head -n 1)

# Check if the environment exists
if [ -n "$EXACT_ENV_EXISTS" ]; then
    echo "✅ Exact environment '$ENV_NAME' already exists"
elif [ -n "$ALT_ENV_NAME" ]; then
    ENV_NAME="$ALT_ENV_NAME"
    echo "✅ Found existing environment with matching suffix: '$ENV_NAME'"
else
    echo "❌ No suitable conda environment found"
    echo "   Expected either:"
    echo "   - Exact name: '$ENV_NAME'"
    echo "   - Or name ending with: 'workflow_16s'"
    exit 1
fi

# Activate the environment
echo "🔄 Activating the conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Check if the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "❌ Python script '$PYTHON_SCRIPT' not found"
    conda deactivate
    exit 1
fi

echo $CONDA_DEFAULT_ENV

# Run the Python script
log "🔄 Running the Python script '$PYTHON_SCRIPT'..."
python "$PYTHON_SCRIPT"

# Deactivate the environment
log "🔄 Deactivating the conda environment..."
conda deactivate

log "✅ Workflow completed successfully!"
