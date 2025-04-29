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
    log "❌ Conda is not installed or not available in PATH."
    exit 1
fi

# Check if the environment exists
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
    log "❌ Conda environment '$ENV_NAME' does not exist."
    exit 1
fi

# Activate the environment
log "🔄 Activating the conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Check if the Python script exists
if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    log "❌ Python script '$PYTHON_SCRIPT' not found."
    conda deactivate
    exit 1
fi

# Run the Python script
log "🔄 Running the Python script '$PYTHON_SCRIPT'..."
python "$PYTHON_SCRIPT"

# Deactivate the environment
log "🔄 Deactivating the conda environment..."
conda deactivate

log "✅ Workflow completed successfully."
