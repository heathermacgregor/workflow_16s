#!/bin/bash

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Set the environment name and the path where the environment will be created
ENV_NAME="workflow_16s"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Ensure mamba is installed
echo "Checking if mamba is installed..."
if ! command -v mamba &> /dev/null
then
    echo "Mamba not found, installing mamba..."
    conda install -y -c conda-forge mamba
fi

# Create qiime2 environment if it doesn't exist
QIIME_ENV="qiime2-amplicon-2024.10"
echo "Checking if $QIIME_ENV environment exists..."
if ! conda env list | grep -q "$QIIME_ENV"
then
    echo "Creating $QIIME_ENV environment..."
    OS=$(uname -s)
    if [ "$OS" = "Linux" ]; then
        YAML_URL="https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-linux-conda.yml"
    elif [ "$OS" = "Darwin" ]; then
        YAML_URL="https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-osx-conda.yml"
    else
        echo "Unsupported operating system: $OS"
        exit 1
    fi
    
    mamba env create -n "$QIIME_ENV" --file "$YAML_URL"
    if [ $? -ne 0 ]; then
        echo "Failed to create $QIIME_ENV environment"
        exit 1
    fi
else
    echo "The $QIIME_ENV environment already exists."
fi

# Check if the workflow environment exists
if conda env list | grep -q "$ENV_NAME"
then
    echo "The environment '$ENV_NAME' already exists. Activating the environment..."
else
    # Create workflow environment
    echo "The environment '$ENV_NAME' does not exist. Creating the environment from environment.yml using mamba..."
    mamba env create --file "$SCRIPT_DIR/references/conda_envs/workflow_16s.yml" --prefix "$ENV_PATH"
fi

# Activate the workflow environment
echo "Activating the conda environment..."
source activate "$ENV_PATH"

# Check if fastqc is available
if ! command -v fastqc &> /dev/null; then
    echo "Error: 'fastqc' executable not found in PATH."
    echo "Please install FastQC or provide the correct path."
    exit 1
fi

# Run Python script
echo "Running the Python script 'run.py'..."
python "$SCRIPT_DIR/src/run.py"

# Deactivate environment
echo "Deactivating the conda environment..."
conda deactivate

echo "Environment setup and script execution complete!"
