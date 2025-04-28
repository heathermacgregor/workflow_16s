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

# Add Silva classifier installation
CLASSIFIER_DIR="$SCRIPT_DIR/references/classifier"
CLASSIFIER_FILE="$CLASSIFIER_DIR/silva-138-99-515-806-classifier.qza"
CLASSIFIER_URL="https://data.qiime2.org/2024.10/common/silva-138-99-515-806-classifier.qza"

echo "Checking for SILVA classifier..."
mkdir -p "$CLASSIFIER_DIR"

if [ ! -f "$CLASSIFIER_FILE" ]; then
    echo "Downloading SILVA classifier..."
    if command -v wget &> /dev/null; then
        wget -O "$CLASSIFIER_FILE" "$CLASSIFIER_URL"
    elif command -v curl &> /dev/null; then
        curl -L "$CLASSIFIER_URL" -o "$CLASSIFIER_FILE"
    else
        echo "Error: Need wget or curl to download classifier"
        exit 1
    fi
    
    # Verify download success
    if [ ! -f "$CLASSIFIER_FILE" ]; then
        echo "Failed to download SILVA classifier"
        exit 1
    fi
    echo "Successfully downloaded SILVA classifier"
else
    echo "SILVA classifier already exists at: $CLASSIFIER_FILE"
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

# Check if fastqc is available and install if missing
if ! command -v fastqc &> /dev/null; then
    echo "FastQC not found. Installing FastQC..."
    mamba install -y -c bioconda fastqc
    
    # Verify installation
    if ! command -v fastqc &> /dev/null; then
        echo "Failed to install FastQC. Please install manually."
        exit 1
    fi
fi

# Run Python script
echo "Running the Python script 'run.py'..."
python "$SCRIPT_DIR/src/run.py"

# Deactivate environment
echo "Deactivating the conda environment..."
conda deactivate

echo "Environment setup and script execution complete!"
