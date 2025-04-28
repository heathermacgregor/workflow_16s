#!/bin/bash

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Set the environment name and the path where the environment will be created
ENV_NAME="workflow_16s"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Ensure mamba is installed (if it's not already in the system, you can install it via conda)
echo "Checking if mamba is installed..."
if ! command -v mamba &> /dev/null
then
    echo "Mamba not found, installing mamba..."
    conda install -c conda-forge mamba
fi

# Check if fastqc is installed
echo "Checking if fastqc is installed..."
if ! command -v fastqc &> /dev/null
then
    echo "FastQC not found, installing FastQC..."
    mamba install -c bioconda fastqc -y
    # Ensure fastqc is added to the path
    export PATH="$CONDA_PREFIX/bin:$PATH"
else
    echo "FastQC is already installed."
fi

# Check if seqkit is installed
echo "Checking if seqkit is installed..."
if ! command -v seqkit &> /dev/null
then
    echo "Seqkit not found, installing Seqkit..."
    mamba install -c bioconda seqkit -y
    # Ensure seqkit is added to the path
    export PATH="$CONDA_PREFIX/bin:$PATH"
else
    echo "Seqkit is already installed."
fi

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"
then
    echo "The environment '$ENV_NAME' already exists. Activating the environment..."
else
    # Create the conda environment from the YAML file using mamba if it doesn't exist
    echo "The environment '$ENV_NAME' does not exist. Creating the environment from environment.yml using mamba..."
    mamba env create --file "$SCRIPT_DIR/references/conda_envs/workflow_16s.yml" --prefix "$ENV_PATH"
fi

# Activate the environment
echo "Activating the conda environment..."
source activate "$ENV_PATH"

# Run the Python script 'run.py'
echo "Running the Python script 'run.py'..."
python "$SCRIPT_DIR/src/run.py"

# Deactivate the environment after running the script
echo "Deactivating the conda environment..."
conda deactivate

echo "Environment setup and script execution complete!"
