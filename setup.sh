#!/bin/bash

# Set the environment name and the path where the environment will be created
ENV_NAME="workflow_16s"
ENV_PATH="./$ENV_NAME"

# Create the conda environment from the YAML file
echo "Creating conda environment from environment.yml..."
conda env create --file ./references/conda_envs/workflow_16s.yml --prefix "$ENV_PATH"

# Activate the environment
echo "Activating the conda environment..."
source activate "$ENV_PATH"

# Run the Python script 'run.py'
echo "Running the Python script 'run.py'..."
python ./src/run.py

# Deactivate the environment after running the script
echo "Deactivating the conda environment..."
conda deactivate

echo "Environment setup and script execution complete!"
