#!/bin/bash

# Set the environment name and the path where the environment will be created
ENV_NAME="workflow_16s"
ENV_PATH="./$ENV_NAME"


# Ensure mamba is installed (if it's not already in the system, you can install it via conda)
echo "Checking if mamba is installed..."
if ! command -v mamba &> /dev/null
then
    echo "Mamba not found, installing mamba..."
    conda install -c conda-forge mamba
fi

# Create the conda environment from the YAML file using mamba
echo "Creating conda environment from environment.yml using mamba..."
mamba env create --file ./references/conda_envs/workflow_16s.yml --prefix "$ENV_PATH"

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
