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

# Add Silva database files installation
CLASSIFIER_DIR="$SCRIPT_DIR/references/classifier/silva-138-99-515-806"
SILVA_FILES=(
    "silva-138-99-seqs-515-806.qza"
    "silva-138-99-tax-515-806.qza"
)
CLASSIFIER_FILE="$CLASSIFIER_DIR/silva-138-99-515-806-classifier.qza"
BASE_URL="https://data.qiime2.org/2024.10/common"

echo "Checking for SILVA database files..."
mkdir -p "$CLASSIFIER_DIR"

# First download the base files
for FILE in "${SILVA_FILES[@]}"; do
    FILE_PATH="$CLASSIFIER_DIR/$FILE"
    FILE_URL="$BASE_URL/$FILE"
    
    if [ ! -f "$FILE_PATH" ]; then
        echo "Downloading $FILE..."
        if command -v wget &> /dev/null; then
            wget --no-verbose --show-progress -O "$FILE_PATH" "$FILE_URL" || { echo "Download failed"; rm -f "$FILE_PATH"; exit 1; }
        elif command -v curl &> /dev/null; then
            curl -# -L "$FILE_URL" -o "$FILE_PATH" || { echo "Download failed"; rm -f "$FILE_PATH"; exit 1; }
        else
            echo "Error: Need wget or curl to download files"
            exit 1
        fi
        
        if [ ! -f "$FILE_PATH" ]; then
            echo "Failed to download $FILE - URL might be incorrect or file unavailable"
            exit 1
        fi
        echo "Successfully downloaded $FILE"
    else
        echo "$FILE already exists at: $FILE_PATH"
    fi
done

# Now create classifier if needed
if [ ! -f "$CLASSIFIER_FILE" ]; then
    echo "Creating classifier artifact..."
    
    # Activate QIIME2 environment
    source activate "$QIIME_ENV"
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    
    # Import sequences and taxonomy
    qiime tools import \
        --type 'FeatureData[Sequence]' \
        --input-path "$CLASSIFIER_DIR/silva-138-99-seqs-515-806.qza" \
        --output-path "$TEMP_DIR/sequences.qza"

    qiime tools import \
        --type 'FeatureData[Taxonomy]' \
        --input-path "$CLASSIFIER_DIR/silva-138-99-tax-515-806.qza" \
        --output-path "$TEMP_DIR/taxonomy.qza"

    # Train classifier
    qiime feature-classifier fit-classifier-naive-bayes \
        --i-reference-reads "$TEMP_DIR/sequences.qza" \
        --i-reference-taxonomy "$TEMP_DIR/taxonomy.qza" \
        --o-classifier "$CLASSIFIER_FILE"
    
    # Cleanup and deactivate
    rm -rf "$TEMP_DIR"
    conda deactivate
    
    if [ ! -f "$CLASSIFIER_FILE" ]; then
        echo "Failed to create classifier artifact"
        exit 1
    fi
    echo "Successfully created classifier artifact"
else
    echo "Classifier artifact already exists at: $CLASSIFIER_FILE"
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
