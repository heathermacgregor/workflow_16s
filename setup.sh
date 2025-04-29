#!/bin/bash

# Get the absolute path of the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# Set the environment name and the path where the environment will be created
ENV_NAME="workflow_16s"
ENV_PATH="$SCRIPT_DIR/$ENV_NAME"

# Ensure mamba is installed
echo "⟲ Checking if mamba is installed..."
if ! command -v mamba &> /dev/null
then
    echo "⟲ Installing mamba..."
    conda install -y -c conda-forge mamba
fi

# Create qiime2 environment if it doesn't exist
QIIME_ENV="qiime2-amplicon-2024.10"
echo "⟲ Checking if $QIIME_ENV environment exists..."
if ! conda env list | grep -q "$QIIME_ENV"
then
    echo "⟲ Creating $QIIME_ENV environment..."
    OS=$(uname -s)
    if [ "$OS" = "Linux" ]; then
        YAML_URL="https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-linux-conda.yml"
    elif [ "$OS" = "Darwin" ]; then
        YAML_URL="https://data.qiime2.org/distro/amplicon/qiime2-amplicon-2024.10-py310-osx-conda.yml"
    else
        echo "🞫 Unsupported operating system: $OS"
        exit 1
    fi

    echo "⟲ Trying to create environment with mamba..."
    mamba env create -n "$QIIME_ENV" --file "$YAML_URL"
    if [ $? -ne 0 ]; then
        echo "⚠ mamba failed, trying with conda instead..."
        conda env create -n "$QIIME_ENV" --file "$YAML_URL"
        if [ $? -ne 0 ]; then
            echo "🞫 Failed to create $QIIME_ENV environment with both mamba and conda"
            exit 1
        fi
    fi
    echo "✅ $QIIME_ENV environment created successfully"
else
    echo "✔ $QIIME_ENV environment already exists"
fi

# Add Silva database files installation
CLASSIFIER_DIR="$SCRIPT_DIR/references/classifier/silva-138-99-515-806"
SILVA_FILES=(
    "silva-138-99-seqs-515-806.qza"
    "silva-138-99-tax-515-806.qza"
)
CLASSIFIER_FILE="$CLASSIFIER_DIR/silva-138-99-515-806-classifier.qza"
QIIME_BASE_URL="https://data.qiime2.org/2024.10/common"
ZENODO_CLASSIFIER_URL="https://zenodo.org/records/15299267/files/silva-138-99-515-806-classifier.qza"

echo "⟲ Checking for SILVA database files..."
mkdir -p "$CLASSIFIER_DIR"

# Download sequence and taxonomy files from QIIME2
for FILE in "${SILVA_FILES[@]}"; do
    FILE_PATH="$CLASSIFIER_DIR/$FILE"
    FILE_URL="$QIIME_BASE_URL/$FILE"
    
    if [ ! -f "$FILE_PATH" ]; then
        echo "⟲ Downloading $FILE..."
        if command -v wget &> /dev/null; then
            wget --no-verbose --show-progress -O "$FILE_PATH" "$FILE_URL" || { echo "Download failed"; rm -f "$FILE_PATH"; exit 1; }
        elif command -v curl &> /dev/null; then
            curl -# -L "$FILE_URL" -o "$FILE_PATH" || { echo "Download failed"; rm -f "$FILE_PATH"; exit 1; }
        else
            echo "🞫 Error: Need wget or curl to download files"
            exit 1
        fi
        
        if [ ! -f "$FILE_PATH" ]; then
            echo "🞫 Failed to download $FILE - URL might be incorrect or file unavailable"
            exit 1
        fi
        echo "✔ Successfully downloaded $FILE"
    else
        echo "$FILE already exists at: $FILE_PATH"
    fi
done

# Try downloading classifier from Zenodo
if [ ! -f "$CLASSIFIER_FILE" ]; then
    echo "⟲ Attempting classifier download from Zenodo..."
    
    if command -v wget &> /dev/null; then
        wget --no-verbose --show-progress -O "$CLASSIFIER_FILE" "$ZENODO_CLASSIFIER_URL" || DL_FAILED=true
    elif command -v curl &> /dev/null; then
        curl -# -L "$ZENODO_CLASSIFIER_URL" -o "$CLASSIFIER_FILE" || DL_FAILED=true
    else
        echo "🞫 Error: Need wget or curl to download classifier"
        exit 1
    fi
    
    # If download failed, generate classifier
    if [ ${DL_FAILED} ] || [ ! -f "$CLASSIFIER_FILE" ]; then
        echo "⟲ Generating classifier..."
        rm -f "$CLASSIFIER_FILE" 2>/dev/null
        
        # Activate QIIME2 environment
        source activate "$QIIME_ENV"
        
        # Generate classifier
        qiime feature-classifier fit-classifier-naive-bayes \
            --i-reference-reads "$CLASSIFIER_DIR/silva-138-99-seqs-515-806.qza" \
            --i-reference-taxonomy "$CLASSIFIER_DIR/silva-138-99-tax-515-806.qza" \
            --o-classifier "$CLASSIFIER_FILE"
        
        # Deactivate environment
        conda deactivate
        
        if [ ! -f "$CLASSIFIER_FILE" ]; then
            echo "🞫 Failed to generate classifier artifact"
            exit 1
        fi
        echo "✔ Successfully generated classifier artifact"
    else
        echo "✔ Successfully downloaded classifier from Zenodo"
    fi
else
    echo "Classifier already exists at: $CLASSIFIER_FILE"
fi

# Check if the workflow environment exists
if conda env list | grep -q "$ENV_NAME"
then
    echo "⟲ Activating the conda environment '$ENV_NAME'..."
else
    # Create workflow environment
    echo "⟲ Creating the environment '$ENV_NAME' from environment.yml using mamba..."
    mamba env create --file "$SCRIPT_DIR/references/conda_envs/workflow_16s.yml" --prefix "$ENV_PATH"
fi

# Activate the workflow environment
echo "⟲ Activating the conda environment '$ENV_NAME'..."
source activate "$ENV_PATH"

# Check if fastqc is available and install if missing
if ! command -v fastqc &> /dev/null; then
    echo "⟲ Installing FastQC..."
    mamba install -y -c bioconda fastqc
    
    # Verify installation
    if ! command -v fastqc &> /dev/null; then
        echo "🞫 Failed to install FastQC. Please install manually."
        exit 1
    fi
fi

# Run Python script
#echo "Running the Python script 'run.py'..."
#python "$SCRIPT_DIR/src/run.py"

# Deactivate environment
echo "⟲ Deactivating the conda environment..."
conda deactivate

echo "Environment setup complete!"
