#!/bin/bash

# This script is the single entry point for the Render web service.
# It handles Kaggle authentication, downloads the large PyTorch models, and starts the server.

# --- 1. Variable Check ---
# Render should automatically provide KAGGLE_USERNAME and KAGGLE_API_TOKEN secrets.
if [ -z "$KAGGLE_API_TOKEN" ] || [ -z "$KAGGLE_USERNAME" ]; then
    echo "ERROR: Missing KAGGLE_USERNAME or KAGGLE_API_TOKEN secret. Cannot proceed."
    exit 1
fi

# --- 2. Kaggle Authentication Setup ---
# Create the .kaggle directory and the configuration file required by the Kaggle CLI.
mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_API_TOKEN\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle authentication file created successfully."

# --- 3. Download Models ---
mkdir -p models

# Your confirmed final dataset slug
DATASET_SLUG="k221030116/vgg16-plant-disease-weights-final" 

echo "Downloading models from $DATASET_SLUG..."

# Execute download command
kaggle datasets download -d $DATASET_SLUG -p models/ --unzip

# Check the exit code of the last command (kaggle datasets download)
if [ $? -ne 0 ]; then
    echo "ERROR: Kaggle download failed (Code $?). Check DATASET_SLUG or token validity."
    # If the download fails, exit the deployment process
    exit 1
fi

# Cleanup and Final Startup Preparation
rm -rf models/*.zip
echo "Kaggle models downloaded and ready."

# --- 4. Start the Gunicorn server ---
# $PORT is automatically set by Render
echo "Starting Gunicorn server on port $PORT..."
exec gunicorn --workers 1 --bind 0.0.0.0:$PORT app:app