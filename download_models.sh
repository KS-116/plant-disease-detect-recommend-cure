#!/bin/bash

# --- 1. Configure Kaggle Authentication ---
# We assume the environment variable KAGGLE_API_TOKEN is set in Azure.
if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "ERROR: KAGGLE_API_TOKEN environment variable is not set in Azure. Cannot authenticate."
    exit 1
fi

# --- 2. Create Directory and Download Models ---
mkdir -p models

# !!! FINAL DEPLOYMENT SLUG IS USED HERE !!!
DATASET_SLUG="k221030116/vgg16-plant-disease-weights-final" 

echo "Downloading models from $DATASET_SLUG..."

# The Kaggle CLI automatically authenticates using the environment variable.
kaggle datasets download -d $DATASET_SLUG -p models/ --unzip

if [ $? -ne 0 ]; then
    echo "ERROR: Kaggle download failed. Check token validity or if the slug is correct."
    exit 1
fi

# Cleanup temporary zip files
rm -rf models/*.zip

echo "Kaggle models downloaded and ready."