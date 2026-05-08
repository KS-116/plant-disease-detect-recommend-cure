if [ -z "$KAGGLE_API_TOKEN" ]; then
    echo "ERROR: KAGGLE_API_TOKEN environment variable is not set in Azure. Cannot authenticate."
    exit 1
fi

mkdir -p models

DATASET_SLUG="k221030116/vgg16-plant-disease-weights-final" 

echo "Downloading models from $DATASET_SLUG..."

kaggle datasets download -d $DATASET_SLUG -p models/ --unzip

if [ $? -ne 0 ]; then
    echo "ERROR: Kaggle download failed. Check token validity or if the slug is correct."
    exit 1
fi

rm -rf models/*.zip

echo "Kaggle models downloaded and ready."