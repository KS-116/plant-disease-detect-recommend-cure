if [ -z "$KAGGLE_API_TOKEN" ] || [ -z "$KAGGLE_USERNAME" ]; then
    echo "ERROR: Missing KAGGLE_USERNAME or KAGGLE_API_TOKEN secret. Cannot proceed."
    exit 1
fi

mkdir -p ~/.kaggle
echo "{\"username\":\"$KAGGLE_USERNAME\",\"key\":\"$KAGGLE_API_TOKEN\"}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
echo "Kaggle authentication file created successfully."

mkdir -p models

DATASET_SLUG="k221030116/vgg16-plant-disease-weights-final" 

echo "Downloading models from $DATASET_SLUG..."

kaggle datasets download -d $DATASET_SLUG -p models/ --unzip

if [ $? -ne 0 ]; then
    echo "ERROR: Kaggle download failed (Code $?). Check DATASET_SLUG or token validity."
    exit 1
fi

rm -rf models/*.zip
echo "Kaggle models downloaded and ready."

echo "Starting Gunicorn server on port $PORT..."
exec gunicorn --workers 1 --bind 0.0.0.0:$PORT app:app