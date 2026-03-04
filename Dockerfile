# Dockerfile

# Use a stable Python base image suitable for PyTorch
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy essential files
COPY requirements.txt .
COPY download_models.sh .
COPY app.py .
COPY index.html .
COPY detector-logic.js .

# Install dependencies and make the script executable
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x download_models.sh

# Hugging Face Spaces uses the PORT environment variable, often 7860
ENV PORT 7860

# Combined startup command: Run model download, then start Gunicorn
CMD ["/bin/bash", "-c", "./download_models.sh && gunicorn --workers 1 --bind 0.0.0.0:$PORT app:app"]