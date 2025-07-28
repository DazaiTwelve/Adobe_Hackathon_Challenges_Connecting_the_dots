# Filename: challenge_1b/Dockerfile

FROM --platform=linux/amd64 python:3.9-slim

WORKDIR /app

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# --- FINAL OFFLINE MODEL SOLUTION ---
# 1. Create a dedicated directory for the model files
RUN mkdir /model_files

# 2. Copy and run the pre-loading script to save the model to that directory
COPY ./preload_models.py .
RUN python preload_models.py

# 3. Copy your main application source code
# NEW, CORRECTED LINE
COPY ./main_1b.py .
