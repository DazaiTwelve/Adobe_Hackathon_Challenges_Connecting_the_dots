# Filename: challenge_1b/preload_models.py
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
SAVE_PATH = "/model_files"

# Download the model from Hugging Face
model = SentenceTransformer(MODEL_NAME)

# Save the model to a local directory
model.save(SAVE_PATH)

print(f"Model '{MODEL_NAME}' saved to '{SAVE_PATH}' successfully.")