import os

# Dataset location
DATASET_PATH = "dataset/ravdess"

# Folder locations
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
FEATURE_DIR = "features"

# Audio configuration
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
N_MELS = 128

# Deep learning configuration
EPOCHS = 40
BATCH_SIZE = 32

# Random state
RANDOM_STATE = 42

# Create directories
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# Emotion mapping
EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}