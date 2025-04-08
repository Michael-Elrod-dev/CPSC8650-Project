# config.py

import os
from pathlib import Path
import torch

# Path configurations
BASE_DIR = Path(__file__).parent
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
EXTRACTED_DATA_DIR = os.path.join(RAW_DATA_DIR, "files")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Dataset configurations
IMAGE_SIZE = (256, 256, 150)  # As specified in the paper
TRAIN_VAL_TEST_SPLIT = (0.7, 0.1, 0.2)

# Model configurations
NUM_CLASSES = 1  # Binary classification
BATCH_SIZE = 4   # Reduced from 10 to handle larger images if needed
LEARNING_RATE = 1e-3  # As specified in the paper
NUM_EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training configurations
RANDOM_SEED = 42
NUM_FOLDS = 10  # 10-fold cross-validation as in the paper
EARLY_STOPPING_PATIENCE = 10

# Task configurations
TASKS = ["facial_features", "brain_tissue_loss"]

# Reference metrics from the paper
REFERENCE_METRICS = {
    "facial_features": {
        "accuracy": 0.9549,
        "sensitivity": 0.9442,
        "specificity": 0.9552,
        "roc_auc": 0.991
    },
    "brain_tissue_loss": {
        "accuracy": 0.9763,
        "sensitivity": 0.9768,
        "specificity": 0.9762,
        "roc_auc": 0.993
    }
}