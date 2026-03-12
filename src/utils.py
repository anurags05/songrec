import logging
import hashlib
import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DEFAULT_DATASET = BASE_DIR / "tracks_features.csv"

# Configuration
LOG_FORMAT = "[%(levelname)s] %(asctime)s %(module)s - %(message)s"

def setup_logging(level=logging.INFO):
    """Configures the logging system."""
    logging.basicConfig(level=level, format=LOG_FORMAT)
    return logging.getLogger("song_recommender")

def get_dataset_hash(file_path: str) -> str:
    """Generates a MD5 hash of the dataset to detect changes."""
    if not os.path.exists(file_path):
        return ""
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def ensure_dirs():
    """Ensures all required project directories exist."""
    for d in [DATA_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
