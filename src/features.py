import joblib
import os
import logging
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from src.utils import MODELS_DIR

logger = logging.getLogger(__name__)

class FeaturePipeline:
    def __init__(self, vectorizer_params: dict = None):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=10000, 
            **(vectorizer_params or {})
        )
        self.scaler = StandardScaler()
        self.feature_matrix: csr_matrix = None

    def fit_transform(self, df: pd.DataFrame, audio_feature_cols: list) -> csr_matrix:
        """Transforms text and numeric columns into a unified sparse matrix."""
        logger.info("Starting feature engineering...")
        
        # 1. Text Features (TF-IDF)
        logger.info("Generating TF-IDF vectors for metadata...")
        text_features = self.vectorizer.fit_transform(df["metadata"])
        
        # 2. Audio Features (Standardization)
        logger.info("Scaling numeric audio features...")
        available_audio = [col for col in audio_feature_cols if col in df.columns]
        if available_audio:
            audio_data = df[available_audio].fillna(0).values
            numeric_features = self.scaler.fit_transform(audio_data)
        else:
            numeric_features = None

        # 3. Combine
        if numeric_features is not None:
            logger.info("Combining sparse text and dense numeric features...")
            self.feature_matrix = hstack([text_features, csr_matrix(numeric_features)]).tocsr()
        else:
            self.feature_matrix = text_features.tocsr()
            
        logger.info(f"Final feature matrix shape: {self.feature_matrix.shape}")
        return self.feature_matrix

    def save_artifacts(self):
        """Saves models to disk for later use."""
        logger.info(f"Saving feature engineering artifacts to {MODELS_DIR}...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, MODELS_DIR / "vectorizer.pkl")
        joblib.dump(self.scaler, MODELS_DIR / "scaler.pkl")

    def load_artifacts(self):
        """Loads models from disk."""
        logger.info(f"Loading feature engineering artifacts from {MODELS_DIR}...")
        self.vectorizer = joblib.load(MODELS_DIR / "vectorizer.pkl")
        self.scaler = joblib.load(MODELS_DIR / "scaler.pkl")
