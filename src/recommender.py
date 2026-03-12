import joblib
import logging
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.neighbors import NearestNeighbors
from rapidfuzz import process, fuzz
from src.utils import MODELS_DIR

logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(
            n_neighbors=n_neighbors, 
            metric="cosine", 
            algorithm="auto"
        )
        self.is_fitted = False

    def fit(self, feature_matrix):
        """Fits the NearestNeighbors model on the feature matrix."""
        logger.info("Fitting NearestNeighbors model...")
        self.model.fit(feature_matrix)
        self.is_fitted = True

    def find_song_fuzzy(self, query: str, song_list: List[str]) -> str:
        """Uses fuzzy matching to find the best song name match."""
        match = process.extractOne(query, song_list, scorer=fuzz.WRatio)
        if match:
            return match[0]
        return None

    def get_recommendations(self, song_name: str, df: pd.DataFrame, feature_matrix, n: int = 10) -> pd.DataFrame:
        """Recommends songs based on a specific song name."""
        song_name = song_name.lower().strip()
        
        if song_name not in df["track_name"].values:
            logger.info(f"Song '{song_name}' not found. Attempting fuzzy match...")
            fuzzy_name = self.find_song_fuzzy(song_name, df["track_name"].unique())
            if not fuzzy_name:
                raise ValueError(f"Song '{song_name}' not found even with fuzzy matching.")
            song_name = fuzzy_name
            logger.info(f"Using fuzzy match: '{song_name}'")

        # Get the row index for the song
        idx = df[df["track_name"] == song_name].index[0]
        target_features = feature_matrix[idx]

        # Query index
        distances, indices = self.model.kneighbors(target_features, n_neighbors=n+1)
        
        # Exclude the input song itself
        rec_indices = indices.flatten()[1:]
        return df.iloc[rec_indices][["track_name", "artist_name", "genre"]]

    def get_recommendations_by_artist(self, artist_name: str, df: pd.DataFrame, feature_matrix, n: int = 10) -> pd.DataFrame:
        """Recommends songs based on similarity to an artist's average profile."""
        artist_name = artist_name.lower().strip()
        
        artist_songs = df[df["artist_name"] == artist_name]
        if artist_songs.empty:
            # Attempt fuzzy artist match
            fuzzy_artist = self.find_song_fuzzy(artist_name, df["artist_name"].unique())
            if not fuzzy_artist:
                raise ValueError(f"Artist '{artist_name}' not found.")
            artist_name = fuzzy_artist
            artist_songs = df[df["artist_name"] == artist_name]

        # Compute average feature vector for the artist
        artist_indices = artist_songs.index
        artist_features = feature_matrix[artist_indices]
        avg_vector = artist_features.mean(axis=0)

        # Query index
        distances, indices = self.model.kneighbors(avg_vector, n_neighbors=n + artist_songs.shape[0])
        
        # Filter out songs by the same artist
        rec_indices = indices.flatten()
        recs = df.iloc[rec_indices]
        filtered_recs = recs[recs["artist_name"] != artist_name].head(n)
        
        return filtered_recs[["track_name", "artist_name", "genre"]]

    def save_model(self):
        logger.info(f"Saving Recommender model to {MODELS_DIR}...")
        joblib.dump(self.model, MODELS_DIR / "model.pkl")

    def load_model(self):
        logger.info(f"Loading Recommender model from {MODELS_DIR}...")
        self.model = joblib.load(MODELS_DIR / "model.pkl")
        self.is_fitted = True
