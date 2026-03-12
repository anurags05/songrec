import pandas as pd
import logging
from typing import Dict, List, Optional
from src.utils import DEFAULT_DATASET

logger = logging.getLogger(__name__)

# Column normalization mapping
COLUMN_MAPPING = {
    "track_name": ["name", "track", "song", "title"],
    "artist_name": ["artists", "artist"],
    "genre": ["genre", "genres", "track_genre"]
}

REQUIRED_COLUMNS = ["track_name", "artist_name"]

# Numeric features typically found in music datasets
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", 
    "speechiness", "acousticness", "instrumentalness", 
    "liveness", "valence", "tempo"
]

class DataLoader:
    def __init__(self, dataset_path: str = str(DEFAULT_DATASET)):
        self.dataset_path = dataset_path
        self.df: Optional[pd.DataFrame] = None
        self.song_index: Dict[str, int] = {}
        self.artist_index: Dict[str, List[int]] = {}

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names based on a predefined mapping."""
        cols = df.columns.tolist()
        new_cols = {}
        
        for standard_name, variants in COLUMN_MAPPING.items():
            for variant in variants:
                if variant in cols:
                    new_cols[variant] = standard_name
                    break
        
        df = df.rename(columns=new_cols)
        
        # Verify required columns exist
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")
            
        return df

    def optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimizes data types to reduce memory usage."""
        logger.info("Optimizing memory usage...")
        
        # Numeric features to float32
        for col in AUDIO_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("float32")
        
        # Text features to string/category
        if "track_name" in df.columns:
            df["track_name"] = df["track_name"].astype("string")
        if "artist_name" in df.columns:
            df["artist_name"] = df["artist_name"].astype("category")
        if "genre" in df.columns:
            df["genre"] = df["genre"].fillna("unknown").astype("category")
        else:
            df["genre"] = "unknown"
            df["genre"] = df["genre"].astype("category")
            
        return df

    def load_and_preprocess(self, sample_n: Optional[int] = None) -> pd.DataFrame:
        """Loads, cleans, and prepares the dataset."""
        logger.info(f"Loading dataset from {self.dataset_path}...")
        
        # Use low_memory=False for large files to avoid dtype guessing issues
        df = pd.read_csv(self.dataset_path, low_memory=False)
        
        if sample_n:
            logger.info(f"Sampling {sample_n} rows...")
            df = df.sample(n=min(sample_n, len(df)), random_state=42)

        df = self.normalize_columns(df)
        df = df.drop_duplicates(subset=["track_name", "artist_name"])
        df = self.optimize_memory(df)
        
        # Text cleaning
        df["track_name"] = df["track_name"].str.lower().str.strip()
        df["artist_name"] = df["artist_name"].astype(str).str.lower().str.strip()
        
        # Combine text features for TF-IDF
        logger.info("Creating metadata for text features...")
        # Ensure all components used in metadata are strings and fill NAs
        track_meta = df["track_name"].fillna("").astype(str)
        artist_meta = df["artist_name"].fillna("").astype(str)
        genre_meta = df["genre"].fillna("unknown").astype(str)
        
        df["metadata"] = (artist_meta + " " + genre_meta + " " + track_meta).str.strip()
        
        # Final safety check: drop rows where metadata might still be empty or NA
        df = df[df["metadata"].notna() & (df["metadata"] != "")]
        
        # CRITICAL: Reset index so it matches the sparse matrix row indices (0 to N-1)
        df = df.reset_index(drop=True)
        
        self.df = df
        self._build_indices()
        
        logger.info(f"Dataset loaded successfully with {len(df)} rows.")
        return df

    def _build_indices(self):
        """Creates lookup dictionaries for fast searching."""
        if self.df is None:
            return
            
        logger.info("Building lookup indices...")
        # Song to index mapping
        self.song_index = {name: idx for idx, name in enumerate(self.df["track_name"])}
        
        # Artist to list of indices mapping
        self.artist_index = self.df.groupby("artist_name").indices
        # Convert indices from numpy arrays to lists for consistency if needed
        self.artist_index = {k: list(v) for k, v in self.artist_index.items()}
