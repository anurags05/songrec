import pytest
import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.features import FeaturePipeline
from src.recommender import Recommender

@pytest.fixture
def sample_df():
    data = {
        "track_name": ["song a", "song b", "song c", "song d"],
        "artist_name": ["artist 1", "artist 1", "artist 2", "artist 3"],
        "genre": ["pop", "pop", "rock", "jazz"],
        "danceability": [0.8, 0.7, 0.2, 0.5],
        "energy": [0.9, 0.8, 0.1, 0.4],
        "metadata": ["artist 1 pop song a", "artist 1 pop song b", "artist 2 rock song c", "artist 3 jazz song d"]
    }
    return pd.DataFrame(data)

def test_data_loader_normalization():
    loader = DataLoader("dummy.csv")
    df = pd.DataFrame({"name": ["s1"], "artists": ["a1"]})
    normalized = loader.normalize_columns(df)
    assert "track_name" in normalized.columns
    assert "artist_name" in normalized.columns

def test_feature_pipeline(sample_df):
    pipeline = FeaturePipeline()
    matrix = pipeline.fit_transform(sample_df, ["danceability", "energy"])
    assert matrix.shape[0] == 4
    # TF-IDF + 2 numeric features
    assert matrix.shape[1] > 2

def test_recommender_logic(sample_df):
    pipeline = FeaturePipeline()
    matrix = pipeline.fit_transform(sample_df, ["danceability", "energy"])
    
    recommender = Recommender(n_neighbors=2)
    recommender.fit(matrix)
    
    # Recommendation for 'song a' should likely be 'song b' (same artist/genre)
    recs = recommender.get_recommendations("song a", sample_df, matrix, n=1)
    assert len(recs) == 1
    assert recs.iloc[0]["track_name"] == "song b"
