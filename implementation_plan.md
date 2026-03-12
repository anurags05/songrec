# Implementation Plan - Song Recommendation System

Build a scalable, memory-efficient content-based recommendation engine for 1M+ songs.

## Proposed Changes

### Project Setup
- Create directory structure: `src/`, `models/`, `data/`, `tests/`.
- Create `requirements.txt` with required libraries.
- **Dataset**: Use [tracks_features.csv](file:///c:/Users/Anurag/Desktop/songrec/tracks_features.csv) located in the root directory.

---

### Core Components

#### [NEW] [utils.py](file:///c:/Users/Anurag/Desktop/songrec/src/utils.py)
- Configure logging with custom format.
- Implement dataset hashing for cache validation.
- Implement standard paths and constants.

#### [NEW] [data_loader.py](file:///c:/Users/Anurag/Desktop/songrec/src/data_loader.py)
- Load CSV using chunking if necessary (or efficient pandas dtypes).
- **Memory Optimization**: Use `float32`, `category`, and `string` dtypes.
- **Normalization**: Map `name` to `track_name`, `artists` to `artist_name`.
- **Merged Data**: Handle missing genres gracefully (default to "unknown" as [tracks_features.csv](file:///c:/Users/Anurag/Desktop/songrec/tracks_features.csv) lacks them explicitly).
- **Features**: Combine text features into a single metadata string.

#### [NEW] [features.py](file:///c:/Users/Anurag/Desktop/songrec/src/features.py)
- Use `TfidfVectorizer` for text features.
- Use `StandardScaler` for numeric audio features.
- Combine using `scipy.sparse.hstack` for memory efficiency.
- Save models to `models/` directory.

#### [NEW] [recommender.py](file:///c:/Users/Anurag/Desktop/songrec/src/recommender.py)
- Use `NearestNeighbors` with cosine similarity on sparse matrices.
- Implement `rapidfuzz` for song/artist query matching.
- Implement song-based and artist-based recommendation logic.

---

### Interface & Entry Point

#### [NEW] [main.py](file:///c:/Users/Anurag/Desktop/songrec/main.py)
- CLI using `argparse`.
- Interactive menu for searching by song or artist.
- Model persistence check (load from cache or rebuild).

---

## Verification Plan

### Automated Tests
- Run `pytest tests/test_recommender.py` to verify recommendation logic.
- Mock dataset with varied column names to test mapping utility.

### Manual Verification
- Generate a small sample dataset (10k rows) and run `main.py`.
- Verify recommendations "make sense" (e.g., similar genres/artists).
- Check memory usage during 100k+ row processing.
