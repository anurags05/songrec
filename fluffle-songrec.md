# Song Recommendation System - Project Context

## Project Overview

A **scalable, content-based music recommendation engine** built with Python, Scikit-Learn, and Pandas. The system analyzes both textual metadata (song name, artist, genre) and numeric audio features (danceability, energy, etc.) to recommend similar songs or artists.

### Key Characteristics
- **Scale**: Designed to handle 1M+ tracks efficiently using sparse matrices
- **Memory Optimized**: Uses specialized dtypes (`float32`, `category`) to reduce memory footprint
- **Hybrid Features**: Combines TF-IDF text vectors with standardized numeric audio features
- **Fuzzy Matching**: Robust search using `rapidfuzz` to handle typos in user queries
- **Model Persistence**: Caches trained models for instant startup on subsequent runs

---

## Directory Structure

```
songrec/
├── main.py                 # CLI entry point and model orchestration
├── tui.py                  # Terminal UI (textual-based)
├── compare.py              # Comparison utilities
├── requirements.txt        # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Data ingestion, cleaning, memory optimization
│   ├── features.py         # TF-IDF + StandardScaler pipeline
│   ├── recommender.py      # KNN-based recommendation engine
│   └── utils.py            # Logging, hashing, directory management
├── tests/
│   └── test_recommender.py # Unit tests for core logic
├── models/                 # Saved model artifacts (auto-generated)
├── data/                   # Dataset storage (auto-generated)
└── docs/
    ├── README.md           # User-facing documentation
    ├── explanation.md      # Detailed walkthrough
    ├── implementation_plan.md
    └── futureplan.md       # Planned enhancements
```

---

## Building and Running

### Prerequisites
- Python 3.8+
- Dataset: `tracks_features.csv` (download from [Kaggle](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs))

### Installation
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
# Basic run (uses cached models if available)
python main.py

# Force rebuild of models
python main.py --rebuild

# Use a sample of N rows for faster development
python main.py --sample 1000

# Specify custom dataset path
python main.py --dataset /path/to/dataset.csv
```

### Running Tests
```bash
pytest tests/test_recommender.py
```

---

## Architecture

### Core Modules

| Module | Responsibility |
|--------|----------------|
| `main.py` | CLI interface, model caching, pipeline orchestration |
| `src/data_loader.py` | CSV loading, column normalization, memory optimization, index building |
| `src/features.py` | TF-IDF vectorization for text, StandardScaler for numeric features |
| `src/recommender.py` | KNN model with cosine similarity, fuzzy matching, recommendation generation |
| `src/utils.py` | Logging setup, dataset hashing (for cache invalidation), directory management |

### Feature Pipeline

1. **Text Features**: `metadata` (artist + genre + track name) → TF-IDF vectorization
2. **Numeric Features**: Audio features (`danceability`, `energy`, `loudness`, etc.) → StandardScaler
3. **Combined**: Sparse matrix via `scipy.sparse.hstack`

### Recommendation Algorithms

- **By Song**: Finds k-nearest neighbors using cosine similarity on the feature matrix
- **By Artist**: Computes average feature vector for artist's songs, then finds similar songs by other artists

---

## Development Conventions

### Code Style
- Type hints used throughout (`Optional`, `List`, `Dict`, `Tuple`)
- Logging via `logging` module with module-level loggers
- Path handling via `pathlib.Path`

### Testing Practices
- Pytest fixtures for sample data (`sample_df`)
- Tests verify core logic: column normalization, feature pipeline output shape, recommendation ordering

### Data Handling
- Column normalization maps variants (`name`, `track`, `song` → `track_name`)
- Memory optimization: `float32` for numeric, `category` for text columns
- Index reset after preprocessing to align DataFrame indices with sparse matrix rows

---

## Key Configuration

### Audio Features Used
```python
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo"
]
```

### Model Artifacts (saved to `models/`)
| File | Content |
|------|---------|
| `vectorizer.pkl` | Fitted TfidfVectorizer |
| `scaler.pkl` | Fitted StandardScaler |
| `model.pkl` | Fitted NearestNeighbors model |
| `processed_df.pkl` | Preprocessed DataFrame |
| `feature_matrix.pkl` | Combined sparse feature matrix |
| `dataset_hash.txt` | MD5 hash for cache invalidation |

---

## Known Limitations & Future Work

### Current Limitations
- Genre data often defaults to "unknown" in the dataset
- No collaborative filtering (content-based only)
- CLI-only interface (GUI planned with `customtkinter`)

### Planned Enhancements
- GUI integration using `customtkinter`
- Improved genre-based filtering
- Dynamic dataset hot-loading
- Advanced hybrid search (potential collaborative filtering)

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dataset not found error | Ensure `tracks_features.csv` exists or use `--dataset` flag |
| Models rebuilding every run | Check `models/dataset_hash.txt` matches current dataset |
| Memory errors with large datasets | Use `--sample N` flag for development |
| Song not found | Fuzzy matching should auto-correct; check spelling |
