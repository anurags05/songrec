# Song Recommendation System - Project Context

## Project Overview

A **scalable, content-based music recommendation engine** built with Python that suggests songs based on audio features and metadata similarity. The system handles large datasets (1M+ tracks) using sparse matrices, memory optimization, and efficient nearest-neighbor search.

### Core Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  data_loader.py │ ──► │  features.py     │ ──► │ recommender.py  │
│  (ETL + Clean)  │     │ (TF-IDF + Scale) │     │ (KNN + Fuzzy)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                         ┌──────────────────┐
                         │  main.py (CLI)   │
                         │  Model Caching   │
                         └──────────────────┘
```

### Key Components

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point, model pipeline orchestration, caching logic |
| `src/data_loader.py` | Data ingestion, column normalization, memory optimization |
| `src/features.py` | TF-IDF vectorization for text, StandardScaler for audio features |
| `src/recommender.py` | KNN with cosine similarity, fuzzy matching via `rapidfuzz` |
| `src/utils.py` | Logging, dataset hashing, directory management |
| `tests/test_recommender.py` | Pytest-based unit tests for core logic |

## Building and Running

### Prerequisites
- **Python**: 3.8+
- **Dependencies**: `pip install -r requirements.txt`

### Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Run with custom dataset
python main.py --dataset /path/to/tracks_features.csv

# Run with sampling (faster for development)
python main.py --sample 1000

# Force rebuild models
python main.py --rebuild

# Run tests
pytest tests/test_recommender.py
```

### CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Path to CSV dataset | `tracks_features.csv` |
| `--sample N` | Use N random rows for testing | `None` |
| `--rebuild` | Force model rebuild | `False` |

### Dataset Requirements

Expected CSV columns (auto-normalized):
- **Required**: `track_name` (or `name`, `song`, `title`), `artist_name` (or `artists`, `artist`)
- **Optional Audio Features**: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`
- **Optional**: `genre` (or `genres`, `track_genre`)

### Model Caching

The system automatically caches:
- `models/model.pkl` - Trained KNN model
- `models/vectorizer.pkl` - TF-IDF vectorizer
- `models/scaler.pkl` - Feature scaler
- `models/processed_df.pkl` - Preprocessed dataframe
- `models/feature_matrix.pkl` - Combined feature matrix
- `models/dataset_hash.txt` - MD5 hash for change detection

Models rebuild automatically if the dataset changes.

## Development Conventions

### Code Style
- **Type hints**: Used throughout (`Optional`, `List`, `Dict`, `Tuple`)
- **Logging**: Structured logging via `src/utils.setup_logging()`
- **Error handling**: Fuzzy matching fallbacks for user input, graceful degradation

### Testing Practices
- **Framework**: pytest
- **Fixtures**: `sample_df` fixture for reusable test data
- **Coverage**: Tests for data loading, feature pipeline, and recommendation logic

### Project Structure

```
songrec/
├── main.py              # Entry point
├── requirements.txt     # Dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py   # ETL pipeline
│   ├── features.py      # Feature engineering
│   ├── recommender.py   # KNN + fuzzy search
│   └── utils.py         # Utilities
├── tests/
│   ├── __init__.py
│   └── test_recommender.py
├── models/              # Cached model artifacts (git-ignored)
├── data/                # Working data directory (git-ignored)
└── docs/
    ├── README.md
    ├── explanation.md
    └── futureplan.md
```

### Key Design Patterns

1. **Pipeline Pattern**: `DataLoader → FeaturePipeline → Recommender`
2. **Sparse Matrix Optimization**: Uses `scipy.sparse.csr_matrix` for memory efficiency
3. **Hybrid Features**: Combines TF-IDF (text) + StandardScaler (numeric) via `hstack`
4. **Fuzzy Matching**: `rapidfuzz` for typo-tolerant search

### Known Limitations / WIP

- GUI integration planned (see `futureplan.md` - mentions `customtkinter`)
- Genre handling: Many tracks default to "unknown"
- No collaborative filtering (content-based only)

## Quick Reference

### Adding a New Feature Column
1. Add to `AUDIO_FEATURES` list in `src/data_loader.py`
2. Ensure column exists in dataset or add normalization mapping
3. Rebuild models with `--rebuild` flag

### Changing Similarity Metric
Edit `src/recommender.py`:
```python
self.model = NearestNeighbors(
    n_neighbors=n_neighbors,
    metric="cosine",  # Change to "euclidean", "manhattan", etc.
    algorithm="auto"
)
```

### Adjusting TF-IDF Parameters
Edit `src/features.py`:
```python
self.vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,  # Adjust vocabulary size
    ngram_range=(1, 2),  # Add bigrams if needed
)
```
