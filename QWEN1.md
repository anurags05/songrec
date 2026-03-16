# Song Recommendation System - Project Context

## Project Overview

A **scalable, content-based music recommendation engine** built with Python that suggests songs based on audio features and metadata similarity. The system handles large datasets (1M+ tracks) using sparse matrices, memory optimization, and efficient nearest-neighbor search.

### Core Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  data_loader.py │ ──►  │  features.py     │ ──►  │ recommender.py   │
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
| `main.py` | CLI entry point, model pipeline orchestration, caching logic. |
| `src/data_loader.py` | Data ingestion, cleaning of data, memory optimization. Also standardizes columns names (e.g., `genre`, `artist`) to reduce memory footprint.|
| `src/features.py` | This is the feature engineering pipeline. Combines -> TF-IDF vectorization for text metadata (`song name`,`artist`), StandardScaler for audio features. |
| `src/recommender.py` | Uses KNN with cosine similarity to find similar songs and artists, fuzzy matching via `rapidfuzz` to handle spelling errors. |
| `src/utils.py` | Utility set for Logging, dataset hashing, directory management. |
| `tests/test_recommender.py` | Pytest-based unit tests to verif core logic and data processing.|

## Key Techniques

1. **Content-Based Filtering**: Recommends items based on their features rather than user behavior, making it effective for cold-start scenarios where no user history is available.
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to convert text metadata (names, genres) into numeric vectors, emphasizing unique keywords.
3. **Feature Scaling**: Standardizes numeric audio features (like `loudness` or `tempo`) to ensure they contribute equally to the similarity calculation.
4. **Sparse Matrix Representation**: Uses `scipy.sparse` to store feature vectors efficiently, significantly reducing memory usage for datasets with high-dimensional text features.
5. **K-Nearest Neighbors (KNN)**: An unsupervised learning algorithm used for similarity search, utilizing **Cosine Similarity** to measure the distance between songs in the feature space.
6. **Fuzzy String Matching**: Uses the `rapidfuzz` library to improve user experience by finding the closest match for song or artist names even if the search query contains typos.
7. **Model Persistence**: Uses `joblib`  to save trained models in cache allowing for instant startup on future runs.

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

or 

# Run the Terminal UI application 
python tui.py

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
| `--tui` | Run project with terminal UI | `False` |

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
