# Song Recommendation System

A scalable, content-based music recommendation engine built with Python, Scikit-Learn, and Pandas.

## Features
- **Scalable**: Efficiently handles 1M+ rows using sparse matrices and index-based search.
- **Memory Optimized**: Uses specialized dtypes to reduce memory footprint.
- **Fuzzy Matching**: Robust search functionality using `rapidfuzz`.
- **Hybrid Features**: Combines text metadata (TF-IDF) with numeric audio features (Standardized).

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Run tests:
   ```bash
   pytest tests/test_recommender.py
   ```
3. Download datasets from kaggle:
[1921-2020 tracks](https://www.kaggle.com/datasets/yamaerenay/spotify-dataset-19212020-600k-tracks)
[Spotify 1.2m](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)

## Usage

Run the main script:
```bash
python main.py
```

### Script Arguments
- `--dataset`: Path to the CSV dataset (defaults to `tracks_features.csv`).
- `--sample N`: Use a random sample of N rows for faster development.
- `--rebuild`: Force the system to rebuild the feature matrix and model.

Example:
```bash
python main.py
```
```
--- Song Recommendation System ---
1. Recommend by Song
2. Recommend by Artist
3. Exit
Select an option (1-3): 1
Enter song name: Bullet In The Head
```



## Dataset Requirements
The system expects a CSV with at least:
- Track Name (or `name`, `song`, `title`)
- Artist Name (or `artists`)
- (Optional) Audio features: `danceability`, `energy`, etc.

## Architecture
- `src/data_loader.py`: Data cleaning and memory optimization.
- `src/features.py`: TF-IDF and Scaling pipeline.
- `src/recommender.py`: NearestNeighbors search and fuzzy matching.
- `main.py`: CLI interface and model persistence.
