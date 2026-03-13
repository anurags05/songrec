# Project Explanation: Song Recommendation System

## Overview
The **Song Recommendation System** is a scalable, memory-efficient, content-based music recommendation engine built with Python. It allows users to discover new music based on song similarity or artist style by analyzing both textual metadata and numeric audio features.

The system is designed to handle large-scale datasets (up to 1M+ rows) using sparse matrices and optimized data processing techniques, making it production-ready and modular for future enhancements like a graphical user interface (GUI).

---

## File Structure and Functions

### Core Application
- **`main.py`**: The entry point of the application. It provides a Command-Line Interface (CLI) using `argparse`, handles high-level orchestration (loading data, training/loading models), and manages model persistence through caching.
- **`src/data_loader.py`**: Handles data ingestion, cleaning, and memory optimization. It standardizes column names, optimizes pandas data types (e.g., `float32`, `category`) to reduce memory footprint, and prepares text features for vectorization.
- **`src/features.py`**: Implements the feature engineering pipeline. It uses TF-IDF for text metadata (song name, artist, genre) and `StandardScaler` for numeric audio features (danceability, energy, etc.), combining them into a unified sparse matrix.
- **`src/recommender.py`**: Contains the recommendation logic. It uses the K-Nearest Neighbors algorithm with cosine similarity to find similar tracks. It also incorporates fuzzy matching to handle user typos.
- **`src/utils.py`**: Provides utility functions for logging, dataset hashing (to detect file changes), and directory management.

### Testing and Support
- **`tests/test_recommender.py`**: Contains automated tests to verify the recommendation logic and data processing steps.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.

---

## Key Techniques Used

1. **Content-Based Filtering**: Recommends items based on their features rather than user behavior, making it effective for cold-start scenarios where no user history is available.
2. **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to convert text metadata (names, genres) into numeric vectors, emphasizing unique keywords.
3. **Feature Scaling**: Standardizes numeric audio features (like `loudness` or `tempo`) to ensure they contribute equally to the similarity calculation.
4. **Sparse Matrix Representation**: Uses `scipy.sparse` to store feature vectors efficiently, significantly reducing memory usage for datasets with high-dimensional text features.
5. **K-Nearest Neighbors (KNN)**: An unsupervised learning algorithm used for similarity search, utilizing **Cosine Similarity** to measure the distance between songs in the feature space.
6. **Fuzzy String Matching**: Uses the `rapidfuzz` library to improve user experience by finding the closest match for song or artist names even if the search query contains typos.
7. **Model Persistence**: Uses `joblib` to serialize and save trained models and processed dataframes, allowing for instant startup on subsequent runs.

---

## Datasets Used

The project is designed to work with large-scale music datasets, specifically:
- **`tracks_features.csv`**: A dataset containing approximately 1.2 million tracks with audio features (danceability, energy, etc.) and metadata (name, artist, album).
- **`artists.csv`**: Supplementary artist metadata.
- *Source*: Primarily sourced from Kaggle's Spotify datasets (e.g., [Spotify 1.2M+ Songs](https://www.kaggle.com/datasets/rodolfofigueroa/spotify-12m-songs)).

---

## Project Requirements

- **Python Version**: 3.8+
- **Core Libraries**:
  - `pandas` & `numpy`: For data manipulation.
  - `scikit-learn`: For TF-IDF, Scaling, and KNN.
  - `scipy`: For sparse matrix operations.
  - `rapidfuzz`: For fuzzy search.
  - `joblib`: For saving/loading models.
  - `pytest`: For running tests.

---

## Work in Progress (WIP)

### Current Status
- Fully functional CLI for song and artist-based recommendations.
- Memory-optimized pipeline capable of processing 1M+ tracks.
- Automated testing suite for core logic.

### Future Enhancements
- **GUI Integration**: Transition from a terminal-based interface to a modern desktop application using `customtkinter`.
- **Genre Support**: Improve genre-based filtering and recommendation as many tracks in the current dataset default to "unknown".
- **Dynamic Dataset Hot-Loading**: Allow users to point to new CSV files through the UI without manual path configuration.
- **Advanced Hybrid Search**: Incorporate collaborative filtering elements or user ratings if available.
