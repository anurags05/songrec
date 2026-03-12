import argparse
import sys
import os
import joblib
import pandas as pd
from src.utils import setup_logging, get_dataset_hash, DEFAULT_DATASET, MODELS_DIR, ensure_dirs
from src.data_loader import DataLoader, AUDIO_FEATURES
from src.features import FeaturePipeline
from src.recommender import Recommender

logger = setup_logging()

def build_pipeline(dataset_path, sample_n=None):
    """Builds and saves the model pipeline."""
    loader = DataLoader(dataset_path)
    df = loader.load_and_preprocess(sample_n=sample_n)
    
    pipeline = FeaturePipeline()
    feature_matrix = pipeline.fit_transform(df, AUDIO_FEATURES)
    
    recommender = Recommender()
    recommender.fit(feature_matrix)
    
    # Save artifacts
    ensure_dirs()
    pipeline.save_artifacts()
    recommender.save_model()
    
    # Save processed dataframe and hash for quick loading
    joblib.dump(df, MODELS_DIR / "processed_df.pkl")
    ds_hash = get_dataset_hash(dataset_path)
    with open(MODELS_DIR / "dataset_hash.txt", "w") as f:
        f.write(ds_hash)
    
    return loader, pipeline, recommender, df, feature_matrix

def load_pipeline(dataset_path):
    """Loads existing model pipeline if hashes match."""
    hash_path = MODELS_DIR / "dataset_hash.txt"
    if not hash_path.exists():
        return None
        
    with open(hash_path, "r") as f:
        saved_hash = f.read().strip()
    
    if saved_hash != get_dataset_hash(dataset_path):
        logger.info("Dataset changed. Rebuilding pipeline...")
        return None
        
    try:
        df = joblib.load(MODELS_DIR / "processed_df.pkl")
        pipeline = FeaturePipeline()
        pipeline.load_artifacts()
        
        recommender = Recommender()
        recommender.load_model()
        
        # Reconstruct feature matrix (actually better to save it too, but let's re-transform to save space if needed)
        # For performance with 1M+ rows, saving the matrix is better.
        # But here we'll assume we can re-generate or we saved it.
        # Let's check for a saved matrix.
        matrix_path = MODELS_DIR / "feature_matrix.pkl"
        if matrix_path.exists():
            feature_matrix = joblib.load(matrix_path)
        else:
            feature_matrix = pipeline.fit_transform(df, AUDIO_FEATURES)
            joblib.dump(feature_matrix, matrix_path)
            
        loader = DataLoader(dataset_path)
        loader.df = df
        loader._build_indices()
        
        return loader, pipeline, recommender, df, feature_matrix
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Terminal-based Song Recommender")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET), help="Path to CSV dataset")
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to sample for development")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of models")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found at {args.dataset}")
        sys.exit(1)

    cached = None if args.rebuild else load_pipeline(args.dataset)
    
    if cached:
        logger.info("Loaded models from cache.")
        loader, pipeline, recommender, df, feature_matrix = cached
    else:
        logger.info("Initializing and training models (this may take a while)...")
        loader, pipeline, recommender, df, feature_matrix = build_pipeline(args.dataset, args.sample)
        # Ensure matrix is saved for next time
        joblib.dump(feature_matrix, MODELS_DIR / "feature_matrix.pkl")

    while True:
        print("\n--- Song Recommendation System ---")
        print("1. Recommend by Song")
        print("2. Recommend by Artist")
        print("3. Exit")
        choice = input("Select an option (1-3): ")

        if choice == "1":
            song = input("Enter song name: ")
            try:
                recs = recommender.get_recommendations(song, df, feature_matrix)
                print(f"\nRecommended Songs for '{song}':")
                print(recs.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "2":
            artist = input("Enter artist name: ")
            try:
                recs = recommender.get_recommendations_by_artist(artist, df, feature_matrix)
                print(f"\nRecommended Songs like '{artist}':")
                print(recs.to_string(index=False))
            except Exception as e:
                print(f"Error: {e}")
        
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
