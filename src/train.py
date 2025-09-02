# src/train.py
import json
import joblib
import pickle
import numpy as np
from pathlib import Path
from src.utils import load_ml100k, idx_mappings
from src.content_based import build_content
from src.collab_svd import train_svd

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

def main():
    ratings, movies = load_ml100k("ml-latest-small")
    movies, id2idx, idx2id, id2title, title2id = idx_mappings(movies)

    # --- Content-based
    mlb, cosine_sim = build_content(movies)

    # --- Collaborative (SVD)
    svd = train_svd(ratings)

    # Save artifacts
    joblib.dump(mlb, MODELS_DIR / "tfidf_content.joblib")
    np.save(MODELS_DIR / "cosine_sim.npy", cosine_sim)
    with open(MODELS_DIR / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    joblib.dump(
        {
            "id2idx": id2idx,
            "idx2id": idx2id,
            "id2title": id2title,
            "title2id": title2id,
        },
        MODELS_DIR / "mappings.joblib"
    )
    (MODELS_DIR / "config.json").write_text(json.dumps({"alpha": 0.7, "top_n": 10}, indent=2))
    print("✅ Models saved in ./models")

if __name__ == "__main__":
    main()
