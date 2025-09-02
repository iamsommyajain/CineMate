# src/test_recommend.py

import pickle
import joblib
import numpy as np
import pandas as pd
from src.collab_svd import recommend_for_user
from src.content_based import similar_by_title
from src.utils import load_ml100k

# ----------------------
# Load dataset
# ----------------------
ratings, movies = load_ml100k("ml-latest-small")

# ----------------------
# Load trained SVD model
# ----------------------
with open("./models/svd.pkl", "rb") as f:
    svd_model = pickle.load(f)

# Example: Recommend movies for a user
user_id = 1  # change to any userId from ratings
all_movie_ids = movies["movieId"].tolist()
user_rated_ids = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())

top_n = 5
svd_recs = recommend_for_user(svd_model, user_id, all_movie_ids, user_rated_ids, top_n=top_n)
print(f"\nTop {top_n} SVD recommendations for user {user_id}:")
for mid, score in svd_recs:
    title = movies[movies["movieId"] == mid]["title"].values[0]
    print(f"{title} (score: {score:.2f})")

# ----------------------
# Load content-based model
# ----------------------
# Mappings
mappings = joblib.load("./models/mappings.joblib")
title2id = mappings["title2id"]
id2idx = mappings["id2idx"]

# Cosine similarity matrix
cosine_sim = np.load("./models/cosine_sim.npy")

# Example: Recommend similar movies for a given title
movie_title = "Toy Story (1995)"  # change to any movie in dataset
top_n = 5
similar_movies = similar_by_title(movie_title, movies, cosine_sim, title2id, id2idx, top_n=top_n)
print(f"\nTop {top_n} movies similar to '{movie_title}':")
for mid, score in similar_movies:
    title = movies[movies["movieId"] == mid]["title"].values[0]
    print(f"{title} (similarity: {score:.2f})")
