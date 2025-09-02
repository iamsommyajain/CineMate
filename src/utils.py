# src/utils.py
import pandas as pd
from pathlib import Path

GENRES_ORDER = [
    "unknown","Action","Adventure","Animation","Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical","Mystery",
    "Romance","Sci-Fi","Thriller","War","Western"
]

import pandas as pd
from pathlib import Path

def load_ml100k(dataset_path="ml-latest-small"):
    base = Path(dataset_path)
    
    # Load ratings
    ratings = pd.read_csv(base / "ratings.csv")  # columns: userId,movieId,rating,timestamp
    
    # Load movies
    movies = pd.read_csv(base / "movies.csv")  # columns: movieId,title,genres

    # If needed, convert genres string to list
    movies["genres_list"] = movies["genres"].apply(lambda x: x.split("|"))
    
    return ratings, movies

def idx_mappings(movies_df):
    # Force zero-based contiguous item index for matrices
    movies_df = movies_df.reset_index(drop=True).copy()
    movies_df["item_idx"] = range(len(movies_df))
    id2idx = dict(zip(movies_df["movieId"], movies_df["item_idx"]))
    idx2id = dict(zip(movies_df["item_idx"], movies_df["movieId"]))
    id2title = dict(zip(movies_df["movieId"], movies_df["title"]))
    title2id = {t.lower(): mid for mid, t in zip(movies_df["movieId"], movies_df["title"])}
    return movies_df, id2idx, idx2id, id2title, title2id
