# src/content_based.py
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

def build_content(movies_df):
    # Split the 'genres' column into a list of genres
    movies_df['genres_list'] = movies_df['genres'].apply(lambda x: x.split('|'))

    # MultiLabelBinarizer to convert genres to binary matrix
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(movies_df["genres_list"])

    from collections import Counter
    unique_rows = [tuple(row) for row in genre_mat]
    counts = Counter(unique_rows)
    print("Number of unique genre combinations:", len(counts))
    print("Most common genre combination:", counts.most_common(5))
    print("Genre matrix shape:", genre_mat.shape)
    print("First 5 rows of genre matrix:\n", genre_mat[:5])

    # Compute cosine similarity between movies
    cosine_sim = cosine_similarity(genre_mat)
    
    # Debug: check first 5 rows
    print("Genre matrix shape:", genre_mat.shape)
    print("First 5 rows:\n", genre_mat[:5])
    print("Cosine similarity sample:\n", cosine_sim[:5, :5])

    return mlb, cosine_sim

def similar_by_title(title, movies_df, cosine_sim, title2id, id2idx, top_n=10, exclude_self=True):
    key = title.strip().lower()
    if key not in title2id:
        raise ValueError(f"Title not found: {title}")
    movie_id = title2id[key]
    i = id2idx[movie_id]
    sims = cosine_sim[i]

    # Rank by similarity
    idxs = np.argsort(-sims)
    out = []
    for j in idxs:
        if exclude_self and j == i:
            continue
        out.append((int(movies_df.loc[j, "movieId"]), float(sims[j])))
        if len(out) >= top_n:
            break
    return out  # list of (movieId, score)
