# src/collab_svd.py
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from typing import List, Tuple

def train_svd(ratings_df, n_factors=100, random_state=42):
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(ratings_df[["userId","movieId","rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.2, random_state=random_state)

    algo = SVD(n_factors=n_factors, random_state=random_state)
    algo.fit(trainset)

    # Optional quick RMSE check
    # from surprise import accuracy
    # preds = algo.test(testset)
    # print("RMSE:", accuracy.rmse(preds, verbose=False))

    return algo

def recommend_for_user(algo, user_id: int, all_movie_ids: List[int], user_rated_ids: set, top_n=10) -> List[Tuple[int, float]]:
    # Score all unseen movies for this user
    candidates = [mid for mid in all_movie_ids if mid not in user_rated_ids]
    ests = []
    for mid in candidates:
        est = algo.predict(user_id, mid).est
        ests.append((mid, float(est)))
    # sort desc by est
    ests.sort(key=lambda x: -x[1])
    return ests[:top_n]
