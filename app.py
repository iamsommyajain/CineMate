# app.py
import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd
from src.collab_svd import recommend_for_user
from src.content_based import similar_by_title
from src.utils import load_ml100k

# -------------------------
# Cached data loading
# -------------------------
@st.cache_data
def load_data():
    st.write("Loading dataset...")
    ratings, movies = load_ml100k("ml-latest-small")
    st.write(f"Loaded {len(movies)} movies and {len(ratings)} ratings")
    return ratings, movies

@st.cache_resource
def load_models():
    st.write("Loading models...")
    # SVD model
    with open("./models/svd.pkl", "rb") as f:
        svd_model = pickle.load(f)

    # Content-based
    mappings = joblib.load("./models/mappings.joblib")
    title2id = mappings["title2id"]
    id2idx = mappings["id2idx"]
    cosine_sim = np.load("./models/cosine_sim.npy")

    st.write("Models loaded successfully!")
    return svd_model, title2id, id2idx, cosine_sim

# -------------------------
# Load everything
# -------------------------
ratings, movies = load_data()
svd_model, title2id, id2idx, cosine_sim = load_models()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎬 Movie Recommender with Debug")
st.write("Choose the type of recommendation below:")

option = st.radio("Recommendation type:", ["User-based (SVD)", "Content-based (Similar Movies)"])

if option == "User-based (SVD)":
    user_id = st.number_input(
        "Enter user ID:",
        min_value=1,
        max_value=int(ratings["userId"].max()),
        value=1,
        step=1
    )
    top_n = st.slider("Number of recommendations:", 1, 20, 5)

    if st.button("Get Recommendations"):
        st.write(f"Fetching top {top_n} recommendations for user {user_id}...")
        user_rated_ids = set(ratings[ratings["userId"] == user_id]["movieId"].tolist())
        recs = recommend_for_user(svd_model, user_id, movies["movieId"].tolist(), user_rated_ids, top_n=top_n)

        if recs:
            st.subheader(f"Top {top_n} recommendations for user {user_id}:")
            for mid, score in recs:
                title = movies[movies["movieId"] == mid]["title"].values[0]
                st.write(f"{title} (score: {score:.2f})")
        else:
            st.warning("No recommendations found for this user.")

elif option == "Content-based (Similar Movies)":
    movie_title = st.text_input("Enter a movie name (exact or partial):", value="")
    top_n = st.slider("Number of similar movies:", 1, 20, 5)
    show_debug = st.checkbox("Show similarity vector for debugging")

    if st.button("Find Similar Movies"):
        if not movie_title.strip():
            st.warning("Please enter a movie name!")
        else:
            st.write(f"Finding top {top_n} movies similar to '{movie_title}'...")
            try:
                similar = similar_by_title(movie_title, movies, cosine_sim, title2id, id2idx, top_n=top_n)
                if similar:
                    st.subheader(f"Top {top_n} movies similar to '{movie_title}':")
                    for mid, score in similar:
                        title = movies[movies["movieId"] == mid]["title"].values[0]
                        st.write(f"{title} (similarity: {score:.2f})")

                    if show_debug:
                        # Show similarity vector for selected movie
                        movie_id = title2id[movie_title.strip().lower()]
                        idx = id2idx[movie_id]
                        sim_vector = cosine_sim[idx]
                        st.subheader(f"Debug: Cosine similarity vector for '{movie_title}'")
                        st.write(sim_vector[:50])  # show first 50 values for brevity
                else:
                    st.warning("No similar movies found.")
            except ValueError as e:
                st.error(str(e))

st.write("✅ App loaded successfully!")
