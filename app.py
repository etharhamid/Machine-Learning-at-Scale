import streamlit as st
import numpy as np
import pandas as pd
import pickle
from numba import njit

# 1. Light Numba Kernel (Only for inference, no training code needed)
@njit(fastmath=True)
def _predict_new_user(movie_indices, rating_values, V, b_m, lamda=0.02, gamma_u=0.8, gamma_b=0.002, k=20, iterations=10):
    # Initialize dummy user
    dummy_emb = np.zeros(k, dtype=np.float64)
    dummy_bias = 0.0
    n_ratings = len(movie_indices)
    eye_k = np.eye(k, dtype=np.float64) * gamma_u

    # Fold-in (Iterative Least Squares for 1 user)
    for _ in range(iterations):
        # Update Bias
        bias_sum = 0.0
        for i in range(n_ratings):
            m_idx = movie_indices[i]
            dot = 0.0
            for f in range(k): dot += dummy_emb[f] * V[f, m_idx]
            bias_sum += lamda * (rating_values[i] - b_m[m_idx] - dot)
        dummy_bias = bias_sum / (lamda * n_ratings + gamma_b)

        # Update Embedding
        rhs = np.zeros(k, dtype=np.float64)
        lhs = eye_k.copy()
        for i in range(n_ratings):
            m_idx = movie_indices[i]
            resid = rating_values[i] - dummy_bias - b_m[m_idx]
            factor = lamda
            for r in range(k):
                rhs[r] += factor * V[r, m_idx] * resid
                for c in range(k):
                    lhs[r, c] += factor * V[r, m_idx] * V[c, m_idx]
        dummy_emb = np.linalg.solve(lhs, rhs)
        
    return dummy_emb, dummy_bias

# 2. Page Config
st.set_page_config(page_title="MovieLens 32M Demo", layout="wide")
st.title("🎬 MovieLens 32M - ALS Model Demo")

# 3. Load Pre-Trained Weights
@st.cache_resource
def load_data():
    # Load Metadata
    df_movies = pd.read_csv("movies.csv")
    
    # Load Model (Match the new filenames)
    V = np.load("movie_embeddings.npy")
    b_m = np.load("movie_biases.npy")
    
    # Load Mappings
    with open("movie_map.pkl", "rb") as f:
        movie_map = pickle.load(f)
        
    # Create Reverse Map (Index -> Real ID)
    # This is needed to look up the movie title after getting a recommendation index
    idx_to_movieid = {v: k for k, v in movie_map.items()}
    
    return df_movies, V, b_m, movie_map, idx_to_movieid

# 4. Sidebar Controls
st.sidebar.header("Try the Model")
selected_title = st.sidebar.selectbox("Select a movie you like:", df_movies['title'].unique())

if st.sidebar.button("Recommend"):
    # Get ID
    movie_row = df_movies[df_movies['title'] == selected_title]
    if movie_row.empty:
        st.error("Movie not found in DB")
    else:
        movie_id = movie_row.iloc[0]['movieId']
        
        if movie_id in movie_map:
            m_idx = movie_map[movie_id]
            
            # Prepare Input
            indices = np.array([m_idx], dtype=np.int32)
            values = np.array([5.0], dtype=np.float64) # Assume 5-star rating
            
            # Run Inference
            user_emb, user_bias = _predict_new_user(indices, values, V, b_m)
            
            # Score all movies (Dot Product)
            # score = user_bias + item_bias + user_emb . item_emb
            scores = user_bias + b_m + np.dot(V.T, user_emb)
            
            # Top 10
            top_indices = np.argsort(scores)[::-1][:11]
            
            st.subheader(f"Recommendations based on '{selected_title}':")
            results = []
            for idx in top_indices:
                if idx == m_idx: continue # Skip input
                real_id = idx_to_movieid.get(idx, -1)
                meta = df_movies[df_movies['movieId'] == real_id]
                if not meta.empty:
                    results.append(meta.iloc[0]['title'])
            
            for i, title in enumerate(results, 1):
                st.write(f"{i}. **{title}**")
        else:
            st.warning("This movie was not in the training set (filtered out due to low ratings).")
