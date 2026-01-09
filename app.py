import streamlit as st
import numpy as np
import pandas as pd
import pickle
from numba import njit

# 1. EXACT Hyperparameters (from your paper)
OPTIMAL_PARAMS = {
    'lamda': 0.0204,
    'gamma_u': 0.8588,
    'gamma_b': 0.0019,
    'k': 20
}

# 2. Updated Numba Kernel (With Filtering)
@njit(fastmath=True)
def _predict_new_user(movie_indices, rating_values, V, b_m, movie_counts):
    # Unpack params
    lamda = 0.0204
    gamma_u = 0.8588
    gamma_b = 0.0019
    k = 20
    
    # Initialize
    dummy_emb = np.zeros(k, dtype=np.float64)
    dummy_bias = 0.0
    n_ratings = len(movie_indices)
    eye_k = np.eye(k, dtype=np.float64) * gamma_u

    # Fold-in (10 iterations)
    for _ in range(10):
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
        
    # Calculate Scores
    # score = bias + item_bias + dot(user, item)
    scores = dummy_bias + b_m + np.dot(V.T, dummy_emb)
    
    # FILTER: Set score to -Infinity if movie has < 50 ratings
    # This matches your original code exactly
    for m in range(len(scores)):
        if movie_counts[m] < 50:
            scores[m] = -np.inf
            
    return scores

# 3. Load Data
@st.cache_resource
def load_data():
    df_movies = pd.read_csv("movies.csv")
    V = np.load("movie_embeddings.npy")
    b_m = np.load("movie_biases.npy")
    movie_counts = np.load("movie_counts.npy") # <--- NEW LOAD
    
    with open("movie_map.pkl", "rb") as f:
        movie_map = pickle.load(f)
        
    idx_to_movieid = {v: k for k, v in movie_map.items()}
    return df_movies, V, b_m, movie_counts, movie_map, idx_to_movieid

# ... [Standard Streamlit Boilerplate] ...
try:
    df_movies, V, b_m, movie_counts, movie_map, idx_to_movieid = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

st.title("🎬 MovieLens 32M ")
st.sidebar.header("User Control")
selected_title = st.sidebar.selectbox("Movie you liked:", df_movies['title'].unique())

if st.sidebar.button("Recommend"):
    movie_id = df_movies[df_movies['title'] == selected_title].iloc[0]['movieId']
    
    if movie_id in movie_map:
        m_idx = movie_map[movie_id]
        
        # Prepare Input
        indices = np.array([m_idx], dtype=np.int32)
        values = np.array([5.0], dtype=np.float64)
        
        # Run Inference with Counts
        scores = _predict_new_user(indices, values, V, b_m, movie_counts)
        
        # Top 10
        top_indices = np.argsort(scores)[::-1][:11]
        
        st.subheader(f"Recommendations for '{selected_title}':")
        for idx in top_indices:
            if idx == m_idx: continue
            real_id = idx_to_movieid[idx]
            title = df_movies[df_movies['movieId'] == real_id].iloc[0]['title']
            st.write(f"**{title}**")
