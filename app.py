import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import gdown

# ==========================================
# 1. PATH & IMPORT SETUP
# ==========================================
# Get the absolute path of the directory where app.py is located
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / 'src'
# DATA_PATH = BASE_DIR / 'data' / 'movies.csv'  <-- Comment this out
DATA_PATH = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/data/movies.csv"
MODEL_PATH = BASE_DIR / 'best_model.npz'


# The rest of your code remains the same:
movies_df = load_movies_data(str(DATA_PATH))

# Add src to system path so we can import modules
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import modules with error handling
try:
    from recommender import MovieRecommender
    from utils import (
        load_movies_data,
        create_rating_distribution_plot,
        create_similarity_plot,
        create_polarization_plot,
        format_genres
    )
except ImportError as e:
    st.error(f"""
    ❌ Import Error: {e}
    
    Please ensure your folder structure is:
    - app.py
    - src/
        - recommender.py
        - utils.py
    """)
    st.stop()

# ==========================================
# 2. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Insert your Google Drive File ID here
# Example Link: https://drive.google.com/file/d/1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq/view?usp=sharing
# The ID is: 1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq
FILE_ID = '1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq' 

# ==========================================
# 3. DATA LOADING & CACHING
# ==========================================
@st.cache_resource
def load_data_and_model():
    """
    Downloads model from Drive if missing, then loads data and model.
    Cached so it only runs once.
    """
    # 1. Download Model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        try:
            with st.spinner("📥 Downloading model from Google Drive (100MB+)..."):
                gdown.download(url, str(MODEL_PATH), quiet=False)
            st.success("✅ Download complete!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    # 2. Load Movies Data
    if not os.path.exists(DATA_PATH):
        st.error(f"❌ Data file not found: {DATA_PATH}")
        st.stop()
        
    movies_df = load_movies_data(str(DATA_PATH))

    # 3. Initialize Recommender
    try:
        recommender = MovieRecommender(str(MODEL_PATH), movies_df)
        return recommender, movies_df
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        # If the file is corrupted, delete it so it downloads again next time
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.stop()

# Load the resources
recommender, movies_df = load_data_and_model()

# ==========================================
# 4. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .main-header { font-size: 3rem; font-weight: bold; text-align: center; color: #FF6B6B; margin-bottom: 2rem; }
    .sub-header { font-size: 1.5rem; font-weight: bold; color: #4ECDC4; margin-top: 2rem; margin-bottom: 1rem; }
    .movie-card { background-color: #262730; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; border-left: 5px solid #FF6B6B; }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. APP INTERFACE
# ==========================================
st.markdown('<div class="main-header">🎬 Movie Recommender System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Navigation")
    page = st.radio("Go to:", ["🏠 Home", "🎬 Recommendations", "🔍 Similar Movies"])
    
    st.markdown("---")
    st.info(f"""
    **Model Info:**
    Movies: {recommender.n_movies:,}
    Users: {len(recommender.user_biases):,}
    Dims: {recommender.k}
    """)

# --- HOME ---
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Welcome")
        st.write("This application uses Matrix Factorization (ALS) to generate personalized movie recommendations.")
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Movies Loaded", recommender.n_movies)
        st.markdown('</div>', unsafe_allow_html=True)

# --- RECOMMENDATIONS ---
elif page == "🎬 Recommendations":
    st.markdown('<div class="sub-header">Get Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        user_id = st.number_input("User ID", min_value=0, max_value=len(recommender.user_biases)-1)
        if st.button("Get Recs", type="primary"):
            recs = recommender.recommend_movies(user_id, n_recommendations=10)
            st.session_state.recs = recs
            
    with col2:
        if 'recs' in st.session_state:
            for rec in st.session_state.recs:
                st.markdown(f"""
                <div class="movie-card">
                    <b>{rec['title']}</b><br>
                    <small>{format_genres(rec['genres'])}</small><br>
                    Score: {rec['predicted_rating']:.2f}
                </div>
                """, unsafe_allow_html=True)

# --- SIMILAR MOVIES ---
elif page == "🔍 Similar Movies":
    st.markdown('<div class="sub-header">Find Similar Movies</div>', unsafe_allow_html=True)
    
    search = st.text_input("Search Movie Title")
    if search:
        matches = movies_df[movies_df['title'].str.contains(search, case=False, na=False)].head(10)
        if not matches.empty:
            movie_id = st.selectbox("Select Movie", matches['movieId'], format_func=lambda x: matches[matches['movieId']==x]['title'].iloc[0])
            
            if st.button("Find Similar"):
                similar = recommender.find_similar_movies(movie_id)
                for s in similar:
                    st.write(f"**{s['title']}** (Sim: {s['similarity']:.2f})")
