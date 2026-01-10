import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import gdown
import requests

# ==========================================
# 1. PATH & IMPORT SETUP
# ==========================================
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / 'src'
MODEL_PATH = BASE_DIR / 'best_model.npz'
DATA_PATH = BASE_DIR / 'data' / 'movies.csv'
LINKS_PATH = BASE_DIR / 'data' / 'links.csv'

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from recommender import MovieRecommender
    from utils import load_movies_data, format_genres
except ImportError as e:
    st.error(f"❌ Import Error: {e}")
    st.stop()

# ==========================================
# 2. CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_FILE_ID = '1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq'
MOVIES_FILE_ID = '1f7ImoZRL4C9x_ZzSG4qhTCunV2lVvpD8'
LINKS_FILE_ID = '1uVWSBWCtCe7YrekhshK_CjK_v_w-Bdkj'

# ==========================================
# 3. DATA LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_data_and_model():
    # Download model
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH.parent, exist_ok=True)
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        try:
            gdown.download(url, str(MODEL_PATH), quiet=False)
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()
    
    # Download movies
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH.parent, exist_ok=True)
        url = f'https://drive.google.com/uc?id={MOVIES_FILE_ID}'
        try:
            gdown.download(url, str(DATA_PATH), quiet=False)
        except Exception as e:
            st.error(f"❌ Failed to download movies: {e}")
            st.stop()
    
    # Download links
    if not os.path.exists(LINKS_PATH):
        os.makedirs(LINKS_PATH.parent, exist_ok=True)
        url = f'https://drive.google.com/uc?id={LINKS_FILE_ID}'
        try:
            gdown.download(url, str(LINKS_PATH), quiet=False)
        except:
            pass
    
    # Load data
    movies_df = load_movies_data(str(DATA_PATH))
    
    if os.path.exists(LINKS_PATH):
        try:
            links_df = pd.read_csv(LINKS_PATH)
            movies_df = movies_df.merge(links_df, on='movieId', how='left')
        except Exception as e:
            pass
    
    recommender = MovieRecommender(str(MODEL_PATH), movies_df)
    return recommender, movies_df

recommender, movies_df = load_data_and_model()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def get_poster_url(movie_id):
    """Get movie poster URL from TMDb with extensive debugging"""
    try:
        movie_info = movies_df[movies_df['movieId'] == movie_id]
        
        if movie_info.empty:
            return None
        
        # Check if tmdbId column exists
        if 'tmdbId' not in movie_info.columns:
            return None
        
        tmdb_id = movie_info.iloc[0]['tmdbId']
        
        # Check if tmdbId is valid
        if pd.isna(tmdb_id) or tmdb_id == '' or tmdb_id == 0:
            return None
        
        # Convert to integer
        tmdb_id = int(float(tmdb_id))
        
        # TMDb image base URLs - try multiple sizes
        base_urls = [
            f"https://image.tmdb.org/t/p/w342/{tmdb_id}.jpg",
            f"https://image.tmdb.org/t/p/w500/{tmdb_id}.jpg",
            f"https://image.tmdb.org/t/p/original/{tmdb_id}.jpg"
        ]
        
        # Return the first URL (we'll let Streamlit handle the loading)
        return base_urls[0]
        
    except Exception as e:
        return None

# ==========================================
# 5. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
    }
    .sub-text {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 3rem;
    }
    
    /* Movie card styles */
    .movie-card {
        background: linear-gradient(145deg, #2a2d3a 0%, #1e2029 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .movie-card:hover {
        border: 2px solid #667eea;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    /* Rating badge */
    .rating-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Poster styles */
    .poster-placeholder {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 3rem 1rem;
        text-align: center;
        font-size: 4rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        min-height: 200px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .poster-image {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        width: 100%;
    }
    
    /* Rated movie item */
    .rated-item {
        background: #2a2d3a;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.8rem;
        border-left: 4px solid #667eea;
    }
    
    /* Button styles */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Info box */
    .info-box {
        background: rgba(102, 126, 234, 0.1);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 6. APP INTERFACE
# ==========================================
st.markdown('<div class="main-header">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Discover your next favorite movie with AI-powered recommendations</div>', unsafe_allow_html=True)

# Initialize session state
if 'rated_movies' not in st.session_state:
    st.session_state.rated_movies = []
if 'recs' not in st.session_state:
    st.session_state.recs = []

# Function to generate recommendations
def generate_recommendations():
    if len(st.session_state.rated_movies) > 0:
        user_ratings = [(m_id, rating) for m_id, rating, _ in st.session_state.rated_movies]
        st.session_state.recs = recommender.recommend_from_ratings(
            user_ratings, 
            n_recommendations=10,
            iterations=20
        )

# Main layout
col_left, col_right = st.columns([2, 3], gap="large")

with col_left:
    st.markdown("### 🎬 Rate Movies")
    
    # Single search box that filters and shows dropdown
    st.markdown("##### Type to search movies")
    
    # Search input
    search_query = st.text_input(
        "Search for a movie", 
        "", 
        placeholder="Start typing a movie name...",
        key="search_input"
    )
    
    # Show dropdown only if user is searching
    if search_query and len(search_query) >= 2:
        filtered_movies = movies_df[
            movies_df['title'].str.contains(search_query, case=False, na=False)
        ].head(20)
        
        if not filtered_movies.empty:
            # Dropdown appears with filtered results
            selected_movie = st.selectbox(
                "Select from results",
                options=filtered_movies['movieId'].tolist(),
                format_func=lambda x: filtered_movies[filtered_movies['movieId']==x]['title'].iloc[0],
                key="movie_select"
            )
            
            # Rating slider
            rating = st.slider("⭐ Your Rating", 0.5, 5.0, 4.0, 0.5, key="rating_slider")
            
            # Add rating button
            if st.button("➕ Add Rating", use_container_width=True, type="primary"):
                movie_title = filtered_movies[filtered_movies['movieId']==selected_movie]['title'].iloc[0]
                if selected_movie not in [m[0] for m in st.session_state.rated_movies]:
                    st.session_state.rated_movies.append((selected_movie, rating, movie_title))
                    st.success(f"✅ Added: {movie_title}")
                    st.rerun()
                else:
                    st.warning("⚠️ Already rated!")
        else:
            st.info("No movies found. Try a different search.")
    elif search_query and len(search_query) < 2:
        st.info("Type at least 2 characters to search...")
    
    # Recommend button (shows when there are rated movies)
    if st.session_state.rated_movies:
        st.markdown("---")
        if st.button("🎬 Get Recommendations", use_container_width=True, type="primary", key="recommend_btn"):
            with st.spinner("🎬 Generating recommendations..."):
                generate_recommendations()
            st.success("✨ Recommendations ready!")
            st.rerun()
    
    # Show rated movies below
    if st.session_state.rated_movies:
        st.markdown("---")
        st.markdown("### 📝 Your Ratings")
        
        for i, (movie_id, rating, title) in enumerate(st.session_state.rated_movies):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"""
                <div class="rated-item">
                    <strong>{title[:50]}{'...' if len(title) > 50 else ''}</strong><br>
                    <span class="rating-badge">⭐ {rating}</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("🗑️", key=f"del_{i}", use_container_width=True):
                    st.session_state.rated_movies.pop(i)
                    st.rerun()
        
        if st.button("🔄 Clear All Ratings", use_container_width=True):
            st.session_state.rated_movies = []
            st.session_state.recs = []
            st.rerun()
    else:
        st.markdown("""
        <div class="info-box">
            <strong>👋 Get Started!</strong><br>
            Search and rate movies you've seen to get personalized recommendations.
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("### 🎥 Recommended for You")
    
    if st.session_state.recs:
        st.caption(f"✨ Based on {len(st.session_state.rated_movies)} rated movie(s)")
        st.markdown("<br>", unsafe_allow_html=True)
        
        for idx, rec in enumerate(st.session_state.recs, 1):
            col_poster, col_info = st.columns([1, 3])
            
            with col_poster:
                poster_url = get_poster_url(rec['movieId'])
                
                if poster_url:
                    # Display image with error handling
                    st.markdown(f'<img src="{poster_url}" class="poster-image" onerror="this.style.display=\'none\'">', unsafe_allow_html=True)
                    # Fallback if image fails to load
                    st.markdown('<div class="poster-placeholder" style="display:none;" id="fallback-{idx}">🎬</div>'.format(idx=idx), unsafe_allow_html=True)
                else:
                    st.markdown('<div class="poster-placeholder">🎬</div>', unsafe_allow_html=True)
            
            with col_info:
                st.markdown(f"""
                <div class="movie-card">
                    <h3 style="margin:0; color:#667eea; font-size:1.3rem;">#{idx} {rec['title']}</h3>
                    <p style="margin:0.5rem 0; color:#aaa; font-size:0.9rem;">
                        {format_genres(rec['genres'])}
                    </p>
                    <span class="rating-badge">📊 Score: {rec['score']:.3f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            if idx < len(st.session_state.recs):
                st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
            <strong>🎯 No recommendations yet</strong><br>
            Rate some movies and click "Get Recommendations" to see personalized suggestions here.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="margin:0;">Built with ❤️ using <strong>Streamlit</strong> • Powered by <strong>Matrix Factorization (ALS)</strong></p>
    <p style="margin:0.5rem 0 0 0; font-size:0.9rem;">
        {movies:,} movies • {dims} latent dimensions
    </p>
</div>
""".format(movies=recommender.n_movies, dims=recommender.k), unsafe_allow_html=True)
