import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import gdown
import requests
import hashlib

# ==========================================
# 1. PAGE CONFIGURATION (MUST BE FIRST)
# ==========================================
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PATH & IMPORT SETUP
# ==========================================
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / 'src'
MODEL_PATH = BASE_DIR / 'best_model.npz'
DATA_PATH = BASE_DIR / 'data' / 'movies.csv'
LINKS_PATH = BASE_DIR / 'data' / 'links.csv'

# Add src to system path
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
    
    Please ensure your folder structure is correct:
```
    project/
    ├── app.py
    ├── src/
    │   ├── recommender.py
    │   └── utils.py
    └── data/
        └── movies.csv
```
    """)
    st.stop()

# ==========================================
# 3. CONFIGURATION & API KEYS
# ==========================================

# Google Drive File IDs (Replace with your actual file IDs)
MODEL_FILE_ID = '1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq'
MOVIES_FILE_ID = '1f7ImoZRL4C9x_ZzSG4qhTCunV2lVvpD8'
LINKS_FILE_ID = '1-bh3vGZ0DR_aOCNMZLLz0YX8KYy5xHxu'

# OMDb API Key (get free key at http://www.omdbapi.com/apikey.aspx)
try:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
except:
    OMDB_API_KEY = None
    st.warning("⚠️ No OMDb API key found. Movie posters will show placeholders.")

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

@st.cache_data(ttl=3600)
def get_poster_url(imdb_id):
    """
    Get movie poster from OMDb API using IMDb ID.
    Cached for 1 hour to reduce API calls.
    """
    if pd.isna(imdb_id) or not OMDB_API_KEY:
        return None
    
    try:
        # Format IMDb ID (should be like tt0111161)
        if isinstance(imdb_id, (int, float)):
            imdb_id = f"tt{int(imdb_id):07d}"
        elif not str(imdb_id).startswith('tt'):
            imdb_id = f"tt{int(float(str(imdb_id))):07d}"
        
        # Call OMDb API
        url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
        response = requests.get(url, timeout=3)
        
        if response.status_code == 200:
            data = response.json()
            poster = data.get('Poster')
            if poster and poster != 'N/A':
                return poster
    except:
        pass
    
    return None


def generate_poster_placeholder(title):
    """Generate a nice-looking poster placeholder with gradient."""
    title_hash = int(hashlib.md5(title.encode()).hexdigest()[:6], 16)
    color1 = f"#{title_hash % 0xFFFFFF:06x}"
    color2 = f"#{(title_hash * 2) % 0xFFFFFF:06x}"
    
    display_title = title[:25] + "..." if len(title) > 25 else title
    
    return f"""
    <div style="
        background: linear-gradient(135deg, {color1} 0%, {color2} 100%);
        padding: 2rem 0.5rem;
        border-radius: 8px;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 3.5rem; margin-bottom: 0.5rem; opacity: 0.9;">🎬</div>
        <div style="color: white; font-weight: bold; font-size: 0.85rem; padding: 0 0.5rem; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
            {display_title}
        </div>
    </div>
    """


# ==========================================
# 5. DATA LOADING & CACHING
# ==========================================

@st.cache_resource
def download_file_from_gdrive(file_id, output_path, file_description):
    """Download a file from Google Drive if it doesn't exist."""
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        if file_size > 0:
            return True
    
    os.makedirs(Path(output_path).parent, exist_ok=True)
    
    url = f'https://drive.google.com/uc?id={file_id}'
    try:
        with st.spinner(f"📥 Downloading {file_description} from Google Drive..."):
            gdown.download(url, str(output_path), quiet=False)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            st.error(f"❌ Failed to download {file_description}")
            return False
    except Exception as e:
        st.error(f"❌ Error downloading {file_description}: {e}")
        st.info("💡 Make sure the Google Drive link has 'Anyone with the link' sharing enabled")
        return False


@st.cache_resource
def load_data_and_model():
    """Downloads and loads all required data and model."""
    
    # Download model
    if not download_file_from_gdrive(MODEL_FILE_ID, MODEL_PATH, "model"):
        st.stop()
    
    # Download movies.csv
    if not download_file_from_gdrive(MOVIES_FILE_ID, DATA_PATH, "movies.csv"):
        st.stop()
    
    # Download links.csv (optional)
    download_file_from_gdrive(LINKS_FILE_ID, LINKS_PATH, "links.csv")
    
    # Load movies data
    try:
        movies_df = load_movies_data(str(DATA_PATH))
        
        if movies_df.empty:
            st.error("❌ Movies data file is empty!")
            st.stop()
        
        # Load and merge links data if available
        if os.path.exists(LINKS_PATH):
            try:
                links_df = pd.read_csv(LINKS_PATH)
                movies_df = movies_df.merge(links_df, on='movieId', how='left')
            except:
                pass
        
    except Exception as e:
        st.error(f"❌ Failed to load movies data: {e}")
        st.stop()
    
    # Initialize recommender
    try:
        recommender = MovieRecommender(str(MODEL_PATH), movies_df)
        return recommender, movies_df
    except Exception as e:
        st.error(f"❌ Error initializing recommender: {e}")
        import traceback
        with st.expander("Show full error traceback"):
            st.code(traceback.format_exc())
        st.stop()


# ==========================================
# 6. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    .main-header { 
        font-size: 2.8rem; 
        font-weight: bold; 
        text-align: center; 
        color: #FF6B6B; 
        margin-top: 0 !important;
        margin-bottom: 1.5rem;
        padding-top: 0 !important;
    }
    
    .sub-header { 
        font-size: 1.5rem; 
        font-weight: bold; 
        color: #4ECDC4; 
        margin-top: 1.5rem; 
        margin-bottom: 1rem; 
    }
    
    .movie-card { 
        background-color: #1e1e1e; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin-bottom: 1rem; 
        border-left: 5px solid #FF6B6B;
        transition: transform 0.2s;
    }
    
    .movie-card:hover {
        transform: translateX(5px);
    }
    
    .metric-card { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        padding: 1rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
    }
    
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #ff5252;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(255,107,107,0.3);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Improve divider */
    hr {
        margin: 1rem 0;
        border: none;
        border-top: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 7. LOAD DATA
# ==========================================

with st.spinner("🎬 Loading recommender system..."):
    recommender, movies_df = load_data_and_model()

# ==========================================
# 8. APP HEADER
# ==========================================

st.markdown('<div class="main-header">🎬 Movie Recommender System</div>', unsafe_allow_html=True)

# ==========================================
# 9. SIDEBAR
# ==========================================

with st.sidebar:
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **Movies:** {recommender.n_movies:,}  
    **Dimensions:** {recommender.k}  
    **Type:** Matrix Factorization (ALS)
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 How It Works")
    st.caption("""
    1. Rate some movies you've seen
    2. Our AI learns your preferences
    3. Get personalized recommendations!
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Tips")
    st.caption("""
    - Rate at least 5 movies for better results
    - Mix different genres
    - Be honest with your ratings!
    """)
    
    if not OMDB_API_KEY:
        st.markdown("---")
        st.warning("⚠️ Posters disabled. Add OMDb API key in Streamlit secrets.")

# ==========================================
# 10. MAIN CONTENT - RECOMMENDATIONS
# ==========================================

st.markdown('<div class="sub-header">Get Personalized Recommendations</div>', unsafe_allow_html=True)

# Initialize session state
if 'rated_movies' not in st.session_state:
    st.session_state.rated_movies = []

col1, col2 = st.columns([1, 2])

# LEFT COLUMN - Rate Movies
with col1:
    st.markdown("#### 🎬 Rate Some Movies")
    st.caption("Search and rate movies to get personalized recommendations")
    
    # Movie search
    search_movie = st.text_input(
        "Search for a movie", 
        placeholder="e.g., The Matrix, Harry Potter...",
        label_visibility="collapsed"
    )
    
    if search_movie:
        matches = movies_df[
            movies_df['title'].str.contains(search_movie, case=False, na=False)
        ].head(15)
        
        if not matches.empty:
            movie_to_rate = st.selectbox(
                "Select Movie",
                options=matches['movieId'].tolist(),
                format_func=lambda x: matches[matches['movieId']==x]['title'].iloc[0],
                label_visibility="collapsed"
            )
            
            col_rating, col_add = st.columns([2, 1])
            
            with col_rating:
                rating = st.slider("Rating", 0.5, 5.0, 3.0, 0.5, label_visibility="collapsed")
            
            with col_add:
                if st.button("➕ Add", use_container_width=True):
                    movie_title = matches[matches['movieId']==movie_to_rate]['title'].iloc[0]
                    if movie_to_rate not in [m[0] for m in st.session_state.rated_movies]:
                        st.session_state.rated_movies.append((movie_to_rate, rating, movie_title))
                        st.success("✅ Added!")
                        st.rerun()
                    else:
                        st.warning("Already rated!")
        else:
            st.info("No movies found. Try a different search.")
    
    st.markdown("---")
    
    # Display rated movies
    st.markdown("#### 📝 Your Ratings")
    
    if st.session_state.rated_movies:
        for i, (movie_id, rating, title) in enumerate(st.session_state.rated_movies):
            col_a, col_b, col_c = st.columns([4, 2, 1])
            with col_a:
                st.markdown(f"**{title[:35]}{'...' if len(title) > 35 else ''}**")
            with col_b:
                st.caption(f"⭐ {rating:.1f}/5.0")
            with col_c:
                if st.button("🗑️", key=f"del_{i}", help="Remove"):
                    st.session_state.rated_movies.pop(i)
                    st.rerun()
        
        st.markdown("---")
        
        # Settings
        n_recs = st.slider("Number of Recommendations", 5, 30, 10, 5)
        iterations = st.slider("Training Quality", 5, 30, 15, 5, 
                              help="More iterations = better quality but slower")
        
        # Get recommendations button
        if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("🔮 Training AI on your preferences..."):
                try:
                    user_ratings = [(m_id, rating) for m_id, rating, _ in st.session_state.rated_movies]
                    
                    recs = recommender.recommend_from_ratings(
                        user_ratings, 
                        n_recommendations=n_recs,
                        iterations=iterations
                    )
                    
                    st.session_state.recs = recs
                    st.success(f"✅ Found {len(recs)} perfect matches!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    with st.expander("Show error details"):
                        import traceback
                        st.code(traceback.format_exc())
        
        # Clear button
        if st.button("🔄 Clear All Ratings", use_container_width=True):
            st.session_state.rated_movies = []
            st.session_state.pop('recs', None)
            st.rerun()
    else:
        st.info("👆 Search and rate some movies to get started!")
        
        with st.expander("💡 Need ideas? Try these popular movies"):
            suggestions = movies_df.head(10)['title'].tolist()
            for movie in suggestions:
                st.caption(f"• {movie}")

# RIGHT COLUMN - Show Recommendations
with col2:
    st.markdown("#### 🎥 Your Personalized Recommendations")
    
    if 'recs' in st.session_state and st.session_state.recs:
        st.caption(f"✨ Based on your {len(st.session_state.rated_movies)} ratings")
        
        for idx, rec in enumerate(st.session_state.recs, 1):
            movie_info = movies_df[movies_df['movieId'] == rec['movieId']]
            
            col_poster, col_info = st.columns([1, 3])
            
            with col_poster:
                poster_displayed = False
                
                if not movie_info.empty and 'imdbId' in movie_info.columns:
                    imdb_id = movie_info.iloc[0]['imdbId']
                    poster_url = get_poster_url(imdb_id)
                    
                    if poster_url:
                        try:
                            st.image(poster_url, use_container_width=True)
                            poster_displayed = True
                        except:
                            pass
                
                if not poster_displayed:
                    st.markdown(
                        generate_poster_placeholder(rec['title']),
                        unsafe_allow_html=True
                    )
            
            with col_info:
                st.markdown(f"""
                <div style="margin-bottom: 0.5rem;">
                    <span style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 0.25rem 0.75rem;
                        border-radius: 20px;
                        font-weight: bold;
                        font-size: 0.9rem;
                        margin-right: 0.5rem;
                    ">#{idx}</span>
                    <span style="color: #FF6B6B; font-weight: bold; font-size: 1.2rem;">
                        {rec['title']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <p style="margin: 0.5rem 0; color: #999; font-size: 0.9rem;">
                    {format_genres(rec['genres'])}
                </p>
                """, unsafe_allow_html=True)
                
                score_color = "#4ECDC4" if rec['score'] > 0.5 else "#FFA500"
                st.markdown(f"""
                <p style="margin: 0.5rem 0;">
                    <span style="color: {score_color}; font-weight: bold;">
                        📊 Match Score: {rec['score']:.2%}
                    </span>
                </p>
                """, unsafe_allow_html=True)
                
                if st.button("➕ Add to Watchlist", key=f"watch_{idx}", use_container_width=True):
                    st.success(f"Added to watchlist!")
            
            st.divider()
    else:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem 1rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 15px;
            margin-top: 2rem;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
            <h3 style="color: #4ECDC4; margin-bottom: 0.5rem;">No Recommendations Yet</h3>
            <p style="color: #999; margin-bottom: 1.5rem;">
                Rate some movies on the left to get personalized recommendations!
            </p>
            <div style="color: #666; font-size: 0.9rem;">
                <p>💡 <strong>Quick Start:</strong></p>
                <p>1. Search for movies you've watched</p>
                <p>2. Rate them honestly (0.5 - 5.0 stars)</p>
                <p>3. Click "Get Recommendations"</p>
                <p>4. Discover your next favorite movie! 🍿</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 11. FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p style="margin: 0.5rem 0;">Built with ❤️ using Streamlit</p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        Powered by Matrix Factorization (ALS) • 
        <a href="https://github.com/yourusername/movie-recommender" target="_blank" style="color: #4ECDC4; text-decoration: none;">View on GitHub</a>
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.8rem; color: #888;">
        Movie data from <a href="https://movielens.org" target="_blank" style="color: #888;">MovieLens</a> • 
        Posters from <a href="http://www.omdbapi.com" target="_blank" style="color: #888;">OMDb API</a>
    </p>
</div>
""", unsafe_allow_html=True)
