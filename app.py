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

# Fixed parameters
N_RECOMMENDATIONS = 10
N_ITERATIONS = 20

# OMDb API Key
try:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
except:
    OMDB_API_KEY = None

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

@st.cache_data(ttl=3600)
def get_poster_url(imdb_id):
    """Get movie poster from OMDb API using IMDb ID."""
    if pd.isna(imdb_id) or not OMDB_API_KEY:
        return None
    
    try:
        if isinstance(imdb_id, (int, float)):
            imdb_id = f"tt{int(imdb_id):07d}"
        elif not str(imdb_id).startswith('tt'):
            imdb_id = f"tt{int(float(str(imdb_id))):07d}"
        
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
        border-radius: 10px;
        text-align: center;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
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
        with st.spinner(f"📥 Downloading {file_description}..."):
            gdown.download(url, str(output_path), quiet=False)
        
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
        else:
            st.error(f"❌ Failed to download {file_description}")
            return False
    except Exception as e:
        st.error(f"❌ Error downloading {file_description}: {e}")
        return False


@st.cache_resource
def load_data_and_model():
    """Downloads and loads all required data and model."""
    
    if not download_file_from_gdrive(MODEL_FILE_ID, MODEL_PATH, "model"):
        st.stop()
    
    if not download_file_from_gdrive(MOVIES_FILE_ID, DATA_PATH, "movies.csv"):
        st.stop()
    
    download_file_from_gdrive(LINKS_FILE_ID, LINKS_PATH, "links.csv")
    
    try:
        movies_df = load_movies_data(str(DATA_PATH))
        
        if movies_df.empty:
            st.error("❌ Movies data file is empty!")
            st.stop()
        
        if os.path.exists(LINKS_PATH):
            try:
                links_df = pd.read_csv(LINKS_PATH)
                movies_df = movies_df.merge(links_df, on='movieId', how='left')
            except:
                pass
        
    except Exception as e:
        st.error(f"❌ Failed to load movies data: {e}")
        st.stop()
    
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
# 6. CUSTOM CSS - EYE-FRIENDLY LIGHT BLUE THEME
# ==========================================

st.markdown("""
<style>
    /* Remove extra top space */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
        max-width: 1200px;
    }
    
    /* Hide default Streamlit header space */
    header {
        background-color: transparent !important;
    }
    
    .main > div {
        padding-top: 1rem !important;
    }
    
    /* Main header - Light blue theme */
    .main-header { 
        font-size: 2.5rem; 
        font-weight: 700; 
        text-align: center; 
        color: #2E86AB;
        background: linear-gradient(135deg, #E8F4F8 0%, #D4E9F7 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(46, 134, 171, 0.1);
    }
    
    .sub-header { 
        font-size: 1.3rem; 
        font-weight: 600; 
        color: #2E86AB; 
        margin-top: 1rem; 
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #A9D6E5;
    }
    
    /* Cards - Light blue theme */
    .movie-card { 
        background: linear-gradient(135deg, #F7FBFC 0%, #EAF4F9 100%);
        padding: 1.2rem; 
        border-radius: 12px; 
        margin-bottom: 0.8rem; 
        border-left: 4px solid #61A5C2;
        box-shadow: 0 2px 6px rgba(46, 134, 171, 0.08);
        transition: all 0.3s ease;
    }
    
    .movie-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.15);
    }
    
    /* Buttons - Light blue theme */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #2E86AB 0%, #61A5C2 100%);
        color: white;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.65rem 1.2rem;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(46, 134, 171, 0.2);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #235F7A 0%, #2E86AB 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #E8F4F8;
        border-left: 4px solid #2E86AB;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F7FBFC 0%, #EAF4F9 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    hr {
        margin: 1rem 0;
        border: none;
        border-top: 1px solid #A9D6E5;
    }
    
    /* Rating display */
    .rating-display {
        background: linear-gradient(135deg, #E8F4F8 0%, #D4E9F7 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        border-left: 3px solid #61A5C2;
        margin: 0.3rem 0;
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
    st.markdown("### 📊 System Info")
    st.info(f"""
    **Movies:** {recommender.n_movies:,}  
    **Dimensions:** {recommender.k}  
    **Algorithm:** ALS Matrix Factorization
    """)
    
    st.markdown("---")
    
    st.markdown("### 🎯 How It Works")
    st.caption("""
    1. **Rate movies** you've watched
    2. **AI analyzes** your preferences  
    3. **Get 10 personalized** recommendations
    4. **Discover** your next favorite!
    """)
    
    st.markdown("---")
    
    st.markdown("### 💡 Quick Tips")
    st.caption("""
    • Rate at least **5 movies** for best results  
    • Mix **different genres**  
    • Be **honest** with ratings  
    • More ratings = Better accuracy
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    st.caption(f"""
    **Iterations:** {N_ITERATIONS} (Auto)  
    **Recommendations:** {N_RECOMMENDATIONS} (Auto)
    """)

# ==========================================
# 10. MAIN CONTENT
# ==========================================

st.markdown('<div class="sub-header">📝 Rate Movies & Get Recommendations</div>', unsafe_allow_html=True)

# Initialize session state
if 'rated_movies' not in st.session_state:
    st.session_state.rated_movies = []

col1, col2 = st.columns([1, 2])

# LEFT COLUMN - Rate Movies
with col1:
    st.markdown("#### 🔍 Search & Rate Movies")
    
    # Movie search
    search_movie = st.text_input(
        "Search", 
        placeholder="Type a movie name (e.g., Inception, Avatar)...",
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
            
            rating = st.slider(
                "Your Rating", 
                0.5, 5.0, 3.0, 0.5,
                help="How much did you like this movie?"
            )
            
            if st.button("⭐ Add Rating", use_container_width=True, type="primary"):
                movie_title = matches[matches['movieId']==movie_to_rate]['title'].iloc[0]
                if movie_to_rate not in [m[0] for m in st.session_state.rated_movies]:
                    st.session_state.rated_movies.append((movie_to_rate, rating, movie_title))
                    st.success(f"✅ Added: {movie_title}")
                    st.rerun()
                else:
                    st.warning("⚠️ You already rated this movie!")
        else:
            st.info("🔍 No movies found. Try a different search term.")
    
    # Get recommendations button (right after rating section)
    st.markdown("---")
    
    if len(st.session_state.rated_movies) >= 3:
        st.markdown("#### 🎬 Ready for Recommendations?")
        st.caption(f"You've rated {len(st.session_state.rated_movies)} movies")
        
        if st.button(
            f"🚀 Get {N_RECOMMENDATIONS} Recommendations", 
            type="primary", 
            use_container_width=True,
            help=f"AI will analyze your ratings using {N_ITERATIONS} iterations"
        ):
            with st.spinner(f"🔮 AI is analyzing your {len(st.session_state.rated_movies)} ratings..."):
                try:
                    user_ratings = [(m_id, rating) for m_id, rating, _ in st.session_state.rated_movies]
                    
                    recs = recommender.recommend_from_ratings(
                        user_ratings, 
                        n_recommendations=N_RECOMMENDATIONS,
                        iterations=N_ITERATIONS
                    )
                    
                    st.session_state.recs = recs
                    st.success(f"✅ Found {len(recs)} perfect matches!")
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    with st.expander("Show error details"):
                        import traceback
                        st.code(traceback.format_exc())
    else:
        st.info(f"📌 Rate at least 3 movies to get recommendations (currently: {len(st.session_state.rated_movies)})")
    
    # My Ratings section (below the button)
    st.markdown("---")
    st.markdown("#### 📋 Your Rated Movies")
    
    if st.session_state.rated_movies:
        for i, (movie_id, rating, title) in enumerate(st.session_state.rated_movies):
            st.markdown(f"""
            <div class="rating-display">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <strong style="color: #2E86AB;">{title[:30]}{'...' if len(title) > 30 else ''}</strong>
                    </div>
                    <div style="color: #61A5C2; font-weight: 600; margin-left: 10px;">
                        ⭐ {rating:.1f}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🗑️ Remove", key=f"del_{i}", use_container_width=True):
                st.session_state.rated_movies.pop(i)
                st.rerun()
        
        st.markdown("---")
        
        if st.button("🔄 Clear All Ratings", use_container_width=True):
            st.session_state.rated_movies = []
            st.session_state.pop('recs', None)
            st.rerun()
    else:
        st.info("👆 No ratings yet. Search and rate movies above!")
        
        with st.expander("💡 Popular movies to try"):
            suggestions = movies_df.head(8)['title'].tolist()
            for movie in suggestions:
                st.caption(f"🎬 {movie}")

# RIGHT COLUMN - Show Recommendations
with col2:
    st.markdown("#### 🎥 Your Personalized Recommendations")
    
    if 'recs' in st.session_state and st.session_state.recs:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #E8F4F8 0%, #D4E9F7 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            text-align: center;
            border: 2px solid #61A5C2;
        ">
            <strong style="color: #2E86AB;">✨ Based on your {len(st.session_state.rated_movies)} ratings</strong>
        </div>
        """, unsafe_allow_html=True)
        
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
                        background: linear-gradient(135deg, #2E86AB 0%, #61A5C2 100%);
                        color: white;
                        padding: 0.3rem 0.8rem;
                        border-radius: 20px;
                        font-weight: bold;
                        font-size: 0.9rem;
                        margin-right: 0.5rem;
                    ">#{idx}</span>
                    <span style="color: #2E86AB; font-weight: bold; font-size: 1.15rem;">
                        {rec['title']}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <p style="margin: 0.5rem 0; color: #61A5C2; font-size: 0.9rem;">
                    {format_genres(rec['genres'])}
                </p>
                """, unsafe_allow_html=True)
                
                score_pct = rec['score'] * 100
                score_color = "#2E86AB" if score_pct > 70 else "#61A5C2"
                st.markdown(f"""
                <p style="margin: 0.5rem 0;">
                    <span style="color: {score_color}; font-weight: 600;">
                        📊 Match Score: {score_pct:.1f}%
                    </span>
                </p>
                """, unsafe_allow_html=True)
            
            st.divider()
    else:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 3rem 1.5rem;
            background: linear-gradient(135deg, #F7FBFC 0%, #EAF4F9 100%);
            border-radius: 15px;
            margin-top: 2rem;
            border: 2px dashed #A9D6E5;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🎬</div>
            <h3 style="color: #2E86AB; margin-bottom: 0.5rem;">No Recommendations Yet</h3>
            <p style="color: #61A5C2; margin-bottom: 1.5rem;">
                Rate some movies to discover your perfect match!
            </p>
            <div style="color: #61A5C2; font-size: 0.95rem; line-height: 1.8;">
                <p><strong style="color: #2E86AB;">🎯 Quick Start Guide:</strong></p>
                <p>1️⃣ Search for movies you've watched</p>
                <p>2️⃣ Rate them honestly (0.5 - 5.0 ⭐)</p>
                <p>3️⃣ Click "Get Recommendations"</p>
                <p>4️⃣ Discover amazing movies! 🍿</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 11. FOOTER
# ==========================================

st.markdown("---")
st.markdown("""
<div style="
    text-align: center; 
    color: #61A5C2; 
    padding: 1.5rem;
    background: linear-gradient(135deg, #F7FBFC 0%, #EAF4F9 100%);
    border-radius: 10px;
    margin-top: 2rem;
">
    <p style="margin: 0.5rem 0; font-weight: 600; color: #2E86AB;">
        Built with ❤️ using Streamlit & Machine Learning
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem;">
        Powered by ALS Matrix Factorization • 
        <a href="https://github.com/yourusername/movie-recommender" target="_blank" 
           style="color: #2E86AB; text-decoration: none; font-weight: 600;">
            View on GitHub ↗
        </a>
    </p>
    <p style="margin: 0.5rem 0; font-size: 0.85rem; color: #89B0C2;">
        Data: <a href="https://movielens.org" target="_blank" style="color: #61A5C2; text-decoration: none;">MovieLens</a> • 
        Posters: <a href="http://www.omdbapi.com" target="_blank" style="color: #61A5C2; text-decoration: none;">OMDb API</a>
    </p>
</div>
""", unsafe_allow_html=True)
