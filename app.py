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
MODEL_PATH = BASE_DIR / 'best_model.npz'

# For GitHub raw content, use this format:
# DATA_PATH = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO_NAME/main/data/movies.csv"
# Or for local file:
DATA_PATH = BASE_DIR / 'data' / 'movies.csv'

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
    - data/
        - movies.csv
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

# Google Drive File IDs
# Get the ID from the share link: https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing
MODEL_FILE_ID = '1L_pXf730fiJsHVyoyHOaDBMFVE2vQ_Dq'  # Your model file
MOVIES_FILE_ID = '1sRGLCqUlZHHIauj8dJK46nfBtGtJq12v?usp=drive_link'  # Replace with your movies.csv file ID 

# ==========================================
# 3. DATA LOADING & CACHING
# ==========================================
@st.cache_resource
def load_data_and_model():
    """
    Downloads model and movies.csv from Google Drive if missing, then loads them.
    Cached so it only runs once.
    """
    # 1. Download Model if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={MODEL_FILE_ID}'
        try:
            with st.spinner("📥 Downloading model from Google Drive (100MB+)..."):
                gdown.download(url, str(MODEL_PATH), quiet=False)
            st.success("✅ Model download complete!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.info("💡 Make sure the Google Drive link has 'Anyone with the link' sharing enabled")
            st.stop()

    # 2. Download Movies CSV if it doesn't exist
    if not os.path.exists(DATA_PATH):
        # Create data directory if it doesn't exist
        os.makedirs(DATA_PATH.parent, exist_ok=True)
        
        url = f'https://drive.google.com/uc?id={MOVIES_FILE_ID}'
        try:
            with st.spinner("📥 Downloading movies.csv from Google Drive..."):
                gdown.download(url, str(DATA_PATH), quiet=False)
            st.success("✅ Movies data download complete!")
        except Exception as e:
            st.error(f"❌ Failed to download movies.csv: {e}")
            st.info("💡 Make sure the Google Drive link has 'Anyone with the link' sharing enabled")
            st.stop()

    # 3. Load Movies Data
    try:
        # Check file size first
        file_size = os.path.getsize(DATA_PATH)
        if file_size == 0:
            st.error(f"❌ Movies data file is empty: {DATA_PATH}")
            # Delete empty file so it re-downloads next time
            os.remove(DATA_PATH)
            st.info("Please restart the app to re-download the file")
            st.stop()
        
        movies_df = load_movies_data(str(DATA_PATH))
        
        if movies_df.empty or len(movies_df.columns) == 0:
            st.error("❌ Movies data file is empty or invalid!")
            os.remove(DATA_PATH)
            st.info("Please restart the app to re-download the file")
            st.stop()
            
    except pd.errors.EmptyDataError:
        st.error(f"❌ Movies data file is empty: {DATA_PATH}")
        if os.path.exists(DATA_PATH):
            os.remove(DATA_PATH)
        st.info("Please restart the app to re-download the file")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load movies data: {e}")
        st.stop()

    # 4. Initialize Recommender
    try:
        recommender = MovieRecommender(str(MODEL_PATH), movies_df)
        return recommender, movies_df
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("💡 Tip: If the model is corrupted, delete 'best_model.npz' and restart the app.")
        # If the file is corrupted, delete it so it downloads again next time
        if os.path.exists(MODEL_PATH):
            try:
                os.remove(MODEL_PATH)
                st.info("🗑️ Corrupted model file deleted. Please restart the app.")
            except:
                pass
        st.stop()

# Load the resources
recommender, movies_df = load_data_and_model()

# ==========================================
# 4. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .main-header { 
        font-size: 3rem; 
        font-weight: bold; 
        text-align: center; 
        color: #FF6B6B; 
        margin-bottom: 2rem; 
    }
    .sub-header { 
        font-size: 1.5rem; 
        font-weight: bold; 
        color: #4ECDC4; 
        margin-top: 2rem; 
        margin-bottom: 1rem; 
    }
    .movie-card { 
        background-color: #262730; 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin-bottom: 1rem; 
        border-left: 5px solid #FF6B6B; 
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
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff5252;
    }
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
    st.markdown("### 📊 Model Info")
    st.info(f"""
    **Loaded Successfully!**
    
    📽️ Movies: {recommender.n_movies:,}  
    👥 Users: {len(recommender.user_biases):,}  
    📐 Dimensions: {recommender.k}
    """)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("This app uses Matrix Factorization (ALS) for personalized movie recommendations.")

# ==========================================
# 6. PAGE CONTENT
# ==========================================

# --- HOME PAGE ---
if page == "🏠 Home":
    st.markdown('<div class="sub-header">Welcome to Movie Recommender</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📽️ Total Movies", f"{recommender.n_movies:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("👥 Total Users", f"{len(recommender.user_biases):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("📐 Model Dims", recommender.k)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.write("""
        - **Personalized Recommendations**: Get movie suggestions tailored to user preferences
        - **Similar Movies**: Find movies similar to ones you like
        - **Advanced ML**: Powered by Matrix Factorization (ALS) algorithm
        """)
    
    with col2:
        st.markdown("### 🚀 How to Use")
        st.write("""
        1. Navigate using the sidebar menu
        2. Enter a User ID for personalized recommendations
        3. Or search for a movie to find similar titles
        4. Enjoy discovering new movies!
        """)

# --- RECOMMENDATIONS PAGE ---
elif page == "🎬 Recommendations":
    st.markdown('<div class="sub-header">Get Personalized Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### ⚙️ Settings")
        user_id = st.number_input(
            "Enter User ID", 
            min_value=0, 
            max_value=len(recommender.user_biases)-1,
            value=0,
            help="Enter a valid user ID to get recommendations"
        )
        
        n_recs = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10)
        
        if st.button("🎬 Get Recommendations", type="primary", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                try:
                    recs = recommender.recommend_movies(user_id, n_recommendations=n_recs)
                    st.session_state.recs = recs
                    st.session_state.user_id = user_id
                    st.success(f"✅ Found {len(recs)} recommendations!")
                except Exception as e:
                    st.error(f"❌ Error generating recommendations: {e}")
    
    with col2:
        st.markdown("#### 🎥 Your Recommendations")
        if 'recs' in st.session_state and st.session_state.recs:
            st.caption(f"Recommendations for User ID: {st.session_state.get('user_id', user_id)}")
            
            for idx, rec in enumerate(st.session_state.recs, 1):
                st.markdown(f"""
                <div class="movie-card">
                    <h4 style="margin:0; color:#FF6B6B;">#{idx} {rec['title']}</h4>
                    <p style="margin:0.5rem 0; color:#999;">
                        <b>Genres:</b> {format_genres(rec['genres'])}
                    </p>
                    <p style="margin:0; color:#4ECDC4;">
                        ⭐ <b>Predicted Rating:</b> {rec['predicted_rating']:.2f} / 5.0
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 Enter a User ID and click 'Get Recommendations' to see results")

# --- SIMILAR MOVIES PAGE ---
elif page == "🔍 Similar Movies":
    st.markdown('<div class="sub-header">Find Similar Movies</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### 🔎 Search")
        search = st.text_input("Search Movie Title", placeholder="e.g., Toy Story, Matrix...")
        
        if search:
            matches = movies_df[movies_df['title'].str.contains(search, case=False, na=False)].head(20)
            
            if not matches.empty:
                st.markdown(f"**Found {len(matches)} matches**")
                
                # Create a better display for selection
                movie_options = {
                    f"{row['title']} ({row['movieId']})": row['movieId'] 
                    for _, row in matches.iterrows()
                }
                
                selected_title = st.selectbox(
                    "Select a Movie", 
                    options=list(movie_options.keys()),
                    help="Choose the movie to find similar titles"
                )
                
                movie_id = movie_options[selected_title]
                
                n_similar = st.slider("Number of Similar Movies", min_value=5, max_value=20, value=10)
                
                if st.button("🔍 Find Similar Movies", type="primary", use_container_width=True):
                    with st.spinner("Finding similar movies..."):
                        try:
                            similar = recommender.find_similar_movies(movie_id, n_similar=n_similar)
                            st.session_state.similar = similar
                            st.session_state.selected_movie = selected_title
                            st.success(f"✅ Found {len(similar)} similar movies!")
                        except Exception as e:
                            st.error(f"❌ Error finding similar movies: {e}")
            else:
                st.warning("No movies found. Try a different search term.")
        else:
            st.info("👆 Start typing to search for movies")
    
    with col2:
        st.markdown("#### 🎬 Similar Movies")
        
        if 'similar' in st.session_state and st.session_state.similar:
            st.caption(f"Movies similar to: **{st.session_state.get('selected_movie', 'N/A')}**")
            
            for idx, s in enumerate(st.session_state.similar, 1):
                # Create a similarity percentage
                similarity_pct = s['similarity'] * 100
                
                st.markdown(f"""
                <div class="movie-card">
                    <h4 style="margin:0; color:#4ECDC4;">#{idx} {s['title']}</h4>
                    <p style="margin:0.5rem 0; color:#999;">
                        <b>Genres:</b> {format_genres(s['genres'])}
                    </p>
                    <p style="margin:0;">
                        🔗 <b>Similarity:</b> {s['similarity']:.3f} ({similarity_pct:.1f}%)
                    </p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("👆 Search for a movie and click 'Find Similar Movies' to see results")

# ==========================================
# 7. FOOTER
# ==========================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | Powered by Matrix Factorization (ALS)</p>
</div>
""", unsafe_allow_html=True)
