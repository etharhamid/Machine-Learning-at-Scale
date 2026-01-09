import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.recommender import MovieRecommender
from src.utils import (
    load_movies_data,
    create_rating_distribution_plot,
    create_similarity_plot,
    create_polarization_plot,
    format_genres
)

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background-color: #f0f2f6;
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
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff5252;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_recommender():
    """Load the recommender system (cached)."""
    movies_df = load_movies_data('data/movies.csv')
    recommender = MovieRecommender('models/best_model.npz', movies_df)
    return recommender, movies_df

# Load data
try:
    recommender, movies_df = load_recommender()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Main header
st.markdown('<div class="main-header">🎬 Movie Recommender System</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/cotton/256/movie-projector.png", width=150)
    st.markdown("### 🎯 Navigation")
    
    page = st.radio(
        "Choose a feature:",
        ["🏠 Home", "🎬 Get Recommendations", "🔍 Find Similar Movies", 
         "📊 Movie Analysis", "ℹ️ About"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Model Info")
    st.info(f"""
    **Latent Dimensions:** {recommender.k}  
    **Total Movies:** {recommender.n_movies:,}  
    **Total Users:** {len(recommender.user_biases):,}
    """)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Movies", f"{recommender.n_movies:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Users", f"{len(recommender.user_biases):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Embedding Dims", recommender.k)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sub-header">🎯 What can you do?</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎬 Get Personalized Recommendations
        - Select a user ID
        - Get top movie recommendations
        - See predicted ratings
        
        ### 🔍 Find Similar Movies
        - Search for any movie
        - Discover similar titles
        - Based on collaborative filtering
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Analyze Movies
        - Most polarizing movies
        - Consensus favorites
        - Embedding visualizations
        
        ### 🎯 Smart Filtering
        - Filter by genre
        - Exclude watched movies
        - Customizable recommendations
        """)

# ============================================================================
# GET RECOMMENDATIONS PAGE
# ============================================================================
elif page == "🎬 Get Recommendations":
    st.markdown('<div class="sub-header">Get Personalized Movie Recommendations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### ⚙️ Settings")
        
        user_idx = st.number_input(
            "User ID",
            min_value=0,
            max_value=len(recommender.user_biases) - 1,
            value=0,
            help="Enter a user ID to get personalized recommendations"
        )
        
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=50,
            value=10,
            step=5
        )
        
        # Genre filter
        all_genres = set()
        for genres in movies_df['genres'].dropna():
            all_genres.update(genres.split('|'))
        all_genres = sorted(list(all_genres))
        
        selected_genres = st.multiselect(
            "Filter by genres (optional)",
            options=all_genres,
            default=None
        )
        
        if st.button("🎬 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = recommender.recommend_movies(
                    user_idx, 
                    n_recommendations=n_recommendations * 2  # Get more then filter
                )
                
                # Filter by genre if selected
                if selected_genres:
                    recommendations = [
                        rec for rec in recommendations
                        if any(genre in rec['genres'] for genre in selected_genres)
                    ][:n_recommendations]
                else:
                    recommendations = recommendations[:n_recommendations]
                
                st.session_state.recommendations = recommendations
    
    with col2:
        if 'recommendations' in st.session_state and st.session_state.recommendations:
            recs = st.session_state.recommendations
            
            st.markdown(f"### 🎯 Top {len(recs)} Recommendations")
            
            # Plot
            fig = create_rating_distribution_plot(recs)
            st.plotly_chart(fig, use_container_width=True)
            
            # List recommendations
            for i, rec in enumerate(recs, 1):
                with st.container():
                    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"**{i}. {rec['title']}**")
                        st.markdown(f"{format_genres(rec['genres'])}")
                    
                    with col_b:
                        st.metric("Predicted Rating", f"{rec['predicted_rating']:.2f}⭐")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Configure settings and click 'Get Recommendations' to see results")

# ============================================================================
# FIND SIMILAR MOVIES PAGE
# ============================================================================
elif page == "🔍 Find Similar Movies":
    st.markdown('<div class="sub-header">Find Movies Similar to Your Favorite</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 🔎 Search")
        
        # Search for movie
        search_query = st.text_input(
            "Search for a movie",
            placeholder="Enter movie title..."
        )
        
        if search_query:
            # Find matching movies
            matches = movies_df[
                movies_df['title'].str.contains(search_query, case=False, na=False)
            ].head(10)
            
            if not matches.empty:
                selected_movie = st.selectbox(
                    "Select a movie",
                    options=matches['movieId'].tolist(),
                    format_func=lambda x: matches[matches['movieId'] == x]['title'].iloc[0]
                )
                
                n_similar = st.slider(
                    "Number of similar movies",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5
                )
                
                if st.button("🔍 Find Similar Movies", type="primary"):
                    with st.spinner("Finding similar movies..."):
                        similar = recommender.find_similar_movies(
                            selected_movie,
                            n_similar=n_similar
                        )
                        st.session_state.similar_movies = similar
                        st.session_state.selected_movie_title = matches[
                            matches['movieId'] == selected_movie
                        ]['title'].iloc[0]
            else:
                st.warning("No movies found matching your search")
    
    with col2:
        if 'similar_movies' in st.session_state and st.session_state.similar_movies:
            similar = st.session_state.similar_movies
            
            st.markdown(f"### 🎬 Movies Similar to '{st.session_state.selected_movie_title}'")
            
            # Plot
            fig = create_similarity_plot(similar)
            st.plotly_chart(fig, use_container_width=True)
            
            # List similar movies
            for i, movie in enumerate(similar, 1):
                with st.container():
                    st.markdown(f'<div class="movie-card">', unsafe_allow_html=True)
                    col_a, col_b = st.columns([3, 1])
                    
                    with col_a:
                        st.markdown(f"**{i}. {movie['title']}**")
                        st.markdown(f"{format_genres(movie['genres'])}")
                    
                    with col_b:
                        st.metric("Similarity", f"{movie['similarity']:.3f}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👆 Search for a movie and click 'Find Similar Movies' to see results")

# ============================================================================
# MOVIE ANALYSIS PAGE
# ============================================================================
elif page == "📊 Movie Analysis":
    st.markdown('<div class="sub-header">Movie Polarization Analysis</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["🔥 Most Polarizing", "🤝 Most Consensus"])
    
    with tab1:
        st.markdown("""
        **Most polarizing movies** have the highest embedding norms, indicating that 
        users have strong (but divided) opinions about them.
        """)
        
        n_polarizing = st.slider(
            "Number of movies to show",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="polarizing_slider"
        )
        
        with st.spinner("Analyzing..."):
            polarizing = recommender.get_most_polarizing_movies(n_movies=n_polarizing)
            
            fig = create_polarization_plot(polarizing, 'polarization')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show as table
            with st.expander("📋 View as table"):
                df_polar = pd.DataFrame(polarizing)
                st.dataframe(
                    df_polar[['title', 'genres', 'polarization']],
                    use_container_width=True
                )
    
    with tab2:
        st.markdown("""
        **Consensus movies** have the lowest embedding norms, indicating that 
        most users agree about these movies (universally liked or disliked).
        """)
        
        n_consensus = st.slider(
            "Number of movies to show",
            min_value=10,
            max_value=50,
            value=20,
            step=5,
            key="consensus_slider"
        )
        
        with st.spinner("Analyzing..."):
            consensus = recommender.get_least_polarizing_movies(n_movies=n_consensus)
            
            fig = create_polarization_plot(consensus, 'consensus')
            st.plotly_chart(fig, use_container_width=True)
            
            # Show as table
            with st.expander("📋 View as table"):
                df_consensus = pd.DataFrame(consensus)
                st.dataframe(
                    df_consensus[['title', 'genres', 'consensus']],
                    use_container_width=True
                )

# ============================================================================
# ABOUT PAGE
# ============================================================================
elif page == "ℹ️ About":
    st.markdown('<div class="sub-header">About This Project</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This movie recommendation system uses **Alternating Least Squares (ALS)** matrix factorization
    to learn latent representations of users and movies from rating data.
    
    ### 🔧 Technical Details
    
    - **Algorithm:** ALS with bias terms
    - **Implementation:** Optimized with Numba JIT compilation
    - **Features:**
        - Personalized recommendations
        - Movie similarity search
        - Polarization analysis
        - Genre filtering
    
    ### 📊 Model Architecture
```
    Rating = user_bias + movie_bias + user_embedding · movie_embedding
```
    
    - **User embeddings:** Learned preferences in latent space
    - **Movie embeddings:** Learned characteristics in latent space
    - **Biases:** Account for user/movie-specific rating tendencies
    
    ### 🚀 Performance
    
    - Trained with early stopping
    - Hyperparameters tuned via random search
    - Optimized for both accuracy and generalization
    
    ### 👨‍💻 Built With
    
    - **NumPy** - Numerical computations
    - **Numba** - JIT compilation for speed
    - **Streamlit** - Web interface
    - **Plotly** - Interactive visualizations
    
    ### 📝 License
    
    This project is open source and available under the MIT License.
    
    ### 🔗 Links
    
    - [GitHub Repository](#)
    - [Documentation](#)
    - [Report an Issue](#)
    """)
    
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        🎬 Movie Recommender System | Powered by ALS Matrix Factorization
    </div>
    """,
    unsafe_allow_html=True
)
