"""
Movie Recommender using Dummy User Training
Matches your Colab notebook implementation exactly
"""

import numpy as np
from numba import njit, prange


class MovieRecommender:
    def __init__(self, model_path, movies_df):
        """
        Initialize the recommender with movie embeddings only.
        Users are trained on-the-fly as "dummy users".
        """
        self.movies_df = movies_df
        
        # Load model
        print("📦 Loading model from:", model_path)
        model_data = np.load(model_path, allow_pickle=True)
        
        print("📦 Available keys in model:", list(model_data.files))
        
        # ====================================================================
        # LOAD MOVIE EMBEDDINGS AND BIASES (Exact structure from your model)
        # ====================================================================
        if 'movie_embeddings' not in model_data.files:
            raise ValueError("❌ movie_embeddings not found in model file!")
        
        if 'movie_biases' not in model_data.files:
            raise ValueError("❌ movie_biases not found in model file!")
        
        self.movie_embeddings = model_data['movie_embeddings']  # Shape: (k, n_movies)
        self.movie_biases = model_data['movie_biases']          # Shape: (n_movies,)
        
        print(f"✓ Loaded movie_embeddings: {self.movie_embeddings.shape}")
        print(f"✓ Loaded movie_biases: {self.movie_biases.shape}")
        
        # Get dimensions
        self.k = self.movie_embeddings.shape[0]      # Latent dimensions
        self.n_movies = self.movie_embeddings.shape[1]  # Number of movies
        
        # ====================================================================
        # LOAD HYPERPARAMETERS
        # ====================================================================
        self.lamda = float(model_data['lamda']) if 'lamda' in model_data.files else 0.02
        self.gamma_u = float(model_data['gamma_u']) if 'gamma_u' in model_data.files else 0.85
        self.gamma_b = float(model_data['gamma_b']) if 'gamma_b' in model_data.files else 0.002
        
        print(f"✓ Hyperparameters: λ={self.lamda:.4f}, γ_u={self.gamma_u:.4f}, γ_b={self.gamma_b:.4f}")
        
        # ====================================================================
        # CREATE MOVIE ID MAPPINGS FROM idx_to_movieid
        # This is the CRITICAL FIX - your model has idx_to_movieid as an ARRAY
        # ====================================================================
        if 'idx_to_movieid' not in model_data.files:
            raise ValueError("❌ idx_to_movieid not found in model file!")
        
        # idx_to_movieid is a NumPy array where index = model index, value = original movie ID
        idx_to_movieid_array = model_data['idx_to_movieid']
        
        # Create BOTH mapping directions
        # 1. idx -> movieid (from array directly)
        self.idx_to_movieid = {idx: int(movie_id) 
                               for idx, movie_id in enumerate(idx_to_movieid_array)}
        
        # 2. movieid -> idx (reverse mapping - THIS IS WHAT WAS MISSING!)
        self.movieid_to_idx = {int(movie_id): idx 
                               for idx, movie_id in enumerate(idx_to_movieid_array)}
        
        print(f"✓ Created movie ID mappings: {len(self.movieid_to_idx)} movies")
        
        # Show sample mapping
        sample_items = list(self.movieid_to_idx.items())[:3]
        print(f"  Sample (movieId -> idx): {sample_items}")
        
        # Verify mapping consistency
        movie_id_range = (min(self.movieid_to_idx.keys()), max(self.movieid_to_idx.keys()))
        print(f"  Movie ID range: {movie_id_range[0]} - {movie_id_range[1]}")
        
        # ====================================================================
        # LOAD MOVIE RATING COUNTS (for filtering low-count movies)
        # ====================================================================
        # Try to extract from movie_ptr if available
        if 'movie_ptr' in model_data.files:
            movie_ptr = model_data['movie_ptr']
            self.movie_counts = np.diff(movie_ptr)
            print(f"✓ Calculated movie_counts from movie_ptr")
        elif 'movie_ptr_train' in model_data.files:
            movie_ptr_train = model_data['movie_ptr_train']
            self.movie_counts = np.diff(movie_ptr_train)
            print(f"✓ Calculated movie_counts from movie_ptr_train")
        else:
            # Fallback: no filtering
            self.movie_counts = np.ones(self.n_movies) * 1000
            print(f"⚠ No movie_ptr found, disabling rating count filtering")
        
        # Reconstruct movie_ptr for scoring (matches your notebook)
        self.movie_ptr = np.concatenate([[0], np.cumsum(self.movie_counts)]).astype(np.int32)
        
        print(f"\n✅ Recommender initialized successfully!")
        print(f"   Movies: {self.n_movies:,}")
        print(f"   Latent dimensions: {self.k}")
        print(f"   Movie ID range: {movie_id_range[0]} - {movie_id_range[1]}")
    
    # ========================================================================
    # NUMBA KERNELS (Exact copies from your Colab notebook)
    # ========================================================================
    @staticmethod
    @njit(fastmath=True)
    def _numba_dummy_train_kernel(movie_indices, rating_values, V, b_m,
                                   lamda, gamma_u, gamma_b, k, iterations):
        """
        Numba kernel to train a single user vector.
        Exact copy from your Colab notebook.
        """
        # Initialize random embedding
        dummy_user_embedding = np.random.normal(0, 0.1, size=(k,)).astype(np.float64)
        dummy_bias = 0.0
        n_ratings = len(movie_indices)
        
        # Pre-allocate identity matrix scaled by gamma_u
        eye_k = np.eye(k, dtype=np.float64) * gamma_u
        
        for _ in range(iterations):
            # 1. Update Bias
            bias_sum = 0.0
            for i in range(n_ratings):
                m_idx = movie_indices[i]
                r = rating_values[i]
                # Dot product: dummy_emb @ V[:, m_idx]
                dot_val = 0.0
                for f in range(k):
                    dot_val += dummy_user_embedding[f] * V[f, m_idx]
                pred = b_m[m_idx] + dot_val
                bias_sum += lamda * (r - pred)
            
            dummy_bias = bias_sum / (lamda * n_ratings + gamma_b)
            
            # 2. Update Embedding
            first_term = eye_k.copy()
            second_term = np.zeros(k, dtype=np.float64)
            
            for i in range(n_ratings):
                m_idx = movie_indices[i]
                r = rating_values[i]
                # Update second term: v_n * (rating - bias - b_m)
                residual = r - dummy_bias - b_m[m_idx]
                # Manual outer product accumulation for speed
                factor = lamda
                for r_idx in range(k):
                    v_r = V[r_idx, m_idx]
                    second_term[r_idx] += factor * v_r * residual
                    for c_idx in range(k):
                        first_term[r_idx, c_idx] += factor * v_r * V[c_idx, m_idx]
            
            # Use lstsq directly (handles both singular and non-singular matrices)
            dummy_user_embedding = np.linalg.lstsq(first_term, second_term, rcond=-1)[0]
        
        return dummy_user_embedding, dummy_bias
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _numba_calculate_scores(dummy_emb, V, dummy_bias, b_m, movie_ptr):
        """
        Calculates scores for all movies in parallel.
        Exact copy from your Colab notebook.
        """
        n_movies = V.shape[1]
        k = V.shape[0]
        scores = np.zeros(n_movies, dtype=np.float64)
        
        for m in prange(n_movies):
            # Check rating count using the pointer structure
            count = movie_ptr[m+1] - movie_ptr[m]
            if count < 100:
                scores[m] = 0.0
            else:
                # Dot product
                dot_val = 0.0
                for f in range(k):
                    dot_val += dummy_emb[f] * V[f, m]
                scores[m] = dot_val + (b_m[m] * 0.05)
        
        return scores
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    def recommend_from_ratings(self, user_ratings, n_recommendations=15, iterations=10):
        """
        Generate recommendations based on user's ratings.
        Matches your Colab notebook implementation exactly.
        
        Args:
            user_ratings: List of tuples [(movie_id, rating), ...]
            n_recommendations: Number of recommendations to return
            iterations: Number of training iterations for dummy user
            
        Returns:
            List of recommended movies with scores
        """
        if len(user_ratings) == 0:
            print("⚠ No ratings provided")
            return []
        
        # Convert movie IDs to indices (like your Colab: movieid_to_idx)
        dummy_movie_indices = []
        dummy_user_ratings = []
        rated_movie_ids = set()
        
        for movie_id, rating in user_ratings:
            if movie_id in self.movieid_to_idx:
                idx = self.movieid_to_idx[movie_id]
                dummy_movie_indices.append(idx)
                dummy_user_ratings.append((idx, rating))
                rated_movie_ids.add(movie_id)
            else:
                print(f"⚠ Movie ID {movie_id} not found in training data, skipping...")
        
        if len(dummy_user_ratings) == 0:
            print("❌ None of the provided movies were found in the model")
            return []
        
        print(f"\n{'='*60}")
        print(f"Training dummy user with {len(dummy_user_ratings)} ratings...")
        print(f"{'='*60}")
        
        # Convert to NumPy arrays for Numba (like your Colab)
        indices = np.array([x[0] for x in dummy_user_ratings], dtype=np.int32)
        values = np.array([x[1] for x in dummy_user_ratings], dtype=np.float64)
        
        # Train dummy user (exact same call as your Colab)
        dummy_user_embedding, dummy_bias = self._numba_dummy_train_kernel(
            indices, values,
            self.movie_embeddings,
            self.movie_biases,
            self.lamda,
            self.gamma_u,
            self.gamma_b,
            self.k,
            iterations
        )
        
        print(f"✓ Dummy user embedding norm: {np.linalg.norm(dummy_user_embedding):.4f}")
        print(f"✓ Dummy user bias: {dummy_bias:.4f}")
        
        # Calculate scores for all movies (exact same call as your Colab)
        scores = self._numba_calculate_scores(
            dummy_user_embedding,
            self.movie_embeddings,
            dummy_bias,
            self.movie_biases,
            self.movie_ptr
        )
        
        print(f"✓ Score statistics: Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}")
        
        # Rank movies by score (like your Colab)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Filter out rated movies and collect top N (like your Colab)
        top_recommendations = []
        for idx in ranked_indices:
            # Skip if already rated (like your Colab: if idx not in dummy_movie_idx)
            if idx in dummy_movie_indices:
                continue
            
            # Skip if score is 0 (filtered out by rating count < 100)
            if scores[idx] <= 0:
                continue
            
            # Convert index back to movie ID (like your Colab: idx_to_movieid)
            movie_id = self.idx_to_movieid[idx]
            
            # Get movie info from dataframe
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if not movie_info.empty:
                top_recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'score': float(scores[idx]),
                    'predicted_rating': float(scores[idx])
                })
            
            if len(top_recommendations) >= n_recommendations:
                break
        
        return top_recommendations
    
    def find_similar_movies(self, movie_id, n_similar=10):
        """
        Find similar movies based on embedding cosine similarity.
        """
        if movie_id not in self.movieid_to_idx:
            print(f"❌ Movie ID {movie_id} not found in model")
            return []
        
        movie_idx = self.movieid_to_idx[movie_id]
        
        # Get movie embedding vector (column from V)
        movie_vec = self.movie_embeddings[:, movie_idx]
        movie_norm = np.linalg.norm(movie_vec)
        
        # Calculate cosine similarity with all movies
        similarities = np.zeros(self.n_movies)
        for i in range(self.n_movies):
            other_vec = self.movie_embeddings[:, i]
            other_norm = np.linalg.norm(other_vec)
            if other_norm > 0 and movie_norm > 0:
                similarities[i] = np.dot(movie_vec, other_vec) / (movie_norm * other_norm)
        
        # Exclude the query movie itself
        similarities[movie_idx] = -1
        
        # Get top N similar
        similar_indices = np.argsort(similarities)[::-1][:n_similar]
        
        # Build results
        similar_movies = []
        for idx in similar_indices:
            if similarities[idx] > 0:
                similar_movie_id = self.idx_to_movieid[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
                
                if not movie_info.empty:
                    similar_movies.append({
                        'movieId': int(similar_movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'genres': movie_info.iloc[0]['genres'],
                        'similarity': float(similarities[idx])
                    })
        
        return similar_movies
    
    def get_movie_title(self, movie_id):
        """Get movie title by ID"""
        movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
        if not movie_info.empty:
            return movie_info.iloc[0]['title']
        return f"Movie ID {movie_id}"


# ============================================================================
# TESTING SCRIPT
# ============================================================================
if __name__ == "__main__":
    import pandas as pd
    
    print("="*70)
    print("TESTING MOVIE RECOMMENDER - DUMMY USER APPROACH")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    movies_df = pd.read_csv('movies.csv')
    
    # Initialize recommender
    print("\n2. Initializing recommender...")
    recommender = MovieRecommender('best_model.npz', movies_df)
    
    # Test with the same movie as your Colab notebook
    print("\n3. Testing with Harry Potter rating (same as your Colab)...")
    dummy_movie_ids = [4896]  # Harry Potter and the Sorcerer's Stone
    dummy_ratings = [5.0]
    
    # Prepare user ratings
    user_ratings = list(zip(dummy_movie_ids, dummy_ratings))
    
    # Get recommendations
    recommendations = recommender.recommend_from_ratings(
        user_ratings,
        n_recommendations=15,
        iterations=10
    )
    
    # Display results
    print(f"\n{'='*70}")
    print("RECOMMENDATION RESULTS")
    print(f"{'='*70}")
    print("\nRated Movies:")
    for movie_id in dummy_movie_ids:
        title = recommender.get_movie_title(movie_id)
        print(f"  {title}")
    
    print(f"\nTop {len(recommendations)} Recommendations:")
    print(f"\n{'Rank':<6} {'Score':<10} {'Title':<60}")
    print("-" * 76)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i:<6} {rec['score']:<10.4f} {rec['title'][:57]}")
    
    print(f"\n{'='*70}")
    
    # Test similar movies
    print("\n4. Finding movies similar to Harry Potter...")
    similar = recommender.find_similar_movies(4896, n_similar=5)
    
    print(f"\n{'Rank':<6} {'Similarity':<12} {'Title':<60}")
    print("-" * 78)
    for i, sim in enumerate(similar, 1):
        print(f"{i:<6} {sim['similarity']:<12.4f} {sim['title'][:57]}")
    
    print("\n✅ All tests completed!")
