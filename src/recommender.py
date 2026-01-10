"""
Recommender System using Dummy User Training
This matches your Colab notebook implementation
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
        model_data = np.load(model_path)
        
        # Movie embeddings and biases (V and b_m in your code)
        self.movie_factors = model_data['movie_factors']  # Shape: (k, n_movies)
        self.movie_biases = model_data['movie_biases']    # Shape: (n_movies,)
        
        # Check if we need to transpose
        if self.movie_factors.shape[0] > self.movie_factors.shape[1]:
            # If shape is (n_movies, k), transpose to (k, n_movies)
            self.movie_factors = self.movie_factors.T
        
        self.k = self.movie_factors.shape[0]
        self.n_movies = self.movie_factors.shape[1]
        
        # Hyperparameters (same as your notebook)
        self.lamda = model_data.get('lamda', 0.05)
        self.gamma_u = model_data.get('gamma_u', 0.05)
        self.gamma_b = model_data.get('gamma_b', 0.05)
        
        # Load movie ID mapping
        if 'movie_id_map' in model_data.files:
            self.movie_id_map = model_data['movie_id_map'].item()
            self.idx_to_movie_id = {v: k for k, v in self.movie_id_map.items()}
        else:
            # Create mapping from dataframe
            self.movie_id_map = {row['movieId']: idx 
                                for idx, row in movies_df.iterrows()}
            self.idx_to_movie_id = {v: k for k, v in self.movie_id_map.items()}
        
        # Load movie rating counts if available (for filtering)
        if 'movie_counts' in model_data.files:
            self.movie_counts = model_data['movie_counts']
        else:
            # Default to no filtering
            self.movie_counts = np.ones(self.n_movies) * 1000
        
        print(f"✓ Recommender initialized:")
        print(f"  - Movies: {self.n_movies:,}")
        print(f"  - Latent dimensions: {self.k}")
        print(f"  - Hyperparameters: λ={self.lamda}, γ_u={self.gamma_u}, γ_b={self.gamma_b}")
    
    @staticmethod
    @njit(fastmath=True)
    def _train_dummy_user(movie_indices, rating_values, V, b_m,
                         lamda, gamma_u, gamma_b, k, iterations):
        """
        Numba kernel to train a single dummy user vector.
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
                residual = r - dummy_bias - b_m[m_idx]
                factor = lamda
                
                for r_idx in range(k):
                    v_r = V[r_idx, m_idx]
                    second_term[r_idx] += factor * v_r * residual
                    for c_idx in range(k):
                        first_term[r_idx, c_idx] += factor * v_r * V[c_idx, m_idx]
            
            # Solve linear system
            dummy_user_embedding = np.linalg.lstsq(first_term, second_term, rcond=-1)[0]
        
        return dummy_user_embedding, dummy_bias
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _calculate_scores(dummy_emb, V, dummy_bias, b_m, movie_counts, min_count=100):
        """
        Calculates scores for all movies in parallel.
        """
        n_movies = V.shape[1]
        k = V.shape[0]
        scores = np.zeros(n_movies, dtype=np.float64)
        
        for m in prange(n_movies):
            # Filter out movies with too few ratings
            if movie_counts[m] < min_count:
                scores[m] = -999.0  # Very low score to exclude
            else:
                # Dot product
                dot_val = 0.0
                for f in range(k):
                    dot_val += dummy_emb[f] * V[f, m]
                scores[m] = dot_val + (b_m[m] * 0.05)
        
        return scores
    
    def recommend_from_ratings(self, user_ratings, n_recommendations=10, iterations=10):
        """
        Generate recommendations based on user's ratings.
        
        Args:
            user_ratings: List of tuples [(movie_id, rating), ...]
            n_recommendations: Number of recommendations to return
            iterations: Number of training iterations for dummy user
            
        Returns:
            List of recommended movies with scores
        """
        if len(user_ratings) == 0:
            # Return popular movies if no ratings provided
            return self._get_popular_movies(n_recommendations)
        
        # Convert movie IDs to indices
        movie_indices = []
        rating_values = []
        rated_indices = set()
        
        for movie_id, rating in user_ratings:
            if movie_id in self.movie_id_map:
                idx = self.movie_id_map[movie_id]
                movie_indices.append(idx)
                rating_values.append(rating)
                rated_indices.add(idx)
        
        if len(movie_indices) == 0:
            return self._get_popular_movies(n_recommendations)
        
        # Convert to numpy arrays
        indices = np.array(movie_indices, dtype=np.int32)
        values = np.array(rating_values, dtype=np.float64)
        
        # Train dummy user
        dummy_embedding, dummy_bias = self._train_dummy_user(
            indices, values,
            self.movie_factors, self.movie_biases,
            self.lamda, self.gamma_u, self.gamma_b,
            self.k, iterations
        )
        
        # Calculate scores for all movies
        scores = self._calculate_scores(
            dummy_embedding,
            self.movie_factors,
            dummy_bias,
            self.movie_biases,
            self.movie_counts,
            min_count=100
        )
        
        # Rank movies
        ranked_indices = np.argsort(scores)[::-1]
        
        # Filter out rated movies and get top N
        recommendations = []
        for idx in ranked_indices:
            if idx not in rated_indices and scores[idx] > -999.0:
                movie_id = self.idx_to_movie_id[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
                
                if not movie_info.empty:
                    recommendations.append({
                        'movieId': int(movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'genres': movie_info.iloc[0]['genres'],
                        'predicted_rating': float(scores[idx]),
                        'score': float(scores[idx])
                    })
                
                if len(recommendations) >= n_recommendations:
                    break
        
        return recommendations
    
    def find_similar_movies(self, movie_id, n_similar=10):
        """
        Find similar movies based on embedding similarity.
        """
        if movie_id not in self.movie_id_map:
            return []
        
        movie_idx = self.movie_id_map[movie_id]
        
        # Get movie embedding (column from V)
        movie_vec = self.movie_factors[:, movie_idx]
        
        # Calculate cosine similarity with all movies
        similarities = np.zeros(self.n_movies)
        movie_norm = np.linalg.norm(movie_vec)
        
        for i in range(self.n_movies):
            other_vec = self.movie_factors[:, i]
            other_norm = np.linalg.norm(other_vec)
            if other_norm > 0 and movie_norm > 0:
                similarities[i] = np.dot(movie_vec, other_vec) / (movie_norm * other_norm)
        
        # Get top N similar movies (excluding the query movie)
        similarities[movie_idx] = -1  # Exclude self
        similar_indices = np.argsort(similarities)[::-1][:n_similar]
        
        # Build results
        similar_movies = []
        for idx in similar_indices:
            if similarities[idx] > 0:  # Only positive similarities
                similar_movie_id = self.idx_to_movie_id[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
                
                if not movie_info.empty:
                    similar_movies.append({
                        'movieId': int(similar_movie_id),
                        'title': movie_info.iloc[0]['title'],
                        'genres': movie_info.iloc[0]['genres'],
                        'similarity': float(similarities[idx])
                    })
        
        return similar_movies
    
    def _get_popular_movies(self, n=10):
        """
        Return most popular movies (highest bias + most ratings).
        """
        # Combine movie bias and rating count for popularity score
        popularity = self.movie_biases * 0.5 + np.log1p(self.movie_counts) * 0.1
        top_indices = np.argsort(popularity)[::-1][:n]
        
        popular_movies = []
        for idx in top_indices:
            movie_id = self.idx_to_movie_id[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if not movie_info.empty:
                popular_movies.append({
                    'movieId': int(movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'predicted_rating': float(popularity[idx]),
                    'score': float(popularity[idx])
                })
        
        return popular_movies

