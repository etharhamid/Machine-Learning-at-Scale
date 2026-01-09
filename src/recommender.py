"""
Fixed MovieRecommender class for your model structure
Works with idx_to_userid and idx_to_movieid arrays
"""

import numpy as np
import pandas as pd


class MovieRecommender:
    def __init__(self, model_path, movies_df):
        """
        Initialize the recommender with proper user/movie ID mapping
        """
        self.movies_df = movies_df
        
        # Load model
        model_data = np.load(model_path)
        
        # Your model uses 'user_embeddings' and 'movie_embeddings' (shape: k x n)
        # Need to transpose them to get (n x k) for easier indexing
        self.user_factors = model_data['user_embeddings'].T  # Shape: (n_users, k)
        self.movie_factors = model_data['movie_embeddings'].T  # Shape: (n_movies, k)
        self.user_biases = model_data['user_biases']
        self.movie_biases = model_data['movie_biases']
        
        # Get dimensions
        self.k = self.user_factors.shape[1]
        self.n_users = len(self.user_biases)
        self.n_movies = len(self.movie_biases)
        
        # Calculate global mean from biases (approximation)
        self.global_mean = float(np.mean(self.user_biases) + np.mean(self.movie_biases))
        
        # ============================================================
        # CRITICAL FIX: Create proper ID mappings
        # ============================================================
        
        # Your model has idx_to_userid and idx_to_movieid
        # We need to create the REVERSE mapping (id -> idx)
        
        if 'idx_to_userid' in model_data.files:
            idx_to_userid = model_data['idx_to_userid']
            # Create reverse mapping: original_user_id -> index
            self.user_id_map = {int(user_id): idx for idx, user_id in enumerate(idx_to_userid)}
            self.user_idx_to_id = {idx: int(user_id) for idx, user_id in enumerate(idx_to_userid)}
            print(f"✓ Created user_id_map from idx_to_userid")
        else:
            # Fallback: assume sequential IDs
            self.user_id_map = {i: i for i in range(self.n_users)}
            self.user_idx_to_id = {i: i for i in range(self.n_users)}
            print("⚠ No idx_to_userid found, assuming sequential user IDs")
        
        if 'idx_to_movieid' in model_data.files:
            idx_to_movieid = model_data['idx_to_movieid']
            # Create reverse mapping: original_movie_id -> index
            self.movie_id_map = {int(movie_id): idx for idx, movie_id in enumerate(idx_to_movieid)}
            self.movie_idx_to_id = {idx: int(movie_id) for idx, movie_id in enumerate(idx_to_movieid)}
            print(f"✓ Created movie_id_map from idx_to_movieid")
        else:
            # Fallback: use movieId from dataframe
            self.movie_id_map = {int(row['movieId']): idx 
                                for idx, row in enumerate(movies_df['movieId'].values)}
            self.movie_idx_to_id = {v: k for k, v in self.movie_id_map.items()}
            print("⚠ No idx_to_movieid found, using movieId from dataframe")
        
        print(f"\n✓ Recommender initialized:")
        print(f"  - Users: {self.n_users:,} (ID range: {min(self.user_id_map.keys())}-{max(self.user_id_map.keys())})")
        print(f"  - Movies: {self.n_movies:,} (ID range: {min(self.movie_id_map.keys())}-{max(self.movie_id_map.keys())})")
        print(f"  - Latent dimensions: {self.k}")
        print(f"  - User factors shape: {self.user_factors.shape}")
        print(f"  - Movie factors shape: {self.movie_factors.shape}")
    
    def _get_user_index(self, user_id):
        """Convert user ID to internal index"""
        if user_id in self.user_id_map:
            return self.user_id_map[user_id]
        else:
            valid_range = f"{min(self.user_id_map.keys())}-{max(self.user_id_map.keys())}"
            raise ValueError(f"User ID {user_id} not found in training data. "
                           f"Valid range: {valid_range}")
    
    def _get_movie_index(self, movie_id):
        """Convert movie ID to internal index"""
        if movie_id in self.movie_id_map:
            return self.movie_id_map[movie_id]
        else:
            valid_range = f"{min(self.movie_id_map.keys())}-{max(self.movie_id_map.keys())}"
            raise ValueError(f"Movie ID {movie_id} not found in training data. "
                           f"Valid range: {valid_range}")
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair using original IDs
        
        Formula: r̂ = U^T @ V + b_u + b_v
        """
        user_idx = self._get_user_index(user_id)
        movie_idx = self._get_movie_index(movie_id)
        
        # Matrix factorization prediction
        prediction = (np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx]) +
                     self.user_biases[user_idx] + 
                     self.movie_biases[movie_idx])
        
        # Clip to valid rating range (1-5 for most rating systems)
        return np.clip(prediction, 1.0, 5.0)
    
    def recommend_movies(self, user_id, n_recommendations=10, exclude_rated=None):
        """
        Get movie recommendations for a user using original user ID
        
        Args:
            user_id: Original user ID from dataset
            n_recommendations: Number of recommendations to return
            exclude_rated: Set or list of movie IDs to exclude (already rated movies)
        
        Returns:
            List of dictionaries with movie recommendations
        """
        try:
            user_idx = self._get_user_index(user_id)
        except ValueError as e:
            print(f"Error: {e}")
            return []
        
        # Calculate predictions for ALL movies (vectorized)
        # predictions = U_user^T @ V_all + b_user + b_movies
        predictions = (np.dot(self.user_factors[user_idx], self.movie_factors.T) +
                      self.user_biases[user_idx] + 
                      self.movie_biases)
        
        # Clip predictions to valid range
        predictions = np.clip(predictions, 1.0, 5.0)
        
        # Exclude already rated movies if provided
        if exclude_rated is not None:
            exclude_rated = set(exclude_rated)  # Convert to set for O(1) lookup
            # Set predictions for excluded movies to -inf so they don't get recommended
            for movie_id in exclude_rated:
                if movie_id in self.movie_id_map:
                    movie_idx = self.movie_id_map[movie_id]
                    predictions[movie_idx] = -np.inf
        
        # Get top N movie indices (sorted by prediction, descending)
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        # Convert indices back to original movie IDs and get movie info
        recommendations = []
        for idx in top_indices:
            # Skip if prediction is -inf (excluded movie)
            if predictions[idx] == -np.inf:
                continue
                
            movie_id = self.movie_idx_to_id[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if not movie_info.empty:
                recommendations.append({
                    'movieId': int(movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'predicted_rating': float(predictions[idx])
                })
        
        return recommendations
    
    def find_similar_movies(self, movie_id, n_similar=10):
        """
        Find similar movies using cosine similarity of movie embeddings
        
        Args:
            movie_id: Original movie ID
            n_similar: Number of similar movies to return
        
        Returns:
            List of dictionaries with similar movies
        """
        try:
            movie_idx = self._get_movie_index(movie_id)
        except ValueError as e:
            print(f"Error: {e}")
            return []
        
        # Get the movie's embedding vector
        movie_vec = self.movie_factors[movie_idx]
        
        # Calculate cosine similarity with all movies
        # cosine_sim = (A · B) / (||A|| * ||B||)
        similarities = np.dot(self.movie_factors, movie_vec) / (
            np.linalg.norm(self.movie_factors, axis=1) * np.linalg.norm(movie_vec) + 1e-8
        )
        
        # Get top N similar movies (excluding the movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        # Convert back to original movie IDs and get info
        similar_movies = []
        for idx in similar_indices:
            similar_movie_id = self.movie_idx_to_id[idx]
            movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
            
            if not movie_info.empty:
                similar_movies.append({
                    'movieId': int(similar_movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'similarity': float(similarities[idx])
                })
        
        return similar_movies
    
    def get_user_top_rated_movies(self, user_id, ratings_df, n=10):
        """
        Get the user's top rated movies (for display purposes)
        
        Args:
            user_id: Original user ID
            ratings_df: DataFrame with user ratings
            n: Number of top rated movies to return
        
        Returns:
            List of dictionaries with user's top rated movies
        """
        user_ratings = ratings_df[ratings_df['userId'] == user_id]
        user_ratings = user_ratings.sort_values('rating', ascending=False).head(n)
        
        top_movies = []
        for _, row in user_ratings.iterrows():
            movie_info = self.movies_df[self.movies_df['movieId'] == row['movieId']]
            if not movie_info.empty:
                top_movies.append({
                    'movieId': int(row['movieId']),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'actual_rating': float(row['rating'])
                })
        
        return top_movies
