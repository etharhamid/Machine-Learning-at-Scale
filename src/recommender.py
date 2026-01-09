class MovieRecommender:
    def __init__(self, model_path, movies_df):
        """
        Initialize the recommender with proper user/movie ID mapping
        """
        self.movies_df = movies_df
        
        # Load model
        model_data = np.load(model_path)
        self.user_factors = model_data['user_factors']
        self.movie_factors = model_data['movie_factors']
        self.user_biases = model_data['user_biases']
        self.movie_biases = model_data['movie_biases']
        self.global_mean = float(model_data['global_mean'])
        
        self.k = self.user_factors.shape[1]
        self.n_users = len(self.user_biases)
        self.n_movies = len(self.movie_biases)
        
        # Load ID mappings if they exist
        if 'user_id_map' in model_data.files:
            self.user_id_map = model_data['user_id_map'].item()
            # Check mapping direction and create reverse if needed
            first_key = list(self.user_id_map.keys())[0]
            if isinstance(first_key, (int, np.integer)) and first_key > 100:
                # It's original_id -> index (correct)
                self.user_idx_to_id = {v: k for k, v in self.user_id_map.items()}
            else:
                # It's index -> original_id (reverse it)
                self.user_idx_to_id = self.user_id_map
                self.user_id_map = {v: k for k, v in self.user_idx_to_id.items()}
        else:
            # No mapping - assume sequential IDs
            self.user_id_map = {i: i for i in range(self.n_users)}
            self.user_idx_to_id = {i: i for i in range(self.n_users)}
        
        if 'movie_id_map' in model_data.files:
            self.movie_id_map = model_data['movie_id_map'].item()
            # Check mapping direction
            first_key = list(self.movie_id_map.keys())[0]
            if isinstance(first_key, (int, np.integer)) and first_key > 100:
                # It's original_id -> index (correct)
                self.movie_idx_to_id = {v: k for k, v in self.movie_id_map.items()}
            else:
                # It's index -> original_id (reverse it)
                self.movie_idx_to_id = self.movie_id_map
                self.movie_id_map = {v: k for k, v in self.movie_idx_to_id.items()}
        else:
            # No mapping - use movieId from dataframe
            self.movie_id_map = {row['movieId']: idx 
                                for idx, row in movies_df.iterrows()}
            self.movie_idx_to_id = {v: k for k, v in self.movie_id_map.items()}
        
        print(f"✓ Recommender initialized:")
        print(f"  - Users: {self.n_users:,} (ID range: {min(self.user_id_map.keys())}-{max(self.user_id_map.keys())})")
        print(f"  - Movies: {self.n_movies:,} (ID range: {min(self.movie_id_map.keys())}-{max(self.movie_id_map.keys())})")
        print(f"  - Latent dimensions: {self.k}")
    
    def _get_user_index(self, user_id):
        """Convert user ID to internal index"""
        if user_id in self.user_id_map:
            return self.user_id_map[user_id]
        else:
            raise ValueError(f"User ID {user_id} not found in training data. "
                           f"Valid range: {min(self.user_id_map.keys())}-{max(self.user_id_map.keys())}")
    
    def _get_movie_index(self, movie_id):
        """Convert movie ID to internal index"""
        if movie_id in self.movie_id_map:
            return self.movie_id_map[movie_id]
        else:
            raise ValueError(f"Movie ID {movie_id} not found in training data")
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair using original IDs
        """
        user_idx = self._get_user_index(user_id)
        movie_idx = self._get_movie_index(movie_id)
        
        prediction = (self.global_mean + 
                     self.user_biases[user_idx] + 
                     self.movie_biases[movie_idx] +
                     np.dot(self.user_factors[user_idx], self.movie_factors[movie_idx]))
        
        return np.clip(prediction, 0.5, 5.0)
    
    def recommend_movies(self, user_id, n_recommendations=10, exclude_rated=True):
        """
        Get movie recommendations for a user using original user ID
        """
        try:
            user_idx = self._get_user_index(user_id)
        except ValueError as e:
            print(f"Error: {e}")
            return []
        
        # Calculate predictions for all movies
        predictions = (self.global_mean + 
                      self.user_biases[user_idx] + 
                      self.movie_biases +
                      np.dot(self.user_factors[user_idx], self.movie_factors.T))
        
        predictions = np.clip(predictions, 0.5, 5.0)
        
        # Get top N
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        # Convert back to original movie IDs
        recommendations = []
        for idx in top_indices:
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
        Find similar movies using original movie ID
        """
        try:
            movie_idx = self._get_movie_index(movie_id)
        except ValueError as e:
            print(f"Error: {e}")
            return []
        
        # Calculate cosine similarity
        movie_vec = self.movie_factors[movie_idx]
        similarities = np.dot(self.movie_factors, movie_vec) / (
            np.linalg.norm(self.movie_factors, axis=1) * np.linalg.norm(movie_vec)
        )
        
        # Get top N (excluding the movie itself)
        similar_indices = np.argsort(similarities)[::-1][1:n_similar+1]
        
        # Convert back to original movie IDs
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
