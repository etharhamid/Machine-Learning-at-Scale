import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

class MovieRecommender:
    """Movie recommendation system using trained embeddings."""
    
    def __init__(self, model_path: str, movies_df: pd.DataFrame):
        """
        Initialize recommender with trained model.
        
        Parameters:
        -----------
        model_path : str
            Path to .npz file with trained embeddings
        movies_df : pd.DataFrame
            DataFrame with movie metadata (movieId, title, genres)
        """
        # Load trained model
        model_data = np.load(model_path)
        self.user_embeddings = model_data['user_embeddings']
        self.movie_embeddings = model_data['movie_embeddings']
        self.user_biases = model_data['user_biases']
        self.movie_biases = model_data['movie_biases']
        
        # Load hyperparameters if available
        self.k = model_data.get('k', self.movie_embeddings.shape[0])
        
        # Store movie metadata
        self.movies_df = movies_df
        self.n_movies = self.movie_embeddings.shape[1]
        
        # Create movie ID mapping (assuming movies are indexed 0 to n-1)
        # Adjust this based on your actual mapping
        self.movie_ids = movies_df['movieId'].values
        self.movieid_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        
    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        """Predict rating for a user-movie pair."""
        if user_idx >= len(self.user_biases) or movie_idx >= self.n_movies:
            return 0.0
        
        prediction = (
            self.user_biases[user_idx] + 
            self.movie_biases[movie_idx] + 
            np.dot(self.user_embeddings[:, user_idx], 
                   self.movie_embeddings[:, movie_idx])
        )
        
        # Clip to valid rating range
        return np.clip(prediction, 0.5, 5.0)
    
    def recommend_movies(self, user_idx: int, n_recommendations: int = 10,
                        exclude_movies: List[int] = None) -> List[Dict]:
        """
        Get top-N movie recommendations for a user.
        
        Parameters:
        -----------
        user_idx : int
            User index
        n_recommendations : int
            Number of recommendations to return
        exclude_movies : List[int]
            Movie indices to exclude from recommendations
        
        Returns:
        --------
        List[Dict]
            List of recommended movies with predictions
        """
        if user_idx >= len(self.user_biases):
            return []
        
        # Predict ratings for all movies
        predictions = np.zeros(self.n_movies)
        for movie_idx in range(self.n_movies):
            predictions[movie_idx] = self.predict_rating(user_idx, movie_idx)
        
        # Exclude movies if provided
        if exclude_movies:
            predictions[exclude_movies] = -np.inf
        
        # Get top-N recommendations
        top_indices = np.argsort(predictions)[::-1][:n_recommendations]
        
        # Format results
        recommendations = []
        for idx in top_indices:
            if idx < len(self.movie_ids):
                movie_id = self.movie_ids[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                
                recommendations.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'predicted_rating': float(predictions[idx])
                })
        
        return recommendations
    
    def find_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Dict]:
        """
        Find movies similar to a given movie based on embeddings.
        
        Parameters:
        -----------
        movie_id : int
            Movie ID
        n_similar : int
            Number of similar movies to return
        
        Returns:
        --------
        List[Dict]
            List of similar movies with similarity scores
        """
        if movie_id not in self.movieid_to_idx:
            return []
        
        movie_idx = self.movieid_to_idx[movie_id]
        movie_emb = self.movie_embeddings[:, movie_idx]
        
        # Compute cosine similarity with all movies
        similarities = np.zeros(self.n_movies)
        for idx in range(self.n_movies):
            other_emb = self.movie_embeddings[:, idx]
            # Cosine similarity
            similarity = np.dot(movie_emb, other_emb) / (
                np.linalg.norm(movie_emb) * np.linalg.norm(other_emb) + 1e-10
            )
            similarities[idx] = similarity
        
        # Exclude the query movie itself
        similarities[movie_idx] = -np.inf
        
        # Get top-N similar movies
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        
        # Format results
        similar_movies = []
        for idx in top_indices:
            if idx < len(self.movie_ids):
                similar_movie_id = self.movie_ids[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id].iloc[0]
                
                similar_movies.append({
                    'movie_id': int(similar_movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'similarity': float(similarities[idx])
                })
        
        return similar_movies
    
    def get_movie_embedding_norm(self, movie_id: int) -> float:
        """Get the norm of a movie's embedding (polarization measure)."""
        if movie_id not in self.movieid_to_idx:
            return 0.0
        
        movie_idx = self.movieid_to_idx[movie_id]
        return float(np.linalg.norm(self.movie_embeddings[:, movie_idx]))
    
    def get_most_polarizing_movies(self, n_movies: int = 30) -> List[Dict]:
        """Get most polarizing movies (highest embedding norms)."""
        norms = np.array([np.linalg.norm(self.movie_embeddings[:, idx]) 
                         for idx in range(self.n_movies)])
        
        top_indices = np.argsort(norms)[::-1][:n_movies]
        
        polarizing_movies = []
        for idx in top_indices:
            if idx < len(self.movie_ids):
                movie_id = self.movie_ids[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                
                polarizing_movies.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'polarization': float(norms[idx])
                })
        
        return polarizing_movies
    
    def get_least_polarizing_movies(self, n_movies: int = 30) -> List[Dict]:
        """Get least polarizing movies (lowest embedding norms)."""
        norms = np.array([np.linalg.norm(self.movie_embeddings[:, idx]) 
                         for idx in range(self.n_movies)])
        
        top_indices = np.argsort(norms)[:n_movies]
        
        consensus_movies = []
        for idx in top_indices:
            if idx < len(self.movie_ids):
                movie_id = self.movie_ids[idx]
                movie_info = self.movies_df[self.movies_df['movieId'] == movie_id].iloc[0]
                
                consensus_movies.append({
                    'movie_id': int(movie_id),
                    'title': movie_info['title'],
                    'genres': movie_info['genres'],
                    'consensus': float(norms[idx])
                })
        
        return consensus_movies
