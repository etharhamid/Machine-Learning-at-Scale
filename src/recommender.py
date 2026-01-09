import numpy as np
import pandas as pd

class MovieRecommender:
    def __init__(self, model_path, movies_df):
        self.movies_df = movies_df
        
        # Load model
        model_data = np.load(model_path, allow_pickle=True)
        
        # Transpose embeddings to shape (n_items, k) for easier dot products
        self.user_factors = model_data['user_embeddings'].T
        self.movie_factors = model_data['movie_embeddings'].T
        self.user_biases = model_data['user_biases']
        self.movie_biases = model_data['movie_biases']
        
        # ============================================================
        # CRITICAL FIX: Load proper ID mappings
        # ============================================================
        # Load the dictionary: { Original_Movie_ID : Internal_Matrix_Index }
        self.movie_id_to_idx = model_data['movie_id_map'].item()
        
        # Create reverse mapping: { Internal_Matrix_Index : Original_Movie_ID }
        self.idx_to_movie_id = {v: k for k, v in self.movie_id_to_idx.items()}
        
        # Handle User ID mapping similarly if needed
        if 'user_id_map' in model_data:
            self.user_id_to_idx = model_data['user_id_map'].item()
        else:
            self.user_id_to_idx = {}

    def find_similar_movies(self, movie_id, n_similar=10):
        """
        Finds similar movies using the INTERNAL MATRIX INDEX, then maps back to Movie IDs.
        """
        # 1. Translate Original ID -> Internal Index
        if movie_id not in self.movie_id_to_idx:
            return [] # Movie was not in training data
            
        movie_idx = self.movie_id_to_idx[movie_id]
        
        # 2. Get the embedding vector for this index
        movie_vec = self.movie_factors[movie_idx]
        
        # 3. Calculate Cosine Similarity
        # (Assuming normalized vectors for pure cosine, or just dot product)
        scores = self.movie_factors.dot(movie_vec)
        
        # 4. Get top N indices
        # We get n_similar + 1 because the movie itself will be the top result
        top_indices = np.argsort(scores)[::-1][:n_similar+1]
        
        similar_movies = []
        for idx in top_indices:
            # Skip the movie itself
            if idx == movie_idx:
                continue
                
            # 5. Translate Internal Index -> Original ID
            similar_movie_id = self.idx_to_movie_id[idx]
            
            # 6. Get Metadata from DataFrame
            movie_info = self.movies_df[self.movies_df['movieId'] == similar_movie_id]
            
            if not movie_info.empty:
                similar_movies.append({
                    'movieId': int(similar_movie_id),
                    'title': movie_info.iloc[0]['title'],
                    'genres': movie_info.iloc[0]['genres'],
                    'similarity': float(scores[idx])
                })
        
        return similar_movies
