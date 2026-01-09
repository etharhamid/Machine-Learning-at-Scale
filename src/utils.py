import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict

def load_movies_data(movies_path: str) -> pd.DataFrame:
    """Load and preprocess movies data."""
    df = pd.read_csv(movies_path)
    
    # Clean data if needed
    df['title'] = df['title'].str.strip()
    df['genres'] = df['genres'].str.strip()
    
    return df

def create_rating_distribution_plot(recommendations: List[Dict]) -> go.Figure:
    """Create a bar plot of predicted ratings."""
    titles = [rec['title'][:30] + '...' if len(rec['title']) > 30 
              else rec['title'] for rec in recommendations]
    ratings = [rec['predicted_rating'] for rec in recommendations]
    
    fig = go.Figure(data=[
        go.Bar(
            x=ratings,
            y=titles,
            orientation='h',
            marker=dict(
                color=ratings,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rating")
            ),
            text=[f"{r:.2f}" for r in ratings],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Predicted Ratings for Recommended Movies",
        xaxis_title="Predicted Rating",
        yaxis_title="Movie",
        height=max(400, len(recommendations) * 40),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    return fig

def create_similarity_plot(similar_movies: List[Dict]) -> go.Figure:
    """Create a bar plot of movie similarities."""
    titles = [movie['title'][:30] + '...' if len(movie['title']) > 30 
              else movie['title'] for movie in similar_movies]
    similarities = [movie['similarity'] for movie in similar_movies]
    
    fig = go.Figure(data=[
        go.Bar(
            x=similarities,
            y=titles,
            orientation='h',
            marker=dict(
                color=similarities,
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Similarity")
            ),
            text=[f"{s:.3f}" for s in similarities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Similar Movies by Embedding Similarity",
        xaxis_title="Cosine Similarity",
        yaxis_title="Movie",
        height=max(400, len(similar_movies) * 40),
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    return fig

def create_polarization_plot(movies: List[Dict], metric_name: str = 'polarization') -> go.Figure:
    """Create a bar plot of movie polarization."""
    titles = [movie['title'][:30] + '...' if len(movie['title']) > 30 
              else movie['title'] for movie in movies[:20]]  # Show top 20
    values = [movie[metric_name] for movie in movies[:20]]
    
    fig = go.Figure(data=[
        go.Bar(
            x=values,
            y=titles,
            orientation='h',
            marker=dict(
                color=values,
                colorscale='RdYlGn_r' if metric_name == 'polarization' else 'RdYlGn',
                showscale=True,
                colorbar=dict(title="Embedding Norm")
            ),
            text=[f"{v:.3f}" for v in values],
            textposition='auto',
        )
    ])
    
    title_text = "Most Polarizing Movies" if metric_name == 'polarization' else "Most Consensus Movies"
    
    fig.update_layout(
        title=title_text,
        xaxis_title="Embedding Norm",
        yaxis_title="Movie",
        height=600,
        yaxis=dict(autorange="reversed"),
        showlegend=False
    )
    
    return fig

def format_genres(genres: str) -> str:
    """Format genre string with emojis."""
    genre_emojis = {
        'Action': '💥',
        'Adventure': '🗺️',
        'Animation': '🎨',
        'Children': '👶',
        'Comedy': '😂',
        'Crime': '🔫',
        'Documentary': '📹',
        'Drama': '🎭',
        'Fantasy': '🧙',
        'Film-Noir': '🕵️',
        'Horror': '👻',
        'Musical': '🎵',
        'Mystery': '🔍',
        'Romance': '❤️',
        'Sci-Fi': '🚀',
        'Thriller': '😱',
        'War': '⚔️',
        'Western': '🤠'
    }
    
    genre_list = genres.split('|')
    formatted = []
    for genre in genre_list:
        emoji = genre_emojis.get(genre.strip(), '🎬')
        formatted.append(f"{emoji} {genre.strip()}")
    
    return ' | '.join(formatted)
