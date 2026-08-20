"""
Model training and evaluation for music recommendation system.
"""
import numpy as np
import os
import joblib
from typing import Tuple, List, Dict, Any


def train_model(features: np.ndarray, 
               similarity_metric: str = 'cosine') -> np.ndarray:
    """
    Compute similarity matrix for music recommendation.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_songs, n_features).
    similarity_metric : str, default 'cosine'
        Similarity metric to use.

    Returns
    -------
    np.ndarray
        Similarity matrix of shape (n_songs, n_songs).
    """
    from .data import compute_similarity_matrix
    return compute_similarity_matrix(features, similarity_metric)


def evaluate_recommendations(similarity_matrix: np.ndarray,
                           song_titles: list,
                           ground_truth: dict = None,
                           top_k: int = 10) -> Dict[str, float]:
    """
    Evaluate recommendation quality (if ground truth available).

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix of shape (n_songs, n_songs).
    song_titles : list
        List of song titles.
    ground_truth : dict, optional
        Dictionary mapping song titles to lists of related songs.
    top_k : int, default 10
        Number of recommendations to consider for evaluation.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    if ground_truth is None:
        # Return basic statistics about the similarity matrix
        return {
            'mean_similarity': np.mean(similarity_matrix),
            'std_similarity': np.std(similarity_matrix),
            'max_similarity': np.max(similarity_matrix),
            'min_similarity': np.min(similarity_matrix),
            'num_songs': len(song_titles)
        }
    
    # If ground truth is provided, calculate precision@k, recall@k, etc.
    # This would require implementing recommendation retrieval and comparison
    # For now, return placeholder
    return {
        'precision_at_k': 0.0,
        'recall_at_k': 0.0,
        'f1_at_k': 0.0,
        'note': 'Ground truth evaluation not implemented'
    }


def save_model(similarity_matrix: np.ndarray, 
              song_titles: list,
              feature_names: list,
              model_dir: str = "models") -> None:
    """
    Save trained model (similarity matrix and metadata) to disk.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix of shape (n_songs, n_songs).
    song_titles : list
        List of song titles.
    feature_names : list
        List of feature names.
    model_dir : str, default "models"
        Directory to save the model.
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Save similarity matrix
    np.save(os.path.join(model_dir, "similarity_matrix.npy"), similarity_matrix)
    
    # Save song titles
    with open(os.path.join(model_dir, "song_titles.txt"), 'w') as f:
        for title in song_titles:
            f.write(title + '\n')
    
    # Save feature names
    with open(os.path.join(model_dir, "feature_names.txt"), 'w') as f:
        for fname in feature_names:
            f.write(fname + '\n')


def load_model(model_dir: str = "models") -> Tuple[np.ndarray, list, list]:
    """
    Load trained model from disk.

    Parameters
    ----------
    model_dir : str, default "models"
        Directory where the model is saved.

    Returns
    -------
    tuple
        (similarity_matrix, song_titles, feature_names)
    """
    # Load similarity matrix
    similarity_matrix = np.load(os.path.join(model_dir, "similarity_matrix.npy"))
    
    # Load song titles
    with open(os.path.join(model_dir, "song_titles.txt"), 'r') as f:
        song_titles = [line.strip() for line in f.readlines()]
    
    # Load feature names
    with open(os.path.join(model_dir, "feature_names.txt"), 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    return similarity_matrix, song_titles, feature_names


def get_recommendations(similarity_matrix: np.ndarray,
                       song_title: str,
                       song_titles: list,
                       top_n: int = 10) -> list:
    """
    Get top-N similar songs based on similarity matrix.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix of shape (n_songs, n_songs).
    song_title : str
        Title of the song to get recommendations for.
    song_titles : list
        List of all song titles in the same order as similarity matrix.
    top_n : int, default 10
        Number of recommendations to return.

    Returns
    -------
    list
        List of tuples (song_title, similarity_score) sorted by similarity descending.
    """
    from .data import get_recommendations
    return get_recommendations(similarity_matrix, song_title, song_titles, top_n)
