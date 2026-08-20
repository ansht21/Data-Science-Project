"""
Utility functions for music recommendation system.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Tuple, Dict, Any


def setup_plotting_style():
    """Set up matplotlib and seaborn plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def load_and_inspect_data(filepath: str) -> pd.DataFrame:
    """
    Load data and display basic information.

    Parameters
    ----------
    filepath : str
        Path to the data file.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    df = pd.read_csv(filepath)
    print("Dataset Shape:", df.shape)
    print("\nColumn Information:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nBasic Statistics:")
    print(df.describe(include='all'))
    return df


def plot_feature_distributions(df: pd.DataFrame,
                             numerical_cols: List[str],
                             save_path: str = None):
    """
    Plot distribution of numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    numerical_cols : list
        List of numerical column names.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3  # Ceiling division for 3 columns per row
    
    plt.figure(figsize=(15, 4 * n_rows))
    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, 3, i + 1)
        plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_categorical_distributions(df: pd.DataFrame,
                                 categorical_cols: List[str],
                                 save_path: str = None):
    """
    Plot distribution of categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    categorical_cols : list
        List of categorical column names.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    n_cols = len(categorical_cols)
    n_rows = (n_cols + 2) // 3  # Ceiling division for 3 columns per row
    
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(categorical_cols):
        plt.subplot(n_rows, 3, i + 1)
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_similarity_distribution(similarity_matrix: np.ndarray,
                               save_path: str = None):
    """
    Plot distribution of similarity scores.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices_from(similarity_matrix, k=1)
    similarities = similarity_matrix[triu_indices]
    
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(similarities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(similarities):.3f}')
    plt.axvline(np.median(similarities), color='green', linestyle='--', 
                label=f'Median: {np.median(similarities):.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_recommendation_example(similarity_matrix: np.ndarray,
                              song_titles: List[str],
                              target_song: str,
                              top_n: int = 10,
                              save_path: str = None):
    """
    Plot an example of recommendations for a target song.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Similarity matrix of shape (n_songs, n_songs).
    song_titles : list
        List of song titles.
    target_song : str
        Title of the target song.
    top_n : int, default 10
        Number of top recommendations to show.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    from .data import get_recommendations
    
    # Get recommendations
    recommendations = get_recommendations(similarity_matrix, target_song, song_titles, top_n)
    
    if not recommendations:
        print(f"No recommendations found for '{target_song}'")
        return
    
    # Prepare data for plotting
    rec_titles = [rec[0] for rec in recommendations]
    rec_scores = [rec[1] for rec in recommendations]
    
    # Create horizontal bar chart
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(rec_titles))
    plt.barh(y_pos, rec_scores, color='skyblue', edgecolor='navy')
    plt.yticks(y_pos, rec_titles)
    plt.xlabel('Similarity Score')
    plt.title(f'Top {top_n} Recommendations for: "{target_song}"')
    plt.gca().invert_yaxis()  # Show highest similarity at top
    
    # Add value labels on bars
    for i, v in enumerate(rec_scores):
        plt.text(v + 0.01, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_directories(directory_paths: list):
    """
    Create directories if they don't exist.

    Parameters
    ----------
    directory_paths : list
        List of directory paths to create.
    """
    for path in directory_paths:
        os.makedirs(path, exist_ok=True)


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    str
        Human-readable file size.
    """
    size_bytes = os.path.getsize(filepath)
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.2f} MB"
    else:
        return f"{size_bytes/(1024**3):.2f} GB"
