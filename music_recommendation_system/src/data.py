"""
Data loading and preprocessing for music recommendation system.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(filepath):
    """Load music dataset from CSV file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    return pd.read_csv(filepath)


def preprocess_features(df, feature_columns=None):
    """Preprocess features for music recommendation."""
    df_features = df.copy()
    if feature_columns is None:
        exclude_cols = ['title', 'artist']
        feature_columns = [col for col in df.columns if col not in exclude_cols]
    X = df_features[feature_columns].copy()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    processed_features = preprocessor.fit_transform(X)
    if len(categorical_cols) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols)
        feature_names = np.concatenate([numerical_cols, cat_feature_names])
    else:
        feature_names = np.array(numerical_cols)
    return processed_features, feature_names, preprocessor


def compute_similarity_matrix(features, metric='cosine'):
    """Compute similarity matrix between all pairs of songs."""
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    if metric == 'cosine':
        return cosine_similarity(features)
    elif metric == 'euclidean':
        distances = euclidean_distances(features)
        mean_dist = np.mean(distances)
        if mean_dist > 0:
            gamma = 1.0 / (2 * mean_dist**2)
            return np.exp(-gamma * distances**2)
        else:
            return np.ones_like(distances)
    elif metric == 'correlation':
        similarity_matrix = np.corrcoef(features)
        return np.nan_to_num(similarity_matrix, nan=0.0)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Choose from 'cosine', 'euclidean', 'correlation'.")


def get_recommendations(similarity_matrix, song_title, song_titles, top_n=10):
    """Get top-N similar songs based on similarity matrix."""
    try:
        song_idx = song_titles.index(song_title)
    except ValueError:
        raise ValueError(f"Song '{song_title}' not found in the dataset.")
    sim_scores = similarity_matrix[song_idx]
    song_similarities = list(enumerate(sim_scores))
    song_similarities.sort(key=lambda x: x[1], reverse=True)
    song_similarities = song_similarities[1:top_n+1]
    return [(song_titles[idx], score) for idx, score in song_similarities]


def save_processed_data(features, feature_names, song_titles, similarity_matrix, 
                       features_path, titles_path, similarity_path):
    """Save processed data to disk."""
    np.save(features_path, features)
    with open(titles_path, 'w') as f:
        for title in song_titles:
            f.write(title + '\n')
    with open(features_path.replace('.npy', '_features.txt'), 'w') as f:
        for fname in feature_names:
            f.write(fname + '\n')
    np.save(similarity_path, similarity_matrix)


def load_processed_data(features_path, titles_path, similarity_path):
    """Load processed data from disk."""
    features = np.load(features_path)
    with open(titles_path, 'r') as f:
        song_titles = [line.strip() for line in f.readlines()]
    feature_names_path = features_path.replace('.npy', '_features.txt')
    with open(feature_names_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    similarity_matrix = np.load(similarity_path)
    return features, feature_names, song_titles, similarity_matrix
