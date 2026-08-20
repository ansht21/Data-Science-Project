"""
Data loading and preprocessing for spam email detection.
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load spam email data from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded data.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df: pd.DataFrame, text_column: str = 'text', label_column: str = 'label'):
    """
    Preprocess features and split data into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    text_column : str, default 'text'
        Name of the text column.
    label_column : str, default 'label'
        Name of the label column.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, vectorizer)
        Where vectorizer is the fitted TfidfVectorizer.
    """
    # Separate features and target
    X = df[text_column]
    y = df[label_column]
    
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    # Transform text data
    X_vectorized = vectorizer.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42)
    
    return (X_train, X_test, y_train, y_test, vectorizer)


def save_processed_data(X_train, X_test, y_train, y_test, 
                       train_path: str, test_path: str):
    """
    Save processed data to CSV files.

    Parameters
    ----------
    X_train, X_test : arrays
        Feature data (will be converted to dense for CSV saving).
    y_train, y_test : arrays
        Target labels.
    train_path, test_path : str
        Paths to save the processed data.
    """
    # For sparse matrices, convert to dense (if needed)
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
        X_test = X_test.toarray()
    
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
