"""
Data loading and preprocessing for sentiment analysis.
"""
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os


def download_nltk_resources():
    """Download required NLTK resources."""
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        WordNetLemmatizer().lemmatize('test')
    except LookupError:
        nltk.download('wordnet')


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load sentiment analysis data from CSV file.

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


def preprocess_text(text: str) -> str:
    """
    Preprocess text for sentiment analysis.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Preprocessed text.
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Tokenize by splitting on whitespace
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join back to string
    return ' '.join(tokens)


def prepare_features(texts, max_features=5000, ngram_range=(1, 2)):
    """
    Convert text data to TF-IDF features.

    Parameters
    ----------
    texts : list or array-like
        Text data to vectorize.
    max_features : int, default 5000
        Maximum number of features (vocabulary size).
    ngram_range : tuple, default (1, 2)
        The lower and upper boundary of the range of n-values for different n-grams.

    Returns
    -------
    tuple
        (vectorizer, features) where vectorizer is the fitted TfidfVectorizer
        and features is the transformed text data.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    features = vectorizer.fit_transform(texts)
    return vectorizer, features


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and test sets.

    Parameters
    ----------
    X : array-like
        Feature data.
    y : array-like
        Target labels.
    test_size : float, default 0.2
        Proportion of data to use for test set.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


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
