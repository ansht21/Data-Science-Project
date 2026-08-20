"""
Model training and evaluation for sentiment analysis.
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Tuple, Dict, Any, List


def train_model(X_train, y_train, 
                C: float = 1.0,
                max_iter: int = 1000,
                random_state: int = 42,
                class_weight: str = None) -> LogisticRegression:
    """
    Train Logistic Regression model for sentiment analysis.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    C : float, default 1.0
        Inverse of regularization strength.
    max_iter : int, default 1000
        Maximum number of iterations.
    random_state : int, default 42
        Random seed for reproducibility.
    class_weight : str or dict, default None
        Weights associated with classes.

    Returns
    -------
    LogisticRegression
        Trained model.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, 
                   X_test, 
                   y_test) -> Dict[str, Any]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test target.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'classification_report': None  # Will be filled if needed
    }


def save_model(model: LogisticRegression, filepath: str) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    filepath : str
        Path where to save the model.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> LogisticRegression:
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model.

    Returns
    -------
    LogisticRegression
        Loaded model.
    """
    return joblib.load(filepath)


def get_feature_importances(model: LogisticRegression, 
                           feature_names: list) -> Dict[str, float]:
    """
    Get feature coefficients from trained logistic model.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    feature_names : list
        List of feature names.

    Returns
    -------
    dict
        Dictionary mapping feature names to coefficient values (for binary case)
        or average absolute coefficient (for multiclass).
    """
    if hasattr(model, 'coef_'):
        if len(model.coef_) == 2:  # Binary classification
            # For binary, we can use the difference or just one class
            coefs = model.coef_[1] - model.coef_[0]  # Difference between classes
        else:  # Multiclass
            # Use average absolute coefficient across classes
            coefs = np.mean(np.abs(model.coef_), axis=0)
        return dict(zip(feature_names, coefs))
    else:
        return {}


def predict_sentiment(model, vectorizer, text: str) -> str:
    """
    Predict sentiment for a single text input.

    Parameters
    ----------
    model : LogisticRegression
        Trained model.
    vectorizer : TfidfVectorizer
        Fitted vectorizer.
    text : str
        Input text to classify.

    Returns
    -------
    str
        Predicted sentiment class.
    """
    # Preprocess text
    from .data import preprocess_text
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    return prediction
