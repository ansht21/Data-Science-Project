"""
Model training and evaluation for spam email detection.
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Tuple, Dict, Any


def train_model(X_train, y_train, 
                C: float = 1.0,
                max_iter: int = 1000,
                random_state: int = 42) -> LogisticRegression:
    """
    Train Logistic Regression model for spam detection.

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

    Returns
    -------
    LogisticRegression
        Trained model.
    """
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LogisticRegression, 
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
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
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
        Dictionary mapping feature names to coefficient values.
    """
    return dict(zip(feature_names, model.coef_[0]))


def predict_spam(model, vectorizer, text: str) -> str:
    """
    Predict if text is spam or not.

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
        Prediction: 'spam' or 'ham'.
    """
    # Vectorize text
    text_vector = vectorizer.transform([text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    return 'spam' if prediction == 1 else 'ham'
