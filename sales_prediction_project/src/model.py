"""
Model training and evaluation for sales prediction.
"""
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict, Any


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Train Linear Regression model.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.

    Returns
    -------
    LinearRegression
        Trained model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : LinearRegression
        Trained model.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test target.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }


def save_model(model: LinearRegression, filepath: str) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : LinearRegression
        Trained model.
    filepath : str
        Path where to save the model.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> LinearRegression:
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model.

    Returns
    -------
    LinearRegression
        Loaded model.
    """
    return joblib.load(filepath)


def get_feature_importances(model: LinearRegression, 
                           feature_names: list) -> Dict[str, float]:
    """
    Get feature coefficients from trained linear model.

    Parameters
    ----------
    model : LinearRegression
        Trained model.
    feature_names : list
        List of feature names.

    Returns
    -------
    dict
        Dictionary mapping feature names to coefficient values.
    """
    return dict(zip(feature_names, model.coef_))
