"""
Model training and evaluation for car price prediction.
"""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict, Any


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                n_estimators: int = 100, 
                max_depth: int = None,
                random_state: int = 42) -> RandomForestRegressor:
    """
    Train Random Forest Regressor model.

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training target.
    n_estimators : int, default 100
        Number of trees in the forest.
    max_depth : int, default None
        Maximum depth of the tree.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    RandomForestRegressor
        Trained model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestRegressor, 
                   X_test: np.ndarray, 
                   y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance.

    Parameters
    ----------
    model : RandomForestRegressor
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


def save_model(model: RandomForestRegressor, filepath: str) -> None:
    """
    Save trained model to disk.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model.
    filepath : str
        Path where to save the model.
    """
    joblib.dump(model, filepath)


def load_model(filepath: str) -> RandomForestRegressor:
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model.

    Returns
    -------
    RandomForestRegressor
        Loaded model.
    """
    return joblib.load(filepath)


def get_feature_importances(model: RandomForestRegressor, 
                           feature_names: list) -> Dict[str, float]:
    """
    Get feature importances from trained model.

    Parameters
    ----------
    model : RandomForestRegressor
        Trained model.
    feature_names : list
        List of feature names.

    Returns
    -------
    dict
        Dictionary mapping feature names to importance scores.
    """
    importances = model.feature_importances_
    return dict(zip(feature_names, importances))
