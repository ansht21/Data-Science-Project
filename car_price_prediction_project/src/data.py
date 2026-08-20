"""
Data loading and preprocessing for car price prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load car price data from CSV file.

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


def preprocess_data(df: pd.DataFrame, target_column: str = 'Selling_Price'):
    """
    Preprocess features and split data into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_column : str, default 'Selling_Price'
        Name of the target column.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor)
        Where preprocessor is the fitted ColumnTransformer.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessing pipelines
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Fit the preprocessor on training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    return (X_train_processed, X_test_processed, y_train, y_test, preprocessor)


def save_processed_data(X_train, X_test, y_train, y_test, 
                        train_path: str, test_path: str):
    """
    Save processed data to CSV files.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : arrays
        Processed data.
    train_path, test_path : str
        Paths to save the processed data.
    """
    train_df = pd.DataFrame(X_train)
    train_df['target'] = y_train
    test_df = pd.DataFrame(X_test)
    test_df['target'] = y_test
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
