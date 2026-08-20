"""
Data loading and preprocessing for sales prediction.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load sales data from CSV file.

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


def preprocess_data(df: pd.DataFrame, target_column: str = 'Item_Outlet_Sales'):
    """
    Preprocess features and split data into training and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_column : str, default 'Item_Outlet_Sales'
        Name of the target column.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, label_encoders, scaler)
        Where label_encoders is a dictionary of fitted LabelEncoders
        and scaler is the fitted StandardScaler.
    """
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Initialize label encoders for categorical columns
    label_encoders = {}
    X_encoded = X.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Initialize scaler for numerical columns
    scaler = StandardScaler()
    
    # Scale numerical features
    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42)
    
    return (X_train, X_test, y_train, y_test, label_encoders, scaler)


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
