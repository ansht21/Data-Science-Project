"""
Utility functions for car price prediction.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
import os


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
    print(df.describe())
    return df


def plot_target_distribution(y: pd.Series, save_path: str = None):
    """
    Plot distribution of target variable.

    Parameters
    ----------
    y : pd.Series
        Target variable.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Selling Price')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.boxplot(y)
    plt.title('Box Plot of Target Variable')
    plt.ylabel('Selling Price')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix of numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    corr_matrix = numerical_df.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numerical Features')
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
