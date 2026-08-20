"""
Utility functions for spam email detection.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict, Any


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
    print(df.describe(include='all'))
    return df


def plot_label_distribution(y: pd.Series, save_path: str = None):
    """
    Plot distribution of target variable (spam/ham).

    Parameters
    ----------
    y : pd.Series
        Target variable.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    y.value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title('Distribution of Email Labels')
    plt.xlabel('Label (0=Ham, 1=Spam)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    # Add value labels on bars
    for i, v in enumerate(y.value_counts()):
        plt.text(i, v + max(y.value_counts())*0.01, str(v), ha='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_text_length_distribution(df: pd.DataFrame, text_column: str = 'text', save_path: str = None):
    """
    Plot distribution of text lengths.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    text_column : str, default 'text'
        Name of the text column.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    text_lengths = df[text_column].astype(str).apply(len)
    
    plt.figure(figsize=(10, 6))
    plt.hist(text_lengths, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(text_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(text_lengths):.1f}')
    plt.axvline(np.median(text_lengths), color='green', linestyle='--', 
                label=f'Median: {np.median(text_lengths):.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
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
