"""
Utility functions for image classification (MNIST).
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, classification_report


def setup_plotting_style():
    """Set up matplotlib and seaborn plotting style."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12


def load_and_inspect_data():
    """
    Load MNIST data and display basic information.

    Returns
    -------
    tuple
        (x_train, y_train, x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print("MNIST Dataset Information:")
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    print("\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  Class {cls}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return x_train, y_train, x_test, y_test


def plot_sample_images(x, y, num_samples=10, save_path=None):
    """
    Plot sample images from the dataset.

    Parameters
    ----------
    x : array
        Images with shape (num_samples, height, width) or (num_samples, height, width, channels)
    y : array
        Labels
    num_samples : int, default 10
        Number of samples to display
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        plt.subplot(2, num_samples//2, i+1)
        if len(x.shape) == 4 and x.shape[3] == 1:
            plt.imshow(x[i].squeeze(), cmap='gray')
        else:
            plt.imshow(x[i], cmap='gray')
        plt.title(f'Label: {y[i]}')
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy and loss.

    Parameters
    ----------
    history : tf.keras.callbacks.History
        Training history
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Plot confusion matrix.

    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    class_names : list, optional
        List of class names. If None, uses 0-9 for MNIST.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print classification report.

    Parameters
    ----------
    y_true : array
        True labels
    y_pred : array
        Predicted labels
    class_names : list, optional
        List of class names. If None, uses 0-9 for MNIST.
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)


def create_directories(directory_paths):
    """
    Create directories if they don't exist.

    Parameters
    ----------
    directory_paths : list
        List of directory paths to create.
    """
    for path in directory_paths:
        os.makedirs(path, exist_ok=True)


def get_file_size(filepath):
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
