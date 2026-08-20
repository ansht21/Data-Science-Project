"""
Data loading and preprocessing for image classification (MNIST).
"""
import tensorflow as tf
import numpy as np
import os


def load_mnist_data():
    """
    Load and preprocess MNIST dataset.

    Returns
    -------
    tuple
        (x_train, y_train, x_test, y_test) where:
        - x_train, x_test: normalized image arrays with shape (num_samples, 28, 28, 1)
        - y_train, y_test: integer label arrays
    """
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    return x_train, y_train, x_test, y_test


def prepare_data_for_cnn(x_train, y_train, x_test, y_test, validation_split=0.1):
    """
    Prepare data for CNN training with optional validation split.

    Parameters
    ----------
    x_train, y_train, x_test, y_test : arrays
        Data from load_mnist_data()
    validation_split : float, default 0.1
        Fraction of training data to use for validation

    Returns
    -------
    tuple
        (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    # Calculate validation set size
    val_size = int(len(x_train) * validation_split)
    
    # Shuffle training data
    indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[indices]
    y_train_shuffled = y_train[indices]
    
    # Split into training and validation
    x_val = x_train_shuffled[:val_size]
    y_val = y_train_shuffled[:val_size]
    x_train_final = x_train_shuffled[val_size:]
    y_train_final = y_train_shuffled[val_size:]
    
    return x_train_final, y_train_final, x_val, y_val, x_test, y_test


def save_preprocessed_data(x_train, y_train, x_val, y_val, x_test, y_test, 
                          train_path, val_path, test_path):
    """
    Save preprocessed data to numpy files.

    Parameters
    ----------
    x_train, y_train, x_val, y_val, x_test, y_test : arrays
        Data arrays to save
    train_path, val_path, test_path : str
        Paths to save the data
    """
    np.savez(train_path, x=x_train, y=y_train)
    np.savez(val_path, x=x_val, y=y_val)
    np.savez(test_path, x=x_test, y=y_test)


def load_preprocessed_data(train_path, val_path, test_path):
    """
    Load preprocessed data from numpy files.

    Parameters
    ----------
    train_path, val_path, test_path : str
        Paths to the saved data

    Returns
    -------
    tuple
        (x_train, y_train, x_val, y_val, x_test, y_test)
    """
    train_data = np.load(train_path)
    val_data = np.load(val_path)
    test_data = np.load(test_path)
    
    return (train_data['x'], train_data['y'], 
            val_data['x'], val_data['y'],
            test_data['x'], test_data['y'])
