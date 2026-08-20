"""
Model building and training for image classification (MNIST).
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a Convolutional Neural Network model for MNIST classification.

    Parameters
    ----------
    input_shape : tuple, default (28, 28, 1)
        Shape of input images (height, width, channels)
    num_classes : int, default 10
        Number of output classes

    Returns
    -------
    tf.keras.Model
        Compiled CNN model
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Dropout for regularization
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model(model, x_train, y_train, x_val=None, y_val=None, 
                epochs=10, batch_size=32, verbose=1):
    """
    Train the CNN model.

    Parameters
    ----------
    model : tf.keras.Model
        Model to train
    x_train, y_train : arrays
        Training data
    x_val, y_val : arrays, optional
        Validation data
    epochs : int, default 10
        Number of training epochs
    batch_size : int, default 32
        Batch size for training
    verbose : int, default 1
        Verbosity mode

    Returns
    -------
    tf.keras.callbacks.History
        Training history
    """
    # Prepare validation data
    validation_data = (x_val, y_val) if x_val is not None and y_val is not None else None
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose
    )
    
    return history


def evaluate_model(model, x_test, y_test):
    """
    Evaluate the trained model on test data.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    x_test, y_test : arrays
        Test data

    Returns
    -------
    dict
        Dictionary containing loss and accuracy
    """
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_acc
    }


def predict_digit(model, image):
    """
    Predict digit for a single image.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    image : array
        Single image with shape (28, 28, 1) or (1, 28, 28, 1)

    Returns
    -------
    int
        Predicted digit (0-9)
    """
    # Ensure image has batch dimension
    if len(image.shape) == 3:
        image = np.expand_dims(image, axis=0)
    
    # Get prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    
    return predicted_class


def save_model(model, filepath):
    """
    Save trained model to disk.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model
    filepath : str
        Path where to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    model.save(filepath)


def load_model(filepath):
    """
    Load trained model from disk.

    Parameters
    ----------
    filepath : str
        Path to the saved model

    Returns
    -------
    tf.keras.Model
        Loaded model
    """
    return tf.keras.models.load_model(filepath)


def get_model_summary(model):
    """
    Get string representation of model architecture.

    Parameters
    ----------
    model : tf.keras.Model
        Model to summarize

    Returns
    -------
    str
        Model summary as string
    """
    import io
    import sys
    
    # Capture model summary
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    model.summary()
    sys.stdout = old_stdout
    
    return buffer.getvalue()
