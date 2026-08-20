# Image Classification — Handwritten Digit Recognition

> Professional TensorFlow/Keras CNN for MNIST handwritten digit recognition.

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)

**Result: 98.83% test accuracy**

---

## Project Structure

```
image_classification_project/
├── src/                     # Source code packages
│   ├── __init__.py
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # Model building and training
│   └── utils.py             # Utility functions
├── data/                    # Data directories
│   ├── raw/                 # Raw data files (MNIST auto-downloaded)
│   └── processed/           # Processed data files
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks for exploration
│   └── image_classification.ipynb
├── tests/                   # Unit tests
│   └── test_basic.py
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Problem Statement

Recognise handwritten digits (0–9) from the MNIST dataset — a foundational computer vision task with applications in postal automation, cheque processing, and form recognition.

---

## Key Features

✅ **Modular Codebase** - Separation of concerns with dedicated modules for data, modeling, and utilities  
✅ **Reproducible Results** - Fixed random seeds and versioned dependencies  
✅ **Unit Testing** - Basic test suite to ensure code correctness  
✅ **Professional Documentation** - Clear docstrings and README  
✅ **Easy Installation** - Single requirements file for dependencies  
✅ **Extensible Design** - Easy to modify architecture or add new datasets  

---

## Installation

```bash
# Clone the repository
git clone https://gitlab.com/ansht_121/machine-learning-project.git
cd machine-learning-project/image_classification_project

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Using Jupyter Notebook (Exploration)
```bash
jupyter notebook notebooks/image_classification.ipynb
```

### 2. Using Python Script (Production)
```python
from image_classification_project.src.data import load_mnist_data, prepare_data_for_cnn
from image_classification_project.src.model import create_cnn_model, train_model, evaluate_model, save_model

# Load and preprocess data
x_train, y_train, x_test, y_test = load_mnist_data()
x_train, y_train, x_val, y_val, x_test, y_test = prepare_data_for_cnn(x_train, y_train, x_test, y_test)

# Create model
model = create_cnn_model(input_shape=(28, 28, 1), num_classes=10)

# Train model
history = train_model(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=32)

# Evaluate model
metrics = evaluate_model(model, x_test, y_test)
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
print(f"Test Loss: {metrics['test_loss']:.4f}")

# Save model
save_model(model, "models/mnist_cnn.h5")
```

---

## Model Details

**Algorithm**: Convolutional Neural Network (CNN)  
**Architecture**:
- Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)
- Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
- Conv2D(64, 3×3, ReLU)
- Flatten → Dense(64, ReLU) → Dropout(0.5) → Dense(10, Softmax)

**Why CNN for MNIST?**
- Spatial hierarchies: CNNs excel at capturing local patterns (edges, curves) and combining them into complex features
- Parameter sharing: Convolutional layers dramatically reduce parameters compared to fully connected networks
- Translation invariance: Pooling layers provide robustness to small shifts in digit position

---

## Evaluation Metrics

The model is evaluated using standard classification metrics:

| Metric | Description |
|--------|-------------|
| **Accuracy** | Proportion of correctly classified images |
| **Loss** | Categorical cross-entropy loss on test set |

---

## Running Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Or run the test file directly
python tests/test_basic.py
```

Sample test output:
```
============================= test session starts ==============================
collected 1 item

tests/test_basic.py::test_directories_exist PASSED
...

============================== 5 passed in 0.23s ==============================
```

---

## Example Output

After running the training script, you should see output similar to:
```
Epoch 1/5
1875/1875 [==============================] - 8s 4ms/step - loss: 0.3450 - accuracy: 0.8897 - val_loss: 0.0937 - val_accuracy: 0.9711
Epoch 2/5
1875/1875 [==============================] - 7s 4ms/step - loss: 0.0506 - accuracy: 0.9843 - val_loss: 0.0305 - val_accuracy: 0.9882
...
Test Accuracy: 0.9883
Test Loss: 0.0491
```

This indicates the model achieves 98.83% accuracy on the MNIST test set.

---

## Data Requirements

No manual download required — the MNIST dataset is automatically fetched and cached by TensorFlow/Keras on first run.

The dataset consists of:
- **Training set**: 60,000 grayscale images (28×28 pixels)
- **Test set**: 10,000 grayscale images (28×28 pixels)
- **Labels**: Integers from 0 to 9 representing the digit shown

---

## Extending the Project

### Changing Architecture
1. Modify `src/model.py` `create_cnn_model` function
2. Add/remove layers, change filter sizes, add dropout/batch normalization
3. Adjust input shape if needed

### Using Different Dataset
1. Update `src/data.py` to load your dataset
2. Ensure output shape matches model expectations
3. Update class count in model creation

### Hyperparameter Tuning
1. Adjust epochs, batch size, learning rate in training script
2. Consider adding learning rate schedules or early stopping

---

## References

- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Documentation: https://keras.io/
- MNIST Dataset: http://yann.lecun.com/exdb/mnist/
- LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.

---

[⬅ Back to all ML projects](../README.md)
