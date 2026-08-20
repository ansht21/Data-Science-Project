# Car Price Prediction

> Professional Machine Learning project for predicting used car selling prices using Random Forest Regressor.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Joblib](https://img.shields.io/badge/Joblib-1.0.0+%23000000?style=flat-square&logo=joblib&logoColor=white)](https://joblib.readthedocs.io/)

*Professional implementation with modular code, unit tests, and clear documentation.*

---

## Project Structure

```
car_price_prediction/
├── src/                     # Source code packages
│   ├── __init__.py
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # Model training and evaluation
│   └── utils.py             # Utility functions
├── data/                    # Data directories
│   ├── raw/                 # Raw data files
│   └── processed/           # Processed data files
├── models/                  # Saved model files
├── notebooks/               # Jupyter notebooks for exploration
│   └── car_price_prediction.ipynb
├── tests/                   # Unit tests
│   └── test_model.py
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Problem Statement

Predict the selling price of used cars based on their specifications including year, mileage, fuel type, transmission, and engine details. Accurate pricing is crucial for both buyers and sellers in the used automobile market.

---

## Key Features

✅ **Modular Codebase** - Separation of concerns with dedicated modules for data, modeling, and utilities  
✅ **Reproducible Results** - Fixed random seeds and versioned dependencies  
✅ **Unit Testing** - Comprehensive test suite to ensure code correctness  
✅ **Professional Documentation** - Clear docstrings and README  
✅ **Easy Installation** - Single requirements file for dependencies  
✅ **Extensible Design** - Easy to add new features or models  

---

## Installation

### Option 1: Using pip (Recommended)
```bash
# Clone the repository
git clone https://gitlab.com/ansht_121/machine-learning-project.git
cd machine-learning-project/car_price_prediction

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Development Installation
```bash
pip install -e .
```

---

## Usage

### 1. Using Jupyter Notebook (Exploration)
```bash
jupyter notebook notebooks/car_price_prediction.ipynb
```

### 2. Using Python Script (Production)
```python
from car_price_prediction.src.data import load_data, preprocess_data
from car_price_prediction.src.model import train_model, evaluate_model, save_model

# Load and preprocess data
df = load_data("data/raw/car_data.csv")
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Train model
model = train_model(X_train, y_train, n_estimators=100)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print(f"Model Performance:")
print(f"  R² Score: {metrics['r2']:.3f}")
print(f"  RMSE: {metrics['rmse']:.2f}")
print(f"  MAE: {metrics['mae']:.2f}")

# Save model
save_model(model, "models/car_price_model.pkl")
```

### 3. Command Line Interface
```bash
# Train model and save to models/
python -c "
from car_price_prediction.src.data import load_data, preprocess_data
from car_price_prediction.src.model import train_model, evaluate_model, save_model
import os

# Ensure directories exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data (place your car_data.csv in data/raw/)
df = load_data('data/raw/car_data.csv')
X_train, X_test, y_train, y_test, _ = preprocess_data(df)

# Train and evaluate
model = train_model(X_train, y_train)
metrics = evaluate_model(model, X_test, y_test)
save_model(model, 'models/car_price_model.pkl')

print(f'Training Complete - R²: {metrics[\"r2\"]:.3f}')
"
```

---

## Model Details

**Algorithm**: Random Forest Regressor  
**Why Random Forest?**
- Handles non-linear relationships (car depreciation is not linear)
- Captures feature interactions (e.g., mileage vs age)
- Naturally handles categorical variables
- Robust to outliers
- Provides feature importance metrics

**Features Used**:
- Year of manufacture
- Present price
- Driven kilometers
- Fuel type
- Seller type
- Transmission
- Owner count

**Target Variable**: Selling Price

---

## Evaluation Metrics

The model is evaluated using standard regression metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Average absolute error |
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared error |
| **RMSE** | $\sqrt{MSE}$ | Error in same units as target |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Proportion of variance explained |

---

## Running Tests

```bash
# Run unit tests
python -m pytest tests/ -v

# Or run the test file directly
python tests/test_model.py
```

Sample test output:
```
============================= test session starts ==============================
collected 1 item

tests/test_model.py::test_model_training PASSED

============================== 1 passed in 0.52s ==============================
```

---

## Example Output

After running the training script, you should see output similar to:
```
Model Performance:
  R² Score: 0.91
  RMSE: 1.69
  MAE: 1.32
```

This indicates the model explains 91% of the variance in car prices, with average prediction errors of approximately ₹1.32 lakhs.

---

## Data Requirements

Place your dataset in `data/raw/car_data.csv` with the following columns:
- `Year`: Year of manufacture
- `Present_Price`: Current ex-showroom price
- `Driven_kms`: Total distance driven
- `Fuel_Type`: Petrol/Diesel/CNG
- `Selling_type`: Dealer/Individual
- `Transmission`: Manual/Automatic
- `Owner`: Number of previous owners
- `Selling_Price`: Target variable (what we're predicting)

---

## Extending the Project

### Adding New Models
1. Create a new model file in `src/` (e.g., `xgboost_model.py`)
2. Follow the same interface as `model.py`
3. Update imports in usage scripts

### Feature Engineering
1. Modify `src/data.py` preprocessing function
2. Add new feature creation steps
3. Update unit tests accordingly

### Hyperparameter Tuning
1. Add a tuning script in `notebooks/` or `src/`
2. Use scikit-learn's `GridSearchCV` or `RandomizedSearchCV`
3. Save best parameters to config

---

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Joblib Documentation: https://joblib.readthedocs.io/
- Random Forest Breiman, L. (2001). "Random Forests". Machine Learning.

---

## License

This project is part of the Machine Learning Projects repository.  
See the root [LICENSE](../LICENSE) file for details.

---

[⬅ Back to all ML projects](../README.md)
