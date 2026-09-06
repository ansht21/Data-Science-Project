# Sentiment Analysis — Tweet Classification

Professional NLP pipeline for tweet sentiment classification using TF-IDF and Logistic Regression.

Result: 71% accuracy, 0.70 weighted F1

## Project Structure

sentiment_Analysis_project/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── model.py
│   └── utils.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── dataset/
├── notebooks/
│   └── sentiment_analysis.ipynb
├── tests/
│   └── test_basic.py
├── requirements.txt
└── README.md

## Problem Statement

Classify social media posts as positive, negative, or neutral.

## Key Features

✅ Modular Codebase
✅ Reproducible Results
✅ Unit Testing
✅ Professional Documentation
✅ Easy Installation
✅ Extensible Design

## Installation

```bash
git clone https://gitlab.com/ansht_121/data-science-projects.git
cd data-science-projects/sentiment_Analysis_project
pip install -r requirements.txt
```

## Usage

### 1. Using Jupyter Notebook
```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### 2. Using Python Script
```python
import pandas as pd
from sentiment_Analysis_project.src.data import load_data, preprocess_text, prepare_features, split_data
from sentiment_Analysis_project.src.model import train_model, evaluate_model, save_model, predict_sentiment

# Load data
df = load_data("dataset/sentiment_analysis.csv")

# Preprocess text
df['processed_text'] = df['text'].apply(preprocess_text)

# Prepare features
vectorizer, X = prepare_features(df['processed_text'].tolist(), max_features=5000, ngram_range=(1, 2))
y = df['sentiment'].values

# Split data
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# Train model
model = train_model(X_train, y_train, class_weight='balanced', max_iter=1000)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1-Score (weighted): {metrics['f1_score']:.3f}")

# Save model and vectorizer
save_model(model, "models/sentiment_model.pkl")
import joblib
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Example prediction
sample_text = "I love this product! It's amazing!"
prediction = predict_sentiment(model, vectorizer, sample_text)
print(f"Sentiment for '{sample_text}': {prediction}")
```

## Model Details

Algorithm: Logistic Regression with TF-IDF features
Pipeline:
1. Text preprocessing (lowercase, remove non-alphabetic, stopword removal, lemmatization)
2. TF-IDF vectorization (unigrams + bigrams, 5,000 features)
3. Logistic Regression classifier (with class_weight='balanced')

Why This Approach?
- TF-IDF over raw counts: Better feature weighting
- NLP preprocessing: Improves feature quality
- Logistic Regression: Interpretable and fast
- Class weighting: Handles class imbalance

## Evaluation Metrics

- Accuracy: Proportion of correctly classified tweets
- Precision: Of predicted positives, how many were actually positive
- Recall: Of actual positives, how many were correctly predicted
- F1-Score: Harmonic mean of precision and recall
- Confusion Matrix: Detailed breakdown of predictions per class

## Running Tests

```bash
python -m pytest tests/ -v
```

## Example Output

After running the training script:
```
Accuracy: 0.710
F1-Score (weighted): 0.700
```

This indicates the model achieves 71% accuracy on the test set.

## Data Requirements

Place dataset in `dataset/sentiment_analysis.csv` with:
- `text`: The tweet/text to analyze
- `sentiment`: Sentiment label (positive, negative, neutral)

## Extending the Project

### Trying Different Models
Replace LogisticRegression with SVM, Naive Bayes, or Random Forest.

### Improving Text Preprocessing
Modify `src/data.py` `preprocess_text` function for better handling of social media text.

### Feature Engineering
Adjust ngram_range, max_features, or add additional features.

### Hyperparameter Tuning
Adjust Logistic Regression parameters (C, penalty, solver, class_weight).

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- NLTK Documentation: https://www.nltk.org/
- TF-IDF: https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Pang, B., Lee, L., & Vaithyanathan, S. (2002). "Thumbs up? Sentiment Classification using Machine Learning Techniques." EMNLP.

[⬅ Back to all ML projects](../README.md)
