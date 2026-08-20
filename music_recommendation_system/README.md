# Music Recommendation System

Content-based music recommendation system using song features.

## Project Structure

music_recommendation_system/
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
│   └── music_recommendation.ipynb
├── tests/
│   └── test_basic.py
├── requirements.txt
├── params.yaml
└── README.md

## Problem Statement

Build a content-based music recommendation system that suggests similar songs based on song attributes.

## Key Features

✅ Modular Codebase
✅ Reproducible Results
✅ Configuration Management (params.yaml)
✅ Unit Testing
✅ Professional Documentation
✅ Easy Installation
✅ Extensible Design

## Installation

```bash
git clone https://gitlab.com/ansht_121/machine-learning-project.git
cd machine-learning-project/music_recommendation_system
pip install -r requirements.txt
```

## Usage

### 1. Using Jupyter Notebook
```bash
jupyter notebook notebooks/music_recommendation.ipynb
```

### 2. Using Python Script
```python
import pandas as pd
from music_recommendation_system.src.data import load_data, preprocess_features, compute_similarity_matrix, get_recommendations
import yaml

# Load configuration
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

# Load and preprocess data
df = load_data(params['data']['raw_path'])
features, feature_names, _ = preprocess_features(df, 
                                              feature_columns=params['data']['feature_columns'])

# Compute similarity matrix
similarity_matrix = compute_similarity_matrix(features, 
                                            metric=params['model']['similarity_metric'])

# Get recommendations
target_song = "Shape of You"
recommendations = get_recommendations(similarity_matrix, target_song, df['title'].tolist(), 
                                    top_n=params['model']['top_n'])

print(f"Recommendations for '{target_song}':")
for i, (song, score) in enumerate(recommendations, 1):
    print(f"{i}. {song} (Similarity: {score:.3f})")
```

## Model Details

Algorithm: Content-based filtering
Pipeline:
1. Load song metadata
2. Preprocess features (normalize numeric, one-hot encode categorical)
3. Compute similarity matrix (cosine similarity)
4. Generate recommendations

Features: title, artist, genre, year, duration, listen_count
Similarity Metric: Cosine similarity (configurable)

## Data Requirements

Place dataset in `dataset/music_dataset.csv` with:
- `title`: Song title
- `artist`: Artist name
- `genre`: Music genre
- `year`: Release year
- `duration`: Song duration in seconds
- `listen_count`: Number of plays

## Extending the Project

### Trying Different Features
Modify params.yaml to change feature_columns

### Changing Similarity Metric
Update params.yaml model.similarity_metric to 'euclidean' or 'correlation'

### Adding New Features
Add columns to dataset and include in feature_columns

## References

- Scikit-learn Documentation: https://scikit-learn.org/
- Pandas Documentation: https://pandas.pydata.org/docs/
- Ricci, F., Rokach, L., & Shapira, B. (2011). "Introduction to Recommender Systems Handbook."
