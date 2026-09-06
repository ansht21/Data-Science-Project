<div align="center">

# Data Science Projects

### Eight end-to-end ML projects — led by ride-demand forecasting & churn retention — spanning time series, churn modeling, NLP, deep learning, regression, and recommendation systems

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br/>

[![Profile](https://img.shields.io/badge/GitLab-Profile-FC6D26?style=for-the-badge&logo=gitlab&logoColor=white)](https://gitlab.com/ansht_121)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://ansh-thakur.netlify.app/)
[![Data Analytics](https://img.shields.io/badge/GitLab-Data_Analytics-6C8CFF?style=for-the-badge&logo=gitlab&logoColor=white)](https://gitlab.com/ansht_121/data-analytics)

</div>

---

## Overview

| # | Project | Problem Type | Approach | Result |
|---|---------|--------------|----------|--------|
| 1 | [Ride Demand & Revenue Forecasting](#1--ride-demand--revenue-forecasting) | Time series · Clustering · Regression | SARIMA · K-Means · Random Forest | **R² = 0.972** fare model · **MAPE 8.3%** forecast |
| 2 | [Subscription Churn Prediction](#2--subscription-churn-prediction--retention-analytics) | Binary classification | LogReg + Random Forest + explainability | **ROC-AUC 0.839** · 80.0% accuracy |
| 3 | [Image Classification](#3--image-classification--handwritten-digit-recognition) | Multi-class classification | CNN (TensorFlow/Keras) | **98.83%** test accuracy |
| 4 | [Spam Email Detection](#4--spam-email-detection) | Binary text classification | TF-IDF + Logistic Regression | **96.32%** test accuracy |
| 5 | [Sales Prediction](#5--sales-prediction) | Regression | Linear Regression | **R² = 0.91** · MSE 2.86 |
| 6 | [Car Price Prediction](#6--car-price-prediction) | Regression | Random Forest Regressor | MAE · MSE · R² evaluated |
| 7 | [Sentiment Analysis](#7--sentiment-analysis--tweet-classification) | NLP / text classification | TF-IDF + Logistic Regression | **71%** accuracy, 0.70 weighted F1 |
| 8 | [Music Recommendation](#8--music-recommendation-system) | Recommendation | Content-based, cosine similarity | Top-K similar songs |

---

## 1 · Ride Demand & Revenue Forecasting

**Problem:** A ride-hailing operator needs to know *when* to surge-price, *where* to position premium vehicles, and *which* fleet tier to grow — 35,000 trip records, five pickup zones, one simulated year.

*Cross-listed from the [Data Analytics repo](https://gitlab.com/ansht_121/data-analytics) — the SQL / Power BI dashboard layer lives there; this folder contains the full predictive modeling layer.*

### Pipeline

```
35,000 trips (Jan–Dec)
  ↓ SQL cleaning pipeline (ROW_NUMBER() dedup safeguard, type coercion)
  ↓ SARIMA(1,1,1)(1,0,1,7) — daily demand forecast, weekly seasonality
  ↓ K-Means over zone×hour cells — demand segmentation (silhouette-selected k)
  ↓ RandomForestRegressor — route-level fare prediction
  ↓ All validated on a held-out Nov–Dec window (no leakage)
```

### Results (held-out)

| Model | Performance |
|-------|-------------|
| **SARIMA demand forecast** | MAE 7.6 · RMSE 9.6 · **MAPE 8.3%** — ~20% error reduction vs the seasonal-naive baseline (MAE 9.5) |
| **K-Means demand clusters** | k=2 · **silhouette 0.675** — premium long-haul cells (avg ₹507 / 24.9 km) vs high-volume short-trip cells (avg ₹331 / 15.6 km) |
| **RF fare prediction** | **R² = 0.972** · RMSE ₹37.4 — prediction error ~13% of a typical fare |

> **Honest read:** SARIMA residuals pass Ljung-Box (p = 0.54 / 0.86) — no leftover autocorrelation. The fare model's high R² reflects that distance and duration genuinely determine fare; the value is the *expected vs charged* fare gap as an operational lever, not the score itself.

📁 [`01_ride-demand-revenue-forecasting/`](./01_ride-demand-revenue-forecasting/)

---

## 2 · Subscription Churn Prediction & Retention Analytics

**Problem:** A subscription business (real-world telco data, 7,043 customers) is losing ~26% of customers. Predict who is at risk, why, and what revenue is on the table.

*Cross-listed from the [Data Analytics repo](https://gitlab.com/ansht_121/data-analytics) — the SQL / EDA / Excel dashboard layer lives there; this folder contains the full ML pipeline.*

### Pipeline

```
Telco Customer Churn (7,043 × 21)
  ↓ Cleaning & feature engineering (tenure bands, service counts, charge ratios)
  ↓ Churn_EDA_Analysis.ipynb — KPI baseline, segment analysis, statistical validation
  ↓ Churn_Prediction_Analysis.ipynb — Logistic Regression + Random Forest
  ↓ Modular src/ pipeline: data_processing → feature_engineering → modeling
    → evaluation → explainability → risk_scoring
  ↓ pytest suite covering the pipeline modules
```

### Results (held-out)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| **Logistic Regression** | **0.800** | 0.640 | **0.561** | 0.598 | **0.839** |
| Random Forest | 0.797 | **0.652** | 0.511 | 0.573 | 0.837 |

> **Honest read:** recall is the metric that matters here — missing a churner costs the full customer value, while a false alarm costs one retention call. Logistic Regression wins on ROC-AUC (0.839) and recall (0.561) despite Random Forest's slight precision edge. Risk scores feed a revenue-at-risk segmentation (`churn_segment_revenue.csv`) rather than stopping at a leaderboard number.

📁 [`02_subscription-retention-prediction/`](./02_subscription-retention-prediction/)

---

## 3 · Image Classification — Handwritten Digit Recognition

**Problem:** Automatically recognize handwritten digits (0–9) from images — the foundational task behind postal code sorting and cheque digitization.

**Dataset:** MNIST — 60,000 training + 10,000 test grayscale images (28×28 px), loaded directly via `keras.datasets`.

### Model Architecture

```
Input (28×28×1)
  ↓ Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)
  ↓ Conv2D(64, 3×3, ReLU)  →  MaxPooling2D(2×2)
  ↓ Conv2D(64, 3×3, ReLU)
  ↓ Flatten  →  Dense(64, ReLU)  →  Dense(10, Softmax)
```

**Optimizer:** Adam · **Loss:** Sparse categorical cross-entropy · **Epochs:** 5

### Results

| Epoch | Train Accuracy | Train Loss |
|-------|----------------|------------|
| 1 | 88.98% | 0.3449 |
| 2 | 98.46% | 0.0504 |
| 3 | 99.01% | 0.0332 |
| 4 | 99.21% | 0.0253 |
| 5 | 99.43% | 0.0185 |

> **Test accuracy: 98.83%** (loss 0.0491) — the small train/test gap indicates the model generalized well without significant overfitting.

**Key steps:** pixel normalization to `[0,1]` · channel-dimension reshaping · prediction visualization with Matplotlib · model exported to `mnist_model.h5`

📓 [`image_classification_project/imageclassification.ipynb`](./image_classification_project/imageclassification.ipynb)

---

## 4 · Spam Email Detection

**Problem:** Classify emails as spam or legitimate — the classic asymmetric-cost problem, where a spam email reaching the inbox is a mild annoyance but a legitimate email wrongly filtered could be a missed job offer.

### Pipeline

```
Raw email text
  ↓ Text preprocessing (NLTK) — cleaning, tokenization
  ↓ TF-IDF vectorization — weight terms by discriminative power
  ↓ Logistic Regression classifier
  ↓ Evaluate on held-out test set
```

### Results

| Split | Accuracy |
|-------|----------|
| Training | 96.11% |
| **Test** | **96.32%** |

> Test accuracy slightly *exceeding* training accuracy is a healthy sign — the model hasn't memorized the training set and generalizes to unseen emails. A large gap in the other direction would signal overfitting.

**Why TF-IDF over raw word counts:** it down-weights words appearing across all emails ("the", "and") and up-weights terms that actually distinguish spam from legitimate mail, so the classifier learns from signal rather than noise.

📓 [`spam_email_detection_project/`](./spam_email_detection_project/)

---

## 5 · Sales Prediction

**Problem:** Predict product sales from outlet and item attributes — MRP, outlet size, item type, and outlet location. Sales forecasting drives inventory planning: under-forecast and you stock out; over-forecast and you tie up capital in unsold goods.

### Pipeline

```
Raw sales data
  ↓ Handle missing values and duplicates
  ↓ Exploratory data analysis (distribution, correlation)
  ↓ Encode categorical variables (outlet type, item category)
  ↓ Train Linear Regression
  ↓ Evaluate — MSE, MAE, RMSE, R²
```

### Results

| Metric | Value |
|--------|-------|
| **R² Score** | **0.91** |
| Mean Squared Error | 2.86 |

> R² of 0.91 means the model explains 91% of the variance in sales. For a linear model on retail data that's strong — it suggests the relationship between MRP, outlet characteristics and sales is largely linear, so a more complex tree-based model would buy little here.

📓 [`sales_prediction_project/`](./sales_prediction_project/)

---

## 6 · Car Price Prediction

**Problem:** Estimate a used car's selling price from year, mileage, fuel type, transmission, and engine specs. Resale pricing is the core of any used-vehicle marketplace — mispricing costs either seller margin or buyer trust.

### Pipeline

```
Vehicle listings data
  ↓ Clean and inspect
  ↓ Feature engineering — derive vehicle age from year
  ↓ Encode categoricals (fuel type, transmission, seller type)
  ↓ Train Random Forest Regressor
  ↓ Evaluate — MAE, MSE, R²
```

**Why Random Forest here:** depreciation isn't linear. A car loses value fastest in its first years and the curve flattens later; mileage interacts with age rather than acting independently. A tree ensemble captures those non-linear interactions without needing them specified in advance — which is precisely why it suits this problem where Linear Regression suited sales prediction.

📓 [`car_price_prediction_project/`](./car_price_prediction_project/)

---

## 7 · Sentiment Analysis — Tweet Classification

**Problem:** Classify social media posts as positive, negative, or neutral — the core of brand monitoring and customer feedback analysis.

**Dataset:** 500 labelled tweets with text, sentiment, timestamp, and platform metadata.

### Pipeline

```
Raw text
  ↓ Lowercase  →  strip non-alphabetic characters
  ↓ Stopword removal (NLTK)  →  WordNet lemmatization
  ↓ TF-IDF vectorization (unigrams + bigrams, 5,000 features)
  ↓ Logistic Regression (class_weight='balanced', max_iter=1000)
```

Stratified 80/20 train-test split preserves class proportions.

### Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.85 | 0.41 | 0.55 | 27 |
| Neutral | 0.63 | 0.90 | 0.74 | 40 |
| Positive | 0.80 | 0.73 | 0.76 | 33 |
| **Accuracy** | | | **0.71** | 100 |
| **Weighted avg** | 0.75 | 0.71 | 0.70 | 100 |

> **Honest read:** the model is precise on negative tweets (0.85) but recalls only 41% of them — it misses many, usually labelling them neutral. With just 27 negative samples in the test set, more data is the clearest path to improvement. `class_weight='balanced'` was used specifically to counteract this imbalance.

**Includes** a `predict_sentiment()` helper for inference on unseen text.

📓 [`sentiment_Analysis_project/notebooks/sentiment_analysis.ipynb`](./sentiment_Analysis_project/notebooks/sentiment_analysis.ipynb)

---

## 8 · Music Recommendation System

**Problem:** Given a song a listener enjoys, suggest similar tracks — the content-based approach used when you lack the user-interaction history that collaborative filtering requires.

**Dataset:** 100 songs with title, artist, genre, release year, duration, listen date, and platform.

### Approach

```
Raw listening data
  ↓ Deduplicate to unique songs
  ↓ Feature engineering — aggregate listen counts per song
  ↓ One-hot encode categorical features (genre, artist)
  ↓ StandardScaler normalization
  ↓ Cosine similarity matrix across all song pairs
  ↓ recommend_song(title, top_k) → K nearest neighbours
```

**Why content-based:** it sidesteps the cold-start problem — a new song can be recommended immediately from its own attributes, with no listening history needed.

📓 [`music_recommendation_system/notebooks/music_recommendation.ipynb`](./music_recommendation_system/notebooks/music_recommendation.ipynb)

---

## Repository Structure

```
data-science-projects/
├── 01_ride-demand-revenue-forecasting/# Forecasting · SARIMA + K-Means + RF fare model
├── 02_subscription-retention-prediction/# Churn · LogReg + RF, modular pipeline with tests
│   ├── src/                            # data_processing → modeling → explainability
│   └── Churn_Prediction_Analysis.ipynb
├── image_classification_project/       # CNN · MNIST digit recognition
│   └── imageclassification.ipynb
├── spam_email_detection_project/       # NLP · TF-IDF + Logistic Regression
├── sales_prediction_project/           # Regression · Linear Regression
├── car_price_prediction_project/       # Regression · Random Forest
├── sentiment_Analysis_project/         # NLP · tweet sentiment classification
│   ├── dataset/sentiment_analysis.csv
│   └── notebooks/sentiment_analysis.ipynb
└── music_recommendation_system/        # Content-based recommender
    ├── dataset/music_dataset.csv
    └── notebooks/music_recommendation.ipynb
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
# Clone the repository
git clone https://gitlab.com/ansht_121/data-science-projects.git
cd data-science-projects

# Install dependencies
pip install numpy pandas matplotlib scikit-learn tensorflow nltk jupyter

# Launch Jupyter
jupyter notebook
```

### Running Each Project

**Ride Demand Forecasting · Subscription Churn Prediction** — each folder ships its own `requirements.txt` and README.

```bash
pip install -r 01_ride-demand-revenue-forecasting/requirements.txt
pip install -r 02_subscription-retention-prediction/requirements.txt

jupyter notebook 01_ride-demand-revenue-forecasting/Forecast_Clustering_Analysis.ipynb
jupyter notebook 02_subscription-retention-prediction/Churn_Prediction_Analysis.ipynb
# or run the modular churn pipeline end-to-end:
python 02_subscription-retention-prediction/src/run_pipeline.py --help
```

**Image Classification** — no setup needed; MNIST downloads automatically on first run.

```bash
jupyter notebook image_classification_project/imageclassification.ipynb
```

**Spam Detection · Sales Prediction · Car Price Prediction** — open the notebook in each project folder and run all cells.

```bash
jupyter notebook spam_email_detection_project/
jupyter notebook sales_prediction_project/
jupyter notebook car_price_prediction_project/
```

> **Datasets not included.** These three notebooks expect `spam_mail.csv`, `sales dataset.csv`, and `car_data.csv` respectively, which aren't committed to this repository. Place your own copy alongside the notebook, or update the `read_csv()` path in the first cell. The notebooks retain their saved outputs, so the results are reproducible reading even without the raw files.

**Sentiment Analysis** — downloads NLTK corpora on first run.

```bash
cd sentiment_Analysis_project/notebooks
jupyter notebook sentiment_analysis.ipynb
```

> The notebook loads `sentiment_analysis.csv`. Either run it from the `dataset/` folder or update the path to `../dataset/sentiment_analysis.csv`.

**Music Recommendation**

```bash
cd music_recommendation_system/notebooks
jupyter notebook music_recommendation.ipynb
```

> The notebook loads `MusicDataset.csv`. Update the path to `../dataset/music_dataset.csv` to match this repository's layout.

---

## Skills Demonstrated

| Area | Techniques |
|------|-----------|
| **Deep Learning** | CNN architecture design, convolution/pooling layers, softmax classification, Adam optimization |
| **NLP** | Text cleaning, stopword removal, lemmatization, TF-IDF with n-grams |
| **Classical ML** | Linear & logistic regression, Random Forest ensembles, class imbalance handling, stratified splitting |
| **Regression Analysis** | R², MSE, MAE, RMSE interpretation; matching model complexity to relationship structure |
| **Recommender Systems** | Content-based filtering, one-hot encoding, cosine similarity, cold-start reasoning |
| **Time Series Forecasting** | SARIMA with weekly seasonality, held-out validation, Ljung-Box residual diagnostics, seasonal-naive baselines |
| **Unsupervised Learning** | K-Means demand segmentation, silhouette-based k selection, cluster profiling |
| **Churn Modeling** | Logistic Regression vs Random Forest, ROC-AUC / precision / recall trade-offs, explainability, risk scoring |
| **Evaluation** | Accuracy, precision, recall, F1-score, per-class analysis, train/test gap interpretation |
| **Data Handling** | Pandas transformations, feature scaling, deduplication, aggregation |

---

## Future Improvements

- **Sentiment Analysis** — expand beyond 500 samples; try SVM and transformer-based models (BERT) as baselines to beat
- **Image Classification** — add data augmentation and dropout; extend to Fashion-MNIST or CIFAR-10
- **Music Recommendation** — introduce collaborative filtering as a hybrid layer once interaction data is available
- **Across all** — refactor notebooks into reusable modules, add `requirements.txt`, and pin dependency versions

---

## Author

**Ansh Thakur** — Data Analyst & Machine Learning Practitioner

[![GitLab](https://img.shields.io/badge/GitLab-ansht__121-FC6D26?style=flat&logo=gitlab&logoColor=white)](https://gitlab.com/ansht_121)
[![Portfolio](https://img.shields.io/badge/Portfolio-Live_Site-00C7B7?style=flat&logo=netlify&logoColor=white)](https://ansh-thakur.netlify.app/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ansh-thakur-5407192b4/)
[![Email](https://img.shields.io/badge/Email-D14836?style=flat&logo=gmail&logoColor=white)](mailto:ansht1194@gmail.com)
