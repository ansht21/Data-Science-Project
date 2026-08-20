# Sentiment Analysis — Tweet Classification

> NLP pipeline that classifies social media posts as positive, negative, or neutral using TF-IDF features and logistic regression.

![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-154F5B?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

**Result: 71% accuracy · 0.70 weighted F1**

---

## Problem

Brands and product teams need to know how people feel about them at scale — reading every mention manually doesn't work past a few hundred posts. Automated sentiment classification turns a stream of unstructured text into a trackable metric.

## Dataset

499 labelled tweets with text, sentiment label, timestamp, and platform. Included at [`dataset/sentiment_analysis.csv`](./dataset/sentiment_analysis.csv).

---

## Approach

```
Raw tweet text
  ↓ Lowercase
  ↓ Strip non-alphabetic characters (regex)
  ↓ Remove English stopwords (NLTK)
  ↓ WordNet lemmatisation
  ↓ TF-IDF vectorisation (unigrams + bigrams, max 5,000 features)
  ↓ Logistic Regression (class_weight='balanced', max_iter=1000)
```

### Why each step

- **Lemmatisation over stemming** — reduces "running" to "running"→"run" while keeping real words, so features stay interpretable. Stemming would produce fragments like "runn".
- **Stopword removal** — words like "the" and "is" appear in every class and carry no sentiment signal.
- **Bigrams alongside unigrams** — "not good" carries the opposite sentiment to "good". A unigram-only model can't see negation; bigrams capture it.
- **`class_weight='balanced'`** — the classes are unevenly sized, so this penalises errors on rarer classes more heavily, preventing the model from simply predicting the majority class.
- **Stratified 80/20 split** — preserves class proportions across train and test, so the test set is representative.

---

## Results

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Negative | 0.85 | 0.41 | 0.55 | 27 |
| Neutral | 0.63 | 0.90 | 0.74 | 40 |
| Positive | 0.80 | 0.73 | 0.76 | 33 |
| **Accuracy** | | | **0.71** | 100 |
| **Macro avg** | 0.76 | 0.68 | 0.68 | 100 |
| **Weighted avg** | 0.75 | 0.71 | 0.70 | 100 |

### Honest reading of these numbers

**The model has a clear weakness on negative sentiment.** Precision is high (0.85) — when it says "negative", it's usually right. But recall is only 0.41, meaning it misses 59% of genuinely negative tweets, most of which it labels neutral.

That failure mode matters in practice: a brand-monitoring tool that misses over half of complaints is not fit for purpose, even at 71% headline accuracy. **This is why per-class metrics matter more than a single accuracy figure.**

**Root cause is most likely data volume.** With only 27 negative examples in the test set and roughly 135 in training, there simply aren't enough examples for the model to learn the vocabulary of negativity — which is more varied and often more subtle (sarcasm, understatement) than positive language.

---

## Running It

```bash
pip install pandas scikit-learn nltk
cd notebooks
jupyter notebook sentiment_analysis.ipynb
```

NLTK corpora (`stopwords`, `wordnet`) download automatically on first run. The notebook reads the dataset from `../dataset/sentiment_analysis.csv`.

The notebook includes a `predict_sentiment()` helper for classifying new text:

```python
predict_sentiment("I love this product")   # → positive
predict_sentiment("This is very bad")      # → negative
```

---

## Improvement Path

Ordered by expected impact:

1. **More data** — the clearest lever. Negative-class recall is data-starved, not model-starved.
2. **Try a linear SVM** — often outperforms logistic regression on high-dimensional sparse text features.
3. **Transformer embeddings (BERT)** — understands context and negation natively rather than relying on bigrams.
4. **Threshold tuning** — if catching complaints matters more than avoiding false alarms, lower the decision threshold for the negative class to trade precision for recall.

---

[⬅ Back to all ML projects](../README.md)
