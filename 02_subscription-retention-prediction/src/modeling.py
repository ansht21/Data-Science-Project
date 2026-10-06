"""Model training, CV, threshold + revenue-at-risk (matches notebook + tests)."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from src.config import TARGET, RANDOM_STATE
from src.data_processing import load_and_clean
from src.feature_engineering import build_features


def make_splits(df: pd.DataFrame, test_size=0.2):
    X = build_features(df)
    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    return {"X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte,
            "columns": list(X.columns)}


def fit_models(splits: dict):
    Xtr, ytr = splits["X_train"], splits["y_train"]
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    logreg = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(Xtr_s, ytr)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5,
                                random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr, ytr)
    return {"Logistic Regression": logreg, "Random Forest": rf, "scaler": scaler}


def evaluate(name, model, X_test, y_test, scaled=False, scaler=None):
    Xt = scaler.transform(X_test) if scaled else X_test
    pred = model.predict(Xt)
    prob = model.predict_proba(Xt)[:, 1]
    return {"model": name, "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
            "roc_auc": roc_auc_score(y_test, prob),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist()}


def cross_validate(splits: dict, cv_folds=5):
    X, y = splits["X_train"], splits["y_train"]
    rows = []
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    for name, est in [("Logistic Regression", LogisticRegression(max_iter=5000)),
                      ("Random Forest", RandomForestClassifier(n_estimators=200,
                       min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1))]:
        scores = cross_val_score(est, X, y, cv=skf, scoring="roc_auc")
        rows.append({"model": name, "cv_mean_roc_auc": float(scores.mean()),
                     "cv_std": float(scores.std()), "folds": cv_folds})
    return pd.DataFrame(rows)


def optimize_threshold(y_true, proba):
    best, best_cost, thrs = 0.5, float("inf"), np.arange(0.1, 0.9, 0.01)
    for t in thrs:
        pred = (proba >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        cost = fp * 1.0 + fn * 5.0  # missed churner costs 5x a false alarm
        if cost < best_cost:
            best, best_cost = t, cost
    return {"cost_optimal": float(best), "min_cost": float(best_cost)}


def risk_band(p: float) -> str:
    if p >= 0.7:
        return "High"
    if p >= 0.4:
        return "Medium"
    return "Low"


def revenue_at_risk(df: pd.DataFrame):
    flagged = df[df["risk_level"] != "Low"] if "risk_level" in df.columns else df
    exp_risk = float((flagged["churn_probability"] * flagged["MonthlyCharges"]).sum()
                     if "churn_probability" in flagged.columns else 0.0)
    return {"flagged_customers": int(len(flagged)),
            "expected_monthly_revenue_at_risk": round(exp_risk, 2)}
