"""Churn scoring API. Run: uvicorn app.main:app --reload (needs models/churn_model.pkl)."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"

app = FastAPI(title="Churn Prediction API", version="1.0.0")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503,
                                detail="Model not trained. Run: python src/train.py")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class CustomerFeatures(BaseModel):
    gender: str = "Female"
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 80.0
    TotalCharges: float = 960.0


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict")
def predict(c: CustomerFeatures):
    b = get_bundle()
    row = pd.DataFrame([{**c.model_dump()}])
    row["charge_ratio"] = row["TotalCharges"] / (row["tenure"] + 1)
    row["tenure_x_monthly"] = row["tenure"] * row["MonthlyCharges"]
    row["monthly_above_avg"] = 0
    cats = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
            "PaperlessBilling", "PaymentMethod"]
    X = pd.get_dummies(row[cats + ["tenure", "MonthlyCharges", "charge_ratio",
                                   "tenure_x_monthly", "monthly_above_avg"]],
                       drop_first=True)
    X = X.reindex(columns=b["columns"], fill_value=0)
    proba = float(b["model"].predict_proba(X)[0, 1])
    thr = float(b.get("threshold", 0.5))
    band = "High" if proba >= 0.7 else ("Medium" if proba >= 0.4 else "Low")
    return {"churn_probability": round(proba, 4),
            "predicted_churn": bool(proba >= thr),
            "risk_band": band, "threshold": round(thr, 4)}
