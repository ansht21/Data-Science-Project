"""Modeling + API smoke tests matching current src/ layout."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import TARGET
from src.data_processing import load_and_clean
from src import modeling as mdl


@pytest.fixture(scope="session")
def df():
    return load_and_clean(ROOT / "data" / "Telco_Customer_Churn.csv")


@pytest.fixture(scope="session")
def splits(df):
    return mdl.make_splits(df)


def test_split_shapes(splits):
    assert len(splits["y_train"]) + len(splits["y_test"]) == 7032


def test_cross_validation_reports_mean_and_std(splits):
    cv = mdl.cross_validate(splits, cv_folds=3)
    assert {"cv_mean_roc_auc", "cv_std", "folds"}.issubset(cv.columns)
    assert cv["cv_std"].between(0, 0.2).all()


def test_risk_band_boundaries():
    assert mdl.risk_band(0.75) == "High"
    assert mdl.risk_band(0.70) == "High"
    assert mdl.risk_band(0.50) == "Medium"
    assert mdl.risk_band(0.40) == "Medium"
    assert mdl.risk_band(0.10) == "Low"


def test_threshold_is_input_range(splits):
    model = mdl.fit_models(splits)["Random Forest"]
    prob = model.predict_proba(splits["X_test"])[:, 1]
    thr = mdl.optimize_threshold(splits["y_test"].values, prob)
    assert 0 <= thr["cost_optimal"] <= 1
    assert thr["min_cost"] >= 0


def test_revenue_at_risk_expectation(df):
    df2 = df.copy()
    df2["churn_probability"] = 1.0
    df2["risk_level"] = "High"
    rav = mdl.revenue_at_risk(df2)
    assert rav["flagged_customers"] == len(df2)
    assert round(float(rav["expected_monthly_revenue_at_risk"]), 2) == round(
        float(df2["MonthlyCharges"].sum()), 2)


def test_api_predict_smoke():
    from app.main import predict, CustomerFeatures
    out = predict(CustomerFeatures())
    assert 0 <= out["churn_probability"] <= 1
    assert out["risk_band"] in ("High", "Medium", "Low")
