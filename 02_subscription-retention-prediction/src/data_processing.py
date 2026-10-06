"""Load + clean Telco churn CSV. Mirrors Churn_EDA_Analysis.ipynb logic."""
import pandas as pd
from src.config import CATEGORICALS, NUMERICS, TARGET


def load_and_clean(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    for c in CATEGORICALS:
        df[c] = df[c].astype(str).str.strip()
    return df
