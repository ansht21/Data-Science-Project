from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "Telco_Customer_Churn.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
CATS = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]
CATEGORICALS = CATS
NUMERICS = ["tenure", "MonthlyCharges"]
TARGET = "Churn"
RANDOM_STATE = 42
