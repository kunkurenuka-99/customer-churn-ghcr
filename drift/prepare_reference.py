from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

SOURCE = PROJECT_ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
]
TARGET = "Churn"

df = pd.read_csv(SOURCE)

missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df[FEATURES + [TARGET]].dropna().copy()
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

# Fixed split: the training part becomes the reference; the holdout part is
# never used for training, so candidate models can be evaluated fairly.
train_part, holdout = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df[TARGET]
)

train_part.to_csv(DRIFT_DIR / "historical_train.csv", index=False)
holdout.to_csv(DRIFT_DIR / "holdout_data.csv", index=False)
train_part[FEATURES].to_csv(DRIFT_DIR / "reference_data.csv", index=False)

print("Reference dataset created.")
print("Reference rows (training part):", len(train_part))
print("Holdout rows (evaluation only):", len(holdout))
print("Files written to:", DRIFT_DIR)
