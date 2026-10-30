from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

HISTORICAL = DRIFT_DIR / "historical_train.csv"
HOLDOUT = DRIFT_DIR / "holdout_data.csv"
RECENT = DRIFT_DIR / "current_labeled_data.csv"
DRIFT_REPORT = DRIFT_DIR / "drift_report.json"

CANDIDATE_MODEL = DRIFT_DIR / "churn_model_candidate.joblib"
METRICS_FILE = DRIFT_DIR / "retraining_metrics.json"

NUMERIC = ["SeniorCitizen", "tenure"]
CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
]
FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
]
TARGET = "Churn"

with open(DRIFT_REPORT, encoding="utf-8") as f:
    drift_report = json.load(f)

if not drift_report["overall_drift"]:
    print("No overall drift detected.")
    print("RETRAINING SKIPPED (this is correct behaviour).")
    raise SystemExit(0)

if not RECENT.exists():
    print("Drift was detected, but current_labeled_data.csv is not available.")
    print("RETRAINING BLOCKED: wait for ground-truth labels.")
    raise SystemExit(0)

historical = pd.read_csv(HISTORICAL)
recent = pd.read_csv(RECENT)
holdout = pd.read_csv(HOLDOUT)

required = FEATURES + [TARGET]
combined = pd.concat(
    [historical[required], recent[required]], ignore_index=True
).dropna(subset=[TARGET])

X_train = combined[FEATURES]
y_train = combined[TARGET].astype(int)
X_test = holdout[FEATURES]
y_test = holdout[TARGET].astype(int)

preprocess = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), NUMERIC),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), CATEGORICAL),
])

candidate = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])

candidate.fit(X_train, y_train)
accuracy = accuracy_score(y_test, candidate.predict(X_test))

joblib.dump(candidate, CANDIDATE_MODEL)

metrics = {
    "candidate_accuracy": round(float(accuracy), 6),
    "training_rows": int(len(X_train)),
    "holdout_rows": int(len(X_test)),
    "historical_rows": int(len(historical)),
    "recent_labeled_rows": int(len(recent)),
    "triggered_by_features": drift_report["drifted_features"],
}

with open(METRICS_FILE, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print(json.dumps(metrics, indent=2))
print("Candidate model:", CANDIDATE_MODEL)
