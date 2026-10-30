from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

REFERENCE = DRIFT_DIR / "reference_data.csv"
CURRENT = DRIFT_DIR / "current_data.csv"
REPORT = DRIFT_DIR / "drift_report.json"

NUMERIC_FEATURES = ["tenure"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
]

P_VALUE_THRESHOLD = 0.05
MIN_DRIFTED_FEATURES = 2

reference = pd.read_csv(REFERENCE)
current = pd.read_csv(CURRENT)

results = {}
drifted_features = []

# Numeric feature: two-sample Kolmogorov-Smirnov test
for feature in NUMERIC_FEATURES:
    ref_values = reference[feature].dropna()
    cur_values = current[feature].dropna()

    statistic, p_value = ks_2samp(ref_values, cur_values)
    drift_detected = bool(p_value < P_VALUE_THRESHOLD)

    results[feature] = {
        "test": "two-sample KS",
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "reference_mean": round(float(ref_values.mean()), 4),
        "current_mean": round(float(cur_values.mean()), 4),
    }
    if drift_detected:
        drifted_features.append(feature)

# Categorical features: chi-square goodness-of-fit against reference proportions
for feature in CATEGORICAL_FEATURES:
    ref_values = reference[feature].dropna().astype(str)
    cur_values = current[feature].dropna().astype(str)

    categories = sorted(set(ref_values) | set(cur_values))
    ref_counts = ref_values.value_counts().reindex(categories, fill_value=0).astype(float)
    cur_counts = cur_values.value_counts().reindex(categories, fill_value=0).astype(float)

    ref_prop = (ref_counts + 1e-6) / (ref_counts + 1e-6).sum()
    expected = ref_prop * cur_counts.sum()

    statistic, p_value = chisquare(f_obs=cur_counts, f_exp=expected)
    drift_detected = bool(p_value < P_VALUE_THRESHOLD)

    top_ref = ref_values.value_counts(normalize=True).idxmax()
    results[feature] = {
        "test": "chi-square",
        "statistic": round(float(statistic), 6),
        "p_value": round(float(p_value), 6),
        "drift_detected": drift_detected,
        "most_common_reference": top_ref,
        "reference_share": round(float((ref_values == top_ref).mean()), 4),
        "current_share": round(float((cur_values == top_ref).mean()), 4),
    }
    if drift_detected:
        drifted_features.append(feature)

overall_drift = len(drifted_features) >= MIN_DRIFTED_FEATURES

report = {
    "methods": "KS test (numeric), chi-square test (categorical)",
    "reference_rows": int(len(reference)),
    "current_rows": int(len(current)),
    "p_value_threshold": P_VALUE_THRESHOLD,
    "min_drifted_features": MIN_DRIFTED_FEATURES,
    "drifted_feature_count": len(drifted_features),
    "drifted_features": drifted_features,
    "overall_drift": overall_drift,
    "features": results,
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
print("\nDrift report:", REPORT)
print("DECISION:", "DRIFT DETECTED" if overall_drift else "NO OVERALL DRIFT - continue monitoring")
