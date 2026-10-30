from pathlib import Path
import argparse
import json
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRIFT_DIR = PROJECT_ROOT / "drift"

CANDIDATE = DRIFT_DIR / "churn_model_candidate.joblib"
METRICS = DRIFT_DIR / "retraining_metrics.json"
APPROVED_DIR = DRIFT_DIR / "approved"
APPROVED = APPROVED_DIR / "churn_model_approved.joblib"
DECISION = DRIFT_DIR / "approval_decision.json"

parser = argparse.ArgumentParser()
parser.add_argument("--min-accuracy", type=float, default=0.75)
args = parser.parse_args()

with open(METRICS, encoding="utf-8") as f:
    metrics = json.load(f)

accuracy = metrics["candidate_accuracy"]

print("Candidate accuracy:", accuracy)
print("Required accuracy :", args.min_accuracy)

decision = {
    "candidate_accuracy": accuracy,
    "required_accuracy": args.min_accuracy,
}

if accuracy < args.min_accuracy:
    decision["quality_gate"] = "FAIL"
    decision["action"] = "Candidate rejected. Current approved model unchanged."
    with open(DECISION, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)
    print("QUALITY GATE: FAIL")
    print("Current approved model is unchanged.")
    raise SystemExit(1)

APPROVED_DIR.mkdir(exist_ok=True)
shutil.copy2(CANDIDATE, APPROVED)

decision["quality_gate"] = "PASS"
decision["action"] = "Candidate approved: " + str(APPROVED.name)
with open(DECISION, "w", encoding="utf-8") as f:
    json.dump(decision, f, indent=2)

print("QUALITY GATE: PASS")
print("Candidate promoted to:", APPROVED)
