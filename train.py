import sys
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

if "Churn" not in df.columns:
    sys.exit("ERROR: no 'Churn' column in the CSV.")

y = (df["Churn"] == "Yes").astype(int)

cat_cols = ["gender", "Partner", "Dependents", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity"]
num_cols = ["SeniorCitizen", "tenure"]
X = df[cat_cols + num_cols]

pre = ColumnTransformer(
    [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
    remainder="passthrough",
)
model = Pipeline([
    ("pre", pre),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
])

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_tr, y_tr)
print("Accuracy:", round(accuracy_score(y_te, model.predict(X_te)), 4))

joblib.dump(model, "churn_model.joblib")
print("Saved churn_model.joblib")
