from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Customer Churn Prediction")
model = joblib.load("churn_model.joblib")


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(c: Customer):
    df = pd.DataFrame([c.model_dump()])
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    return {"churn": "Yes" if pred == 1 else "No",
            "churn_probability": round(prob, 4)}
