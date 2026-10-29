import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

# ---------- Logger configuration (stdout so Docker and Render capture it) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("churn-api")

MODEL_FILE = "churn_model.joblib"

app = FastAPI(title="Customer Churn Prediction v3")

# ---------- Model loading with logs ----------
try:
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise


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


# ---------- Request / response logging middleware ----------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        if response.status_code >= 500:
            level = logging.ERROR
        elif response.status_code >= 400:
            level = logging.WARNING
        else:
            level = logging.INFO

        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# ---------- Validation error logging (422) ----------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = [".".join(str(part) for part in error.get("loc", [])) for error in exc.errors()]

    logger.warning(
        "event=validation_error path=%s field_locations=%s error_count=%s",
        request.url.path,
        fields,
        len(exc.errors()),
    )

    return await request_validation_exception_handler(request, exc)


# ---------- Routes ----------
@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running", "version": "v3-logging"}


@app.get("/health")
def health():
    return {"status": "ok"}


# Only add non-sensitive features here if input values must be logged.
SAFE_INPUT_FIELDS = set()  # Example: {"InternetService", "tenure"}


@app.post("/predict")
def predict(c: Customer):
    try:
        payload = c.model_dump()

        safe_input = {k: v for k, v in payload.items() if k in SAFE_INPUT_FIELDS}

        logger.info(
            "event=prediction_request feature_names=%s safe_input=%s",
            sorted(payload.keys()),
            safe_input,
        )

        df = pd.DataFrame([payload])
        pred = int(model.predict(df)[0])
        prob = float(model.predict_proba(df)[0][1])

        logger.info(
            "event=prediction_success churn=%s churn_probability=%.4f",
            "Yes" if pred == 1 else "No",
            prob,
        )

        return {"churn": "Yes" if pred == 1 else "No", "churn_probability": round(prob, 4)}

    except Exception:
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
