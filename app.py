import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

# ---------- Logger configuration (stdout so Docker and Render capture it) ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("churn-api")

# ---------- Prometheus metrics ----------
HTTP_REQUESTS = Counter(
    "churn_api_http_requests_total",
    "Total HTTP requests received by the Churn Prediction API",
    ["method", "path", "status"],
)

HTTP_LATENCY = Histogram(
    "churn_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
)

PREDICTIONS = Counter(
    "churn_api_predictions_total",
    "Total churn prediction attempts",
    ["result"],
)

MODEL_LOADED = Gauge(
    "churn_api_model_loaded",
    "Whether the churn prediction model is loaded (1 = loaded, 0 = not loaded)",
)

MODEL_FILE = "churn_model.joblib"

app = FastAPI(title="Customer Churn Prediction v4")

# ---------- Model loading with logs and gauge ----------
try:
    model = joblib.load(MODEL_FILE)
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    MODEL_LOADED.set(0)
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


# ---------- Request / response logging + metrics middleware ----------
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    path = request.url.path
    method = request.method
    start = time.perf_counter()

    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        duration_ms = elapsed * 1000

        # Do not count Prometheus' own scrape requests
        if path != "/metrics":
            HTTP_REQUESTS.labels(
                method=method, path=path, status=str(response.status_code)
            ).inc()
            HTTP_LATENCY.labels(method=method, path=path).observe(elapsed)

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
            method,
            path,
            response.status_code,
            duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        elapsed = time.perf_counter() - start
        duration_ms = elapsed * 1000

        if path != "/metrics":
            HTTP_REQUESTS.labels(method=method, path=path, status="500").inc()
            HTTP_LATENCY.labels(method=method, path=path).observe(elapsed)

        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            method,
            path,
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
    return {"message": "Customer Churn Prediction API is running", "version": "v4-monitoring"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


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

        PREDICTIONS.labels(result="success").inc()

        logger.info(
            "event=prediction_success churn=%s churn_probability=%.4f",
            "Yes" if pred == 1 else "No",
            prob,
        )

        return {"churn": "Yes" if pred == 1 else "No", "churn_probability": round(prob, 4)}

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
