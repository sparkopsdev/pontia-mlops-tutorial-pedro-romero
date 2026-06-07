import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse

FEATURE_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model-api")

model = None
scaler = None
encoders = None
metrics = {
    "prediction_requests_total": 0,
    "prediction_errors_total": 0,
    "prediction_latency_seconds_sum": 0.0,
}


def env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def build_model_uri() -> str:
    explicit_model_uri = env("MODEL_URI")
    if explicit_model_uri:
        return explicit_model_uri
    model_name = env("MODEL_NAME")
    model_alias = env("MODEL_ALIAS", "champion")
    if not model_name:
        raise RuntimeError("Set MODEL_URI or MODEL_NAME/MODEL_ALIAS")
    return f"models:/{model_name}@{model_alias}"


def run_id_from_model_uri(model_uri: str) -> str:
    if model_uri.startswith("models:/"):
        client = mlflow.MlflowClient()
        name_and_alias = model_uri.replace("models:/", "", 1)
        if "@" not in name_and_alias:
            raise RuntimeError("MODEL_URI with registry must use alias syntax: models:/name@alias")
        model_name, alias = name_and_alias.split("@", 1)
        version_info = client.get_model_version_by_alias(model_name, alias)
        return version_info.run_id
    if model_uri.startswith("runs:/"):
        return model_uri.replace("runs:/", "", 1).split("/", 1)[0]
    raise RuntimeError("Unsupported MODEL_URI. Use models:/name@alias or runs:/run_id/model")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, encoders

    tracking_uri = env("MLFLOW_TRACKING_URI") or env("MLFLOW_URL", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = build_model_uri()

    logger.info("Loading model from %s", model_uri)
    model = mlflow.pyfunc.load_model(model_uri)

    run_id = run_id_from_model_uri(model_uri)
    scaler_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/scaler.pkl")
    encoders_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="preprocessing/encoders.pkl")

    scaler = joblib.load(scaler_path)
    encoders = joblib.load(encoders_path)
    logger.info("Model and preprocessing artifacts loaded")
    yield


app = FastAPI(title="Adult Income Model API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    missing = [column for column in FEATURE_COLUMNS if column not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing fields: {missing}")
    df = pd.DataFrame([{column: payload[column] for column in FEATURE_COLUMNS}])
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col])
    return pd.DataFrame(scaler.transform(df), columns=FEATURE_COLUMNS)


@app.post("/predict")
async def predict(request: Request) -> dict[str, list[Any]]:
    start = time.time()
    metrics["prediction_requests_total"] += 1
    try:
        payload = await request.json()
        df = payload_to_dataframe(payload)
        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except HTTPException:
        metrics["prediction_errors_total"] += 1
        raise
    except Exception as exc:
        metrics["prediction_errors_total"] += 1
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        metrics["prediction_latency_seconds_sum"] += time.time() - start


@app.get("/metrics", response_class=PlainTextResponse)
def metrics_endpoint() -> str:
    lines = [
        "# HELP prediction_requests_total Total prediction requests.",
        "# TYPE prediction_requests_total counter",
        f"prediction_requests_total {metrics['prediction_requests_total']}",
        "# HELP prediction_errors_total Total prediction errors.",
        "# TYPE prediction_errors_total counter",
        f"prediction_errors_total {metrics['prediction_errors_total']}",
        "# HELP prediction_latency_seconds_sum Total prediction latency in seconds.",
        "# TYPE prediction_latency_seconds_sum counter",
        f"prediction_latency_seconds_sum {metrics['prediction_latency_seconds_sum']:.6f}",
    ]
    return "\n".join(lines) + "\n"
