import logging
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from src.data_loader import load_data, preprocess_data
from src.evaluate import evaluate
from src.model import train_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger("adult-income")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def main() -> None:
    script_start = time.time()
    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    mlflow_tracking_uri = _env("MLFLOW_TRACKING_URI") or _env("MLFLOW_URL", "http://localhost:5000")
    experiment_name = _env("EXPERIMENT_NAME", "adult-income")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("System info: %s", platform.platform())
    logger.info("MLflow tracking URI: %s", mlflow_tracking_uri)
    logger.info("MLflow experiment: %s", experiment_name)

    train_df, test_df = load_data(DATA_DIR / "adult.data", DATA_DIR / "adult.test")
    X_train, X_test, y_train, y_test, scaler, encoders = preprocess_data(train_df, test_df)

    # We log the model ourselves so the registry path is stable: runs:/<run_id>/model
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name=run_name) as run:
        start_time = time.time()
        model = train_model(X_train, y_train)
        elapsed = time.time() - start_time
        logger.info("Model training complete. Time taken: %.2f seconds", elapsed)

        accuracy, report = evaluate(model, X_test, y_test)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("training_seconds", float(elapsed))
        mlflow.log_text(report, "reports/classification_report.txt")

        model_path = MODEL_DIR / "model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"
        encoders_path = MODEL_DIR / "encoders.pkl"
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(encoders, encoders_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        mlflow.log_artifact(encoders_path, artifact_path="preprocessing")
        mlflow.log_artifact(model_path, artifact_path="local_model")
        mlflow.log_artifact("training.log")

        run_id = run.info.run_id
        Path("run_id.txt").write_text(run_id, encoding="utf-8")
        logger.info("Saved run_id: %s", run_id)

    total_time = time.time() - script_start
    logger.info("Script completed in %.2f seconds.", total_time)


if __name__ == "__main__":
    main()
