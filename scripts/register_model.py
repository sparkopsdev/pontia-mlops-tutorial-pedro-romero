import os
import time
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def get_run_id() -> str:
    env_run_id = os.getenv("RUN_ID")
    if env_run_id:
        return env_run_id.strip()
    run_id_file = Path("run_id.txt")
    if not run_id_file.exists():
        raise RuntimeError("run_id.txt not found and RUN_ID is not set")
    return run_id_file.read_text(encoding="utf-8").strip()


def wait_until_ready(client: MlflowClient, model_name: str, version: str, timeout_seconds: int = 180) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        model_version = client.get_model_version(model_name, version)
        if model_version.status == "READY":
            return
        if model_version.status == "FAILED_REGISTRATION":
            raise RuntimeError(f"Model registration failed: {model_version.status_message}")
        time.sleep(5)
    raise TimeoutError(f"Model version {model_name} v{version} was not READY after {timeout_seconds}s")


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or os.getenv("MLFLOW_URL") or "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)

    model_name = required_env("MODEL_NAME")
    model_alias = os.getenv("MODEL_ALIAS", "champion")
    run_id = get_run_id()

    model_uri = f"runs:/{run_id}/model"
    print(f"Registering {model_uri} as {model_name}")
    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    client = MlflowClient(tracking_uri=tracking_uri)
    wait_until_ready(client, model_name, result.version)
    client.set_registered_model_alias(model_name, model_alias, result.version)

    print(f"Registered {model_name} version {result.version} with alias {model_alias}")


if __name__ == "__main__":
    main()
