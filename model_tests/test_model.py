import os
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

MODEL_DIR = Path("models")
DATA_DIR = Path("data/raw")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]


def load_adult_test_data():
    test_path = DATA_DIR / "adult.test"
    assert test_path.exists(), f"Dataset not found at {test_path}"
    df = pd.read_csv(
        test_path,
        header=0,
        names=COLUMNS,
        na_values=" ?",
        skipinitialspace=True,
        skiprows=1,
    ).dropna()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    y = df["income"].apply(lambda x: 1 if x == ">50K" else 0).to_numpy()
    X = df.drop("income", axis=1)
    return X, y


def transform_input(X):
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    encoders = joblib.load(MODEL_DIR / "encoders.pkl")
    X = X.copy()
    for col, encoder in encoders.items():
        X[col] = encoder.transform(X[col])
    return scaler.transform(X)


def test_model_artifacts_exist():
    for file_name in ["model.pkl", "scaler.pkl", "encoders.pkl"]:
        assert (MODEL_DIR / file_name).exists(), f"Missing artifact: {MODEL_DIR / file_name}"


def test_model_loading():
    try:
        joblib.load(MODEL_DIR / "model.pkl")
        joblib.load(MODEL_DIR / "scaler.pkl")
        joblib.load(MODEL_DIR / "encoders.pkl")
    except Exception as exc:
        pytest.fail(f"Failed to load model artifacts: {exc}")


def test_prediction_shape_and_values():
    model = joblib.load(MODEL_DIR / "model.pkl")
    X_raw, _ = load_adult_test_data()
    X = transform_input(X_raw.head(5))
    predictions = model.predict(X)
    assert predictions.shape == (5,)
    assert set(predictions).issubset({0, 1})


def test_model_accuracy():
    model = joblib.load(MODEL_DIR / "model.pkl")
    X_raw, y = load_adult_test_data()
    X = transform_input(X_raw)
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    assert accuracy >= float(os.getenv("MIN_MODEL_ACCURACY", "0.80")), f"Accuracy too low: {accuracy:.3f}"
