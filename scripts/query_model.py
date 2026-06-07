import argparse
import requests

PAYLOAD = {
    "age": 38,
    "workclass": "Private",
    "fnlwgt": 89814,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Farming-fishing",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base API URL, for example http://name.eastus.azurecontainer.io:8080")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    health = requests.get(f"{base_url}/health", timeout=15)
    print("Health:", health.status_code, health.text)
    health.raise_for_status()

    response = requests.post(f"{base_url}/predict", json=PAYLOAD, timeout=30)
    print("Prediction:", response.status_code, response.text)
    response.raise_for_status()

    metrics = requests.get(f"{base_url}/metrics", timeout=15)
    print("Metrics:\n", metrics.text)
    metrics.raise_for_status()


if __name__ == "__main__":
    main()
