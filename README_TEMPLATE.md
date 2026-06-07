# pontia-mlops-tutorial-tu-nombre-apellido

## Integrantes
- Nombre Apellido
- Nombre Apellido

## Objetivo
Este repositorio entrena, registra y despliega un modelo de clasificación Adult Income usando Python, MLflow, GitHub Actions, Docker, Azure Container Registry y Azure Container Instances.

## Estructura
- `src/`: carga de datos, preprocesamiento, entrenamiento y evaluación.
- `tests/`: tests unitarios originales del repositorio base.
- `model_tests/`: tests del artefacto entrenado.
- `scripts/register_model.py`: registra en MLflow el modelo entrenado y asigna alias.
- `deployment/`: imagen Docker y API FastAPI.
- `.github/workflows/integration.yml`: CI para pull requests.
- `.github/workflows/build.yml`: entrenamiento, tests del modelo y registro en MLflow.
- `.github/workflows/deploy.yml`: build/push de imagen y despliegue en Azure Container Instances.

## Configuración requerida en GitHub

### Secrets
- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_CREDENTIALS`
- `ACR_NAME`
- `ACR_USERNAME`
- `ACR_PASSWORD`
- `AZURE_RESOURCE_GROUP`

### Variables
- `MLFLOW_URL`
- `EXPERIMENT_NAME`
- `MODEL_NAME`
- `MODEL_ALIAS`
- `AZURE_CONTAINER_NAME`
- `IMAGE_NAME`
- `AZURE_REGION`

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -L -o data/raw/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
export MLFLOW_URL="http://localhost:5000"
export EXPERIMENT_NAME="adult-income"
export MODEL_NAME="adult-income-classifier-tu-nombre"
export MODEL_ALIAS="champion"
python src/main.py
pytest model_tests/ -q
python scripts/register_model.py
```

## API

Endpoints:
- `GET /health`
- `POST /predict`
- `GET /metrics`

Ejemplo:

```bash
python scripts/query_model.py --url http://TU_FQDN:8080
```

## Evidencia
Añadir aquí capturas o enlace a video mostrando:
1. Workflows en verde.
2. PRs con code review.
3. API respondiendo `/health`, `/predict` y `/metrics`.
4. Logs del contenedor de Azure.
