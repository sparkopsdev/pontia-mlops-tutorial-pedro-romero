# Comandos rápidos

```bash
# Repo base
mkdir pontia-mlops-tutorial-tu-nombre-apellido
cd pontia-mlops-tutorial-tu-nombre-apellido
git init
git remote add origin git@github.com:TU_USUARIO/pontia-mlops-tutorial-tu-nombre-apellido.git

# Copia primero el repo original dentro de esta carpeta y luego copia estos archivos encima.
mkdir -p .github/workflows
cp pipelines/integration.yml .github/workflows/integration.yml 2>/dev/null || true
cp pipelines/build.yml .github/workflows/build.yml 2>/dev/null || true
cp pipelines/deploy.yml .github/workflows/deploy.yml 2>/dev/null || true

# Base inicial
git add .
git commit -m "chore: bootstrap evaluation repository"
git push -u origin main

# PR 1
git checkout -b ci/integration
# añadir .github/workflows/integration.yml
git add .github/workflows/integration.yml requirements.txt .gitignore
git commit -m "ci: add integration workflow"
git push -u origin ci/integration

# PR 2
git checkout main
git pull
git checkout -b ci/build-model
# añadir src/main.py src/evaluate.py scripts/register_model.py model_tests/test_model.py .github/workflows/build.yml
git add src/main.py src/evaluate.py scripts/register_model.py model_tests/test_model.py .github/workflows/build.yml
git commit -m "ci: add model build and registration workflow"
git push -u origin ci/build-model

# PR 3
git checkout main
git pull
git checkout -b ci/deploy-api
# añadir deployment/ .github/workflows/deploy.yml scripts/query_model.py
git add deployment .github/workflows/deploy.yml scripts/query_model.py README.md
git commit -m "ci: add azure deployment workflow"
git push -u origin ci/deploy-api
```
