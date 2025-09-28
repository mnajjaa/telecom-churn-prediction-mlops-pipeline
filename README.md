# Telecom Churn Prediction MLOps Pipeline

## 📌 Project Overview
This repository implements an end-to-end MLOps workflow for predicting telecom customer churn. The project combines robust data preprocessing, automated training and evaluation, experiment tracking, containerised deployment, continuous integration, and operational monitoring so that data scientists and DevOps engineers can collaborate on a production-ready churn prediction service.

## 🧱 Repository Structure
- `model_pipeline.py` / `model_pipeline1.py` – Feature engineering, model training, evaluation utilities, and model persistence helpers used throughout the workflow.【F:model_pipeline.py†L1-L88】【F:model_pipeline1.py†L1-L92】
- `main.py` – Lightweight script that prepares the data, trains a model, and logs metrics/models to MLflow for quick experimentation.【F:main.py†L1-L31】
- `main1.py` – Production-grade entry point orchestrated by Jenkins that trains or evaluates the model, logs artefacts to MLflow, and forwards metrics to Elasticsearch.【F:main1.py†L1-L85】
- `app.py` – FastAPI service exposing prediction, health-check, and retraining endpoints backed by the persisted churn model.【F:app.py†L1-L56】
- `docker-compose.yml` – Multi-service stack for MLflow tracking, PostgreSQL storage, Elasticsearch/Kibana observability, and the model-serving API.【F:docker-compose.yml†L1-L52】
- `jenkinsfile` – CI/CD pipeline that provisions dependencies, runs tests, trains the model, validates outputs, restarts the serving container, and archives artefacts.【F:jenkinsfile†L1-L206】
- `Makefile` – Automation entry points for installing dependencies, running the pipeline, launching the API, starting MLflow, and building Docker images.【F:Makefile†L1-L62】
- `test_pipeline.py` – Pytest-based unit tests covering data preparation, model training, evaluation, and persistence workflows.【F:test_pipeline.py†L1-L38】

## 🔧 Key Tools and How They Are Used
| Tool | Role in the Pipeline | Why It Matters |
| ---- | -------------------- | -------------- |
| **Python 3 & scikit-learn** | Powers the RandomForest churn model plus preprocessing logic within the training scripts.【F:model_pipeline.py†L1-L88】【F:model_pipeline1.py†L1-L92】 | Offers a mature ML ecosystem with reliable tabular modelling capabilities.
| **Pandas & NumPy** | Handle CSV ingestion, feature manipulation, and numerical arrays across the pipeline scripts.【F:model_pipeline1.py†L1-L48】 | Enable expressive data wrangling and vectorised computation for telecom metrics.
| **MLflow** | Tracks experiments, parameters, metrics, and serialised models in both experimentation (`main.py`) and production runs (`main1.py`).【F:main.py†L1-L31】【F:main1.py†L36-L84】 | Ensures reproducibility, experiment comparison, and centralised model registry.
| **Joblib** | Saves and loads trained RandomForest models for reuse by the API and evaluation workflows.【F:model_pipeline.py†L68-L88】【F:model_pipeline1.py†L70-L92】 | Provides fast serialisation for scikit-learn estimators.
| **FastAPI** | Serves real-time predictions, health checks, and retraining endpoints through `app.py`.【F:app.py†L1-L56】 | Enables high-performance asynchronous APIs suitable for deployment.
| **Docker & docker-compose** | Bundle the application, MLflow UI, PostgreSQL, Elasticsearch, and Kibana into reproducible containers for local or CI execution.【F:docker-compose.yml†L1-L52】 | Guarantees consistent runtime environments across teams and stages.
| **Jenkins** | Automates CI/CD: dependency setup, testing, training, validation, service restarts, and artefact archiving via `jenkinsfile`.【F:jenkinsfile†L1-L206】 | Provides reliable build automation and deployment governance.
| **Elasticsearch & Kibana** | Receive metrics logged from `main1.py` and expose them through Kibana dashboards for monitoring model health.【F:main1.py†L24-L85】【F:docker-compose.yml†L4-L52】 | Deliver observability for tracking accuracy, precision, recall, and F1 over time.
| **PostgreSQL** | Acts as the backing store for the serving layer to persist predictions or metadata when required by the Flask/FastAPI service.【F:docker-compose.yml†L27-L39】 | Supplies durable storage for downstream analytics and auditing.
| **Pytest** | Validates data preparation, training, evaluation, and model persistence routines to catch regressions early.【F:test_pipeline.py†L1-L38】 | Supports automated, repeatable testing integral to CI pipelines.

## 🔄 Pipeline Walkthrough
1. **Data Ingestion & Preprocessing**  
   `prepare_data` merges training and test datasets, removes uninformative columns, encodes categorical variables, normalises numeric features, and optionally filters outliers to ensure clean inputs for modelling.【F:model_pipeline.py†L17-L66】【F:model_pipeline1.py†L11-L52】

2. **Model Training**  
   `train_model` fits a `RandomForestClassifier` on the prepared feature set, chosen for its robustness to feature scaling and interpretability.【F:model_pipeline.py†L68-L74】【F:model_pipeline1.py†L54-L70】

3. **Model Evaluation**  
   `evaluate_model` generates accuracy, precision, recall, and F1 metrics that are logged to MLflow and Elasticsearch for tracking and monitoring.【F:model_pipeline.py†L76-L86】【F:main1.py†L56-L85】

4. **Experiment Tracking**  
   `main.py` and `main1.py` register parameters, metrics, models, and an input example with MLflow so experiments remain reproducible and shareable.【F:main.py†L11-L31】【F:main1.py†L36-L84】

5. **Model Persistence & Serving**  
   Trained models are persisted via Joblib and consumed by the FastAPI application (`app.py`) to service prediction, health, and retraining requests.【F:model_pipeline1.py†L70-L92】【F:app.py†L1-L56】

6. **CI/CD Automation**  
   Jenkins orchestrates the end-to-end workflow: starting containers, running tests, executing training, verifying artefacts, and restarting the serving stack to deploy new models.【F:jenkinsfile†L1-L206】

7. **Observability & Monitoring**  
   Metrics forwarded from `main1.py` are stored in Elasticsearch, visualised through Kibana, and complemented by MLflow’s experiment UI for comprehensive monitoring.【F:main1.py†L24-L85】【F:docker-compose.yml†L4-L52】

## 🚀 Getting Started
1. **Clone & Create a Virtual Environment**
   ```bash
   git clone <repo-url>
   cd telecom-churn-prediction-mlops-pipeline
   make install
   ```

2. **Run the Training Pipeline Locally**
   ```bash
   make run
   ```
   This executes `main.py`, training a RandomForest model and logging the run to MLflow.【F:Makefile†L15-L21】【F:main.py†L1-L31】

3. **Execute Unit Tests**
   ```bash
   make test
   ```
   The tests validate data preparation, training, evaluation, and model persistence flows.【F:Makefile†L23-L34】【F:test_pipeline.py†L1-L38】

4. **Launch the FastAPI Service**
   ```bash
   make api
   ```
   The service loads the persisted model (`churn_model.pkl`) to expose `/predict`, `/healthcheck`, and `/retrain` endpoints.【F:Makefile†L44-L52】【F:app.py†L1-L56】

5. **Start MLflow UI**
   ```bash
   make mlflow-ui
   ```
   Access the tracking dashboard at `http://127.0.0.1:5000` for local experiments.【F:Makefile†L54-L58】

## 🐳 Containerised Deployment
1. Build and run the container locally:
   ```bash
   make docker-build
   make docker-run
   ```
   These commands wrap the FastAPI app into a Docker image and expose it on port 8005.【F:Makefile†L60-L70】

2. For the full stack (MLflow, database, observability, API) run:
   ```bash
   docker-compose up -d
   ```
   This launches all services defined in `docker-compose.yml` with a shared network and persistent storage for PostgreSQL.【F:docker-compose.yml†L1-L52】

## 🤖 Jenkins Automation
The `jenkinsfile` provisions Python dependencies, manages Docker services, runs Pytest, trains and logs the model via `main1.py`, validates the exported artefact, restarts the serving container, smoke-tests the API, and archives generated logs/models for traceability.【F:jenkinsfile†L1-L206】 This pipeline ensures continuous delivery of updated churn models in a controlled manner.

## 📊 Monitoring & Observability
- **MLflow UI** – Inspect runs, compare metrics, and download models logged by `main.py` and `main1.py`.
- **Elasticsearch & Kibana** – `main1.py` pushes metrics into Elasticsearch (`mlflow-metrics` index), enabling Kibana dashboards for real-time performance visualisation.【F:main1.py†L24-L85】【F:docker-compose.yml†L4-L52】
- **Health Checks** – The FastAPI `/healthcheck` endpoint reports the readiness of the serving model for external monitoring tools.【F:app.py†L38-L45】

## ✅ Testing Strategy
Pytest unit tests (`test_pipeline.py`) cover critical stages: data preparation, model training, metric evaluation, and Joblib persistence to guarantee pipeline stability before deployment.【F:test_pipeline.py†L1-L38】 Additional smoke tests are executed in Jenkins against the running API to confirm end-to-end functionality.【F:jenkinsfile†L145-L194】

## 🗺️ Roadmap Ideas
- Add drift detection by monitoring feature distributions via Elasticsearch.
- Automate retraining triggers based on Kibana alerts or degraded metrics.
- Extend the FastAPI service with authentication and request logging.

---
Crafted for engineers who need a repeatable, observable, and production-ready telecom churn prediction workflow.
