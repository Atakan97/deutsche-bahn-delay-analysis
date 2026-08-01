# Deutsche Bahn Delay Analysis

> A data engineering and machine learning pipeline that collects real-time train departure/arrival data from the official Deutsche Bahn Timetables API, transforms it through a dbt schema, trains an XGBoost delay prediction model, and serves predictions with a FastAPI microservice. All of them are visualised in a live Streamlit dashboard.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live Demo:** [Streamlit Dashboard](https://db-delay-analysis.streamlit.app/)

---

## Features

- Automated pipeline: Prefect flow triggered every 15 minutes with GitHub Actions to fetch departures and arrivals for 10 German stations
- dbt transformation layer: Staging views, incremental `fct_delays` fact table, `dim_stations` / `dim_routes` dimension tables with surrogate keys, data quality tests, and full documentation
- XGBoost delay prediction: Feature engineering (7 features including `prev_delay`), RandomizedSearchCV hyperparameter tuning, MLflow experiment tracking, and SHAP explainability
- FastAPI microservice: Deployed to Render (Docker). Validates requests with Pydantic, encodes categoricals with saved LabelEncoders, returns predictions
- Streamlit dashboard: Three pages: KPI Overview, Map, and a live prediction form that calls the FastAPI endpoint
- CI/CD: GitHub Actions lints with `ruff` and runs unit tests on every push and pull request

---

## Screenshots

| Interactive Map | Delay Prediction | Overview Dashboard |
| :---: | :---: | :---: |
| ![Interactive Map](docs/images/Map.png) | ![Delay Prediction](docs/images/Predict.png) | ![Overview Dashboard](docs/images/Overview.png) |

---

## ML Results

- Model: XGBoost Regressor benchmarked against historical mean baseline
- Features: `prev_delay`, `train_type`, `event_type`, `hour_of_day`, `station_category`, `day_of_week`, `is_weekend`
- Explainability: SHAP TreeExplainer plot saved to `docs/images/shap_summary.png`
- Experiment Check: MLflow logs saved to `ml/mlruns/` (`mlflow ui`)

| Metric | Baseline | XGBoost | Improvement |
|---|---|---|---|
| **MAE** | 5.22 min | 4.14 min | **+20.8%** |
| **RMSE** | 11.17 min | 9.30 min | **+16.7%** |

---

## Tech Stack

- Data & Pipeline: Python 3.11, Prefect, dbt, PostgreSQL (Supabase)
- ML & Serving: XGBoost, scikit-learn, SHAP, MLflow, FastAPI, Docker
- Dashboard: Streamlit
- DevOps & Testing: GitHub Actions, Pytest, Ruff, Render

---

## Architecture

**Data flow:**

```
DB StaDa (daily) → raw.stations
DB Timetables (every 15 min) → Python Extract → raw.train_events (JSONB)
             → dbt Staging  → staging.stg_train_events
             → dbt Marts    → marts.fct_delays + dims
             → ML Training  → api/model.pkl (XGBoost + encoders)
             → FastAPI      → POST /predict
             → Streamlit    → Live dashboard
```

---

## How to Use

### Prerequisites

- Python 3.11+
- PostgreSQL 15 (or a [Supabase](https://supabase.com) account)
- Docker (for local)

### 1. Clone the repository

```bash
git clone https://github.com/Atakan97/deutsche-bahn-delay-analysis.git
cd deutsche-bahn-delay-analysis
```

### 2. Install dependencies

```bash
pip install -r requirements-dev.txt
```

### 3. Configure environment

Fill in your values:

```bash
cp .env.example .env
```

Required variables:

- `DATABASE_URL`: PostgreSQL connection string (Supabase)
- `API_URL`: FastAPI base URL (default: `http://localhost:8000`)
- `DB_CLIENT_ID` & `DB_API_KEY`: DB API Marketplace credentials (for StaDa & Timetables)

### 4. Set up the database

```bash
# Create raw/staging/marts schemas, then fetch the 10 monitored stations
# from the official StaDa API, requires DB_CLIENT_ID and DB_API_KEY
python -m data_pipeline.extract.seed_stations

# Install dbt packages
cd transform && dbt deps
```

### 5. Run the pipeline

```bash
# Run the full pipeline once
python -m data_pipeline.orchestration.flows

# Or trigger it with GitHub Actions (runs every 15 minutes)
```

### 6. Run dbt transformations

```bash
cd transform
dbt run        # Build staging and mart models
dbt test       # Run data quality tests
dbt docs serve # Browse model documentation
```

### 7. Train the model

```bash
# Train baseline and XGBoost, log to MLflow, save model artifact
python -m ml.train

# Generate SHAP explainability plot to docs/images/shap_summary.png
python -m ml.explain

# Browse MLflow experiments
mlflow ui --backend-store-uri ml/mlruns
# Open http://localhost:5000
```

### 8. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

### 9. Start the dashboard

```bash
streamlit run dashboard/app.py
# Open http://localhost:8501
```

### Local with Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```
---

## Configuration

Secrets are managed with environment variables (`.env`):

- `DATABASE_URL`: PostgreSQL connection string (GitHub Actions, Streamlit Cloud)
- `DB_CLIENT_ID` & `DB_API_KEY`: DB API Marketplace credentials (GitHub Actions)
- `API_URL`: Deployed FastAPI URL (Streamlit Cloud, GitHub Actions keep-alive)

---

## Deployment

- API: Deployed to Render and kept alive with GitHub Actions
- Dashboard: Hosted on Streamlit Community Cloud connected to Supabase PostgreSQL

---

## CI/CD

- `ci.yml` — Linting & unit tests (on push/PR)
- `elt_pipeline.yml` — Automated flow (every 15 min)
- `station_catalog.yml` — Station data sync (daily)
- `keep_alive.yml` — Uptime pings for Streamlit Cloud & Render API (every 14 min)

---
## License

See [LICENSE](LICENSE).
