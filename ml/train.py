"""
Train and compare baseline vs. XGBoost delay prediction models
"""

import os
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from ml.evaluate import compare_models, compute_metrics
from ml.features import FEATURE_COLUMNS, engineer_features, load_feature_data

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
API_DIR = PROJECT_ROOT / "api"

# MLflow tracking, all experiment data stored locally under ml/mlruns/
MLFLOW_TRACKING_URI = (ML_DIR / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = "deutsche-bahn-delay-prediction"

# Fixed random seed for reproducibility
RANDOM_STATE = 42

# Train/test split ratio
TEST_SIZE = 0.2

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Train a baseline model, predict the mean delay per (route + hour) group
    """
    # Combine features + target for groupby
    train_with_target = X_train.copy()
    train_with_target["delay_minutes"] = y_train.values

    # Group by station_category + hour_of_day and compute the mean delay
    group_cols = ["station_category", "hour_of_day"]
    group_means = (
        train_with_target
        .groupby(group_cols)["delay_minutes"]
        .mean()
    )

    # Global mean for unseen groups
    global_mean = float(y_train.mean())

    # Predict, look up each test row's group mean, or use global mean
    predictions = []
    for _, row in X_test.iterrows():
        key = (row["station_category"], row["hour_of_day"])
        if key in group_means.index:
            predictions.append(group_means[key])
        else:
            predictions.append(global_mean)

    return np.array(predictions)


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBRegressor:
    """Train an XGBoost model with light hyperparameter tuning
    """
    # Base model with sensible defaults
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        # Suppress verbose training output
        verbosity=0,
    )

    # Small parameter grid for RandomizedSearchCV
    param_distributions = {
        "max_depth": [3, 4, 5, 6, 7], # controls tree complexity (deeper = more complex)
        "n_estimators": [50, 100, 150, 200], # number of boosting rounds
        "learning_rate": [0.01, 0.05, 0.1, 0.2], # step size shrinkage
        "subsample": [0.7, 0.8, 0.9, 1.0], # fraction of training data used per tree
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0], # fraction of features used per tree
    }

    # RandomizedSearchCV, try 10 random combinations, evaluate with 3-fold cross validation
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=10,
        cv=3,
        scoring="neg_mean_absolute_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )

    print("Running RandomizedSearchCV (10 iterations × 3-fold cross validation)...")
    search.fit(X_train, y_train)

    print(f"Best hyperparameters: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.4f}")

    return search.best_estimator_


def save_model_artifact(
    model: XGBRegressor,
    label_encoders: dict,
    feature_columns: list[str],
) -> Path:
    """Save the trained model, encoders, and feature list as a single artifact
    """
    artifact = {
        "model": model,
        "label_encoders": label_encoders,
        "feature_columns": feature_columns,
    }

    # Create the artifacts directory if it doesn't exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = ARTIFACTS_DIR / "model.pkl"

    joblib.dump(artifact, artifact_path)
    print(f"\nModel artifact saved to: {artifact_path}")

    # Copy to api/model.pkl for API deployment (FastAPI / Render)
    API_DIR.mkdir(parents=True, exist_ok=True)
    api_model_path = API_DIR / "model.pkl"
    shutil.copy2(artifact_path, api_model_path)
    print(f"Model artifact copied to: {api_model_path}")

    return artifact_path


def run_training() -> None:
    """Main training pipeline, load data, train models, compare, and save
    """
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it in .env (local) or as an environment variable."
        )

    print("=" * 60)
    print("Deutsche Bahn Delay Prediction — Model Training")
    print("=" * 60)

    # Load and engineer features
    print("\n[1/6] Loading feature data from marts tables...")
    df = load_feature_data(database_url)

    print("\n[2/6] Engineering features...")
    X, y, label_encoders = engineer_features(df)

    # Train/test split
    print("\n[3/6] Splitting data (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train set: {len(X_train):,} rows")
    print(f"  Test set:  {len(X_test):,} rows")

    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Train and evaluate BASELINE
    print("\n[4/6] Training baseline model (group-by mean)...")
    baseline_preds = train_baseline(X_train, y_train, X_test)
    baseline_metrics = compute_metrics(y_test, baseline_preds)
    print(f"  Baseline MAE:  {baseline_metrics['mae']:.4f}")
    print(f"  Baseline RMSE: {baseline_metrics['rmse']:.4f}")

    # Log baseline to MLflow
    with mlflow.start_run(run_name="baseline"):
        mlflow.log_param("model_type", "group_mean")
        mlflow.log_param("group_columns", "station_category,hour_of_day")
        mlflow.log_metric("mae", baseline_metrics["mae"])
        mlflow.log_metric("rmse", baseline_metrics["rmse"])
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))

    # Train and evaluate XGBOOST
    print("\n[5/6] Training XGBoost model...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_metrics = compute_metrics(y_test, xgb_preds)
    print(f"  XGBoost MAE:  {xgb_metrics['mae']:.4f}")
    print(f"  XGBoost RMSE: {xgb_metrics['rmse']:.4f}")

    # Log XGBoost to MLflow
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_params(xgb_model.get_params())
        mlflow.log_metric("mae", xgb_metrics["mae"])
        mlflow.log_metric("rmse", xgb_metrics["rmse"])
        mlflow.log_metric("train_size", len(X_train))
        mlflow.log_metric("test_size", len(X_test))

    # Compare and save
    comparison = compare_models(baseline_metrics, xgb_metrics)
    print(comparison)

    print("\n[6/6] Saving model artifact...")
    save_model_artifact(xgb_model, label_encoders, FEATURE_COLUMNS)

    print("\nTraining complete")
    print("Next steps:")
    print("  1. Run SHAP analysis:  python -m ml.explain")
    print("  2. Browse MLflow UI:   mlflow ui --backend-store-uri ml/mlruns")
    print("  3. Start API locally:  uvicorn api.main:app --reload")


if __name__ == "__main__":
    run_training()
