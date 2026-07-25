"""
SHAP explainability analysis for the XGBoost delay prediction model

Generates a SHAP summary plot showing which features have the most effect
on the model's predictions
"""

import os
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from ml.features import engineer_features, load_feature_data
from ml.train import RANDOM_STATE, TEST_SIZE

matplotlib.use("Agg")

# Output paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "ml" / "artifacts"
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
SHAP_PLOT_PATH = IMAGES_DIR / "shap_summary.png"


def run_explain() -> None:
    """Load the trained model, compute SHAP values, and save the summary plot
    """
    # Load .env for DATABASE_URL
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set."
        )

    # Load the trained model artifact
    artifact_path = ARTIFACTS_DIR / "model.pkl"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {artifact_path}. "
            "Run 'python -m ml.train' first."
        )

    print("Loading model artifact...")
    artifact = joblib.load(artifact_path)
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    print(f"  Model type: {type(model).__name__}")
    print(f"  Features: {feature_columns}")

    # Load and prepare data for SHAP analysis
    print("\nLoading feature data...")
    df = load_feature_data(database_url)
    X, y, _ = engineer_features(df)

    # Use the same test split as training for consistency
    _, X_test, _, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Limit to a sample if the dataset is large
    max_shap_samples = 2000
    if len(X_test) > max_shap_samples:
        X_shap = X_test.sample(n=max_shap_samples, random_state=RANDOM_STATE)
        print(f"  Sampled {max_shap_samples} rows from test set for SHAP analysis.")
    else:
        X_shap = X_test
        print(f"  Using full test set ({len(X_test)} rows) for SHAP analysis.")

    # Compute SHAP values
    print("\nComputing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    # Save the summary plot
    print("\nGenerating SHAP summary plot...")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Create the summary plot, shows the impact of each feature on predictions
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_shap,
        feature_names=feature_columns,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SHAP summary plot saved to: {SHAP_PLOT_PATH}")

    # Print top features by mean absolute SHAP value
    mean_abs_shap = pd.Series(
        data=abs(shap_values).mean(axis=0),
        index=feature_columns,
    ).sort_values(ascending=False)

    print("\nTop features by mean |SHAP value|:")
    for feature_name, importance in mean_abs_shap.items():
        print(f"  {feature_name:<20s} {importance:.4f}")

    print("\nSHAP analysis complete")


if __name__ == "__main__":
    run_explain()
