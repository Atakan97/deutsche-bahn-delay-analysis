"""
Model evaluation utilities

Provides functions to compute standard regression metrics (MAE, RMSE) and
to format a comparison between two models

These are used by train.py to evaluate the baseline and XGBoost models

-MAE (Mean Absolute Error) is easy to interpret, treats all errors equally
-RMSE (Root Mean Squared Error) penalises large errors more heavily
They give a more precise result of model performance

"""

import numpy as np
from numpy.typing import ArrayLike


# y_true -> Actual delay values (ground truth)
# y_pred -> Predicted delay values from the model
def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Compute MAE and RMSE for a set of predictions"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    # rounded to 4 decimal places
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
    }


def compare_models(
    baseline_metrics: dict[str, float],
    xgboost_metrics: dict[str, float],
) -> str:
    """Format a comparison between baseline and XGBoost

    Calculates the percentage improvement of XGBoost over the baseline for
    both MAE and RMSE, positive percentage shows XGBoost is better
    """
    # Calculate percentage improvement
    if baseline_metrics["mae"] > 0:
        mae_improvement = (
            (baseline_metrics["mae"] - xgboost_metrics["mae"])
            / baseline_metrics["mae"]
            * 100
        )
    else:
        mae_improvement = 0.0

    if baseline_metrics["rmse"] > 0:
        rmse_improvement = (
            (baseline_metrics["rmse"] - xgboost_metrics["rmse"])
            / baseline_metrics["rmse"]
            * 100
        )
    else:
        rmse_improvement = 0.0

    # Formatted multi-line string showing both models' metrics side by side
    comparison = (
        "\n"
        "=" * 60 + "\n"
        "Model Comparison: Baseline vs. XGBoost\n"
        "=" * 60 + "\n"
        "\n"
        f"{'Metric':<10} {'Baseline':>12} {'XGBoost':>12} {'Improvement':>14}\n"
        f"{'-' * 10} {'-' * 12} {'-' * 12} {'-' * 14}\n"
        f"{'MAE':<10} {baseline_metrics['mae']:>12.4f} {xgboost_metrics['mae']:>12.4f} {mae_improvement:>+13.1f}%\n"
        f"{'RMSE':<10} {baseline_metrics['rmse']:>12.4f} {xgboost_metrics['rmse']:>12.4f} {rmse_improvement:>+13.1f}%\n"
        "\n"
        f"XGBoost improved MAE by {mae_improvement:.1f}% over baseline.\n"
        f"XGBoost improved RMSE by {rmse_improvement:.1f}% over baseline.\n"
        "=" * 60
    )

    return comparison
