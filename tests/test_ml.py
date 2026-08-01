"""
Unit tests for the ML module
"""

import numpy as np
import pandas as pd
import pytest

from ml.evaluate import compare_models, compute_metrics
from ml.features import FEATURE_COLUMNS, engineer_features


# Test fixtures
@pytest.fixture()
def sample_feature_df() -> pd.DataFrame:
    """Create a minimal DataFrame that imitates load_feature_data() output

    Synthetic data is used, so database is not needed
    """
    return pd.DataFrame(
        {
            "delay_minutes": [5.0, 10.0, 0.0, 3.0, 8.0, 2.0],
            "hour_of_day": [8, 8, 17, 17, 12, 12],
            "day_of_week": [1, 1, 5, 5, 0, 0],
            "is_weekend": [False, False, False, False, True, True],
            "event_type": [
                "departure",
                "arrival",
                "departure",
                "arrival",
                "departure",
                "arrival",
            ],
            "trip_id": ["T1", "T1", "T2", "T2", "T3", "T3"],
            "station_id": ["S1", "S2", "S1", "S3", "S2", "S3"],
            "planned_time": pd.to_datetime(
                [
                    "2025-07-16 08:00",
                    "2025-07-16 08:30",
                    "2025-07-16 17:00",
                    "2025-07-16 17:45",
                    "2025-07-16 12:00",
                    "2025-07-16 12:20",
                ]
            ),
            "station_category": [
                "major_hub",
                "regional_hub",
                "major_hub",
                "local_station",
                "regional_hub",
                "local_station",
            ],
            "train_type": [
                "nationalExpress",
                "nationalExpress",
                "regional",
                "regional",
                "national",
                "national",
            ],
        }
    )


@pytest.fixture()
def sample_df_with_nulls() -> pd.DataFrame:
    """DataFrame with some null categorical values to test NaN handling"""
    return pd.DataFrame(
        {
            "delay_minutes": [5.0, 10.0],
            "hour_of_day": [8, 8],
            "day_of_week": [1, 1],
            "is_weekend": [False, False],
            "event_type": ["departure", "arrival"],
            "trip_id": ["T1", "T1"],
            "station_id": ["S1", "S2"],
            "planned_time": pd.to_datetime(
                ["2025-07-16 08:00", "2025-07-16 08:30"]
            ),
            # Null values
            "station_category": ["major_hub", None],
            "train_type": [None, "regional"],
        }
    )


# Tests, feature engineering
class TestEngineerFeatures:
    """Tests for the engineer_features function"""

    def test_returns_correct_shapes(self, sample_feature_df: pd.DataFrame) -> None:
        """X should have the expected number of feature columns, y should be 1D"""
        X, y, encoders = engineer_features(sample_feature_df)

        assert X.shape == (6, len(FEATURE_COLUMNS))
        assert y.shape == (6,)

    def test_returns_expected_columns(self, sample_feature_df: pd.DataFrame) -> None:
        """X should contain exactly the columns listed in FEATURE_COLUMNS"""
        X, _, _ = engineer_features(sample_feature_df)

        assert list(X.columns) == FEATURE_COLUMNS

    def test_target_is_delay_minutes(self, sample_feature_df: pd.DataFrame) -> None:
        """y should contain the delay_minutes values from the input"""
        _, y, _ = engineer_features(sample_feature_df)

        # The values should be present
        assert set(y.values) == {0.0, 2.0, 3.0, 5.0, 8.0, 10.0}

    def test_label_encoders_returned(self, sample_feature_df: pd.DataFrame) -> None:
        """Label encoders should be returned for the three categorical columns"""
        _, _, encoders = engineer_features(sample_feature_df)

        assert "station_category" in encoders
        assert "train_type" in encoders
        assert "event_type" in encoders

    def test_categoricals_are_encoded_as_integers(
        self, sample_feature_df: pd.DataFrame
    ) -> None:
        """Encoded categorical columns should contain integer values"""
        X, _, _ = engineer_features(sample_feature_df)

        for col in ["station_category", "train_type", "event_type"]:
            # All values should be non-negative integers
            assert (X[col] >= 0).all()
            assert X[col].dtype in [np.int32, np.int64, np.intp]

    def test_is_weekend_is_int(self, sample_feature_df: pd.DataFrame) -> None:
        """is_weekend should be converted from bool to int"""
        X, _, _ = engineer_features(sample_feature_df)

        assert X["is_weekend"].dtype in [np.int32, np.int64, np.intp, int]
        assert set(X["is_weekend"].unique()).issubset({0, 1})

    def test_prev_delay_is_computed(self, sample_feature_df: pd.DataFrame) -> None:
        """prev_delay should be 0 for the first event of each trip"""
        X, _, _ = engineer_features(sample_feature_df)

        assert "prev_delay" in X.columns
        assert (X["prev_delay"] == 0).any()

    def test_prev_delay_within_trip(self, sample_feature_df: pd.DataFrame) -> None:
        """prev_delay for the second event of a trip should equal the first event's delay"""
        X, _, _ = engineer_features(sample_feature_df)

        # First event: delay=5, prev_delay=0
        # Second event: delay=10, prev_delay=5
        # Check that at least one prev_delay equals 5.0
        assert 5.0 in X["prev_delay"].values

    def test_null_categoricals_filled_with_unknown(
        self, sample_df_with_nulls: pd.DataFrame
    ) -> None:
        """Null categorical values should be filled with 'unknown' before encoding"""
        X, _, encoders = engineer_features(sample_df_with_nulls)

        assert not X.isnull().any().any()

        # "unknown" should be one of the learned classes
        assert "unknown" in encoders["station_category"].classes_
        assert "unknown" in encoders["train_type"].classes_

    def test_original_dataframe_not_modified(
        self, sample_feature_df: pd.DataFrame
    ) -> None:
        """engineer_features should not modify the input DataFrame"""
        original_columns = list(sample_feature_df.columns)
        original_len = len(sample_feature_df)

        engineer_features(sample_feature_df)

        assert list(sample_feature_df.columns) == original_columns
        assert len(sample_feature_df) == original_len

# Tests, Evaluation metrics
class TestComputeMetrics:
    """Tests for compute_metrics"""

    def test_perfect_predictions(self) -> None:
        """If predictions exactly match actuals, MAE and RMSE should both be 0"""
        y_true = [1.0, 2.0, 3.0]
        y_pred = [1.0, 2.0, 3.0]
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_known_mae(self) -> None:
        """Verify MAE calculation with a known example

        y_true = [1, 2, 3], y_pred = [2, 2, 2]
        errors = [1, 0, 1], MAE = (1 + 0 + 1) / 3 = 0.6667
        """
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 2.0, 2.0]
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == pytest.approx(0.6667, abs=0.001)

    def test_known_rmse(self) -> None:
        """Verify RMSE calculation with a known example

        y_true = [1, 2, 3], y_pred = [2, 2, 2]
        errors = [1, 0, 1], squared = [1, 0, 1], mean = 0.6667, RMSE = 0.8165
        """
        y_true = [1.0, 2.0, 3.0]
        y_pred = [2.0, 2.0, 2.0]
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["rmse"] == pytest.approx(0.8165, abs=0.001)

    def test_negative_errors_handled(self) -> None:
        """Predictions above actuals should contribute to MAE correctly"""
        y_true = [5.0]
        y_pred = [10.0]
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == 5.0
        assert metrics["rmse"] == 5.0

    def test_returns_dict_with_expected_keys(self) -> None:
        """Result should have exactly 'mae' and 'rmse' keys"""
        metrics = compute_metrics([1.0], [2.0])

        assert set(metrics.keys()) == {"mae", "rmse"}

    def test_accepts_numpy_arrays(self) -> None:
        """Should work with numpy arrays, not just lists"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = compute_metrics(y_true, y_pred)

        assert metrics["mae"] == pytest.approx(0.5, abs=0.001)


class TestCompareModels:
    """Tests for compare_models"""

    def test_returns_string(self) -> None:
        """compare_models should return a string"""
        baseline = {"mae": 5.0, "rmse": 7.0}
        xgboost = {"mae": 4.0, "rmse": 5.5}
        result = compare_models(baseline, xgboost)

        assert isinstance(result, str)

    def test_contains_improvement_percentage(self) -> None:
        """The comparison string should mention the percentage improvement"""
        baseline = {"mae": 10.0, "rmse": 15.0}
        xgboost = {"mae": 8.0, "rmse": 12.0}
        result = compare_models(baseline, xgboost)

        # 20% MAE improvement, 20% RMSE improvement
        assert "20.0%" in result

    def test_negative_improvement_shown(self) -> None:
        """If XGBoost is worse, the improvement should be negative"""
        baseline = {"mae": 5.0, "rmse": 7.0}
        xgboost = {"mae": 6.0, "rmse": 8.0}
        result = compare_models(baseline, xgboost)

        # Should show negative improvement
        assert "-20.0%" in result

    def test_zero_baseline_no_division_error(self) -> None:
        """If baseline metrics are 0, should not raise ZeroDivisionError"""
        baseline = {"mae": 0.0, "rmse": 0.0}
        xgboost = {"mae": 1.0, "rmse": 1.0}

        result = compare_models(baseline, xgboost)
        assert isinstance(result, str)
