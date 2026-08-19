"""
Unit tests for the FastAPI prediction API
"""

from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Create a synthetic model artifact and inject it into the API

@pytest.fixture()
def _mock_model_artifact(tmp_path: Path):
    """Create a synthetic model artifact and inject it directly into api.main
    """
    import api.main as api_module

    # Train a minimal XGBRegressor on synthetic data
    X_train = np.array(
        [
            [8, 1, 0, 0, 0, 0, 0.0],
            [17, 5, 1, 1, 1, 1, 3.0],
        ]
    )
    y_train = np.array([2.0, 8.0])
    model = XGBRegressor(n_estimators=2, max_depth=2, verbosity=0)
    model.fit(X_train, y_train)

    # Create LabelEncoders with known classes
    le_station = LabelEncoder()
    le_station.fit(["local_station", "major_hub", "regional_hub"])

    le_train_type = LabelEncoder()
    le_train_type.fit(["IC", "ICE", "RE", "S"])

    le_event = LabelEncoder()
    le_event.fit(["arrival", "departure"])

    artifact = {
        "model": model,
        "label_encoders": {
            "station_category": le_station,
            "train_type": le_train_type,
            "event_type": le_event,
        },
        "feature_columns": [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "station_category",
            "train_type",
            "event_type",
            "prev_delay",
        ],
    }

    artifact_path = tmp_path / "model.pkl"
    joblib.dump(artifact, artifact_path)

    api_module._artifact = artifact

    yield

    api_module._artifact = None

# Create a FastAPI TestClient with the injected model artifact
@pytest.fixture()
def client(_mock_model_artifact) -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


# Tests, health endpoint
class TestHealthEndpoint:

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}

# Tests, model options endpoint
class TestModelOptionsEndpoint:
    def test_model_options_returns_200(self, client: TestClient) -> None:
        response = client.get("/model-options")
        assert response.status_code == 200

    def test_model_options_returns_encoder_values(self, client: TestClient) -> None:
        response = client.get("/model-options")
        data = response.json()

        assert data["station_categories"] == [
            "local_station",
            "major_hub",
            "regional_hub",
        ]
        assert data["train_types"] == ["IC", "ICE", "RE", "S"]
        assert data["event_types"] == ["arrival", "departure"]

    def test_model_options_without_model_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import api.main as api_module

        monkeypatch.setattr(api_module, "_artifact", None)

        response = client.get("/model-options")
        assert response.status_code == 503


# Tests, predict endpoint, valid requests
class TestPredictEndpoint:

    def test_predict_returns_200(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 17,
            "day_of_week": 3,
            "is_weekend": False,
            "station_category": "major_hub",
            "train_type": "ICE",
            "event_type": "departure",
            "prev_delay": 3.5,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_returns_float_delay(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 8,
            "day_of_week": 1,
            "is_weekend": False,
            "station_category": "regional_hub",
            "train_type": "RE",
            "event_type": "arrival",
            "prev_delay": 0.0,
        }
        response = client.post("/predict", json=payload)
        data = response.json()

        assert "predicted_delay_minutes" in data
        assert isinstance(data["predicted_delay_minutes"], float)

    def test_predict_default_prev_delay(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 12,
            "day_of_week": 4,
            "is_weekend": False,
            "station_category": "local_station",
            "train_type": "S",
            "event_type": "departure",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_predict_weekend_day(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 10,
            "day_of_week": 0,
            "is_weekend": True,
            "station_category": "major_hub",
            "train_type": "IC",
            "event_type": "arrival",
            "prev_delay": 5.0,
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200


# Tests, predict endpoint, validation errors
class TestPredictValidation:

    def test_missing_required_field_returns_422(self, client: TestClient) -> None:
        payload = {
            "day_of_week": 3,
            "is_weekend": False,
            "station_category": "major_hub",
            "train_type": "ICE",
            "event_type": "departure",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_hour_out_of_range_returns_422(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 25,
            "day_of_week": 3,
            "is_weekend": False,
            "station_category": "major_hub",
            "train_type": "ICE",
            "event_type": "departure",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_day_out_of_range_returns_422(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 8,
            "day_of_week": 7,
            "is_weekend": False,
            "station_category": "major_hub",
            "train_type": "ICE",
            "event_type": "departure",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_unknown_station_category_returns_422(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 8,
            "day_of_week": 1,
            "is_weekend": False,
            "station_category": "mega_station",
            "train_type": "ICE",
            "event_type": "departure",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

        detail = response.json()["detail"]
        assert "mega_station" in detail
        assert "major_hub" in detail

    def test_unknown_train_type_returns_422(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 8,
            "day_of_week": 1,
            "is_weekend": False,
            "station_category": "major_hub",
            "train_type": "hyperloop",
            "event_type": "departure",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_unknown_event_type_returns_422(self, client: TestClient) -> None:
        payload = {
            "hour_of_day": 8,
            "day_of_week": 1,
            "is_weekend": False,
            "station_category": "major_hub",
            "train_type": "ICE",
            "event_type": "passing_through",
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422

    def test_empty_body_returns_422(self, client: TestClient) -> None:
        response = client.post("/predict", json={})
        assert response.status_code == 422
