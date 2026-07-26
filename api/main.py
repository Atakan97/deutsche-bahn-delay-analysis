"""
FastAPI application for train delay predictions

Serves the trained XGBoost model with two HTTP endpoints:

  POST /predict
    Accepts train journey features (hour, day, station type, train type, etc.)
    and returns a predicted delay in minutes

  GET /health
    Returns {"status": "healthy"} to confirm the service is running. 
"""

from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Model loading, happens once at application startup
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
_artifact: dict | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact at startup, release at shutdown
    """
    global _artifact  # noqa: PLW0603

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run 'python -m ml.train' to train and save the model first."
        )

    _artifact = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
    print(f"  Model type: {type(_artifact['model']).__name__}")
    print(f"  Features: {_artifact['feature_columns']}")

    yield  # Application is running and serving requests

    # Cleanup
    _artifact = None
    print("Model unloaded.")


# FastAPI app
app = FastAPI(
    title="Deutsche Bahn Delay Prediction API",
    description=(
        "Predicts train delays at German railway stations using an XGBoost model "
        "trained on real-time data."
        "Part of the [deutsche-bahn-delay-analysis]"
        "(https://github.com/Atakan97/deutsche-bahn-delay-analysis) project."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Request / Response models
class PredictionRequest(BaseModel):
    """Input features for a delay prediction
    """

    hour_of_day: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of the scheduled departure/arrival (0–23).",
        examples=[17],
    )
    day_of_week: int = Field(
        ...,
        ge=0,
        le=6,
        description="Day of the week (0=Sunday, 1=Monday, ..., 6=Saturday).",
        examples=[3],
    )
    is_weekend: bool = Field(
        ...,
        description="Whether this is a weekend day (Saturday or Sunday).",
        examples=[False],
    )
    station_category: str = Field(
        ...,
        description=(
            "Station classification: 'major_hub', 'regional_hub', or 'local_station'."
        ),
        examples=["major_hub"],
    )
    train_type: str = Field(
        ...,
        description=(
            "Train product type: "
            "'ICE' (Intercity-Express), 'IC' (Intercity), 'EC' (Eurocity), "
            "'RE' (Regional-Express), 'RB' (Regionalbahn), 'S' (S-Bahn), 'FLX' (Flixtrain), etc."
        ),
        examples=["ICE"],
    )
    event_type: str = Field(
        ...,
        description="Either 'departure' or 'arrival'.",
        examples=["departure"],
    )
    prev_delay: float = Field(
        default=0.0,
        description=(
            "Delay at the previous station in minutes. "
            "Use 0 if unknown or if this is the first station."
        ),
        examples=[3.5],
    )

class PredictionResponse(BaseModel):
    """Prediction result returned by the /predict endpoint"""

    predicted_delay_minutes: float = Field(
        ...,
        description="Predicted delay in minutes. Negative means early.",
        examples=[4.2],
    )


class HealthResponse(BaseModel):
    """Health check response"""

    status: str = Field(
        ...,
        description="Service health status.",
        examples=["healthy"],
    )


# Endpoints
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict train delay",
    description=(
        "Given train journey features (time, station type, train type, etc.), "
        "returns the predicted delay in minutes using the trained XGBoost model"
    ),
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Run inference on the loaded XGBoost model
    """
    if _artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. The service is starting up.",
        )

    model = _artifact["model"]
    label_encoders = _artifact["label_encoders"]
    feature_columns = _artifact["feature_columns"]

    # Encode categorical features
    raw_features = {
        "hour_of_day": request.hour_of_day,
        "day_of_week": request.day_of_week,
        "is_weekend": int(request.is_weekend),
        "station_category": request.station_category,
        "train_type": request.train_type,
        "event_type": request.event_type,
        "prev_delay": request.prev_delay,
    }

    # Encode each categorical column using the saved LabelEncoder
    # If the value is unknown, return a clear error
    for col, le in label_encoders.items():
        value = raw_features[col]
        if value not in le.classes_:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Unknown value '{value}' for '{col}'. "
                    f"Valid values: {list(le.classes_)}"
                ),
            )
        raw_features[col] = int(le.transform([value])[0])

    # Build feature vector in the correct column order
    feature_vector = pd.DataFrame([raw_features], columns=feature_columns)

    # Predict
    prediction = model.predict(feature_vector)
    predicted_delay = float(np.round(prediction[0], 2))

    return PredictionResponse(predicted_delay_minutes=predicted_delay)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status. Used by keep-alive cron and monitoring.",
)
async def health() -> HealthResponse:
    """Simple health check, confirms the service is running and the model is loaded."""
    return HealthResponse(status="healthy")
