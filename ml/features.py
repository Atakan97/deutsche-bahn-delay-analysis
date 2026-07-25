"""
Feature engineering for the delay prediction model

- Queries the marts tables (fct_delays, dim_stations, dim_routes)
  in PostgreSQL and returns a pandas DataFrame with all raw features

- Transforms the raw features into the final feature matrix (X)
  and target vector (y) that the ML models expect
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# SQL query to load all features from the marts star schema
# ORDER BY is written for computing the previous stop delay feature correctly
FEATURE_QUERY = """
    SELECT
        f.delay_minutes,
        f.hour_of_day,
        f.day_of_week,
        f.is_weekend,
        f.event_type,
        f.trip_id,
        f.station_id,
        f.planned_time,
        s.station_category,
        r.train_type
    FROM marts.fct_delays f
    LEFT JOIN marts.dim_stations s ON f.station_id = s.station_id
    LEFT JOIN marts.dim_routes r ON f.route_id = r.route_id
    WHERE f.delay_minutes IS NOT NULL
    ORDER BY f.trip_id, f.planned_time
"""

# The columns that the trained model expects as input features
# List is saved with the model so the API knows the exact
# feature order at prediction time
FEATURE_COLUMNS = [
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "station_category",
    "train_type",
    "event_type",
    "prev_delay",
]

# The column is tried to predict
TARGET_COLUMN = "delay_minutes"

def load_feature_data(database_url: str) -> pd.DataFrame:
    """Load feature data from the marts star schema with SQL

    Queries fct_delays joined with dim_stations and dim_routes to get all
    the columns
    """
    df = pd.read_sql(FEATURE_QUERY, database_url)

    if df.empty:
        raise RuntimeError(
            "No feature data found in marts.fct_delays. "
            "Run the ELT pipeline and dbt first to populate the marts tables."
        )

    print(f"Loaded {len(df):,} rows from marts tables.")
    return df

def engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, LabelEncoder]]:
    """Transform raw feature data into ready inputs
    """
    # Make a copy as avoid modifying the original DataFrame
    df = df.copy()

    # Compute previous-stop delay feature
    # Sort by trip and time, then within each trip group, shift delay_minutes
    # by one row so it gives each event the delay from the prior stop
    df = df.sort_values(["trip_id", "planned_time"])
    df["prev_delay"] = (
        df.groupby("trip_id")["delay_minutes"]
        .shift(1)  # Previous row within the same trip
        .fillna(0)  # First stop of a trip has no previous delay → 0
    )

    # Label-encode categorical columns
    # Fill any NaN categoricals with "unknown" before encoding
    categorical_columns = ["station_category", "train_type", "event_type"]
    label_encoders: dict[str, LabelEncoder] = {}

    for col in categorical_columns:
        df[col] = df[col].fillna("unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Convert boolean to int
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Select feature columns and target
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"Engineered features: {list(X.columns)}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    return X, y, label_encoders
