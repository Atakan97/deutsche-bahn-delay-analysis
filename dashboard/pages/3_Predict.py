"""
Delay prediction form

- Lets the user select journey parameters (station type, train type, hour, etc.)
- Sends the features to the FastAPI /predict endpoint
- Displays the predicted delay with a visual metric card and context
"""

import sys
from pathlib import Path

import streamlit as st

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from dashboard.utils.api_client import predict_delay
except ModuleNotFoundError:
    from utils.api_client import predict_delay

st.set_page_config(page_title="Predict: DB Delays", page_icon="🔮", layout="wide")

st.title("Delay Prediction")
st.markdown(
    "Enter journey details below to get a predicted delay from the trained XGBoost model. "
    "The model was trained on real Deutsche Bahn data collected every 15 minutes."
)

# Prediction form
STATION_CATEGORIES = ["major_hub", "regional_hub", "local_station"]
TRAIN_TYPES = [
    "ICE",  # Intercity-Express
    "IC",   # Intercity / Eurocity
    "RE",   # Regional-Express
    "RB",   # Regionalbahn
    "S",    # S-Bahn
    "FLX",  # Flixtrain
]
EVENT_TYPES = ["departure", "arrival"]
DAY_NAMES = [
    "Sunday",     # 0
    "Monday",     # 1
    "Tuesday",    # 2
    "Wednesday",  # 3
    "Thursday",   # 4
    "Friday",     # 5
    "Saturday",   # 6
]

with st.form("prediction_form"):
    st.subheader("Journey Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        hour_of_day = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=17,
            help="Scheduled departure/arrival hour (0–23)",
        )
        day_name = st.selectbox(
            "Day of the week",
            options=DAY_NAMES,
            index=3,  # Wednesday
            help="Day of the scheduled journey",
        )

    with col2:
        station_category = st.selectbox(
            "Station category",
            options=STATION_CATEGORIES,
            index=0,  # major_hub
            help=(
                "major_hub = large central stations, "
                "regional_hub = medium stations, "
                "local_station = smaller stations"
            ),
        )
        train_type = st.selectbox(
            "Train type",
            options=TRAIN_TYPES,
            index=0,  # ICE
            help=(
                "ICE = Intercity-Express, IC = Intercity/Eurocity, "
                "RE = Regional-Express, RB = Regionalbahn, S = S-Bahn, "
                "FLX = Flixtrain"
            ),
        )

    with col3:
        event_type = st.selectbox(
            "Event type",
            options=EVENT_TYPES,
            index=0,  # departure
            help="Whether this is a departure or arrival event",
        )
        prev_delay = st.number_input(
            "Previous stop delay (minutes)",
            min_value=-5.0,
            max_value=120.0,
            value=0.0,
            step=1.0,
            help="Delay at the previous station (0 if unknown or first stop)",
        )

    submitted = st.form_submit_button("Predict Delay", use_container_width=True)

# Process prediction
if submitted:
    # Convert day name back to day_of_week integer
    day_of_week = DAY_NAMES.index(day_name)
    is_weekend = day_of_week in (0, 6)  # 0=Sunday, 6=Saturday

    features = {
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "station_category": station_category,
        "train_type": train_type,
        "event_type": event_type,
        "prev_delay": prev_delay,
    }

    with st.spinner("Calling prediction API..."):
        predicted_delay = predict_delay(features)

    if predicted_delay is not None:
        st.divider()

        # Display prediction with context
        col_metric, col_context = st.columns([1, 2])

        with col_metric:
            # Color the metric based on delay severity
            if predicted_delay <= 0:
                st.metric("Predicted Delay", f"{predicted_delay:.1f} min", delta="On time")
            elif predicted_delay <= 5:
                st.metric("Predicted Delay", f"{predicted_delay:.1f} min", delta="Minor delay")
            elif predicted_delay <= 15:
                st.metric(
                    "Predicted Delay", f"{predicted_delay:.1f} min", delta="Moderate delay"
                )
            else:
                st.metric(
                    "Predicted Delay",
                    f"{predicted_delay:.1f} min",
                    delta="Significant delay",
                    delta_color="inverse",
                )

        with col_context:
            st.markdown("**Journey Summary**")
            st.markdown(
                f"- **Train**: {train_type} ({event_type})\n"
                f"- **Station**: {station_category}\n"
                f"- **Time**: {day_name} at {hour_of_day:02d}:00\n"
                f"- **Previous delay**: {prev_delay:.1f} min\n"
                f"- **Weekend**: {'Yes' if is_weekend else 'No'}"
            )

        # Interpretation help
        st.info(
            "- Negative values mean the train is predicted to arrive/depart early.\n"
            "- 0 minutes means on time.\n"
            "- Positive values indicate the predicted delay in minutes.\n\n"
            "The prediction is based on historical patterns in the data. "
            "Actual delays may vary due to weather, disruptions, or special events."
        )
