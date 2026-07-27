"""
Interactive map of delay density for Germany

Displays a Plotly scatter_mapbox showing the 10 monitored stations
- Size: average delay (bigger = more delay)
- Color: average delay (red = high delay, green = low delay)
"""

import plotly.express as px
import streamlit as st

try:
    from utils.db_connection import run_query
except ModuleNotFoundError:
    from dashboard.utils.db_connection import run_query

st.set_page_config(page_title="Map: DB Delays", page_icon="🗺️", layout="wide")

st.title("Delay Density Map")
st.markdown(
    "Interactive map showing average delay at each monitored station. "
    "Larger and redder markers show longer average delays."
)

# Load station data with coordinates and delay metrics
map_query = """
    SELECT
        s.station_name,
        s.latitude,
        s.longitude,
        s.station_category,
        ROUND(AVG(f.delay_minutes)::numeric, 2) AS avg_delay,
        COUNT(*) AS event_count,
        COUNT(DISTINCT f.trip_id) AS unique_trips
    FROM marts.dim_stations s
    LEFT JOIN marts.fct_delays f ON s.station_id = f.station_id
    GROUP BY s.station_name, s.latitude, s.longitude, s.station_category
    ORDER BY avg_delay DESC
"""

try:
    stations = run_query(map_query)

    if stations.empty:
        st.warning("No station data found. Run seed_stations.py and the pipeline first.")
        st.stop()

    # Adjust stations with no delay data
    stations["avg_delay"] = stations["avg_delay"].fillna(0)
    stations["event_count"] = stations["event_count"].fillna(0).astype(int)
    stations["unique_trips"] = stations["unique_trips"].fillna(0).astype(int)

    # Map
    fig = px.scatter_mapbox(
        stations,
        lat="latitude",
        lon="longitude",
        size="avg_delay",
        size_max=30,
        color="avg_delay",
        color_continuous_scale="RdYlGn_r",
        hover_name="station_name",
        hover_data={
            "avg_delay": ":.2f",
            "event_count": ":,",
            "unique_trips": ":,",
            "station_category": True,
            "latitude": False,
            "longitude": False,
        },
        labels={
            "avg_delay": "Avg Delay (min)",
            "event_count": "Total Events",
            "unique_trips": "Unique Trips",
            "station_category": "Category",
        },
        zoom=5,
        center={"lat": 51.1657, "lon": 10.4515},
        mapbox_style="open-street-map",
    )
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Station detail table
    st.subheader("Station Details")
    st.dataframe(
        stations[["station_name", "station_category", "avg_delay", "event_count", "unique_trips"]]
        .rename(columns={
            "station_name": "Station",
            "station_category": "Category",
            "avg_delay": "Avg Delay (min)",
            "event_count": "Total Events",
            "unique_trips": "Unique Trips",
        }),
        hide_index=True,
        use_container_width=True,
    )

except Exception as e:
    st.error(f"Could not load map data: {e}")
    st.info("Make sure DATABASE_URL is configured and stations are seeded.")
