"""
KPIs, delay trends, and information
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import plotly.express as px
import streamlit as st

try:
    from utils.db_connection import run_query
except ModuleNotFoundError:
    from dashboard.utils.db_connection import run_query

st.set_page_config(page_title="Overview: DB Delays", page_icon="📊", layout="wide")

st.title("Delay Overview")
st.markdown("Real-time delay analytics for 10 major German railway stations.")

# Daily KPI data
daily_kpi_query = """
    SELECT
        (fetched_at AT TIME ZONE 'Europe/Berlin')::date                AS event_date,
        COUNT(*)                                                        AS total_events,
        ROUND(AVG(delay_minutes)::numeric, 1)                           AS avg_delay,
        ROUND(
            COUNT(CASE WHEN delay_minutes <= 0 THEN 1 END)::numeric
            / NULLIF(COUNT(*), 0)::numeric * 100, 1
        )                                                               AS on_time_pct
    FROM marts.fct_delays
    GROUP BY event_date
    ORDER BY event_date
"""

station_count_query = """
    SELECT COUNT(DISTINCT station_id) AS stations
    FROM marts.fct_delays
"""

try:
    daily_kpis = run_query(daily_kpi_query)
    station_count = run_query(station_count_query)

    if daily_kpis.empty:
        st.warning(
            "No data found in marts.fct_delays. "
            "Run the ELT pipeline and dbt to populate the tables first."
        )
        st.stop()

    berlin_today = datetime.now(ZoneInfo("Europe/Berlin")).date()
    earliest_date = daily_kpis["event_date"].min()

    date_column, _ = st.columns([1, 3])
    selected_date = date_column.date_input(
        "Event Date",
        value=berlin_today,
        min_value=earliest_date,
        max_value=berlin_today,
        format="DD.MM.YYYY",
    )

    selected_kpi = daily_kpis[daily_kpis["event_date"] == selected_date]
    daily_kpi_area, station_kpi_area = st.columns([3, 1])

    with daily_kpi_area:
        if selected_kpi.empty:
            st.warning("No event data is available for the selected date.")
        else:
            selected_values = selected_kpi.iloc[0]
            event_column, delay_column, on_time_column = st.columns(3)
            event_column.metric(
                "Daily Events", f"{int(selected_values['total_events']):,}"
            )
            delay_column.metric(
                "Avg Delay", f"{float(selected_values['avg_delay']):.1f} min"
            )
            on_time_column.metric(
                "On-Time Rate", f"{float(selected_values['on_time_pct']):.1f}%"
            )

    with station_kpi_area:
        st.metric("Stations Monitored", int(station_count["stations"].iloc[0]))

except Exception as e:
    st.error(f"Could not load KPIs: {e}")
    st.info("Make sure DATABASE_URL is configured and marts tables are populated.")
    st.stop()

st.divider()

# Delay trend over time (daily average)
st.subheader("Daily Average Delay Trend")

trend_query = """
    SELECT
        DATE_TRUNC('day', fetched_at)::date AS day,
        ROUND(AVG(delay_minutes)::numeric, 2) AS avg_delay,
        COUNT(*) AS event_count
    FROM marts.fct_delays
    GROUP BY day
    ORDER BY day
"""

try:
    trend = run_query(trend_query)
    if not trend.empty:
        fig_trend = px.line(
            trend,
            x="day",
            y="avg_delay",
            labels={"day": "Date", "avg_delay": "Avg Delay (min)"},
            markers=True,
        )
        fig_trend.update_layout(
            hovermode="x unified",
            height=350,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("Not enough data for a trend chart yet.")
except Exception as e:
    st.error(f"Could not load trend data: {e}")

st.divider()

# Delay by station and delay by hour
col_left, col_right = st.columns(2)

# Delay by station
with col_left:
    st.subheader("Average Delay by Station")

    station_query = """
        SELECT
            s.station_name,
            ROUND(AVG(f.delay_minutes)::numeric, 2) AS avg_delay,
            COUNT(*) AS event_count
        FROM marts.fct_delays f
        JOIN marts.dim_stations s ON f.station_id = s.station_id
        GROUP BY s.station_name
        ORDER BY avg_delay DESC
    """

    try:
        stations = run_query(station_query)
        if not stations.empty:
            fig_station = px.bar(
                stations,
                x="avg_delay",
                y="station_name",
                orientation="h",
                labels={"avg_delay": "Avg Delay (min)", "station_name": "Station"},
                color="avg_delay",
                color_continuous_scale="RdYlGn_r",
            )
            fig_station.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=10, b=20),
                yaxis=dict(categoryorder="total ascending"),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_station, use_container_width=True)
        else:
            st.info("No station data available.")
    except Exception as e:
        st.error(f"Could not load station data: {e}")

# Delay by hour of day
with col_right:
    st.subheader("Average Delay by Hour")

    hour_query = """
        SELECT
            hour_of_day,
            ROUND(AVG(delay_minutes)::numeric, 2) AS avg_delay,
            COUNT(*) AS event_count
        FROM marts.fct_delays
        GROUP BY hour_of_day
        ORDER BY hour_of_day
    """

    try:
        hours = run_query(hour_query)
        if not hours.empty:
            fig_hour = px.bar(
                hours,
                x="hour_of_day",
                y="avg_delay",
                labels={"hour_of_day": "Hour of Day", "avg_delay": "Avg Delay (min)"},
                color="avg_delay",
                color_continuous_scale="RdYlGn_r",
            )
            fig_hour.update_layout(
                showlegend=False,
                height=400,
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis=dict(dtick=1),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_hour, use_container_width=True)
        else:
            st.info("No hourly data available.")
    except Exception as e:
        st.error(f"Could not load hourly data: {e}")

st.divider()

# Delay by train type
st.subheader("Average Delay by Train Type")

train_type_query = """
    SELECT
        r.train_type,
        ROUND(AVG(f.delay_minutes)::numeric, 2) AS avg_delay,
        COUNT(*) AS event_count
    FROM marts.fct_delays f
    JOIN marts.dim_routes r ON f.route_id = r.route_id
    WHERE r.train_type IS NOT NULL
    GROUP BY r.train_type
    ORDER BY avg_delay DESC
"""

try:
    train_types = run_query(train_type_query)
    if not train_types.empty:
        fig_type = px.bar(
            train_types,
            x="train_type",
            y="avg_delay",
            labels={"train_type": "Train Type", "avg_delay": "Avg Delay (min)"},
            color="avg_delay",
            color_continuous_scale="RdYlGn_r",
            text="event_count",
        )
        fig_type.update_traces(texttemplate="%{text:,} events", textposition="outside")
        fig_type.update_layout(
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=10, b=40),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("No train type data available.")
except Exception as e:
    st.error(f"Could not load train type data: {e}")
