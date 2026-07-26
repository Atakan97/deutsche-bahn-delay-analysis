"""
Streamlit dashboard entry point for Deutsche Bahn Delay Analysis
"""

import streamlit as st

# Page config
st.set_page_config(
    page_title="Deutsche Bahn Delay Analysis",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.title("DB Delay Analysis")
    st.markdown(
        """
        Real-time delay analytics for 10 major German
        railway stations.
        """
    )

# Landing page
st.title("Deutsche Bahn Delay Analysis Dashboard")
st.markdown(
    """
    Welcome to the Deutsche Bahn Delay Analysis dashboard.
    Use the sidebar to navigate between pages:

    - **Overview** : KPIs, delay trends, information by station and hour.
    - **Map** : Interactive map showing delay density in Germany.
    - **Predict** : Get a real-time delay prediction from the ML model.

    ---
    *Data is refreshed every 15 minutes by the automated pipeline.*
    """
)
