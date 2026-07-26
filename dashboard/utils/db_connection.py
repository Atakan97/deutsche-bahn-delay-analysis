"""
Database connection helper for the Streamlit dashboard
"""

import os

import pandas as pd
import psycopg2
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


@st.cache_resource
def _get_database_url() -> str:
    """Resolve the DATABASE_URL from Streamlit secrets or env vars
    """
    # Environment variable
    url = os.environ.get("DATABASE_URL")
    if url:
        return url

    # Streamlit secrets
    try:
        return st.secrets["DATABASE_URL"]
    except (KeyError, FileNotFoundError):
        pass

    raise RuntimeError(
        "DATABASE_URL not found. Set it in .streamlit/secrets.toml "
        "(for Streamlit Cloud) or as an environment variable (for local dev)."
    )


@st.cache_data(ttl=300) # Results are cached for 5 minutes
def run_query(sql: str) -> pd.DataFrame:
    """Execute a SQL query and return the result as a DataFrame
    """
    database_url = _get_database_url()
    conn = psycopg2.connect(database_url)
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()
    return df
