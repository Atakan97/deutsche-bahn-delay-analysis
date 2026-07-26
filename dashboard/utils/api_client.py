"""
HTTP client for the FastAPI prediction endpoint
"""

import os
import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
def _get_api_url() -> str:
    """Resolve the prediction API base URL
    """
    # Environment variable
    url = os.environ.get("API_URL")
    if url:
        return url

    # Streamlit secrets
    try:
        return st.secrets["API_URL"]
    except (KeyError, FileNotFoundError):
        pass

    # Default for local development
    return "http://localhost:8000"

def predict_delay(features: dict) -> float | None:
    """Call the /predict endpoint and return the predicted delay in minutes
    """
    api_url = _get_api_url()
    try:
        response = httpx.post(
            f"{api_url}/predict",
            json=features,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()["predicted_delay_minutes"]
    except httpx.ConnectError:
        st.error(
            f"Could not connect to the prediction API at {api_url}. "
            "Make sure the API is running (uvicorn api.main:app --port 8000)."
        )
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API returned an error: {e.response.status_code} — {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Unexpected error calling prediction API: {e}")
        return None
