"""
HTTP client for the FastAPI prediction endpoint
"""

import os

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Resolve the prediction API base URL
def _get_api_url() -> str:

    url = os.environ.get("API_URL")
    if url:
        return url

    try:
        return st.secrets["API_URL"]
    except (KeyError, FileNotFoundError):
        pass

    return "http://localhost:8000"

# Request the category values supported by the model
def _request_model_options(api_url: str) -> dict[str, list[str]]:
    response = httpx.get(
        f"{api_url}/model-options",
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()

# Cache model options for 5 minutes
@st.cache_data(ttl=300)
def _get_cached_model_options(api_url: str) -> dict[str, list[str]]:
    return _request_model_options(api_url)

# Return the category values supported by the prediction API
def get_model_options() -> dict[str, list[str]] | None:
    api_url = _get_api_url()
    try:
        return _get_cached_model_options(api_url)
    except httpx.ConnectError:
        st.error(
            f"Could not connect to the prediction API at {api_url}. "
            "Make sure the API is running (uvicorn api.main:app --port 8000)."
        )
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"Could not load model options: {e.response.status_code}")
        return None
    except Exception as e:
        st.error(f"Unexpected error loading model options: {e}")
        return None

# Call the /predict endpoint and return the predicted delay in minutes
def predict_delay(features: dict) -> float | None:
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
