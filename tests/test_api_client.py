"""Unit tests for the dashboard API client"""

import httpx
import pytest

from dashboard.utils import api_client


# The client should return model options from the API
def test_request_model_options_returns_response_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_options = {
        "station_categories": ["major_hub", "regional_hub"],
        "train_types": ["ICE", "RE"],
        "event_types": ["arrival", "departure"],
    }
    response = httpx.Response(
        status_code=200,
        json=expected_options,
        request=httpx.Request("GET", "http://test-api/model-options"),
    )

    monkeypatch.setattr(api_client.httpx, "get", lambda *args, **kwargs: response)

    result = api_client._request_model_options("http://test-api")

    assert result == expected_options


# The client should raise an error for a failed API response
def test_request_model_options_raises_for_api_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = httpx.Response(
        status_code=503,
        request=httpx.Request("GET", "http://test-api/model-options"),
    )

    monkeypatch.setattr(api_client.httpx, "get", lambda *args, **kwargs: response)

    with pytest.raises(httpx.HTTPStatusError):
        api_client._request_model_options("http://test-api")
