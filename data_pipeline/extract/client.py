"""
HTTP client for the v6.db.transport.rest API

Covers all interactions with the Deutsche Bahn REST API into a single class: TransportApiClient 
"""

import logging
import time
import httpx

logger = logging.getLogger(__name__)

# Base URL for the community-maintained DB HAFAS REST API
DEFAULT_BASE_URL = "https://v6.db.transport.rest"

# How long to wait for a single HTTP response before timing out
DEFAULT_TIMEOUT_SECONDS = 30.0

# How many times to retry a failed request before giving up
MAX_RETRIES = 3

# Seconds to wait between retries, 1st retry = 1s, 2nd retry = 2s, 3rd retry = 3s.
RETRY_BACKOFF_SECONDS = 1.0

# Seconds to sleep between sequential API calls to avoid overwhelming the community server
RATE_LIMIT_SECONDS = 0.3


class TransportApiClient:
    """HTTP client for v6.db.transport.rest (Deutsche Bahn HAFAS API)
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        """Initialise the client with a base URL and shared httpx.Client
        """
        self._client = httpx.Client(
            base_url=base_url,
            timeout=DEFAULT_TIMEOUT_SECONDS,
        )

    # Context manager support

    def __enter__(self) -> "TransportApiClient":
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection pool"""
        self._client.close()

    # Private helpers

    def _request_with_retry(
        self,
        method: str,
        path: str,
        params: dict | None = None,
    ) -> list | dict:
        """Make an HTTP request with simple retry logic

          The v6.db.transport.rest API sometimes returns 5xx errors or times out under load
          Retrying 2–3 times handles most temporary failures
        """
        last_exception: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.request(method, path, params=params)
                response.raise_for_status()

                # Pause briefly after each successful call
                time.sleep(RATE_LIMIT_SECONDS)

                return response.json()

            except httpx.HTTPStatusError as e:
                last_exception = e
                status_code = e.response.status_code

                # Only retry on server errors (5xx), client errors (4xx) are not temporary
                if status_code < 500:
                    logger.error(
                        "Client error %d on %s %s — not retrying.",
                        status_code,
                        method,
                        path,
                    )
                    raise

                logger.warning(
                    "Server error %d on %s %s (attempt %d/%d). Retrying in %ds...",
                    status_code,
                    method,
                    path,
                    attempt,
                    MAX_RETRIES,
                    attempt * RETRY_BACKOFF_SECONDS,
                )

            except httpx.RequestError as e:
                last_exception = e
                logger.warning(
                    "Network error on %s %s (attempt %d/%d): %s. Retrying in %ds...",
                    method,
                    path,
                    attempt,
                    MAX_RETRIES,
                    str(e),
                    attempt * RETRY_BACKOFF_SECONDS,
                )

            # Wait longer on each retry attempt
            time.sleep(attempt * RETRY_BACKOFF_SECONDS)
        logger.error(
            "All %d retries exhausted for %s %s.", MAX_RETRIES, method, path
        )
        raise last_exception  # type: ignore[misc]

    # Public API methods

    def search_stations(self, query: str, results: int = 5) -> list[dict]:
        """Search for stations by name

        Calls GET /locations with the given query
        Used once during initial station seeding (seed_stations.py)
        """
        response_data = self._request_with_retry(
            method="GET",
            path="/locations",
            params={
                "query": query,
                "results": results,
                "stops": "true",
                "addresses": "false",
                "poi": "false",
            },
        )

        # The API returns a list of location objects
        if not isinstance(response_data, list):
            logger.warning(
                "Unexpected response type from /locations: %s", type(response_data)
            )
            return []

        return response_data

    def get_departures(self, station_id: str, results: int = 60) -> list[dict]:
        """Fetch upcoming departures from a station

        Calls GET /stops/{station_id}/departures
        """
        response_data = self._request_with_retry(
            method="GET",
            path=f"/stops/{station_id}/departures",
            params={"results": results},
        )

        if not isinstance(response_data, list):
            logger.warning(
                "Unexpected response type from /departures for station %s: %s",
                station_id,
                type(response_data),
            )
            return []

        return response_data

    def get_arrivals(self, station_id: str, results: int = 60) -> list[dict]:
        """Fetch upcoming arrivals at a station

        Calls GET /stops/{station_id}/arrivals
        """
        response_data = self._request_with_retry(
            method="GET",
            path=f"/stops/{station_id}/arrivals",
            params={"results": results},
        )

        if not isinstance(response_data, list):
            logger.warning(
                "Unexpected response type from /arrivals for station %s: %s",
                station_id,
                type(response_data),
            )
            return []

        return response_data
