"""Client for DB's StaDa (Station Data) API

StaDa is the source for the station data used by the
project, it runs separately from the timetable client, because
station data changes slowly and is synchronised once per day
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://apis.deutschebahn.com/db-api-marketplace/apis/station-data/v2"
MAX_RETRIES = 3


@dataclass(frozen=True)
class Station:
    """The station fields consumed by the warehouse"""

    station_id: str  # EVA number, also used by the Timetables API
    name: str
    latitude: float
    longitude: float


def _normalise_name(name: str) -> str:
    """Normalise station name for clean comparison"""
    return name.lower().replace("hauptbahnhof", "hbf").replace(" ", "").replace("-", "")


def _get_search_token(name: str) -> str:
    """Extract the first word to use as a generic search token (e.g. 'Berlin' from 'Berlin Hbf')"""
    words = name.split()
    if words:
        return words[0]
    return name


OFFICIAL_STADA_NAMES = {
    "Frankfurt(Main)Hbf": "Frankfurt (Main) Hbf",
    "München Hbf": "München Hbf",
    "Berlin Hbf": "Berlin Hauptbahnhof",
    "Köln Hbf": "Köln Hbf",
    "Hamburg Hbf": "Hamburg Hbf",
    "Passau Hbf": "Passau Hbf",
    "Stuttgart Hbf": "Stuttgart Hbf",
    "Leipzig Hbf": "Leipzig Hbf",
    "Nürnberg Hbf": "Nürnberg Hbf",
    "Dresden Hbf": "Dresden Hbf",
}


EVA_OVERRIDES = {
    "Berlin Hauptbahnhof": "8098160",
    "Berlin Hbf": "8098160",
}


class StadaClient:
    """Fetch DB station data using Marketplace credentials"""

    def __init__(
        self,
        client_id: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        if not client_id or not api_key:
            raise ValueError("DB_CLIENT_ID and DB_API_KEY must both be configured.")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=30.0,
            headers={"DB-Client-ID": client_id, "DB-Api-Key": api_key},
        )

    def __enter__(self) -> "StadaClient":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def find_stations(self, names: list[str]) -> dict[str, Station]:
        """Resolve all monitored stations using DB's StaDa API"""
        resolved: dict[str, Station] = {}
        for name in names:
            search_term = OFFICIAL_STADA_NAMES.get(name, name)
            try:
                payload = self._get_json("/stations", params={"searchstring": search_term})
                results = payload.get("result", payload.get("results", []))
                if isinstance(results, list) and results:
                    for item in results:
                        station = self._to_station(item)
                        if _normalise_name(station.name) == _normalise_name(name):
                            resolved[name] = station
                            break
            except Exception as e:
                logger.warning("Exact search for %r failed: %s", search_term, e)

            # If the exact search fails, use a wildcard search (e.g. *Berlin*)
            # to match name variations like "Berlin Hbf" and "Berlin Hauptbahnhof"
            if name not in resolved:
                search_term_wc = f"*{_get_search_token(name)}*"
                try:
                    payload_wc = self._get_json(
                        "/stations",
                        params={"searchstring": search_term_wc, "limit": "100"},
                    )
                    results_wc = payload_wc.get("result", payload_wc.get("results", []))
                    if isinstance(results_wc, list):
                        for item in results_wc:
                            station = self._to_station(item)
                            if _normalise_name(station.name) == _normalise_name(name):
                                resolved[name] = station
                                break
                except Exception as e:
                    logger.warning("Fallback search for %r failed: %s", search_term_wc, e)

            if name not in resolved:
                logger.error("StaDa did not return an exact match for %r.", name)
        return resolved

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.get(path, params=params)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("StaDa response is not a JSON object.")
                return payload
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                if attempt == MAX_RETRIES:
                    break
                logger.warning("StaDa request failed (attempt %d/%d): %s", attempt, MAX_RETRIES, error)
                time.sleep(attempt)
        raise RuntimeError("StaDa station sync failed after retries.") from last_error

    @staticmethod
    def _to_station(record: dict[str, Any]) -> Station:
        eva_numbers = record.get("evaNumbers", [])
        if not isinstance(eva_numbers, list):
            raise ValueError("StaDa station has no EVA number list.")

        # Find main EVA number
        main_eva = None
        for item in eva_numbers:
            if item.get("isMain"):
                main_eva = item
                break

        if main_eva is None and len(eva_numbers) == 1:
            main_eva = eva_numbers[0]

        if not isinstance(main_eva, dict) or main_eva.get("number") is None:
            raise ValueError(f"StaDa station has no usable main EVA number: {record!r}")

        point = main_eva.get("geographicCoordinates", {})
        coordinates = point.get("coordinates") if isinstance(point, dict) else None
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            raise ValueError(f"StaDa station has no usable coordinates: {record!r}")

        name = record.get("name")
        if not isinstance(name, str):
            raise ValueError(f"StaDa station has no name: {record!r}")

        station_id = str(main_eva["number"])
        if name in EVA_OVERRIDES:
            station_id = EVA_OVERRIDES[name]

        return Station(
            station_id=station_id,
            name=name,
            longitude=float(coordinates[0]),
            latitude=float(coordinates[1]),
        )
