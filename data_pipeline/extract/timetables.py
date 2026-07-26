"""Client and normaliser for DB's free Timetables API"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
BERLIN = ZoneInfo("Europe/Berlin")
MAX_RETRIES = 3
RATE_LIMIT_SECONDS = 0.35  # Below the free plan's 60 requests/minute limit


class TimetablesClient:
    """Read planned and changed station boards, returned event dicts"""

    def __init__(self, client_id: str, api_key: str, base_url: str = DEFAULT_BASE_URL) -> None:
        if not client_id or not api_key:
            raise ValueError("DB_CLIENT_ID and DB_API_KEY must both be configured.")
        self._client = httpx.Client(
            base_url=base_url,
            timeout=30.0,
            headers={"DB-Client-ID": client_id, "DB-Api-Key": api_key},
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_station_events(self, station_id: str, now=None):
        """Return departure and arrival events for the current and next hour"""
        if now is None:
            now = datetime.now(BERLIN)
        current = now.astimezone(BERLIN).replace(minute=0, second=0, microsecond=0)
        next_hour = current + timedelta(hours=1)

        plan_roots = []
        for hour in (current, next_hour):
            path = f"/plan/{station_id}/{hour.strftime('%y%m%d')}/{hour.strftime('%H')}"
            plan_roots.append(self._get_xml(path))

        changes = _read_changes(self._get_xml(f"/fchg/{station_id}"))
        changes.update(_read_changes(self._get_xml(f"/rchg/{station_id}")))

        events = {"departure": [], "arrival": []}
        seen = set()
        for root in plan_roots:
            for stop in root.findall(".//s"):
                trip_id = stop.get("id")
                if not trip_id:
                    continue
                for tag, event_type in (("dp", "departure"), ("ar", "arrival")):
                    node = stop.find(tag)
                    if node is None or (trip_id, event_type) in seen:
                        continue
                    event = _normalise_event(trip_id, event_type, node, changes.get((trip_id, event_type)), stop)
                    events[event_type].append(event)
                    seen.add((trip_id, event_type))
        return events

    def _get_xml(self, path: str):
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.get(path)
                response.raise_for_status()
                time.sleep(RATE_LIMIT_SECONDS)
                return ET.fromstring(response.content)
            except (httpx.HTTPError, ET.ParseError) as error:
                last_error = error
                if attempt == MAX_RETRIES:
                    break
                logger.warning("Timetables request %s failed (attempt %d/%d): %s", path, attempt, MAX_RETRIES, error)
                time.sleep(attempt)
        raise RuntimeError(f"Timetables request failed for {path}.") from last_error


def _read_changes(root):
    """Build a dict of the latest change nodes, keyed by (trip_id, event_type)"""
    changes = {}
    for stop in root.findall(".//s"):
        trip_id = stop.get("id")
        if not trip_id:
            continue
        for tag, event_type in (("dp", "departure"), ("ar", "arrival")):
            node = stop.find(tag)
            if node is not None:
                changes[(trip_id, event_type)] = node
    return changes


def _normalise_event(trip_id, event_type, planned, change, stop):
    """Build a single event dict from planned and changed XML nodes"""
    planned_time = _parse_time(planned.get("pt"))
    changed_time = _parse_time(change.get("ct")) if change is not None else None
    cancelled = change is not None and change.get("cs") == "c"
    actual_time = None if cancelled else (changed_time or planned_time)

    if actual_time and planned_time:
        delay = int((actual_time - planned_time).total_seconds())
    else:
        delay = None

    line = stop.find("tl")
    category = line.get("c") if line is not None else None
    number = line.get("n") if line is not None else None

    if category and number:
        line_name = f"{category} {number}"
    elif category:
        line_name = category
    elif number:
        line_name = number
    else:
        line_name = None

    return {
        "tripId": trip_id,
        "plannedWhen": planned_time.isoformat() if planned_time else None,
        "when": actual_time.isoformat() if actual_time else None,
        "delay": delay,
        "line": {"name": line_name, "fahrtNr": number, "id": trip_id, "product": category},
        "direction": planned.get("p") or planned.get("pp"),
        "platform": (change.get("cp") if change is not None else None) or planned.get("pp"),
        "source": "db-timetables-v1",
    }


def _parse_time(value):
    """Parse YYMMDDHHMM timetable timestamp in the Berlin timezone"""
    if not value:
        return None
    return datetime.strptime(value[:10], "%y%m%d%H%M").replace(tzinfo=BERLIN)
