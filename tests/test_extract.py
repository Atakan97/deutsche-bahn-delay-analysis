"""Tests for the official DB extraction and raw writer"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import httpx
import pytest

from data_pipeline.extract.stada import StadaClient
from data_pipeline.extract.timetables import TimetablesClient
from data_pipeline.load.db_writer import write_train_events


def test_stada_resolves_main_eva_and_coordinates(httpx_mock) -> None:
    response = {
        "result": [
            {
                "name": "Berlin Hauptbahnhof",
                "evaNumbers": [
                    {
                        "number": 8011160,
                        "isMain": True,
                        "geographicCoordinates": {"coordinates": [13.3694, 52.5251]},
                    }
                ],
            }
        ]
    }
    httpx_mock.add_response(
        url=httpx.URL(
            "https://apis.deutschebahn.com/db-api-marketplace/apis/station-data/v2/stations",
            params=[("searchstring", "Berlin Hauptbahnhof")],
        ),
        json=response,
    )

    with StadaClient("client-id", "api-key") as client:
        stations = client.find_stations(["Berlin Hbf"])

    station = stations["Berlin Hbf"]
    assert station.station_id == "8098160"
    assert station.latitude == pytest.approx(52.5251)
    assert station.longitude == pytest.approx(13.3694)


def test_stada_requires_credentials() -> None:
    with pytest.raises(ValueError, match="DB_CLIENT_ID"):
        StadaClient("", "api-key")


@patch("data_pipeline.extract.timetables.time.sleep")
def test_timetables_normalises_plan_and_realtime_change(mock_sleep, httpx_mock) -> None:
    plan = b"""
        <timetable><s id="trip-123"><tl c="ICE" n="123"/>
        <dp pt="2607191200" pp="12" p="Muenchen Hbf"/>
        <ar pt="2607191155" pp="7" p="Berlin Hbf"/></s></timetable>
    """
    change = b"""
        <timetable><s id="trip-123"><dp ct="2607191205" cp="13"/></s></timetable>
    """
    base = "https://apis.deutschebahn.com/db-api-marketplace/apis/timetables/v1"
    httpx_mock.add_response(url=f"{base}/plan/8011160/260719/12", content=plan)
    httpx_mock.add_response(url=f"{base}/plan/8011160/260719/13", content=b"<timetable />")
    httpx_mock.add_response(url=f"{base}/fchg/8011160", content=change)
    httpx_mock.add_response(url=f"{base}/rchg/8011160", content=b"<timetable />")

    now = datetime(2026, 7, 19, 12, 4, tzinfo=ZoneInfo("Europe/Berlin"))
    with TimetablesClient("client-id", "api-key") as client:
        events = client.get_station_events("8011160", now=now)

    departure = events["departure"][0]
    assert departure["tripId"] == "trip-123"
    assert departure["plannedWhen"] == "2026-07-19T12:00:00+02:00"
    assert departure["when"] == "2026-07-19T12:05:00+02:00"
    assert departure["delay"] == 300
    assert departure["line"]["name"] == "ICE 123"
    assert departure["platform"] == "13"
    assert events["arrival"][0]["delay"] == 0
    mock_sleep.assert_called()


@patch("data_pipeline.load.db_writer.psycopg2.connect")
def test_writer_records_the_official_source(mock_connect: MagicMock) -> None:
    cursor = MagicMock()
    connection = MagicMock()
    connection.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    connection.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mock_connect.return_value = connection

    result = write_train_events(
        database_url="postgresql://fake",
        events=[{"tripId": "trip-123", "delay": 0}],
        station_id="8011160",
        event_type="departure",
    )

    assert result == 1
    _, params = cursor.execute.call_args.args
    assert json.loads(params[2]) == {"tripId": "trip-123", "delay": 0}
    assert params[3] == "db-timetables-v1"
