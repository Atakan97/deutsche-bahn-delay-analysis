"""
Unit tests for the extract layer
Use pytest-httpx to mock HTTP responses to avoild real API during tests
"""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError

from data_pipeline.extract.client import TransportApiClient
from data_pipeline.extract.schemas import (
    StationResponse,
    TrainEvent,
    TrainEventLine,
)
from data_pipeline.load.db_writer import write_train_events

# Test fixtures, reusable sample data
@pytest.fixture()
def sample_station_dict() -> dict:
    """A valid station response from the /locations endpoint"""
    return {
        "type": "stop",
        "id": "8000105",
        "name": "Frankfurt(Main)Hbf",
        "location": {
            "type": "location",
            "id": "8000105",
            "latitude": 50.106817,
            "longitude": 8.663003,
        },
        "products": {
            "nationalExpress": True,
            "national": True,
            "regional": True,
        },
    }


@pytest.fixture()
def sample_event_dict() -> dict:
    """A valid departure event from the /stops/{id}/departures endpoint"""
    return {
        "tripId": "1|12345|0|80|16072025",
        "stop": {
            "type": "stop",
            "id": "8000105",
            "name": "Frankfurt(Main)Hbf",
        },
        "when": "2025-07-16T14:07:00+02:00",
        "plannedWhen": "2025-07-16T14:05:00+02:00",
        "delay": 120,
        "line": {
            "type": "line",
            "id": "ice-123",
            "fahrtNr": "123",
            "name": "ICE 123",
            "product": "nationalExpress",
        },
        "direction": "München Hbf",
        "platform": "12",
    }


@pytest.fixture()
def sample_event_no_trip_id(sample_event_dict: dict) -> dict:
    """An event with tripId set to None (e.g. a replacement bus service)"""
    event = sample_event_dict.copy()
    event["tripId"] = None
    return event


class TestStationResponse:
    """Tests for the StationResponse Pydantic model"""

    def test_valid_station_parses_correctly(self, sample_station_dict: dict) -> None:
        """A station response should parse without errors"""
        station = StationResponse.model_validate(sample_station_dict)

        assert station.id == "8000105"
        assert station.name == "Frankfurt(Main)Hbf"
        assert station.location.latitude == pytest.approx(50.106817)
        assert station.location.longitude == pytest.approx(8.663003)

    def test_station_without_location_raises_error(self) -> None:
        """A station missing the required 'location' field should fail validation"""
        incomplete = {"id": "8000105", "name": "Frankfurt(Main)Hbf"}

        with pytest.raises(ValidationError) as exc_info:
            StationResponse.model_validate(incomplete)

        # Verify the error mentions the missing field
        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "location" in error_fields

    def test_station_with_extra_fields_is_allowed(
        self, sample_station_dict: dict
    ) -> None:
        """Extra fields should be accepted"""
        sample_station_dict["extraField"] = "should be ignored"
        station = StationResponse.model_validate(sample_station_dict)
        assert station.id == "8000105"

    def test_station_with_invalid_id_type_raises_error(self) -> None:
        """Passing a non-string id should raise a validation error
        """
        bad_data = {
            "id": 12345,  # should be a string
            "name": "Test Station",
            "location": {"latitude": 50.0, "longitude": 8.0},
        }
        with pytest.raises(ValidationError) as exc_info:
            StationResponse.model_validate(bad_data)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "id" in error_fields


class TestTrainEvent:
    """Tests for the TrainEvent Pydantic model."""

    def test_valid_event_parses_correctly(self, sample_event_dict: dict) -> None:
        """A departure event should parse with all fields present"""
        event = TrainEvent.model_validate(sample_event_dict)

        assert event.tripId == "1|12345|0|80|16072025"
        assert event.when == "2025-07-16T14:07:00+02:00"
        assert event.plannedWhen == "2025-07-16T14:05:00+02:00"
        assert event.delay == 120
        assert event.direction == "München Hbf"

    def test_event_line_parsed_correctly(self, sample_event_dict: dict) -> None:
        """The nested 'line' object should be parsed into a TrainEventLine model"""
        event = TrainEvent.model_validate(sample_event_dict)

        assert event.line is not None
        assert event.line.name == "ICE 123"
        assert event.line.product == "nationalExpress"
        assert event.line.fahrtNr == "123"

    def test_event_with_null_trip_id_is_valid(
        self, sample_event_no_trip_id: dict
    ) -> None:
        """Events with tripId=None should still parse"""
        event = TrainEvent.model_validate(sample_event_no_trip_id)

        assert event.tripId is None
        assert event.has_valid_trip_id() is False

    def test_has_valid_trip_id_with_valid_id(self, sample_event_dict: dict) -> None:
        """has_valid_trip_id() should return True when tripId is a non-empty string"""
        event = TrainEvent.model_validate(sample_event_dict)
        assert event.has_valid_trip_id() is True

    def test_has_valid_trip_id_with_blank_string(self, sample_event_dict: dict) -> None:
        """has_valid_trip_id() should return False for whitespace-only tripId"""
        sample_event_dict["tripId"] = "   "
        event = TrainEvent.model_validate(sample_event_dict)
        assert event.has_valid_trip_id() is False

    def test_event_with_null_delay_is_valid(self, sample_event_dict: dict) -> None:
        """delay=None is normal (e.g. when delay info is unavailable)"""
        sample_event_dict["delay"] = None
        event = TrainEvent.model_validate(sample_event_dict)
        assert event.delay is None

    def test_event_with_extra_fields_is_allowed(self, sample_event_dict: dict) -> None:
        """Extra fields like 'platform' should be accepted and ignored"""
        event = TrainEvent.model_validate(sample_event_dict)
        assert event.tripId == "1|12345|0|80|16072025"

    def test_minimal_event_with_all_nulls(self) -> None:
        """An event with all optional fields set to None should still parse"""
        minimal = {
            "tripId": None,
            "stop": None,
            "when": None,
            "plannedWhen": None,
            "delay": None,
            "line": None,
            "direction": None,
        }
        event = TrainEvent.model_validate(minimal)
        assert event.tripId is None
        assert event.stop is None
        assert event.delay is None

    def test_event_with_invalid_delay_type_raises_error(self) -> None:
        """delay must be int or None — a non-numeric string should fail"""
        bad_event = {
            "tripId": "1|1|0|80|123",
            "delay": "not_a_number",
        }
        with pytest.raises(ValidationError) as exc_info:
            TrainEvent.model_validate(bad_event)

        errors = exc_info.value.errors()
        error_fields = [e["loc"][0] for e in errors]
        assert "delay" in error_fields


class TestTrainEventLine:
    """Tests for the TrainEventLine Pydantic model."""

    def test_valid_line_parses(self) -> None:
        """A complete line object should parse correctly"""
        line_data = {
            "name": "ICE 123",
            "fahrtNr": "123",
            "id": "ice-123",
            "product": "nationalExpress",
        }
        line = TrainEventLine.model_validate(line_data)
        assert line.name == "ICE 123"
        assert line.product == "nationalExpress"

    def test_line_with_all_nulls(self) -> None:
        """All line fields are optional — a dict of Nones should parse"""
        line = TrainEventLine.model_validate({})
        assert line.name is None
        assert line.product is None


class TestTransportApiClient:
    """Tests for the TransportApiClient using pytest-httpx mocking"""

    def test_get_departures_returns_list(self, httpx_mock) -> None:
        """Successful /departures call should return the parsed JSON list"""
        mock_response = [
            {"tripId": "1|1|0|80|123", "delay": 60},
            {"tripId": "1|2|0|80|456", "delay": 0},
        ]
        httpx_mock.add_response(
            url="https://v6.db.transport.rest/stops/8000105/departures?results=60",
            json=mock_response,
        )

        with TransportApiClient() as client:
            result = client.get_departures("8000105")

        assert result == mock_response
        assert len(result) == 2

    def test_get_arrivals_returns_list(self, httpx_mock) -> None:
        """Successful /arrivals call should return the parsed JSON list"""
        mock_response = [{"tripId": "1|3|0|80|789", "delay": 300}]
        httpx_mock.add_response(
            url="https://v6.db.transport.rest/stops/8000105/arrivals?results=60",
            json=mock_response,
        )

        with TransportApiClient() as client:
            result = client.get_arrivals("8000105")

        assert result == mock_response

    def test_search_stations_returns_list(self, httpx_mock) -> None:
        """Successful /locations call should return a list of station dicts"""
        mock_response = [
            {
                "id": "8000105",
                "name": "Frankfurt(Main)Hbf",
                "location": {"latitude": 50.1, "longitude": 8.6},
            }
        ]
        httpx_mock.add_response(
            url=httpx.URL(
                "https://v6.db.transport.rest/locations",
                params={
                    "query": "Frankfurt",
                    "results": "5",
                    "stops": "true",
                    "addresses": "false",
                    "poi": "false",
                },
            ),
            json=mock_response,
        )

        with TransportApiClient() as client:
            result = client.search_stations("Frankfurt")

        assert len(result) == 1
        assert result[0]["name"] == "Frankfurt(Main)Hbf"

    def test_client_raises_on_4xx_without_retry(self, httpx_mock) -> None:
        """A 4xx client error should fail without retrying"""
        httpx_mock.add_response(
            url="https://v6.db.transport.rest/stops/INVALID/departures?results=60",
            status_code=404,
        )

        with TransportApiClient() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                client.get_departures("INVALID")

        assert exc_info.value.response.status_code == 404
        # Should have been called exactly once
        assert len(httpx_mock.get_requests()) == 1

    def test_client_retries_on_5xx(self, httpx_mock) -> None:
        """A 5xx server error should trigger retries, then raise if all fail"""
        target_url = "https://v6.db.transport.rest/stops/8000105/departures?results=60"
        for _ in range(3):  # MAX_RETRIES = 3
            httpx_mock.add_response(url=target_url, status_code=503)

        with TransportApiClient() as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                client.get_departures("8000105")

        assert exc_info.value.response.status_code == 503
        # Should have been called MAX_RETRIES (3) times
        assert len(httpx_mock.get_requests()) == 3

    def test_empty_response_returns_empty_list(self, httpx_mock) -> None:
        """If the API returns an empty array, we should return an empty list"""
        httpx_mock.add_response(
            url="https://v6.db.transport.rest/stops/8000105/departures?results=60",
            json=[],
        )

        with TransportApiClient() as client:
            result = client.get_departures("8000105")

        assert result == []


class TestDbWriter:
    """Tests for the db_writer module
    Mock psycopg2.connect to avoid requiring a live PostgreSQL instance
    """

    def test_invalid_event_type_raises_value_error(self) -> None:
        """event_type must be 'departure' or 'arrival', anything else fails"""
        with pytest.raises(ValueError, match="event_type must be"):
            write_train_events(
                database_url="postgresql://fake",
                events=[{"tripId": "1|1|0|80|123"}],
                station_id="8000105",
                event_type="invalid",
            )

    def test_empty_events_returns_zero(self) -> None:
        """An empty event list should return 0 without touching the database"""
        result = write_train_events(
            database_url="postgresql://fake",
            events=[],
            station_id="8000105",
            event_type="departure",
        )
        assert result == 0

    @patch("data_pipeline.load.db_writer.psycopg2.connect")
    def test_successful_insert(self, mock_connect: MagicMock) -> None:
        """Verify that events are inserted with correct SQL and parameters"""
        # Set up mock cursor that tracks execute calls
        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        events = [
            {"tripId": "1|1|0|80|123", "delay": 60},
            {"tripId": "1|2|0|80|456", "delay": 0},
        ]

        result = write_train_events(
            database_url="postgresql://fake",
            events=events,
            station_id="8000105",
            event_type="departure",
        )

        # Should have inserted 2 rows
        assert result == 2

        # Verify execute was called twice (once per event)
        assert mock_cursor.execute.call_count == 2

        # Verify the first call has correct parameters
        first_call_args = mock_cursor.execute.call_args_list[0]
        sql = first_call_args[0][0]
        params = first_call_args[0][1]
        assert "INSERT INTO raw.train_events" in sql
        assert params[0] == "8000105"
        assert params[1] == "departure"
        # Third param is the JSON string of the event
        assert json.loads(params[2]) == events[0]

        # Verify commit was called
        mock_conn.commit.assert_called_once()
        mock_conn.close.assert_called_once()
