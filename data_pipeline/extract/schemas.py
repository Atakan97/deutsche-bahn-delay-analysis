"""
Pydantic models for v6.db.transport.rest API responses
These models validate the JSON that comes back from the API before storing it in the database
"""

from pydantic import BaseModel, Field

class StationLocation(BaseModel):
    """Geographic coordinates of a station
    """
    latitude: float
    longitude: float

class StationResponse(BaseModel):
    """A single station returned by GET /locations?query={name}
    """
    # Only validate the fields we actually need
    id: str
    name: str
    location: StationLocation

    # Allow extra fields from the API
    model_config = {"extra": "allow"}

class TrainEventLine(BaseModel):
    """
    Contains the line name (e.g. "ICE 123") and the product type (e.g. "nationalExpress")
    Extract line_name and train_type from this in the dbt staging layer
    """

    name: str | None = None
    fahrtNr: str | None = None
    id: str | None = None

    # Product field tells the broad category of train
    # Will be used as train_type in the staging layer
    product: str | None = None

    model_config = {"extra": "allow"}


class TrainEvent(BaseModel):
    """A single departure or arrival event from the API
    """

    # tripId is the unique identifier for a train journey
    # It can be None for replacement services, which we filter out later in dbt
    tripId: str | None = Field(default=None, alias="tripId")

    # The stop dict contains the station where this event was recorded
    stop: dict | None = None

    # Departure/arrival time, null if the train was cancelled
    when: str | None = None

    # Scheduled departure/arrival time, should always be present
    plannedWhen: str | None = Field(default=None, alias="plannedWhen")

    # Delay in seconds, null means delay is unknown
    delay: int | None = None

    # Line info (train number, product type), can be absent for some events
    line: TrainEventLine | None = None

    # Departures or arrivals station name
    direction: str | None = None

    model_config = {"extra": "allow", "populate_by_name": True}

    def has_valid_trip_id(self) -> bool:
        """Check if this event has a usable trip identifier
        Events without a tripId will be filtered out by the dbt staging model
        """
        return self.tripId is not None and self.tripId.strip() != ""
