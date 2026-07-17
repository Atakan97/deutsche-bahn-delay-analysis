"""
Prefect flow and task definitions for the pipeline
This module defines the pipeline that runs every 15 minutes

- Fetch departure and arrival data from the v6.db.transport.rest API 
for each of the 10 monitored stations

- Insert the raw JSON responses into raw.train_events table in PostgreSQL
"""

import logging
import os
import psycopg2
from dotenv import load_dotenv
from prefect import flow, task

from data_pipeline.extract.client import TransportApiClient
from data_pipeline.load.db_writer import write_train_events

logger = logging.getLogger(__name__)

# Event types collection for each station
EVENT_TYPES = ["departure", "arrival"]

@task(
    name="fetch_station_events",
    retries=2,
    retry_delay_seconds=5,
    log_prints=True,
)
def fetch_station_events(
    client: TransportApiClient,
    station_id: str,
    event_type: str,
) -> list[dict]:
    """Fetch departure or arrival events for a single station from the API
    Extract step, the returned list contains raw JSON dicts as the API returned them
    """
    if event_type == "departure":
        events = client.get_departures(station_id)
    elif event_type == "arrival":
        events = client.get_arrivals(station_id)
    else:
        raise ValueError(f"Unknown event_type: {event_type}")

    print(f"  Fetched {len(events)} {event_type}s for station {station_id}")
    return events

@task(
    name="load_events",
    retries=1,
    retry_delay_seconds=3,
    log_prints=True,
)
def load_events(
    database_url: str,
    events: list[dict],
    station_id: str,
    event_type: str,
) -> int:
    """Write raw API events into the raw.train_events table
    Load step, each event dict is stored as-is in the raw_data JSONB column
    """
    count = write_train_events(
        database_url=database_url,
        events=events,
        station_id=station_id,
        event_type=event_type,
    )
    print(f"  Loaded {count} {event_type}s for station {station_id}")
    return count

@task(
    name="get_station_ids",
    log_prints=True,
)
def get_station_ids(database_url: str) -> list[str]:
    """Load the list of monitored station IDs from raw.stations
    
    The flow always uses stations are actually seeded, adding or removing stations 
    only requires re-running seed_stations.py, not editing this flow code
    """
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT station_id FROM raw.stations ORDER BY station_id;")
            rows = cur.fetchall()
    finally:
        conn.close()

    station_ids = [row[0] for row in rows]

    # If no stations are found
    if not station_ids:
        raise RuntimeError(
            "No stations found in raw.stations. "
            "Run 'python -m data_pipeline.extract.seed_stations' first."
        )

    print(f"Loaded {len(station_ids)} station IDs from raw.stations.")
    return station_ids

@flow(
    name="deutsche-bahn-etl-pipeline",
    log_prints=True,
)
def etl_pipeline() -> None:
    """Main pipeline flow: Extract raw data from the API, Load into PostgreSQL

    This flow is designed to run every 15 minutes with a GitHub Actions cron
    job, each run typically takes 30–60 seconds depending on API response times
    """
    # Load .env for local development
    load_dotenv()

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it in .env (local) or GitHub Actions secrets (CI)."
        )

    print("=" * 60)
    print("Deutsche Bahn ELT Pipeline — Starting")
    print("=" * 60)

    # Get the list of stations to monitor
    station_ids = get_station_ids(database_url)

    # Fetch and load events for each station × event type
    total_fetched = 0
    total_loaded = 0

    # Use a single shared client for all API calls so the TCP connection is reused for requests
    with TransportApiClient() as client:
        for station_id in station_ids:
            for event_type in EVENT_TYPES:
                # Extract, fetch events from the API
                events = fetch_station_events(
                    client=client,
                    station_id=station_id,
                    event_type=event_type,
                )
                total_fetched += len(events)

                # Load, insert raw JSON into PostgreSQL
                if events:
                    count = load_events(
                        database_url=database_url,
                        events=events,
                        station_id=station_id,
                        event_type=event_type,
                    )
                    total_loaded += count

    # Log summary
    print("")
    print("=" * 60)
    print("Pipeline complete.")
    print(f"  Stations processed : {len(station_ids)}")
    print(f"  Events fetched     : {total_fetched}")
    print(f"  Events loaded      : {total_loaded}")
    print("=" * 60)
    
if __name__ == "__main__":
    etl_pipeline()
