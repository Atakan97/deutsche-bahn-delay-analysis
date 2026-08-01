"""
Prefect flow and task definitions for the pipeline
This module defines the pipeline that runs every 15 minutes

Fetch departure and arrival data from the official DB Timetables API
for each of the 10 stations

Insert the raw JSON responses into raw.train_events table in PostgreSQL
"""

import logging
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import psycopg2
from dotenv import load_dotenv
from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from data_pipeline.extract.timetables import TimetablesClient
from data_pipeline.load.db_writer import write_train_events

logger = logging.getLogger(__name__)

# Event types collection for each station
EVENT_TYPES = ["departure", "arrival"]

@task(
    name="fetch_station_events",
    cache_policy=NO_CACHE,
    retries=2,
    retry_delay_seconds=5,
    log_prints=True,
)
def fetch_station_events(
    client: TimetablesClient,
    station_id: str,
) -> dict[str, list[dict]]:
    """Fetch both station boards in one Timetables API"""
    events = client.get_station_events(station_id)
    print(
        f"  Fetched {len(events['departure'])} departures and "
        f"{len(events['arrival'])} arrivals for station {station_id}"
    )
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
        source="db-timetables-v1",
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


@task(
    name="run_dbt_transformations",
    retries=1,
    retry_delay_seconds=5,
    log_prints=True,
)
def run_dbt_transformations() -> None:
    """Run dbt transformations for staging and marts tables"""
    transform_dir = Path(__file__).resolve().parent.parent.parent / "transform"

    db_url = os.environ.get("DATABASE_URL")
    if db_url and not os.environ.get("SUPABASE_HOST"):
        parsed = urlparse(db_url)
        if parsed.hostname:
            os.environ["SUPABASE_HOST"] = parsed.hostname
        if parsed.port:
            os.environ["SUPABASE_PORT"] = str(parsed.port)
        if parsed.username:
            os.environ["SUPABASE_USER"] = parsed.username
        if parsed.password:
            os.environ["SUPABASE_PASSWORD"] = parsed.password
        if parsed.path:
            os.environ["SUPABASE_DB"] = parsed.path.lstrip("/")

    print("Installing dbt package dependencies (dbt deps)...")
    deps_result = subprocess.run(
        ["dbt", "deps", "--profiles-dir", "."],
        cwd=transform_dir,
        capture_output=True,
        text=True,
    )
    if deps_result.returncode != 0:
        print(f"dbt deps failed:\n{deps_result.stderr}")
        raise RuntimeError(
            f"dbt deps failed with return code {deps_result.returncode}:\n{deps_result.stderr}\n{deps_result.stdout}"
        )

    print("Running dbt transformations (dbt run)...")
    result = subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd=transform_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"dbt run failed:\n{result.stderr}")
        raise RuntimeError(
            f"dbt run failed with return code {result.returncode}:\n{result.stderr}\n{result.stdout}"
        )

    print("dbt transformations completed successfully.")


@flow(
    name="deutsche-bahn-elt-pipeline",
    log_prints=True,
)
def elt_pipeline() -> None:
    """Extract raw data from the API, load into PostgreSQL,
    and transform with dbt into staging and marts models

    Designed to run every 15 minutes with a GitHub Actions
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

    total_fetched = 0
    total_loaded = 0

    # Use a single client for all API calls so the TCP connection is reused
    client_id = os.environ.get("DB_CLIENT_ID")
    api_key = os.environ.get("DB_API_KEY")
    if not client_id or not api_key:
        raise RuntimeError("DB_CLIENT_ID and DB_API_KEY must be set for the Timetables API.")

    with TimetablesClient(client_id=client_id, api_key=api_key) as client:
        for station_id in station_ids:
            station_events = fetch_station_events(client=client, station_id=station_id)
            for event_type in EVENT_TYPES:
                events = station_events[event_type]
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

    # Run dbt transformations
    run_dbt_transformations()

    print("")
    print("=" * 60)
    print("Pipeline complete.")
    print(f"  Stations processed : {len(station_ids)}")
    print(f"  Events fetched     : {total_fetched}")
    print(f"  Events loaded      : {total_loaded}")
    print("=" * 60)

if __name__ == "__main__":
    elt_pipeline()
