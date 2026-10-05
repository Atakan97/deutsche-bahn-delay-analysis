"""
Airflow DAG and task definitions for the Deutsche Bahn ELT pipeline

The DAG runs every 15 minutes locally or through an external trigger:
- Fetch departure and arrival data from the official DB Timetables API
  for each of the 10 monitored stations
- Insert the raw JSON responses into raw.train_events table in PostgreSQL
- Run dbt transformations
"""

import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import psycopg2
from airflow.decorators import dag, task  # pyright: ignore[reportMissingImports]

from data_pipeline.extract.timetables import TimetablesClient
from data_pipeline.load.db_writer import write_train_events

logger = logging.getLogger(__name__)
EVENT_TYPES = ["departure", "arrival"]


def get_pipeline_schedule() -> str | None:
    """Read the DAG schedule from the environment"""
    schedule = os.environ.get("PIPELINE_SCHEDULE", "*/15 * * * *").strip()
    if schedule.lower() == "none":
        return None
    return schedule


def get_station_ids(database_url: str) -> list[str]:
    """Load the list of monitored station IDs from raw.stations
    """
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT station_id FROM raw.stations ORDER BY station_id;")
            rows = cur.fetchall()
    finally:
        conn.close()

    station_ids = [row[0] for row in rows]
    if not station_ids:
        raise RuntimeError(
            "No stations found in raw.stations. "
            "Run 'python -m data_pipeline.extract.seed_stations' first."
        )

    logger.info("Loaded %d station IDs from raw.stations.", len(station_ids))
    return station_ids

def fetch_station_events(
    client: TimetablesClient,
    station_id: str,
) -> dict[str, list[dict]]:
    """Fetch both departure and arrival boards for a single station"""
    events = client.get_station_events(station_id)
    logger.info(
        "Fetched %d departures and %d arrivals for station %s",
        len(events["departure"]),
        len(events["arrival"]),
        station_id,
    )
    return events

def load_events(
    database_url: str,
    events: list[dict],
    station_id: str,
    event_type: str,
) -> int:
    """Write raw API events into the raw.train_events table
    """
    count = write_train_events(
        database_url=database_url,
        events=events,
        station_id=station_id,
        event_type=event_type,
        source="db-timetables-v1",
    )
    logger.info("Loaded %d %ss for station %s", count, event_type, station_id)
    return count

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
            os.environ["SUPABASE_USER"] = unquote(parsed.username)
        if parsed.password:
            os.environ["SUPABASE_PASSWORD"] = unquote(parsed.password)
        if parsed.path:
            os.environ["SUPABASE_DB"] = unquote(parsed.path.lstrip("/"))
        query = parse_qs(parsed.query)
        if query.get("sslmode"):
            os.environ["SUPABASE_SSLMODE"] = query["sslmode"][0]

    logger.info("Running dbt transformations (dbt run)...")
    result = subprocess.run(
        ["dbt", "run", "--profiles-dir", "."],
        cwd=transform_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("dbt run failed:\n%s", result.stderr)
        raise RuntimeError(
            f"dbt run failed with return code {result.returncode}:\n{result.stderr}\n{result.stdout}"
        )

    logger.info("dbt transformations completed successfully.")

DAG_DEFAULT_ARGS = {
    "owner": "atakan",
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

@dag(
    dag_id="deutsche-bahn-elt-pipeline",
    description="Extract train data from DB Timetables API, load into PostgreSQL, transform with dbt",
    schedule=get_pipeline_schedule(),
    start_date=datetime(2024, 1, 1),
    catchup=False,
    is_paused_upon_creation=False,
    max_active_runs=1,
    default_args=DAG_DEFAULT_ARGS,
    tags=["elt", "deutsche-bahn", "dbt"],
)
def elt_pipeline_dag() -> None:
    """Extract raw data from the API, load into PostgreSQL,
    and transform with dbt into staging and marts models
    """

    @task()
    def extract_and_load() -> None:
        """Fetch events from every station and load them into PostgreSQL."""
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Set it in the .env file or Docker Compose environment."
            )

        client_id = os.environ.get("DB_CLIENT_ID")
        api_key = os.environ.get("DB_API_KEY")
        if not client_id or not api_key:
            raise RuntimeError(
                "DB_CLIENT_ID and DB_API_KEY must be set for the Timetables API."
            )

        station_ids = get_station_ids(database_url)
        total_fetched = 0
        total_loaded = 0

        with TimetablesClient(client_id=client_id, api_key=api_key) as client:
            for station_id in station_ids:
                station_events = fetch_station_events(
                    client=client, station_id=station_id
                )
                for event_type in EVENT_TYPES:
                    events = station_events[event_type]
                    total_fetched += len(events)

                    if events:
                        count = load_events(
                            database_url=database_url,
                            events=events,
                            station_id=station_id,
                            event_type=event_type,
                        )
                        total_loaded += count

        logger.info(
            "Extract phase completed. Stations: %d, fetched: %d, loaded: %d",
            len(station_ids),
            total_fetched,
            total_loaded,
        )

    @task()
    def transform() -> None:
        """Run dbt transformations after extract_and_load completes"""
        run_dbt_transformations()

    extract_task = extract_and_load()
    transform_task = transform()
    extract_task >> transform_task

elt_pipeline = elt_pipeline_dag()
