"""Initialization and synchronization station data from StaDa"""

from __future__ import annotations

import os

import psycopg2
from dotenv import load_dotenv

from data_pipeline.extract.stada import StadaClient, Station

STATIONS_TO_SEED = [
    "Frankfurt(Main)Hbf",
    "München Hbf",
    "Berlin Hbf",
    "Köln Hbf",
    "Hamburg Hbf",
    "Passau Hbf",
    "Stuttgart Hbf",
    "Leipzig Hbf",
    "Nürnberg Hbf",
    "Dresden Hbf",
]


def get_db_connection(database_url: str) -> psycopg2.extensions.connection:
    """Open and return a PostgreSQL connection."""
    return psycopg2.connect(database_url)


def create_schemas(conn: psycopg2.extensions.connection) -> None:
    """Create the raw, staging, and marts schemas"""
    conn.autocommit = True
    with conn.cursor() as cur:
        for schema in ("raw", "staging", "marts"):
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
            print(f"  Schema '{schema}' is ready.")
    conn.autocommit = False


def create_raw_tables(conn: psycopg2.extensions.connection) -> None:
    """Create the two source tables used by the flow"""
    statements = [
        """
        CREATE TABLE IF NOT EXISTS raw.stations (
            station_id TEXT PRIMARY KEY,
            station_name TEXT NOT NULL,
            latitude NUMERIC(9,6) NOT NULL,
            longitude NUMERIC(9,6) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS raw.train_events (
            id BIGSERIAL PRIMARY KEY,
            station_id TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK (event_type IN ('departure', 'arrival')),
            raw_data JSONB NOT NULL,
            fetched_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            source TEXT NOT NULL DEFAULT 'db-timetables-v1'
        );
        """,
        "CREATE INDEX IF NOT EXISTS idx_train_events_fetched_at ON raw.train_events (fetched_at);",
        "CREATE INDEX IF NOT EXISTS idx_train_events_station_id ON raw.train_events (station_id);",
        "ALTER TABLE raw.train_events ALTER COLUMN source SET DEFAULT 'db-timetables-v1';",
    ]
    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)
    conn.commit()


def upsert_station(conn: psycopg2.extensions.connection, station: Station) -> None:
    """Refresh a station directly, retaining last known catalogue on failure"""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO raw.stations (station_id, station_name, latitude, longitude)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (station_id) DO UPDATE
            SET station_name = EXCLUDED.station_name,
                latitude = EXCLUDED.latitude,
                longitude = EXCLUDED.longitude;
            """,
            (station.station_id, station.name, station.latitude, station.longitude),
        )


def seed_stations(database_url: str, client_id: str, api_key: str) -> None:
    """Atomically refresh all monitored stations using DB's StaDa API"""
    conn = get_db_connection(database_url)
    try:
        print("[1/4] Creating schemas...")
        create_schemas(conn)
        print("[2/4] Creating source tables...")
        create_raw_tables(conn)
        print("[3/4] Fetching the daily StaDa station catalogue...")
        with StadaClient(client_id=client_id, api_key=api_key) as client:
            stations = client.find_stations(STATIONS_TO_SEED)

        missing = sorted(set(STATIONS_TO_SEED) - set(stations))
        if missing:
            raise RuntimeError(f"StaDa did not resolve every monitored station: {missing}")

        print("[4/4] Updating raw.stations...")
        for requested_name in STATIONS_TO_SEED:
            station = stations[requested_name]
            upsert_station(conn, station)
            print(f"  Synced {station.name}: EVA {station.station_id}")
        conn.commit()
        print(f"Station catalogue sync complete: {len(STATIONS_TO_SEED)} stations.")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    load_dotenv()
    database_url = os.environ.get("DATABASE_URL")
    client_id = os.environ.get("DB_CLIENT_ID")
    api_key = os.environ.get("DB_API_KEY")
    if not database_url or not client_id or not api_key:
        raise RuntimeError("DATABASE_URL, DB_CLIENT_ID and DB_API_KEY must be set.")
    seed_stations(database_url, client_id, api_key)
