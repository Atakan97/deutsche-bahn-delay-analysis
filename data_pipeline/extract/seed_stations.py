"""
Database initialisation script
"""

import os
import time

import httpx
import psycopg2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Station list — the 10 major German stations
# ---------------------------------------------------------------------------
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

# Base URL for the Deutsche Bahn HAFAS REST API
API_BASE_URL = "https://v6.db.transport.rest"

# Wait between API calls
API_RATE_LIMIT_SECONDS = 0.5


def get_db_connection(database_url: str) -> psycopg2.extensions.connection:
    """Open and return a psycopg2 connection to the PostgreSQL database
    """
    conn = psycopg2.connect(database_url)
    return conn


def create_schemas(conn: psycopg2.extensions.connection) -> None:
    """Create the raw, staging, and marts schemas

      - raw: Python writes raw API data here. dbt reads but never writes here
      - staging: dbt writes clean, typed views here. Python never touches this
      - marts: dbt writes star-schema tables here. The dashboard and ML read here
    """
    conn.autocommit = True
    with conn.cursor() as cur:
        for schema in ("raw", "staging", "marts"):
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
            print(f"  Schema '{schema}' is ready.")
    conn.autocommit = False


def create_raw_tables(conn: psycopg2.extensions.connection) -> None:
    """Create raw.stations and raw.train_events

       The fetched_at index on raw.train_events (incremental dbt model) is used to know where it fetched, 
       so this index prevents a full table scan on every dbt run
    """
    create_stations_sql = """
        CREATE TABLE IF NOT EXISTS raw.stations (
            station_id   TEXT        PRIMARY KEY,
            station_name TEXT        NOT NULL,
            latitude     NUMERIC(9,6) NOT NULL,
            longitude    NUMERIC(9,6) NOT NULL,
            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """

    create_train_events_sql = """
        CREATE TABLE IF NOT EXISTS raw.train_events (
            id           BIGSERIAL    PRIMARY KEY,
            station_id   TEXT         NOT NULL,
            event_type   TEXT         NOT NULL CHECK (event_type IN ('departure', 'arrival')),
            raw_data     JSONB        NOT NULL,
            fetched_at   TIMESTAMPTZ  NOT NULL DEFAULT now(),
            source       TEXT         NOT NULL DEFAULT 'v6.db.transport.rest'
        );
    """

    create_fetched_at_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_train_events_fetched_at
        ON raw.train_events (fetched_at);
    """

    # Used by queries that filter or aggregate by station
    create_station_id_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_train_events_station_id
        ON raw.train_events (station_id);
    """

    with conn.cursor() as cur:
        cur.execute(create_stations_sql)
        print("  Table 'raw.stations' is ready.")

        cur.execute(create_train_events_sql)
        print("  Table 'raw.train_events' is ready.")

        cur.execute(create_fetched_at_index_sql)
        print("  Index 'idx_train_events_fetched_at' is ready.")

        cur.execute(create_station_id_index_sql)
        print("  Index 'idx_train_events_station_id' is ready.")

    conn.commit()


def search_station(client: httpx.Client, query: str) -> dict | None:
    """Search for a station by name using the v6.db.transport.rest /locations endpoint
    """
    try:
        response = client.get(
            "/locations",
            params={
                "query": query,
                "results": 5,
                # Only return stops (train stations)
                "stops": "true",
                "addresses": "false",
                "poi": "false",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        results = response.json()

        if not results:
            print(f"  No results found for '{query}'.")
            return None

        # First match is almost always the correct major station
        station = results[0]

        # Validate that we got a location with coordinates
        location = station.get("location", {})
        latitude = location.get("latitude")
        longitude = location.get("longitude")

        if latitude is None or longitude is None:
            print(f"  Station '{query}' found but has no coordinates. Skipping.")
            return None

        return {
            "id": station["id"],
            "name": station["name"],
            "latitude": latitude,
            "longitude": longitude,
        }

    except httpx.HTTPStatusError as e:
        print(f"  Error: HTTP {e.response.status_code} when searching for '{query}'.")
        return None
    except httpx.RequestError as e:
        print(f"  Error: Network error when searching for '{query}': {e}")
        return None


def insert_station(
    conn: psycopg2.extensions.connection, station: dict
) -> bool:
    """Insert a single station record into raw.stations
    """
    insert_sql = """
        INSERT INTO raw.stations (station_id, station_name, latitude, longitude)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (station_id) DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.execute(
            insert_sql,
            (station["id"], station["name"], station["latitude"], station["longitude"]),
        )
        # rowcount == 1 means a new row was inserted, 0 means it already existed
        inserted = cur.rowcount == 1

    conn.commit()
    return inserted


def seed_stations(database_url: str) -> None:
    """Main management function, set up DB schema and all 10 stations
    """
    print("=" * 60)
    print("Deutsche-Bahn-Delay-Analysis — DB Initialisation")
    print("=" * 60)

    # Connect to the database
    print("\n[1/4] Connecting to PostgreSQL...")
    conn = get_db_connection(database_url)
    print("  Connected successfully.")

    # Create schemas
    print("\n[2/4] Creating schemas (raw, staging, marts)...")
    create_schemas(conn)

    # Create raw tables and indexes
    print("\n[3/4] Creating raw tables and indexes...")
    create_raw_tables(conn)

    # Fetch station data from the API and seed raw.stations
    print(f"\n[4/4] Seeding {len(STATIONS_TO_SEED)} stations into raw.stations...")

    inserted_count = 0
    skipped_count = 0

    # Use a single shared httpx.Client for all API calls so TCP connections are reused
    with httpx.Client(base_url=API_BASE_URL) as api_client:
        for station_name in STATIONS_TO_SEED:
            print(f"\n  Searching for: '{station_name}'")

            station_data = search_station(api_client, station_name)

            if station_data is None:
                print(f"  Skip: Could not find valid data for '{station_name}'.")
                skipped_count += 1
            else:
                was_inserted = insert_station(conn, station_data)
                if was_inserted:
                    print(
                        f"  Inserted: {station_data['name']} "
                        f"(id={station_data['id']}, "
                        f"lat={station_data['latitude']:.4f}, "
                        f"lon={station_data['longitude']:.4f})"
                    )
                    inserted_count += 1
                else:
                    print(f"  Already exists: {station_data['name']} — skipped.")
                    skipped_count += 1
            time.sleep(API_RATE_LIMIT_SECONDS)

    conn.close()

    # Print summary
    print("\n" + "=" * 60)
    print("Seeding complete.")
    print(f"  Stations inserted : {inserted_count}")
    print(f"  Stations skipped  : {skipped_count}")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Verify data in Supabase: SELECT * FROM raw.stations;")
    print("  2. Run the pipeline:   python -m data_pipeline.orchestration.flows")


if __name__ == "__main__":
    # Load .env file
    load_dotenv()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set.\n"
            "Set your Supabase connection string."
        )

    seed_stations(db_url)
