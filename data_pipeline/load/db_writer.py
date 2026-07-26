"""
Writes provider events into the raw.train_events table
"""

import json
import logging

import psycopg2

logger = logging.getLogger(__name__)


def write_train_events(
    database_url: str,
    events: list[dict],
    station_id: str,
    event_type: str,
    source: str = "db-timetables-v1",
) -> int:
    """Insert a batch of raw API events into raw.train_events
    Each event dict is stored in the raw_data JSONB column
    """
    # Validate event_type
    if event_type not in ("departure", "arrival"):
        raise ValueError(
            f"event_type must be 'departure' or 'arrival', got '{event_type}'."
        )

    if not events:
        logger.info(
            "No events to insert for station %s (%s). Skipping.",
            station_id,
            event_type,
        )
        return 0

    # SQL for inserting a single event
    insert_sql = """
        INSERT INTO raw.train_events (station_id, event_type, raw_data, source)
        VALUES (%s, %s, %s, %s);
    """

    inserted_count = 0

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            for event in events:
                # Serialize the event dict to a JSON string for the JSONB column
                raw_json = json.dumps(event, ensure_ascii=False)

                cur.execute(insert_sql, (station_id, event_type, raw_json, source))
                inserted_count += 1

        # Commit all inserts in a single transaction
        conn.commit()

        logger.info(
            "Inserted %d %s events for station %s.",
            inserted_count,
            event_type,
            station_id,
        )

    except psycopg2.Error:
        conn.rollback()
        logger.exception(
            "Database error inserting %s events for station %s. "
            "Transaction rolled back.",
            event_type,
            station_id,
        )
        raise

    finally:
        conn.close()

    return inserted_count
