"""
Writes provider events into the raw.train_events table
"""

import hashlib
import json
import logging

import psycopg2

logger = logging.getLogger(__name__)


def build_raw_event_id(
    event: dict,
    station_id: str,
    event_type: str,
    source: str,
) -> str:
    """Build a identifier for one provider event

    Key lets PostgreSQL update that raw event during
    Airflow retries and later polls instead of appending indistinguishable
    duplicates
    """
    trip_id = event.get("tripId")
    if not trip_id:
        raise ValueError("Every train event must contain a non-empty tripId.")

    planned_when = event.get("plannedWhen") or ""
    identity = "\x1f".join(
        (source, station_id, event_type, str(trip_id), str(planned_when))
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def write_train_events(
    database_url: str,
    events: list[dict],
    station_id: str,
    event_type: str,
    source: str = "db-timetables-v1",
) -> int:
    """Upsert a batch of raw API events into raw.train_events"""

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

    insert_sql = """
        INSERT INTO raw.train_events
            (raw_id, station_id, event_type, raw_data, source)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (raw_id) DO UPDATE
        SET raw_data = EXCLUDED.raw_data,
            fetched_at = now(),
            source = EXCLUDED.source;
    """

    inserted_count = 0

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            for event in events:
                # Serialize the event dict to a JSON string for the JSONB column
                raw_json = json.dumps(event, ensure_ascii=False)
                raw_id = build_raw_event_id(
                    event=event,
                    station_id=station_id,
                    event_type=event_type,
                    source=source,
                )

                cur.execute(
                    insert_sql,
                    (raw_id, station_id, event_type, raw_json, source),
                )
                inserted_count += 1
        conn.commit()

        logger.info(
            "Upserted %d %s events for station %s.",
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
