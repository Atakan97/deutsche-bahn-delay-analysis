/*
  Staging view that parses raw JSONB into typed columns

  Transform part of the ELT process
*/

with raw_events as (
    select
        id,
        station_id,
        event_type,
        raw_data,
        fetched_at
    from {{ source('raw', 'train_events') }}
)

select
    -- trip_id: unique identifier for a train journey across stations
    raw_data ->> 'tripId'                               as trip_id,

    -- station_id comes from the table column, not the JSON
    station_id,

    -- event_type: 'departure' or 'arrival', also a table column
    event_type,

    -- planned_time: the scheduled departure/arrival time
    (raw_data ->> 'plannedWhen')::timestamptz           as planned_time,

    -- actual_time: the real departure/arrival time, NULL if the train was cancelled
    (raw_data ->> 'when')::timestamptz                  as actual_time,

    -- delay_minutes: converted from seconds (API) to minutes
    -- COALESCE to 0 because null delay means "no delay info available", which
    -- for analysis purposes treating as "on time"
    coalesce(
        (raw_data ->> 'delay')::numeric / 60.0,
        0
    )                                                   as delay_minutes,

    -- line_name: the train line identifier (e.g. "ICE 123", "RE 5")
    raw_data -> 'line' ->> 'name'                       as line_name,

    -- train_type: the broad product category
    -- Used to distinguish ICE, IC/EC, RE, S-Bahn, etc.
    raw_data -> 'line' ->> 'product'                    as train_type,

    -- fetched_at: when the Python pipeline fetched this data, critical for
    -- incremental processing in fct_delays, the incremental model uses
    -- MAX(fetched_at) to know where it left off
    fetched_at

from raw_events

-- Filter out events without a trip identifier
where raw_data ->> 'tripId' is not null
