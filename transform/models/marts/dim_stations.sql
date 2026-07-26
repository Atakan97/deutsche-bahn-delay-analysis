/*
  Station dimension table

  Support the static raw.stations reference data with derived metrics:
    - daily_trip_count: how many distinct trips have been observed at this station
    - station_category: a simple classification based on volume

  The station_category classification:
    - major_hub : stations with > 200 distinct trips (e.g. Frankfurt, München)
    - regional_hub : stations with > 50 distinct trips (e.g. Passau)
    - local_station : stations with <= 50 distinct trips
*/

with stations as (
    select
        station_id,
        station_name,
        latitude,
        longitude
    from {{ source('raw', 'stations') }}
),

-- Count distinct trips per station for all fetched events
trip_counts as (
    select
        station_id,
        count(distinct trip_id) as daily_trip_count
    from {{ ref('stg_train_events') }}
    group by station_id
)

select
    s.station_id,
    s.station_name,
    s.latitude,
    s.longitude,

    coalesce(tc.daily_trip_count, 0) as daily_trip_count,

    -- Classify stations by volume
    case
        when coalesce(tc.daily_trip_count, 0) > 90  then 'major_hub'
        when coalesce(tc.daily_trip_count, 0) > 40  then 'regional_hub'
        else 'local_station'
    end as station_category,

    now() as updated_at

from stations s
left join trip_counts tc
    on s.station_id = tc.station_id
