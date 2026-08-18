/*
  Station dimension table

  Add recent trip volume and a category to the station data

  The station_category classification:
    - major_hub: median daily trips > 200
    - regional_hub: median daily trips > 50
    - local_station: median daily trips <= 50
*/

with stations as (
    select
        station_id,
        station_name,
        latitude,
        longitude
    from {{ source('raw', 'stations') }}
),

date_range as (
    select
        (now() at time zone 'Europe/Berlin')::date - 7 as start_date,
        (now() at time zone 'Europe/Berlin')::date as end_date
),

-- Count trips for each station and day
daily_trip_counts as (
    select
        station_id,
        (planned_time at time zone 'Europe/Berlin')::date as service_date,
        count(distinct trip_id) as trip_count
    from {{ ref('stg_train_events') }}
    where planned_time is not null
    group by
        station_id,
        (planned_time at time zone 'Europe/Berlin')::date
),

-- Use the last 7 complete days
recent_trip_counts as (
    select
        daily.station_id,
        daily.service_date,
        daily.trip_count
    from daily_trip_counts daily
    cross join date_range dates
    where daily.service_date >= dates.start_date
      and daily.service_date < dates.end_date
),

-- Find the typical daily trip count
station_volume as (
    select
        station_id,
        percentile_cont(0.5) within group (order by trip_count)
            as median_daily_trip_count,
        count(*) as classification_days
    from recent_trip_counts
    group by station_id
)

select
    s.station_id,
    s.station_name,
    s.latitude,
    s.longitude,

    coalesce(
        round(volume.median_daily_trip_count::numeric, 1),
        0
    ) as median_daily_trip_count,
    coalesce(volume.classification_days, 0) as classification_days,

    -- Classify stations by volume
    case
        when coalesce(volume.median_daily_trip_count, 0) > 200 then 'major_hub'
        when coalesce(volume.median_daily_trip_count, 0) > 50
            then 'regional_hub'
        else 'local_station'
    end as station_category,

    now() as updated_at

from stations s
left join station_volume volume
    on s.station_id = volume.station_id
