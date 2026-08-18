-- Return rows that do not match the category rules

select
    station_id,
    station_name,
    median_daily_trip_count,
    station_category
from {{ ref('dim_stations') }}
where
    (
        median_daily_trip_count > 200
        and station_category != 'major_hub'
    )
    or (
        median_daily_trip_count > 50
        and median_daily_trip_count <= 200
        and station_category != 'regional_hub'
    )
    or (
        median_daily_trip_count <= 50
        and station_category != 'local_station'
    )
