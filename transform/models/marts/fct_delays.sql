/*
  Fact table for train delay events (INCREMENTAL)

  The pipeline appends new rows to raw.train_events every 15 minutes,
  without incremental processing

  Each run processes only the events fetched in the last
  15 minutes, not full dataset
*/

{{
    config(
        materialized='incremental',
        unique_key='delay_id'
    )
}}

with stg_events as (
    select *
    from {{ ref('stg_train_events') }}

    /*
      Incremental filter: only process rows that arrived after the most recent
      row that already processed
    */
    {% if is_incremental() %}
    where fetched_at > (select max(fetched_at) from {{ this }})
    {% endif %}
),

deduplicated_events as (
    select
        *,
        row_number() over (
            partition by trip_id, station_id, event_type
            order by fetched_at desc
        ) as rn
    from stg_events
),

routes as (
    select
        route_id,
        line_name,
        train_type
    from {{ ref('dim_routes') }}
)

select
    -- Surrogate key: hash of trip + station + event type
    -- Verifies that each unique delay observation gets a reproducible ID
    {{ dbt_utils.generate_surrogate_key(['deduplicated_events.trip_id', 'deduplicated_events.station_id', 'deduplicated_events.event_type']) }} as delay_id,

    deduplicated_events.trip_id,
    deduplicated_events.station_id,

    -- route_id links to dim_routes, uses a LEFT JOIN so events with unknown
    -- routes (null line_name) still appear in the fact table with route_id = NULL
    routes.route_id,

    deduplicated_events.event_type,
    deduplicated_events.planned_time,
    deduplicated_events.actual_time,
    deduplicated_events.delay_minutes,

    -- Derived time features for ML and analysis
    extract(hour from deduplicated_events.planned_time)::int     as hour_of_day,

    -- day_of_week: 0 = Sunday, 1 = Monday, ..., 6 = Saturday
    extract(dow from deduplicated_events.planned_time)::int      as day_of_week,

    -- is_weekend: boolean flag derived from day_of_week
    case
        when extract(dow from deduplicated_events.planned_time) in (0, 6)
        then true
        else false
    end                                                 as is_weekend,

    deduplicated_events.fetched_at

from deduplicated_events
left join routes
    on deduplicated_events.line_name = routes.line_name
    and deduplicated_events.train_type = routes.train_type
where deduplicated_events.rn = 1
