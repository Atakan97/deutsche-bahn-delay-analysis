/*
  Fact table for train delay events (INCREMENTAL)

  The pipeline appends new rows to raw.train_events every 15 minutes,
  without incremental processing, every dbt run would reprocess the entire
  history of raw events, parsing JSON, joining dimensions, computing
  derived fields for potentially millions of rows

  With incremental, each dbt run only processes rows where:
    fetched_at > (select max(fetched_at) from this_table)

  This means each run processes only the events fetched in the last
  15 minutes, not the full historical dataset
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
    {{ dbt_utils.generate_surrogate_key(['stg_events.trip_id', 'stg_events.station_id', 'stg_events.event_type']) }} as delay_id,

    stg_events.trip_id,
    stg_events.station_id,

    -- route_id links to dim_routes. Uses a LEFT JOIN so events with unknown
    -- routes (null line_name) still appear in the fact table with route_id = NULL.
    routes.route_id,

    stg_events.event_type,
    stg_events.planned_time,
    stg_events.actual_time,
    stg_events.delay_minutes,

    -- Derived time features for ML and analysis
    extract(hour from stg_events.planned_time)::int     as hour_of_day,

    -- day_of_week: 0 = Sunday, 1 = Monday, ..., 6 = Saturday
    extract(dow from stg_events.planned_time)::int      as day_of_week,

    -- is_weekend: boolean flag derived from day_of_week
    case
        when extract(dow from stg_events.planned_time) in (0, 6)
        then true
        else false
    end                                                 as is_weekend,

    stg_events.fetched_at

from stg_events
left join routes
    on stg_events.line_name = routes.line_name
    and stg_events.train_type = routes.train_type
