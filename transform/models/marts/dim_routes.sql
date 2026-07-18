/*
  Route dimension table

  Creates a unique list of train routes, identified by the combination of
  line_name (e.g. "ICE 123") and train_type (e.g. "nationalExpress")
*/

with distinct_routes as (
    select distinct
        line_name,
        train_type
    from {{ ref('stg_train_events') }}
    -- Filter out events with no line information
    where line_name is not null
)

select
    {{ dbt_utils.generate_surrogate_key(['line_name', 'train_type']) }} as route_id,
    line_name,
    train_type,
    now() as updated_at

from distinct_routes
