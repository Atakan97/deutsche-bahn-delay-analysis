{#
  generate_schema_name.sql — Custom schema name resolution for dbt.

  WHY this macro exists:
    By default, dbt concatenates the target schema with the custom schema name
    using an underscore. For example, if target.schema = 'public' and a model
    has +schema: staging, dbt would write to 'public_staging'.

    We DON'T want that. We already created exact schemas (raw, staging, marts)
    in Phase 1 (seed_stations.py) and want dbt to write directly to them.

    This macro overrides the default behavior:
      - If a model specifies +schema (e.g. staging or marts), use that name
        exactly, WITHOUT any prefix.
      - If no +schema is specified, fall back to target.schema (public).

  This is a standard dbt pattern documented at:
    https://docs.getdbt.com/docs/build/custom-schemas
#}

{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- if custom_schema_name is not none -%}
        {{ custom_schema_name | trim }}
    {%- else -%}
        {{ target.schema }}
    {%- endif -%}
{%- endmacro %}
