{{
  config(
    materialized='view',
    schema='analytics'
  )
}}

with source as (
    select * from {{ source('reddit', 'reddit_marketing') }}
),

staged as (
    select
        id,
        title,
        score,
        to_timestamp(to_number(created_utc, 38, 0)) as created_utc,
        url,
        num_comments,
        author,
        loaded_at
    from source
)

select * from staged
