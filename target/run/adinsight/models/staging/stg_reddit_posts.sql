
  create or replace   view ADINSIGHT_DB.public_analytics.stg_reddit_posts
  
   as (
    

with source as (
    select * from ADINSIGHT_DB.public.reddit_marketing
),

staged as (
    select
        id,
        title,
        score,
        to_timestamp_ntz(cast(created_utc as float)) as created_utc,
        url,
        num_comments,
        author,
        loaded_at
    from source
)

select * from staged
  );

