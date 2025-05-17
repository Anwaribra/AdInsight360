
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
        to_timestamp(to_number(created_utc, 38, 0)) as created_utc,
        url,
        num_comments,
        author,
        loaded_at
    from source
)

select * from staged
  );

