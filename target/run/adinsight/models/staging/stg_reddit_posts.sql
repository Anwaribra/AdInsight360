
  create or replace   view ADINSIGHT_DB.public_analytics.stg_reddit_posts
  
   as (
    

SELECT
    'post_1' AS id,
    'Test Post' AS title,
    100 AS score,
    CURRENT_TIMESTAMP() AS created_at,
    'https://example.com' AS url,
    5 AS num_comments,
    'test_user' AS author
  );

