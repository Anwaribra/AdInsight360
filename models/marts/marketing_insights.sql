with posts as (
    select * from {{ ref('stg_reddit_posts') }}
),

daily_metrics as (
    select
        date_trunc('day', created_utc) as date,
        count(*) as total_posts,
        sum(score) as total_score,
        sum(num_comments) as total_comments,
        avg(score) as avg_score,
        avg(num_comments) as avg_comments
    from posts
    group by 1
),

author_metrics as (
    select
        author,
        count(*) as post_count,
        sum(score) as total_score,
        sum(num_comments) as total_comments,
        avg(score) as avg_score,
        avg(num_comments) as avg_comments
    from posts
    where author is not null
    group by 1
)

select
    'daily' as metric_type,
    date as dimension,
    total_posts,
    total_score,
    total_comments,
    avg_score,
    avg_comments
from daily_metrics

union all

select
    'author' as metric_type,
    author as dimension,
    post_count as total_posts,
    total_score,
    total_comments,
    avg_score,
    avg_comments
from author_metrics 