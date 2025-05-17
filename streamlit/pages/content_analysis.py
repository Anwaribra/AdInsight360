import streamlit as st
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse
import sys
import os
from pathlib import Path

# Add config directory to path
config_path = Path(__file__).parents[1] / "config"
sys.path.append(str(config_path))

try:
    from local_db import DB_CONFIG
except ImportError:
    st.error("Please create streamlit/config/local_db.py with your database configuration.")
    st.stop()

import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection
def get_db_connection():
    return psycopg2.connect(
        **DB_CONFIG,
        cursor_factory=RealDictCursor
    )

# Page config
st.set_page_config(
    page_title="Content Analysis - AdInsight360",
    page_icon="",
    layout="wide"
)


st.title("Content Analysis")
st.markdown("""
Analyze content patterns and identify successful post characteristics.
""")

st.sidebar.title("Filters")
date_range = st.sidebar.selectbox(
    "Time Period",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Year", "All Time"],
    index=1
)
min_score = st.sidebar.slider("Minimum Score", 0, 1000, 10)

@st.cache_data(ttl=3600)
def get_content_data(date_range, min_score):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    date_filters = {
        "Last 24 Hours": "NOW() - INTERVAL '24 hours'",
        "Last 7 Days": "NOW() - INTERVAL '7 days'",
        "Last 30 Days": "NOW() - INTERVAL '30 days'",
        "Last Year": "NOW() - INTERVAL '1 year'",
        "All Time": "NOW() - INTERVAL '100 years'"
    }
    
    query = f"""
    SELECT *
    FROM stg_reddit_posts
    WHERE created_utc >= {date_filters[date_range]}
    AND score >= {min_score}
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    df = pd.DataFrame(data)
    
    cursor.close()
    conn.close()
    return df

try:
    df = get_content_data(date_range, min_score)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_title_length = df["title"].str.len().mean()
        st.metric(
            "Avg. Title Length",
            f"{avg_title_length:.1f} chars",
            delta=f"{df['title'].str.len().std():.1f} std dev"
        )
    
    with col2:
        avg_score = df["score"].mean()
        st.metric(
            "Avg. Post Score",
            f"{avg_score:.1f}",
            delta=f"{df['score'].std():.1f} std dev"
        )
    
    with col3:
        engagement_rate = (df["score"] / df["num_comments"]).mean()
        st.metric(
            "Engagement Rate",
            f"{engagement_rate:.2f}",
            delta=f"{(df['score'] / df['num_comments']).std():.2f} std dev"
        )

    st.subheader("Popular Content Sources")
    df["domain"] = df["url"].apply(lambda x: urlparse(x).netloc if pd.notnull(x) else "text")
    domain_stats = df.groupby("domain").agg({
        "id": "count",
        "score": "mean",
        "num_comments": "mean"
    }).reset_index()
    
    domain_stats.columns = ["Domain", "Posts", "Avg Score", "Avg Comments"]
    domain_stats = domain_stats.sort_values("Posts", ascending=False).head(10)
    
    fig_domains = px.bar(
        domain_stats,
        x="Domain",
        y=["Posts", "Avg Score", "Avg Comments"],
        title="Top Content Sources and Their Performance"
    )
    st.plotly_chart(fig_domains, use_container_width=True)
    # Post timing analysis
    st.subheader("Post Timing Analysis")
    df["hour"] = pd.to_datetime(df["created_utc"]).dt.hour
    df["day"] = pd.to_datetime(df["created_utc"]).dt.day_name()
    
    timing_stats = df.groupby(["day", "hour"]).agg({
        "score": "mean"
    }).reset_index()
    
    fig_timing = px.density_heatmap(
        timing_stats,
        x="hour",
        y="day",
        z="score",
        title="Best Times to Post (Average Score)",
        labels={"score": "Avg Score"}
    )
    st.plotly_chart(fig_timing, use_container_width=True)
    # Top performing posts
    st.subheader("Top Performing Posts")
    top_posts = df.nlargest(10, "score")[["title", "score", "num_comments", "author"]]
    st.dataframe(top_posts, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure your database connection is properly configured in streamlit/config/local_db.py") 