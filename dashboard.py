import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import sys
from pathlib import Path
from urllib.parse import urlparse

config_path = Path(__file__).parent / "config"
sys.path.append(str(config_path))

try:
    from local_db import DB_CONFIG
except ImportError:
    st.error("Missing database configuration. Please create the file 'streamlit/config/local_db.py' with your DB_CONFIG.")
    st.stop()

# db connection
def get_db_connection():
    return psycopg2.connect(
        **DB_CONFIG,
        cursor_factory=RealDictCursor
    )

# Page config
st.set_page_config(
    page_title="AdInsight360 - Reddit Marketing Analytics",
    page_icon="",
    layout="wide"
)

# Title 
st.title("AdInsight360 Dashboard")
st.markdown("""
Analyze Reddit marketing trends and engagement patterns with real-time insights.
""")

# Create tabs for different analytics sections
tab1, tab2 = st.tabs(["Marketing Overview", "Content Analysis"])

with tab1:
    st.sidebar.title("Filters")
    
    date_range = st.sidebar.selectbox(
        "Time Period",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Year", "All Time"],
        index=1,
        key="date_range_tab1"
    )
    
    @st.cache_data(ttl=3600)
    def get_data(date_range):
        conn = get_db_connection()
        cursor = conn.cursor()
        
        date_filters = {
            "Last 24 Hours": "NOW() - INTERVAL '24 hours'",
            "Last 7 Days": "NOW() - INTERVAL '7 days'",
            "Last 30 Days": "NOW() - INTERVAL '30 days'",
            "Last Year": "NOW() - INTERVAL '1 year'",
            "All Time": "NOW() - INTERVAL '100 years'"  
        }
        daily_query = f"""
        SELECT *
        FROM marketing_insights
        WHERE metric_type = 'daily'
        AND created_at >= {date_filters[date_range]}
        ORDER BY dimension
        """
        cursor.execute(daily_query)
        daily_data = cursor.fetchall()
        # Get author metrics
        author_query = f"""
        SELECT *
        FROM marketing_insights
        WHERE metric_type = 'author'
        ORDER BY total_score DESC
        LIMIT 10
        """
        
        cursor.execute(author_query)
        author_data = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return pd.DataFrame(daily_data), pd.DataFrame(author_data)
    
    # Load data
    try:
        daily_df, author_df = get_data(date_range)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total Posts",
                daily_df["total_posts"].sum(),
                delta=f"{daily_df['total_posts'].mean():.1f} avg/day"
            )
        
        with col2:
            st.metric(
                "Total Engagement",
                daily_df["total_score"].sum(),
                delta=f"{daily_df['total_score'].mean():.1f} avg/day"
            )
        
        with col3:
            st.metric(
                "Total Comments",
                daily_df["total_comments"].sum(),
                delta=f"{daily_df['total_comments'].mean():.1f} avg/day"
            )
        
        with col4:
            st.metric(
                "Avg. Score per Post",
                f"{daily_df['avg_score'].mean():.1f}",
                delta=f"{daily_df['avg_score'].std():.1f} std dev"
            )
        
        # Engagement over time
        st.subheader("Engagement Trends")
        fig_trends = px.line(
            daily_df,
            x="dimension",
            y=["total_posts", "total_score", "total_comments"],
            title="Engagement Metrics Over Time",
            labels={
                "dimension": "Date",
                "value": "Count",
                "variable": "Metric"
            }
        )
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Top authors
        st.subheader("Top Contributors")
        fig_authors = px.bar(
            author_df,
            x="dimension",
            y="total_score",
            title="Top Authors by Total Score",
            labels={
                "dimension": "Author",
                "total_score": "Total Score"
            }
        )
        st.plotly_chart(fig_authors, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Check if streamlit/config/local_db.py exists with valid database configuration")

with tab2:
    st.markdown("""
    Analyze content patterns and identify successful post characteristics.
    """)
    
    st.sidebar.title("Content Filters")
    date_range_content = st.sidebar.selectbox(
        "Time Period",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Year", "All Time"],
        index=1,
        key="date_range_tab2"
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
        df = get_content_data(date_range_content, min_score)
        
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
        st.error(f"Error loading content data: {str(e)}")
        st.info("Please ensure your database connection is properly configured in streamlit/config/local_db.py") 