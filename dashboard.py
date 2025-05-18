import streamlit as st
import pandas as pd
import plotly.express as px
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import sys
from pathlib import Path

config_path = Path(__file__).parent / "config"
sys.path.append(str(config_path))

try:
    from local_db import DB_CONFIG
except ImportError:
    st.error("")
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
st.sidebar.title("Filters")

date_range = st.sidebar.selectbox(
    "Time Period",
    ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Year", "All Time"],
    index=1
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
    st.info("streamlit/config/local_db.py") 