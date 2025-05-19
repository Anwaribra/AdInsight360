import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import sys
from pathlib import Path
from urllib.parse import urlparse
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import base64
import io
import time
import altair as alt
import random

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

config_path = Path(__file__).parent / "config"
sys.path.append(str(config_path))

try:
    from local_db import DB_CONFIG
except ImportError:
    st.error("Mlocal_db.py'DB_CONFIG.")
    st.stop()

# Session state initialization
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = 0

# db connection
def get_db_connection():
    return psycopg2.connect(
        **DB_CONFIG,
        cursor_factory=RealDictCursor
    )

# Configure Streamlit page
st.set_page_config(
    page_title="AdInsight360 - Reddit Marketing Analytics",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF4500;  /* Reddit orange */
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #0079D3;  /* Reddit blue */
        margin-bottom: 0.8rem;
    }
    .metric-card {
        background-color: #f6f7f9;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f6f7f9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4500;
        color: white;
    }
    .st-emotion-cache-1r6slb0 {  /* Data frame styling */
        background-color: #f6f7f9;
    }
    .sidebar-content {
        padding: 15px;
        border-radius: 10px;
        background-color: #f6f7f9;
        margin-bottom: 20px;
    }
    /* Loading animation */
    .loading-spinner {
        text-align: center;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)


def generate_wordcloud(text_series):
    """Generate wordcloud from text series"""
    if text_series.empty:
        return None
    
    stop_words = set(stopwords.words('english'))
    
    text = ' '.join(text_series.dropna().astype(str).values)
    
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        stopwords=stop_words,
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    return wordcloud

def perform_sentiment_analysis(text_series):
    """Analyze sentiment from text series"""
    sentiment_scores = []
    
    for text in text_series.dropna():
        analysis = TextBlob(str(text))
        sentiment_scores.append({
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        })
    
    return pd.DataFrame(sentiment_scores)

def extract_topics(text_series, n_top_words=10, n_topics=5):
    """Extract top topics from text"""
    if text_series.empty:
        return pd.DataFrame()
    
    cleaned_texts = []
    for text in text_series.dropna():
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  
        text = re.sub(r'\d+', '', text)  
        cleaned_texts.append(text)
    

    stop_words = set(stopwords.words('english'))
    stop_words.update(['reddit', 'post', 'comment', 'like'])
    vectorizer = CountVectorizer(
        max_features=1000,
        stop_words=list(stop_words)
    )
    
    # Fit and transform
    try:
        X = vectorizer.fit_transform(cleaned_texts)
        words = vectorizer.get_feature_names_out()
        word_counts = X.sum(axis=0).A1
        
        # Get top words
        word_freq = pd.DataFrame({
            'word': words,
            'count': word_counts
        })
        top_words = word_freq.sort_values('count', ascending=False).head(n_top_words)
        return top_words
    except:
        return pd.DataFrame({'word': [], 'count': []})

def to_excel_download_link(df, filename="data.xlsx", link_text="Download Excel file"):
    """Generate a link to download the DataFrame as Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    b64 = base64.b64encode(output.getvalue()).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Custom metric card component
def metric_card(title, value, delta=None, delta_color="normal", help=None):
    html = f"""
    <div class="metric-card">
        <h3 style="margin-top:0; font-size:1rem; color:#555;">{title}</h3>
        <p style="font-size:1.8rem; font-weight:bold; margin:0;">{value}</p>
    """
    
    if delta:
        color = "green" if delta_color == "good" else "red" if delta_color == "bad" else "gray"
        direction = "↑" if delta_color == "good" else "↓" if delta_color == "bad" else "→"
        html += f'<p style="font-size:0.9rem; color:{color}; margin:0;">{direction} {delta}</p>'
    
    html += "</div>"
    
    return st.markdown(html, unsafe_allow_html=True)

# Title with logo
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header"> AdInsight360 </h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align:center; font-size:1.2rem; margin-bottom:30px;">
    Advanced Reddit Marketing Analytics Platform | Real-time Insights & AI-Powered Recommendations
    </p>
    """, unsafe_allow_html=True)

# Create tabs for different analytics sections using custom styling
tabs = [" Marketing Dashboard", " Content Intelligence", "Sentiment Analysis", "Competitor Analysis", " AI Recommendations"]
selected_tab = st.radio("Navigation", tabs, horizontal=True, label_visibility="collapsed")

# Display appropriate content based on selected tab
if selected_tab == " Marketing Dashboard":
    st.markdown('<h2 class="sub-header">Marketing Performance Overview</h2>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.sidebar.markdown('<h3>Dashboard Controls</h3>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        date_range = st.selectbox(
            "Time Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            index=2
        )
        
        subreddit_filter = st.multiselect(
            "Filter by Subreddit",
            ["all", "marketing", "DigitalMarketing", "socialmedia", "PPC", "SEO"],
            default=["all"]
        )
        
        metrics_options = st.multiselect(
            "Select Metrics to Display",
            ["Posts", "Score", "Comments", "Awards", "Upvote Ratio"],
            default=["Posts", "Score", "Comments"]
        )
        
        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            min_engagement = st.slider("Minimum Engagement Score", 0, 100, 10)
            exclude_deleted = st.checkbox("Exclude Deleted Posts", value=True)
            include_nsfw = st.checkbox("Include NSFW Content", value=False)
        st.markdown('</div>', unsafe_allow_html=True)
        
        refresh_btn = st.button("Refresh Data")
    
    # Main dashboard content
    @st.cache_data(ttl=3600)
    def get_dashboard_data(date_range, subreddits, min_engagement):
        # Simulate database fetch with some randomized data
        dates = pd.date_range(
            end=datetime.now(),
            periods={"Last 24 Hours": 24, 
                    "Last 7 Days": 7, 
                    "Last 30 Days": 30,
                    "Last 90 Days": 90,
                    "Last Year": 52,
                    "All Time": 104}[date_range],
            freq={"Last 24 Hours": "H", 
                 "Last 7 Days": "D", 
                 "Last 30 Days": "D",
                 "Last 90 Days": "3D",
                 "Last Year": "W",
                 "All Time": "W"}[date_range]
        )
        
        # Generate sample data for demonstration
        data = []
        for date in dates:
            base_posts = np.random.randint(50, 150)
            data.append({
                'date': date,
                'total_posts': base_posts,
                'total_score': base_posts * np.random.randint(5, 20),
                'total_comments': base_posts * np.random.randint(3, 12),
                'upvote_ratio': np.random.uniform(0.7, 0.95),
                'awards': np.random.randint(0, base_posts // 5)
            })
        
        daily_df = pd.DataFrame(data)
        
        # Simulate author data
        authors = ['RedditPro', 'MarketerXYZ', 'ContentCreator', 'SEOmaster', 
                  'SocialGuru', 'StrategyPro', 'Analyst101', 'DataDriven',
                  'RedditExpert', 'DigitalGuru']
        
        author_data = []
        for author in authors:
            posts = np.random.randint(5, 50)
            score = posts * np.random.randint(10, 100)
            author_data.append({
                'author': author,
                'posts': posts,
                'total_score': score,
                'avg_score': score / posts,
                'total_comments': posts * np.random.randint(5, 20),
                'engagement_rate': np.random.uniform(0.1, 0.9)
            })
        
        author_df = pd.DataFrame(author_data).sort_values('total_score', ascending=False)
        
        return daily_df, author_df
    
    # Load and display data
    with st.spinner("Loading dashboard data..."):
        try:
            daily_df, author_df = get_dashboard_data(date_range, subreddit_filter, min_engagement)
            
            # Key metrics in nice cards
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            with metrics_col1:
                metric_card(
                    "Total Posts",
                    f"{daily_df['total_posts'].sum():,}",
                    f"{daily_df['total_posts'].mean():.1f} avg/day",
                    "normal"
                )
            
            with metrics_col2:
                metric_card(
                    "Total Engagement",
                    f"{daily_df['total_score'].sum():,}",
                    f"{daily_df['total_score'].mean():.1f} avg/day",
                    "good" if daily_df['total_score'].pct_change().mean() > 0 else "bad"
                )
            
            with metrics_col3:
                metric_card(
                    "Total Comments",
                    f"{daily_df['total_comments'].sum():,}",
                    f"{daily_df['total_comments'].mean():.1f} avg/day",
                    "good"
                )
            
            with metrics_col4:
                metric_card(
                    "Avg. Score per Post",
                    f"{(daily_df['total_score'].sum() / daily_df['total_posts'].sum()):.1f}",
                    f"{daily_df['total_score'].div(daily_df['total_posts']).std():.1f} std dev",
                    "normal"
                )
            
            # Engagement Trends visualization
            st.markdown('<h3 class="sub-header">Engagement Trends</h3>', unsafe_allow_html=True)
            
            trend_metrics = st.multiselect(
                "Select metrics to display on trend chart:",
                ["total_posts", "total_score", "total_comments", "awards"],
                default=["total_posts", "total_score", "total_comments"]
            )
            
            fig_trends = px.line(
                daily_df,
                x="date",
                y=trend_metrics,
                title="",
                labels={
                    "date": "Date",
                    "value": "Count",
                    "variable": "Metric"
                },
                template="plotly_white"
            )
            
            fig_trends.update_layout(
                legend_title="Metrics",
                xaxis_title="Date",
                yaxis_title="Value",
                height=450,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
            
            # Two column layout for remaining charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Top authors
                st.markdown('<h3 class="sub-header">Top Contributors</h3>', unsafe_allow_html=True)
                
                fig_authors = px.bar(
                    author_df.head(10),
                    x="author",
                    y="total_score",
                    color="engagement_rate",
                    color_continuous_scale="RdYlGn",
                    labels={
                        "author": "Author",
                        "total_score": "Total Score",
                        "engagement_rate": "Engagement Rate"
                    },
                    template="plotly_white"
                )
                
                fig_authors.update_layout(height=400)
                st.plotly_chart(fig_authors, use_container_width=True)
            
            with col2:
                # Engagement by Day/Hour
                st.markdown('<h3 class="sub-header">Performance by Time</h3>', unsafe_allow_html=True)
                
                # Generate sample day/hour heatmap data for the demo
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                hours = list(range(24))
                
                heatmap_data = []
                for day in days:
                    for hour in hours:
                        base_score = np.random.randint(20, 100)
                        if day in ['Saturday', 'Sunday'] and 10 <= hour <= 18:
                            # Weekend daytime boost
                            base_score *= 1.5
                        elif 17 <= hour <= 22:
                            # Evening boost
                            base_score *= 1.3
                        
                        heatmap_data.append({
                            'day': day,
                            'hour': hour,
                            'score': base_score
                        })
                
                heatmap_df = pd.DataFrame(heatmap_data)
                
                # Create custom day order
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                heatmap_df['day'] = pd.Categorical(heatmap_df['day'], categories=day_order, ordered=True)
                heatmap_df = heatmap_df.sort_values(['day', 'hour'])
                
                fig_heatmap = px.density_heatmap(
                    heatmap_df,
                    x="hour",
                    y="day",
                    z="score",
                    title="",
                    labels={"score": "Engagement Score"},
                    color_continuous_scale="Viridis"
                )
                
                fig_heatmap.update_layout(
                    xaxis_title="Hour of Day (24h)",
                    yaxis_title="Day of Week",
                    height=400
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown('<h3 class="sub-header">Performance Trend Analysis</h3>', unsafe_allow_html=True)
            
            # Create advanced performance metrics
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                # Calculate growth rate
                start_val = daily_df['total_score'].iloc[0]
                end_val = daily_df['total_score'].iloc[-1]
                growth_rate = ((end_val - start_val) / start_val) * 100 if start_val > 0 else 0
                
                metric_card(
                    "Growth Rate",
                    f"{growth_rate:.1f}%",
                    "Positive trend" if growth_rate > 0 else "Declining trend",
                    "good" if growth_rate > 0 else "bad"
                )
            
            with metrics_col2:
                # Calculate engagement ratio
                engagement_ratio = daily_df['total_comments'].sum() / daily_df['total_posts'].sum()
                
                metric_card(
                    "Comments per Post",
                    f"{engagement_ratio:.2f}",
                    f"{engagement_ratio - 3.5:.2f} vs. benchmark" if engagement_ratio > 3.5 else f"{3.5 - engagement_ratio:.2f} below benchmark",
                    "good" if engagement_ratio > 3.5 else "bad"
                )
            
            with metrics_col3:
                # Calculate average upvote ratio
                avg_upvote_ratio = daily_df['upvote_ratio'].mean() * 100
                
                metric_card(
                    "Avg. Upvote Ratio",
                    f"{avg_upvote_ratio:.1f}%",
                    f"{avg_upvote_ratio - 80:.1f}% vs. benchmark" if avg_upvote_ratio > 80 else f"{80 - avg_upvote_ratio:.1f}% below benchmark",
                    "good" if avg_upvote_ratio > 80 else "bad"
                )
            
            # Export options
            st.markdown("### Export Dashboard Data")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(to_excel_download_link(daily_df, "reddit_metrics.xlsx", "📥 Download Metrics Data"), unsafe_allow_html=True)
            with col2:
                st.markdown(to_excel_download_link(author_df, "reddit_authors.xlsx", "📥 Download Author Data"), unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading dashboard data: {str(e)}")
            st.info("Check if your database connection is properly configured.")

elif selected_tab == "🔍 Content Intelligence":
    st.markdown('<h2 class="sub-header">Content Performance Analysis</h2>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.sidebar.markdown('<h3>Content Filters</h3>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        date_range_content = st.selectbox(
            "Time Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            index=1
        )
        
        min_score = st.slider("Minimum Score", 0, 1000, 10)
        content_type = st.multiselect(
            "Content Type",
            ["Text", "Image", "Video", "Link", "Poll"],
            default=["Text", "Image", "Link"]
        )
        
        analysis_focus = st.radio(
            "Analysis Focus",
            ["Title Analysis", "Content Topics", "Post Timing", "Source Analysis"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    @st.cache_data(ttl=3600)
    def get_content_analysis_data(date_range, min_score, content_types):
        # Generate sample data for demonstration
        n_posts = 200
        
        # Create random dates
        end_date = datetime.now()
        if date_range == "Last 24 Hours":
            start_date = end_date - timedelta(days=1)
        elif date_range == "Last 7 Days":
            start_date = end_date - timedelta(days=7)
        elif date_range == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif date_range == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
        elif date_range == "Last Year":
            start_date = end_date - timedelta(days=365)
        else:  # All Time
            start_date = end_date - timedelta(days=730)
        
        dates = [start_date + (end_date - start_date) * random.random() for _ in range(n_posts)]
        
        # Sample domains
        domains = [
            "i.redd.it", "v.redd.it", "imgur.com", "medium.com", 
            "youtube.com", "twitter.com", "linkedin.com", "forbes.com",
            "business.com", "marketingweek.com", "adweek.com"
        ]
        
        # Sample titles
        title_templates = [
            "How to {action} your {object} for better {result}",
            "{number} ways to improve your {object} strategy",
            "The ultimate guide to {action} {object}",
            "Why your {object} strategy isn't {result}",
            "{object} tips that will change your {result}",
            "The secret to {result} with {object}",
            "How we achieved {number}% growth in our {object}"
        ]
        
        actions = ["optimize", "improve", "enhance", "analyze", "leverage", "transform", "scale"]
        objects = ["marketing", "social media", "content", "Reddit ads", "SEO", "brand", "ROI"]
        results = ["engagement", "conversions", "growth", "revenue", "visibility", "success", "ROI"]
        numbers = ["5", "7", "10", "12", "15", "20", "25", "30"]
        
        titles = []
        for _ in range(n_posts):
            template = random.choice(title_templates)
            title = template.format(
                action=random.choice(actions),
                object=random.choice(objects),
                result=random.choice(results),
                number=random.choice(numbers)
            )
            titles.append(title)
        
        # Generate content types
        content_types_list = ["Text", "Image", "Video", "Link", "Poll"]
        post_types = [random.choice(content_types_list) for _ in range(n_posts)]
        
        # Generate other metrics
        scores = [max(0, int(random.normalvariate(50, 30))) for _ in range(n_posts)]
        comments = [max(0, int(s * random.uniform(0.1, 0.5))) for s in scores]
        upvote_ratios = [min(1.0, max(0.5, random.normalvariate(0.8, 0.1))) for _ in range(n_posts)]
        
        # Create URLs based on content type
        urls = []
        for content_type in post_types:
            if content_type == "Link":
                domain = random.choice(domains)
                urls.append(f"https://{domain}/article-{random.randint(1000, 9999)}")
            elif content_type == "Image":
                urls.append("https://i.redd.it/image-" + str(random.randint(1000, 9999)))
            elif content_type == "Video":
                urls.append("https://v.redd.it/video-" + str(random.randint(1000, 9999)))
            else:
                urls.append(None)
        
        # Create dataframe
        df = pd.DataFrame({
            'created_utc': dates,
            'title': titles,
            'score': scores,
            'num_comments': comments,
            'upvote_ratio': upvote_ratios,
            'url': urls,
            'post_type': post_types,
            'domain': [urlparse(url).netloc if url else "self.reddit" for url in urls]
        })
        
        # Filter based on inputs
        df = df[df['score'] >= min_score]
        if content_types and content_types != ["Text", "Image", "Video", "Link", "Poll"]:
            df = df[df['post_type'].isin(content_types)]
        
        return df
    
    # Load content data
    with st.spinner("Analyzing content data..."):
        try:
            df = get_content_analysis_data(date_range_content, min_score, content_type)
            
            # Show basic stats
            metric_cols = st.columns(4)
            with metric_cols[0]:
                metric_card(
                    "Total Posts",
                    f"{len(df):,}",
                    f"Analyzed content"
                )
            
            with metric_cols[1]:
                metric_card(
                    "Avg. Score",
                    f"{df['score'].mean():.1f}",
                    f"{df['score'].std():.1f} std dev"
                )
            
            with metric_cols[2]:
                metric_card(
                    "Avg. Title Length",
                    f"{df['title'].str.len().mean():.1f} chars",
                    f"{df['title'].str.len().std():.1f} std dev"
                )
            
            with metric_cols[3]:
                comment_ratio = df['num_comments'].sum() / len(df) 
                metric_card(
                    "Comments per Post",
                    f"{comment_ratio:.1f}",
                    f"{comment_ratio - 5:.1f} vs. benchmark" if comment_ratio > 5 else f"{5 - comment_ratio:.1f} below benchmark",
                    "good" if comment_ratio > 5 else "bad"
                )
            
            # Display specific analysis based on selected focus
            if analysis_focus == "Title Analysis":
                st.markdown('<h3 class="sub-header">Title Analysis</h3>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Show correlation between title length and score
                    df['title_length'] = df['title'].str.len()
                    fig_title_corr = px.scatter(
                        df,
                        x="title_length",
                        y="score",
                        color="num_comments",
                        color_continuous_scale="Viridis",
                        title="Title Length vs. Engagement",
                        trendline="ols",
                        labels={
                            "title_length": "Title Length (characters)",
                            "score": "Post Score",
                            "num_comments": "Number of Comments"
                        }
                    )
                    st.plotly_chart(fig_title_corr, use_container_width=True)
                
                with col2:
                    # Generate word cloud
                    st.subheader("Title Word Cloud")
                    
                    word_cloud = generate_wordcloud(df['title'])
                    if word_cloud:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.imshow(word_cloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.info("Not enough text data to generate word cloud.")
                
                # Common title patterns
                st.subheader("Top Words in Successful Titles")
                
                # Extract top topics from titles
                top_words = extract_topics(df['title'], n_top_words=15)
                
                if not top_words.empty:
                    fig_words = px.bar(
                        top_words,
                        x="count",
                        y="word",
                        orientation='h',
                        title="Most Common Words in Post Titles",
                        labels={
                            "count": "Frequency",
                            "word": "Word"
                        }
                    )
                    fig_words.update_layout(
                        yaxis=dict(categoryorder='total ascending'),
                        height=500
                    )
                    st.plotly_chart(fig_words, use_container_width=True)
                
                # Title pattern analysis
                st.subheader("Title Pattern Analysis")
                
                # Analyze length vs performance
                length_bins = [0, 30, 40, 50, 60, 70, 80, 90, 100, 150]
                df['title_length_bin'] = pd.cut(df['title_length'], bins=length_bins)
                
                length_analysis = df.groupby('title_length_bin').agg({
                    'score': 'mean',
                    'num_comments': 'mean',
                    'title': 'count'
                }).reset_index()
                
                length_analysis.columns = ['Title Length', 'Avg Score', 'Avg Comments', 'Count']
                
                fig_length = px.bar(
                    length_analysis,
                    x='Title Length',
                    y=['Avg Score', 'Avg Comments'],
                    barmode='group',
                    title="Performance by Title Length",
                    labels={
                        'value': 'Average Value',
                        'variable': 'Metric'
                    }
                )
                st.plotly_chart(fig_length, use_container_width=True)
                
                # Title patterns with high engagement
                high_performing = df[df['score'] > df['score'].quantile(0.75)]
                st.subheader("High-Performing Title Examples")
                st.dataframe(
                    high_performing[['title', 'score', 'num_comments']].sort_values('score', ascending=False).head(10),
                    use_container_width=True
                )
                
            elif analysis_focus == "Content Topics":
                st.markdown('<h3 class="sub-header">Content Topic Analysis</h3>', unsafe_allow_html=True)
                
                # Content type distribution
                fig_content_type = px.pie(
                    df,
                    names='post_type',
                    title="Content Type Distribution",
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_content_type, use_container_width=True)
                
                # Content performance by type
                content_perf = df.groupby('post_type').agg({
                    'score': ['mean', 'median', 'count'],
                    'num_comments': ['mean', 'median']
                }).reset_index()
                
                content_perf.columns = ['Post Type', 'Mean Score', 'Median Score', 'Count', 'Mean Comments', 'Median Comments']
                
                fig_content_perf = px.bar(
                    content_perf,
                    x='Post Type',
                    y=['Mean Score', 'Mean Comments'],
                    barmode='group',
                    title="Performance by Content Type",
                    labels={
                        'value': 'Average Value',
                        'variable': 'Metric'
                    }
                )
                st.plotly_chart(fig_content_perf, use_container_width=True)
                
                # Topic extraction
                st.subheader("Popular Topics and Themes")
                
                col1, col2 = st.columns(2)
                with col1:
                    # Topic distribution visualization
                    topic_data = extract_topics(df['title'], n_top_words=10)
                    if not topic_data.empty:
                        fig_topics = px.pie(
                            topic_data,
                            values='count',
                            names='word',
                            title="Topic Distribution",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        st.plotly_chart(fig_topics, use_container_width=True)
                    else:
                        st.info("Not enough data to extract topics.")
                
                with col2:
                    # Topic correlation with engagement
                    st.write("Topic Correlation with Engagement")
                    # This would require more complex analysis with real data
                    # For now, display a placeholder chart
                    
                    # Create sample data for the demo
                    topics = ["marketing", "social", "content", "strategy", "growth"]
                    engagement_data = pd.DataFrame({
                        'Topic': topics,
                        'Avg Score': [random.uniform(30, 70) for _ in topics],
                        'Avg Comments': [random.uniform(5, 20) for _ in topics]
                    })
                    
                    fig_topic_eng = px.scatter(
                        engagement_data, 
                        x="Avg Score",
                        y="Avg Comments",
                        size="Avg Score",
                        color="Topic",
                        text="Topic",
                        title="Topics by Engagement Metrics"
                    )
                    
                    fig_topic_eng.update_traces(textposition='top center')
                    fig_topic_eng.update_layout(height=400)
                    st.plotly_chart(fig_topic_eng, use_container_width=True)
                
                # Content recommendations based on analysis
                st.subheader("Content Strategy Recommendations")
                
                recommendations = [
                    "Focus on 'how-to' content as it generates 35% more engagement",
                    "Posts with 'ultimate guide' in the title receive 28% more comments",
                    "Image posts perform better than text posts for engagement",
                    "Content about 'marketing strategy' is trending upward",
                    "Posts with list formats (e.g., '10 ways to...') generate more shares"
                ]
                
                for i, rec in enumerate(recommendations):
                    st.markdown(f"**{i+1}.** {rec}")
            
            elif analysis_focus == "Post Timing":
                st.markdown('<h3 class="sub-header">Post Timing Analysis</h3>', unsafe_allow_html=True)
                
                # Extract hour and day of week
                df['hour'] = pd.to_datetime(df['created_utc']).dt.hour
                df['day'] = pd.to_datetime(df['created_utc']).dt.day_name()
                
                # Day of week performance
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                df['day'] = pd.Categorical(df['day'], categories=day_order, ordered=True)
                
                day_perf = df.groupby('day').agg({
                    'score': 'mean',
                    'num_comments': 'mean',
                    'title': 'count'
                }).reset_index()
                
                day_perf.columns = ['Day', 'Avg Score', 'Avg Comments', 'Count']
                
                fig_day = px.bar(
                    day_perf,
                    x='Day',
                    y=['Avg Score', 'Avg Comments'],
                    barmode='group',
                    title="Performance by Day of Week",
                    labels={
                        'value': 'Average Value',
                        'variable': 'Metric'
                    }
                )
                st.plotly_chart(fig_day, use_container_width=True)
                
                # Hour of day performance
                hour_perf = df.groupby('hour').agg({
                    'score': 'mean',
                    'num_comments': 'mean',
                    'title': 'count'
                }).reset_index()
                
                hour_perf.columns = ['Hour', 'Avg Score', 'Avg Comments', 'Count']
                
                fig_hour = px.line(
                    hour_perf,
                    x='Hour',
                    y=['Avg Score', 'Avg Comments'],
                    title="Performance by Hour of Day",
                    labels={
                        'Hour': 'Hour of Day (24h)',
                        'value': 'Average Value',
                        'variable': 'Metric'
                    }
                )
                st.plotly_chart(fig_hour, use_container_width=True)
                
                # Day/hour heatmap
                timing_stats = df.groupby(['day', 'hour']).agg({
                    'score': 'mean'
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
                
                # Optimal posting time recommendations
                st.subheader("Recommended Posting Times")
                
                # Find best times
                best_times = timing_stats.sort_values('score', ascending=False).head(5)
                
                for i, row in enumerate(best_times.itertuples()):
                    st.markdown(f"**{i+1}.** {row.day} at {row.hour}:00 (Avg Score: {row.score:.1f})")
                
                # Create posting calendar visualization
                st.subheader("Posting Schedule Calendar")
                
                calendar_data = timing_stats.pivot(index='day', columns='hour', values='score')
                fig_calendar = px.imshow(
                    calendar_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Avg Score"),
                    x=list(range(24)),
                    y=day_order,
                    aspect="auto",
                    color_continuous_scale="Viridis"
                )
                
                fig_calendar.update_xaxes(side="top")
                st.plotly_chart(fig_calendar, use_container_width=True)
            
            elif analysis_focus == "Source Analysis":
                st.markdown('<h3 class="sub-header">Content Source Analysis</h3>', unsafe_allow_html=True)
                
                # Domain distribution
                domain_counts = df['domain'].value_counts().reset_index()
                domain_counts.columns = ['Domain', 'Count']
                domain_counts = domain_counts.sort_values('Count', ascending=False).head(10)
                
                fig_domains = px.bar(
                    domain_counts,
                    x='Count',
                    y='Domain',
                    orientation='h',
                    title="Top Content Sources",
                    labels={
                        'Count': 'Number of Posts',
                        'Domain': 'Source Domain'
                    }
                )
                
                fig_domains.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_domains, use_container_width=True)
                
                # Domain performance
                domain_perf = df.groupby('domain').agg({
                    'score': ['mean', 'sum', 'count'],
                    'num_comments': ['mean', 'sum']
                }).reset_index()
                
                domain_perf.columns = ['Domain', 'Avg Score', 'Total Score', 'Count', 'Avg Comments', 'Total Comments']
                domain_perf = domain_perf.sort_values('Total Score', ascending=False).head(10)
                
                fig_domain_perf = px.bar(
                    domain_perf,
                    x='Domain',
                    y=['Avg Score', 'Avg Comments'],
                    barmode='group',
                    title="Performance by Content Source",
                    labels={
                        'value': 'Average Value',
                        'variable': 'Metric'
                    }
                )
                st.plotly_chart(fig_domain_perf, use_container_width=True)
                
                # Source recommendations
                st.subheader("Content Source Strategy")
                
                st.markdown("""
                ### Source Recommendations:
                
                1. **Top-performing sources** - Continue to share content from your top 3 performing domains as they generate 45% more engagement than average
                
                2. **Content diversity** - Aim for a balanced mix of content sources to maintain audience interest
                
                3. **Self-posts vs. Links** - Self-posts are generating 30% more comments, while link posts receive 20% more upvotes
                
                4. **Video content** - YouTube links are underperforming compared to native Reddit video uploads
                
                5. **Competitive analysis** - Begin tracking competitor sources to identify untapped content opportunities
                """)
                
                # Show detailed source stats
                st.subheader("Detailed Source Statistics")
                st.dataframe(domain_perf, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error analyzing content: {str(e)}")
            st.info("Check if your database is properly configured.")
    
elif selected_tab == "🧠 Sentiment Analysis":
    st.markdown('<h2 class="sub-header">Sentiment Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.sidebar.markdown('<h3>Sentiment Filters</h3>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        sentiment_date_range = st.selectbox(
            "Time Period",
            ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year", "All Time"],
            index=1,
            key="sentiment_date_range"
        )
        
        sentiment_subreddit = st.multiselect(
            "Filter by Subreddit",
            ["all", "marketing", "DigitalMarketing", "socialmedia", "PPC", "SEO"],
            default=["all"],
            key="sentiment_subreddit"
        )
        
        sentiment_view = st.radio(
            "View",
            ["Overall Sentiment", "Sentiment Trends", "Comment Analysis", "Topic Sentiment"]
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    @st.cache_data(ttl=3600)
    def get_sentiment_data(date_range, subreddits):
        # Create sample date range
        end_date = datetime.now()
        if date_range == "Last 24 Hours":
            start_date = end_date - timedelta(hours=24)
            freq = "H"
            periods = 24
        elif date_range == "Last 7 Days":
            start_date = end_date - timedelta(days=7)
            freq = "D"
            periods = 7
        elif date_range == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
            freq = "D"
            periods = 30
        elif date_range == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
            freq = "3D"
            periods = 30
        elif date_range == "Last Year":
            start_date = end_date - timedelta(days=365)
            freq = "M"
            periods = 12
        else:  # All Time
            start_date = end_date - timedelta(days=730)
            freq = "M"
            periods = 24
        
        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        # Generate sample sentiment data
        data = []
        for date in dates:
            # Create base sentiment with some randomness and trend
            base_polarity = 0.2 + 0.4 * random.random()
            # Add weekly seasonality - more positive on weekends
            if date.dayofweek >= 5:  # Weekend
                base_polarity += 0.05
            
            # Generate sample posts
            n_posts = random.randint(50, 200)
            
            # Calculate sentiment metrics
            pos_pct = base_polarity + random.uniform(-0.05, 0.05)
            neg_pct = 0.3 + random.uniform(-0.1, 0.1)
            neu_pct = 1 - pos_pct - neg_pct
            
            avg_polarity = (pos_pct * 0.8 - neg_pct * 0.8)
            avg_subjectivity = 0.4 + random.uniform(-0.1, 0.1)
            
            data.append({
                'date': date,
                'avg_polarity': avg_polarity,
                'avg_subjectivity': avg_subjectivity,
                'positive_pct': pos_pct,
                'negative_pct': neg_pct,
                'neutral_pct': neu_pct,
                'post_count': n_posts
            })
        
        # Generate sample topic sentiment data
        topics = ["Marketing Strategy", "Social Media", "Content Creation", 
                 "Analytics", "Advertising", "SEO", "Email Marketing", 
                 "Brand Awareness", "Lead Generation", "Customer Engagement"]
        
        topic_data = []
        for topic in topics:
            topic_data.append({
                'topic': topic,
                'polarity': random.uniform(-0.3, 0.7),
                'subjectivity': random.uniform(0.3, 0.8),
                'post_count': random.randint(10, 100)
            })
        
        # Generate sample comment sentiment
        comments_data = []
        for _ in range(50):
            polarity = random.uniform(-0.8, 0.8)
            sentiment_class = "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"
            
            comments_data.append({
                'comment_text': f"Sample comment with {sentiment_class.lower()} sentiment about marketing strategy.",
                'polarity': polarity,
                'subjectivity': random.uniform(0.2, 0.9),
                'sentiment_class': sentiment_class,
                'upvotes': random.randint(1, 50) if sentiment_class == "Positive" else random.randint(1, 20)
            })
        
        return pd.DataFrame(data), pd.DataFrame(topic_data), pd.DataFrame(comments_data)
    
    # Load sentiment data
    with st.spinner("Analyzing sentiment data..."):
        try:
            sentiment_df, topic_sentiment_df, comments_df = get_sentiment_data(sentiment_date_range, sentiment_subreddit)
            
            # Show appropriate view based on selection
            if sentiment_view == "Overall Sentiment":
                st.markdown('<h3 class="sub-header">Overall Sentiment Analysis</h3>', unsafe_allow_html=True)
                
                # Aggregate sentiment metrics
                avg_polarity = sentiment_df['avg_polarity'].mean()
                avg_subjectivity = sentiment_df['avg_subjectivity'].mean()
                pos_pct = sentiment_df['positive_pct'].mean() * 100
                neg_pct = sentiment_df['negative_pct'].mean() * 100
                neu_pct = sentiment_df['neutral_pct'].mean() * 100
                
                # Display sentiment gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = avg_polarity * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Average Sentiment Score", 'font': {'size': 24}},
                    delta = {'reference': 20, 'increasing': {'color': "green"}},
                    gauge = {
                        'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [-100, -20], 'color': 'red'},
                            {'range': [-20, 20], 'color': 'yellow'},
                            {'range': [20, 100], 'color': 'green'}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 40}}))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Sentiment distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create sentiment distribution pie chart
                    sentiment_dist = pd.DataFrame({
                        'Sentiment': ['Positive', 'Neutral', 'Negative'],
                        'Percentage': [pos_pct, neu_pct, neg_pct]
                    })
                    
                    fig_dist = px.pie(
                        sentiment_dist,
                        values='Percentage',
                        names='Sentiment',
                        title="Sentiment Distribution",
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': 'green',
                            'Neutral': 'gray',
                            'Negative': 'red'
                        },
                        hole=0.4
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                
                with col2:
                    # Create polarity/subjectivity scatter plot
                    fig_scatter = px.scatter(
                        sentiment_df,
                        x="avg_subjectivity",
                        y="avg_polarity",
                        size="post_count",
                        color="avg_polarity",
                        color_continuous_scale=px.colors.diverging.RdBu,
                        color_continuous_midpoint=0,
                        title="Sentiment vs. Subjectivity",
                        labels={
                            "avg_subjectivity": "Subjectivity Score",
                            "avg_polarity": "Sentiment Score",
                            "post_count": "Number of Posts"
                        }
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Display key metrics
                st.subheader("Sentiment Metrics")
                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    metric_card(
                        "Avg. Sentiment Score",
                        f"{avg_polarity:.2f}",
                        "Scale from -1 to 1"
                    )
                
                with metric_cols[1]:
                    metric_card(
                        "Positive Content",
                        f"{pos_pct:.1f}%",
                        f"{pos_pct - 50:.1f}% vs target" if pos_pct > 50 else f"{50 - pos_pct:.1f}% below target",
                        "good" if pos_pct > 50 else "bad"
                    )
                
                with metric_cols[2]:
                    metric_card(
                        "Negative Content",
                        f"{neg_pct:.1f}%",
                        f"{20 - neg_pct:.1f}% below threshold" if neg_pct < 20 else f"{neg_pct - 20:.1f}% above threshold",
                        "good" if neg_pct < 20 else "bad"
                    )
                
                with metric_cols[3]:
                    metric_card(
                        "Subjectivity Score",
                        f"{avg_subjectivity:.2f}",
                        "Scale from 0 to 1"
                    )
                
                # Sentiment summary and insights
                st.subheader("Sentiment Insights")
                
                st.markdown("""
                ### Key Findings:
                
                1. **Overall sentiment is positive** with an average sentiment score that's trending upward
                
                2. **Negative sentiment decreased** by 7.5% compared to the previous time period
                
                3. **Weekends show more positive sentiment** than weekdays, suggesting timing strategy opportunities
                
                4. **High subjectivity** indicates emotional responses to your content rather than factual discussions
                
                5. **Action items:**
                   - Analyze negative comments for constructive feedback
                   - Replicate content strategy from highest positive sentiment periods
                   - Consider weekend posting for maximum positive engagement
                """)
            
            elif sentiment_view == "Sentiment Trends":
                st.markdown('<h3 class="sub-header">Sentiment Trends Over Time</h3>', unsafe_allow_html=True)
                
                # Create line chart of sentiment over time
                fig_trend = px.line(
                    sentiment_df,
                    x="date",
                    y=["avg_polarity", "avg_subjectivity"],
                    title="Sentiment Trends",
                    labels={
                        "date": "Date",
                        "value": "Score",
                        "variable": "Metric"
                    }
                )
                
                fig_trend.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Neutral"
                )
                
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Sentiment composition over time
                st.subheader("Sentiment Composition")
                
                sentiment_comp = sentiment_df.copy()
                fig_comp = px.area(
                    sentiment_comp,
                    x="date",
                    y=["positive_pct", "neutral_pct", "negative_pct"],
                    title="Sentiment Distribution Over Time",
                    labels={
                        "date": "Date",
                        "value": "Percentage",
                        "variable": "Sentiment"
                    },
                    color_discrete_map={
                        "positive_pct": "green",
                        "neutral_pct": "gray",
                        "negative_pct": "red"
                    }
                )
                
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Volume vs. Sentiment chart
                st.subheader("Post Volume vs. Sentiment")
                
                fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add post count line
                fig_vol.add_trace(
                    go.Bar(
                        x=sentiment_df["date"],
                        y=sentiment_df["post_count"],
                        name="Post Count",
                        marker_color="lightblue",
                        opacity=0.7
                    ),
                    secondary_y=False
                )
                
                # Add sentiment line
                fig_vol.add_trace(
                    go.Scatter(
                        x=sentiment_df["date"],
                        y=sentiment_df["avg_polarity"],
                        name="Sentiment Score",
                        line=dict(color="darkblue", width=3)
                    ),
                    secondary_y=True
                )
                
                # Set axis titles
                fig_vol.update_layout(
                    title_text="Post Volume vs. Sentiment Score",
                    xaxis_title="Date"
                )
                
                fig_vol.update_yaxes(title_text="Post Count", secondary_y=False)
                fig_vol.update_yaxes(title_text="Sentiment Score", secondary_y=True)
                
                st.plotly_chart(fig_vol, use_container_width=True)
                
                # Identify sentiment shifts
                st.subheader("Significant Sentiment Shifts")
                
                # Calculate sentiment change
                sentiment_df['polarity_change'] = sentiment_df['avg_polarity'].diff()
                
                # Identify significant shifts
                significant_shifts = sentiment_df.copy()
                significant_shifts = significant_shifts[abs(significant_shifts['polarity_change']) > 0.1]
                significant_shifts = significant_shifts.sort_values('date', ascending=False)
                
                if not significant_shifts.empty:
                    for i, row in enumerate(significant_shifts.itertuples()):
                        direction = "increased" if row.polarity_change > 0 else "decreased"
                        st.markdown(f"**{row.date.strftime('%B %d, %Y')}:** Sentiment {direction} by {abs(row.polarity_change):.2f} points")
                        if i >= 4:  # Show only top 5 shifts
                            break
                else:
                    st.info("No significant sentiment shifts detected in this time period.")
            
            elif sentiment_view == "Comment Analysis":
                st.markdown('<h3 class="sub-header">Comment Sentiment Analysis</h3>', unsafe_allow_html=True)
                
                # Sort comments by polarity
                comments_df = comments_df.sort_values('polarity', ascending=False)
                
                # Sentiment distribution in comments
                st.subheader("Comment Sentiment Distribution")
                
                fig_comment_dist = px.histogram(
                    comments_df,
                    x="polarity",
                    title="Distribution of Comment Sentiment",
                    labels={"polarity": "Sentiment Score"},
                    color_discrete_sequence=["lightblue"],
                    nbins=20
                )
                
                fig_comment_dist.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Neutral"
                )
                
                st.plotly_chart(fig_comment_dist, use_container_width=True)
                
                # Most positive and negative comments
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Most Positive Comments")
                    positive_comments = comments_df[comments_df['polarity'] > 0.3].sort_values('polarity', ascending=False).head(5)
                    
                    for i, row in enumerate(positive_comments.itertuples()):
                        st.markdown(f"""
                        **Comment {i+1} (Score: {row.polarity:.2f}):**
                        > "{row.comment_text}"
                        """)
                
                with col2:
                    st.subheader("Most Negative Comments")
                    negative_comments = comments_df[comments_df['polarity'] < -0.3].sort_values('polarity', ascending=True).head(5)
                    
                    for i, row in enumerate(negative_comments.itertuples()):
                        st.markdown(f"""
                        **Comment {i+1} (Score: {row.polarity:.2f}):**
                        > "{row.comment_text}"
                        """)
                
                # Comment engagement correlation with sentiment
                st.subheader("Comment Engagement by Sentiment")
                
                fig_comment_eng = px.scatter(
                    comments_df,
                    x="polarity",
                    y="upvotes",
                    color="sentiment_class",
                    title="Comment Engagement vs. Sentiment",
                    labels={
                        "polarity": "Sentiment Score",
                        "upvotes": "Upvotes",
                        "sentiment_class": "Sentiment"
                    },
                    color_discrete_map={
                        "Positive": "green",
                        "Neutral": "gray",
                        "Negative": "red"
                    }
                )
                
                fig_comment_eng.update_layout(height=500)
                st.plotly_chart(fig_comment_eng, use_container_width=True)
                
                # Comment sentiment summary
                st.subheader("Comment Analysis Summary")
                
                comment_metrics = comments_df.groupby('sentiment_class').agg({
                    'comment_text': 'count',
                    'upvotes': ['mean', 'sum'],
                    'polarity': 'mean',
                    'subjectivity': 'mean'
                }).reset_index()
                
                comment_metrics.columns = ['Sentiment', 'Count', 'Avg Upvotes', 'Total Upvotes', 'Avg Polarity', 'Avg Subjectivity']
                
                st.dataframe(comment_metrics, use_container_width=True)
                
                # Comment sentiment insights
                st.markdown("""
                ### Key Comment Insights:
                
                1. **Positive comments receive 42% more upvotes** than negative ones
                
                2. **Neutral comments are least engaging** with lowest average upvotes
                
                3. **Questions and constructive criticism** generate more discussion than pure negative feedback
                
                4. **Most negative comments focus on** pricing, customer service, and technical issues
                
                5. **Action items:**
                   - Address common negative feedback themes proactively in your content
                   - Engage more with positive commenters to boost visibility
                   - Monitor sentiment shifts after implementing changes
                """)
            
            elif sentiment_view == "Topic Sentiment":
                st.markdown('<h3 class="sub-header">Topic Sentiment Analysis</h3>', unsafe_allow_html=True)
                
                # Topic sentiment barplot
                topic_sentiment_df = topic_sentiment_df.sort_values('polarity', ascending=False)
                
                fig_topic = px.bar(
                    topic_sentiment_df,
                    x='topic',
                    y='polarity',
                    color='polarity',
                    color_continuous_scale=px.colors.diverging.RdBu,
                    color_continuous_midpoint=0,
                    title="Sentiment by Topic",
                    labels={
                        'topic': 'Topic',
                        'polarity': 'Sentiment Score'
                    },
                    height=500
                )
                
                fig_topic.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Neutral"
                )
                
                st.plotly_chart(fig_topic, use_container_width=True)
                
                # Topic engagement vs. sentiment
                st.subheader("Topic Engagement vs. Sentiment")
                
                fig_topic_eng = px.scatter(
                    topic_sentiment_df,
                    x='polarity',
                    y='post_count',
                    size='post_count',
                    color='subjectivity',
                    hover_name='topic',
                    text='topic',
                    title="Topic Engagement vs. Sentiment",
                    labels={
                        'polarity': 'Sentiment Score',
                        'post_count': 'Post Count',
                        'subjectivity': 'Subjectivity'
                    }
                )
                
                fig_topic_eng.update_traces(textposition='top center')
                st.plotly_chart(fig_topic_eng, use_container_width=True)
                
                # Topic sentiment table
                st.subheader("Topic Sentiment Details")
                
                # Add sentiment classification
                topic_sentiment_df['sentiment_class'] = topic_sentiment_df['polarity'].apply(
                    lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
                )
                
                # Display as table
                st.dataframe(
                    topic_sentiment_df[['topic', 'polarity', 'subjectivity', 'post_count', 'sentiment_class']].sort_values('polarity', ascending=False),
                    use_container_width=True
                )
                
                # Topic insights
                st.subheader("Topic Sentiment Insights")
                
                # Get most positive and negative topics
                most_positive = topic_sentiment_df.loc[topic_sentiment_df['polarity'].idxmax()]['topic']
                most_negative = topic_sentiment_df.loc[topic_sentiment_df['polarity'].idxmin()]['topic']
                
                # Most discussed topic
                most_discussed = topic_sentiment_df.loc[topic_sentiment_df['post_count'].idxmax()]['topic']
                most_subjective = topic_sentiment_df.loc[topic_sentiment_df['subjectivity'].idxmax()]['topic']
                
                st.markdown(f"""
                ### Key Topic Insights:
                
                1. **Most positive topic:** {most_positive}
                   - Continue creating content around this topic to maintain positive sentiment
                
                2. **Most negative topic:** {most_negative}
                   - Address concerns and consider a different approach to this topic
                
                3. **Most discussed topic:** {most_discussed}
                   - High volume indicates strong audience interest, regardless of sentiment
                
                4. **Most subjective topic:** {most_subjective}
                   - This topic generates the most opinion-based rather than fact-based engagement
                
                5. **Sentiment gap opportunities:**
                   - Topics with negative sentiment but high engagement represent opportunities for improvement
                   - Consider addressing common concerns in these topics with new content
                """)
                
                # Topic recommendations
                st.subheader("Content Topic Recommendations")
                
                # Simulate AI-generated recommendations
                recommendations = [
                    f"Focus more on '{most_positive}' content to build on positive reception",
                    f"Create educational content around '{most_negative}' to address misconceptions",
                    f"Leverage high interest in '{most_discussed}' with a comprehensive guide",
                    "Consider a weekly round-up post covering all positive sentiment topics",
                    "Address price-related concerns with a value proposition post"
                ]
                
                for i, rec in enumerate(recommendations):
                    st.markdown(f"**{i+1}.** {rec}")
        
        except Exception as e:
            st.error(f"Error analyzing sentiment: {str(e)}")
            st.info("Check if your database connection is properly configured.")

elif selected_tab == "🎯 Competitor Analysis":
    st.markdown('<h2 class="sub-header">Competitor Analysis</h2>', unsafe_allow_html=True)
    
    st.info("This module is currently under development. Check back soon for competitive analysis features!")
    
    # Placeholder for future competitor analysis features
    st.markdown("""
    ### Coming Soon:
    
    - **Competitor content tracking** - Monitor your competitors' Reddit posts and engagement
    - **Share of voice analysis** - Compare your brand mentions against competitors
    - **Competitive sentiment comparison** - See how audience sentiment differs between brands
    - **Content strategy insights** - Identify gaps and opportunities based on competitor performance
    - **Automated alerts** - Get notified when competitors launch new campaigns or receive unusual engagement
    """)

elif selected_tab == "🤖 AI Recommendations":
    st.markdown('<h2 class="sub-header">AI-Powered Recommendations</h2>', unsafe_allow_html=True)
    
    st.info("The AI Recommendations module is currently in beta. Some features may be limited.")
    
    # Placeholder for AI recommendations
    st.markdown("""
    ### AI Content Strategy Recommendations:
    
    1. **Content Calendar Optimization** - AI-generated posting schedule based on your historical performance
    
    2. **Topic Recommendations** - Data-driven content ideas for your next Reddit posts
    
    3. **Engagement Boosting Tips** - Personalized suggestions to increase engagement metrics
    
    4. **Title Optimization** - AI-generated title suggestions based on high-performing patterns
    
    5. **Audience Insights** - Deep learning analysis of your audience's preferences and behaviors
    """)

# Add footer
st.markdown("""
---
<p style="text-align: center; color: gray; font-size: 0.8rem;">
AdInsight360 PRO | Advanced Reddit Marketing Analytics Platform | © 2023
</p>
""", unsafe_allow_html=True)