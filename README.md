# AdInsight360





This project automates the collection of marketing-related content from Reddit and loads it into Snowflake for analysis. The pipeline extracts posts from various Reddit categories (hot, new, and top posts) and maintains a structured database of marketing insights.

## Project Structure

```
AdInsight360/
├── airflow/               # Airflow DAGs for orchestration
│   └── dags/
├── analyses/             # dbt analyses (ad hoc SQL)
├── macros/              # dbt macros
├── models/              # dbt models
│   ├── marts/           # Business-level marts
│   ├── staging/         # Staging models
│   ├── utils/           # Utility models/macros
│   └── sources.yml      # Source definitions
├── scripts/             # Data extraction and loading scripts
│   ├── extract_data.py
│   └── load_to_snowflake.py
├── streamlit/           # Streamlit analytics dashboard
│   ├── dashboard.py     # Main dashboard application
│   ├── pages/          # Dashboard pages
│   └── utils/          # Dashboard utilities
├── data/               # Data storage and documentation
│   ├── raw/            # Raw JSON data
│   └── doc/            # Documentation assets
├── tests/              # dbt tests
├── .env                # Environment variables
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Data Pipeline

The project consists of three main components:

1. **Data Extraction** (`scripts/extract_data.py`)
   - Fetches posts from Reddit's marketing subreddit
   - Collects posts from multiple categories (hot, new, top)
   - Handles duplicate prevention
   - Saves data in JSON format

2. **Data Loading** (`scripts/load_to_snowflake.py`)
   - Loads extracted data into Snowflake
   - Maintains data integrity with primary keys
   - Handles data type conversions
   - Provides error handling and logging

3. **Analytics Dashboard** (`streamlit/dashboard.py`)
   - Interactive data exploration
   - Real-time metrics and KPIs
   - Custom visualizations
   - Trend analysis and insights










## Dashboard Preview

![Engagement Metrics](data/doc/newplot.png)
- *Real-time engagement metrics showing post performance and community interaction*

![Author Analysis](data/doc/Author_analysis.png)
- *Author contribution analysis and trending metrics*

## Database Schema

![Database Schema](data/doc/ADINSIGHT_DB.png)
- *AdInsight360 database schema showing relationships between staging and mart layers*


## Features

- Automated Reddit data extraction
- Duplicate post prevention
- Structured data storage in Snowflake
- Comprehensive post metadata collection:
  - Post titles and content
  - Engagement metrics (scores, comments)
  - Author information
  - Timestamps
  - URLs
- Interactive Analytics Dashboard:
  - Real-time engagement metrics
  - Content performance analysis
  - Author influence tracking
  - Trend visualization
  - Custom filtering and exploration



## Orchestration & Transformation

- **Airflow** orchestrates the end-to-end pipeline, automating extraction, loading, and transformation tasks.
- **dbt** is used for data transformation and analytics modeling. Models are organized into staging and marts layers for clean, analytics-ready data.
- **Streamlit** provides an interactive web interface for data exploration and analysis.


## Analytics Dashboard

The Streamlit dashboard provides interactive analytics and insights:

1. **Engagement Metrics**
   - Post performance tracking
   - Comment activity analysis
   - Score distribution patterns

2. **Content Analysis**
   - Popular topics and themes
   - Title sentiment analysis
   - URL domain statistics

3. **Author Analytics**
   - Top contributors
   - Posting patterns
   - Engagement rates

4. **Trend Analysis**
   - Time-based patterns
   - Seasonal trends
   - Growth metrics


