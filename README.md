# AdInsight360

A data pipeline for extracting and analyzing marketing insights from Reddit data.

## Overview

This project automates the collection of marketing-related content from Reddit and loads it into Snowflake for analysis. The pipeline extracts posts from various Reddit categories (hot, new, and top posts) and maintains a structured database of marketing insights.

## Project Structure

```
AdInsight360/
├── airflow/               
│   └── dags/
├── analyses/              
├── macros/                 
├── models/               
│   ├── marts/            
│   ├── staging/
│   ├── utils/            
│   └── sources.yml       
├── scripts/               
│   ├── extract_data.py
│   └── load_to_snowflake.py
├── tests/                
├── .env                    
├── requirements.txt       
└── README.md               
```










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

## Data Pipeline

The project consists of two main components:

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

## Orchestration & Transformation

- **Airflow** orchestrates the end-to-end pipeline, automating extraction, loading, and transformation tasks.
- **dbt** is used for data transformation and analytics modeling. Models are organized into staging and marts layers for clean, analytics-ready data.

## Environment Setup

1. **Clone the repository** and install dependencies from `requirements.txt`.
2. **Configure environment variables** by creating a `.env` file in the project root with the following keys:
   - `REDDIT_CLIENT_ID`
   - `REDDIT_CLIENT_SECRET`
   - `REDDIT_USER_AGENT`
   - `SNOWFLAKE_USER`
   - `SNOWFLAKE_PASSWORD`
   - `SNOWFLAKE_ACCOUNT`
   - `SNOWFLAKE_DATABASE`
   - `SNOWFLAKE_SCHEMA`
   - `SNOWFLAKE_WAREHOUSE`
   - `SNOWFLAKE_ROLE`
3. **Set up Airflow** and ensure it can access the project directory.
4. **Configure dbt** profiles as needed for your Snowflake connection.



## Data Structure

The Snowflake table `reddit_marketing` contains the following fields:
- `id` (Primary Key): Unique post identifier
- `title`: Post title
- `score`: Post score/upvotes
- `created_utc`: Post creation timestamp
- `url`: Post URL
- `num_comments`: Number of comments
- `author`: Post author username


