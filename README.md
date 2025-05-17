# AdInsight360

A data pipeline for extracting and analyzing marketing insights from Reddit data.

## Overview

This project automates the collection of marketing-related content from Reddit and loads it into Snowflake for analysis. The pipeline extracts posts from various Reddit categories (hot, new, and top posts) and maintains a structured database of marketing insights.

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


## Data Structure

The Snowflake table `reddit_marketing` contains the following fields:
- `id` (Primary Key): Unique post identifier
- `title`: Post title
- `score`: Post score/upvotes
- `created_utc`: Post creation timestamp
- `url`: Post URL
- `num_comments`: Number of comments
- `author`: Post author username


