import snowflake.connector
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Snowflake configuration
sf_config = {
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_PASSWORD"),
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": os.getenv("SNOWFLAKE_SCHEMA"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "role": os.getenv("SNOWFLAKE_ROLE")
}

def load_to_snowflake():
    """Minimal data loader that respects dbt workflow"""
    try:
        # Connect to Snowflake
        conn = snowflake.connector.connect(**sf_config)
        cursor = conn.cursor()
        
        # Create raw table (if not exists)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_reddit_posts (
            id VARCHAR(255),
            title TEXT,
            score INTEGER,
            created_utc TIMESTAMP_NTZ,
            url TEXT,
            num_comments INTEGER,
            author VARCHAR(255),
            loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
        )
        """)
        
        # Load JSON data
        with open("data/raw/reddit_marketing.json", "r") as f:
            posts = json.load(f)
        
        # Simple batch insert
        cursor.executemany("""
            INSERT INTO raw_reddit_posts 
            (id, title, score, created_utc, url, num_comments, author)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, [
            (
                str(p["id"]),
                str(p["title"]),
                int(p["score"]),
                datetime.utcfromtimestamp(p["created_utc"]),
                str(p["url"]),
                int(p["num_comments"]),
                str(p["author"]) if p["author"] else None
            )
            for p in posts
        ])
        
        conn.commit()
        print(f"Loaded {len(posts)} records to raw_reddit_posts")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    load_to_snowflake()