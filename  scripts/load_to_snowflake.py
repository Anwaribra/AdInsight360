import snowflake.connector
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Snowflake credentials
sf_user = os.getenv("SNOWFLAKE_USER")
sf_password = os.getenv("SNOWFLAKE_PASSWORD")
sf_account = os.getenv("SNOWFLAKE_ACCOUNT")
sf_database = os.getenv("SNOWFLAKE_DATABASE")
sf_schema = os.getenv("SNOWFLAKE_SCHEMA")
sf_warehouse = os.getenv("SNOWFLAKE_WAREHOUSE")
sf_role = os.getenv("SNOWFLAKE_ROLE")

conn = snowflake.connector.connect(
    user=sf_user,
    password=sf_password,
    account=sf_account,
    database=sf_database,
    schema=sf_schema,
    warehouse=sf_warehouse,
    role=sf_role
)

cursor = conn.cursor()

# Create table 
cursor.execute("""
CREATE TABLE IF NOT EXISTS reddit_marketing (
    title STRING,
    score INT,
    created_utc FLOAT,
    url STRING,
    num_comments INT,
    author STRING
)
""")

# Load JSON data
with open("data/raw/reddit_marketing.json", "r") as f:
    posts = json.load(f)

# Insert into Snowflake
for post in posts:
    cursor.execute("""
        INSERT INTO reddit_marketing (title, score, created_utc, url, num_comments, author)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        post["title"],
        post["score"],
        post["created_utc"],
        post["url"],
        post["num_comments"],
        post["author"]
    ))

print(f"Loaded {len(posts)} records into Snowflake.")

cursor.close()
conn.close()
