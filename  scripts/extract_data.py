import praw
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Reddit credentials from .env
client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

# Connect to Reddit
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Subreddit to extract from
subreddit = reddit.subreddit("marketing")

# Store posts
posts = []
seen_ids = set()

# Helper to add posts safely
def add_post(post):
    if post.id not in seen_ids:
        posts.append({
            "id": post.id,
            "title": post.title,
            "score": post.score,
            "created_utc": post.created_utc,
            "url": post.url,
            "num_comments": post.num_comments,
            "author": str(post.author) if post.author else None
        })
        seen_ids.add(post.id)

# Extract from hot
print("Fetching hot posts...")
for post in subreddit.hot(limit=1000):
    add_post(post)

# Extract from new
print("Fetching new posts...")
for post in subreddit.new(limit=1000):
    add_post(post)

# Extract from top 
time_filters = ["week", "month", "year", "all"]
for tf in time_filters:
    print(f"Fetching top posts from: {tf}")
    for post in subreddit.top(limit=1000, time_filter=tf):
        add_post(post)

# Save to JSON
os.makedirs("data/raw", exist_ok=True)
with open("data/raw/reddit_marketing.json", "w") as f:
    json.dump(posts, f, indent=2)

print(f"\n Saved {len(posts)} unique posts to data/raw/reddit_marketing.json")
