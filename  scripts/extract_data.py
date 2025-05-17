import praw
import json
import os
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

subreddit = reddit.subreddit("marketing")
posts = []

print("Fetching posts from r/marketing...")
for post in subreddit.hot(limit=100):
    posts.append({
        "title": post.title,
        "score": post.score,
        "created_utc": post.created_utc,
        "url": post.url,
        "num_comments": post.num_comments,
        "author": str(post.author)
    })

os.makedirs("data/raw", exist_ok=True)
with open("data/raw/reddit_marketing.json", "w") as f:
    json.dump(posts, f, indent=2)

print(f"{len(posts)} posts saved to data/raw/reddit_marketing.json")
