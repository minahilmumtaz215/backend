import os
import re
import praw
from dotenv import load_dotenv

load_dotenv()

# Load Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

# Initialize PRAW Reddit instance
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def extract_post_id(url: str) -> str:
    """
    Extracts the post ID from a Reddit URL.
    Example: https://www.reddit.com/r/AskReddit/comments/abc123/example_title/
    """
    pattern = r"comments/([a-z0-9]{6,})"
    match = re.search(pattern, url)
    return match.group(1) if match else ""

def fetch_reddit_comments(post_id: str):
    """
    Fetch all top-level comments and all replies recursively from a Reddit post.
    """
    submission = reddit.submission(id=post_id)
    submission.comments.replace_more(limit=None)  # Fetch all comments, no limit

    comments_data = []

    def gather_replies(comment):
        replies = []
        for reply in comment.replies:
            replies.append({
                "text": reply.body,
                "scores": reply.score,
                "publishedAt": reply.created_utc,
                "replies": gather_replies(reply)  # Recursive replies
            })
        return replies

    for top_level_comment in submission.comments:
        comments_data.append({
            "text": top_level_comment.body,
            "scores": top_level_comment.score,
            "publishedAt": top_level_comment.created_utc,
            "replies": gather_replies(top_level_comment)
        })

    return comments_data

