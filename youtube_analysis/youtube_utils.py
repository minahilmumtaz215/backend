import os
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def extract_video_id(url: str) -> str:
    # Supports normal YouTube URLs and Shorts
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})(?:\?|&|\/|$)"
    match = re.search(pattern, url)
    return match.group(1) if match else ""

def fetch_youtube_comments(video_id: str):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None

    while True:
        request = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            text = snippet["textDisplay"]
            likes = snippet.get("likeCount", 0)
            published_at = snippet.get("publishedAt", "")

            # Collect replies (optional)
            replies = []
            for reply in item.get("replies", {}).get("comments", []):
                reply_snippet = reply["snippet"]
                replies.append({
                    "text": reply_snippet["textDisplay"],
                    "likes": reply_snippet.get("likeCount", 0),
                    "publishedAt": reply_snippet.get("publishedAt", "")
                })

            comments.append({
                "text": text,
                "likes": likes,
                "publishedAt": published_at,
                "replies": replies
            })

        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return comments
