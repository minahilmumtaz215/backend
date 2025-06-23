from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
from dotenv import load_dotenv
from datetime import datetime
import os

# YouTube imports
from youtube_analysis.youtube_utils import fetch_youtube_comments, extract_video_id
from youtube_analysis.top_comments import get_top_comments as get_top_youtube_comments

# Reddit imports
from reddit_analysis.reddit_utils import extract_post_id, fetch_reddit_comments
from reddit_analysis.top_comments import get_top_comments as get_top_reddit_comments

# Shared
from shared.sentiment_analysis import analyze_sentiments, categorize_words_by_sentiment

load_dotenv()

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================
# MODELS
# =======================

class ReplyItem(BaseModel):
    text: str
    likes: int | None = None
    scores: int | None = None
    publishedAt: str

class CommentItem(BaseModel):
    text: str
    likes: int | None = None
    scores: int | None = None
    publishedAt: str
    replies: List[ReplyItem] = []

class WordFrequency(BaseModel):
    word: str
    count: int

class AnalysisResult(BaseModel):
    sentiment_distribution: Dict[str, int]
    frequent_words: Dict[str, List[WordFrequency]]
    top_comments: List[CommentItem]

# =======================
# HEALTH CHECK
# =======================

@app.get("/")
def health_check():
    return {"status": "Unified Comment Analyzer is running."}

# =======================
# YOUTUBE ENDPOINT (UPDATED)
# =======================

@app.get("/analyze", response_model=AnalysisResult)
def analyze_youtube(video_url: str = Query(..., description="Full YouTube Video URL")):
    video_id = extract_video_id(video_url)
    if not video_id:
        return empty_response("Invalid video URL")

    comments_data = fetch_youtube_comments(video_id)
    all_texts = collect_all_texts(comments_data)

    if not all_texts:
        return empty_response("No comments found.")

    sentiments = analyze_sentiments(all_texts)
    sentiment_counts = count_sentiments(sentiments)
    frequent_words = categorize_words_by_sentiment(all_texts, sentiments)

    top_comments_data = get_top_youtube_comments(comments_data)

    def convert_timestamp(ts):
        if isinstance(ts, (int, float)):
            return datetime.utcfromtimestamp(ts).isoformat()
        return ts if isinstance(ts, str) else datetime.utcnow().isoformat()

    cleaned_top_comments = []
    for c in top_comments_data:
        cleaned_replies = [
            {
                "text": r["text"],
                "likes": r.get("likes"),
                "publishedAt": convert_timestamp(r.get("publishedAt", 0))
            } for r in c.get("replies", [])
        ]
        cleaned_top_comments.append({
            "text": c["text"],
            "likes": c.get("likes"),
            "publishedAt": convert_timestamp(c.get("publishedAt", 0)),
            "replies": cleaned_replies
        })

    return {
        "sentiment_distribution": sentiment_counts,
        "frequent_words": frequent_words,
        "top_comments": cleaned_top_comments
    }

# =======================
# REDDIT ENDPOINT
# =======================

@app.get("/analyze_reddit_post", response_model=AnalysisResult)
def analyze_reddit_post(url: str = Query(..., description="Reddit post URL")):
    try:
        for k in ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]:
            if not os.getenv(k):
                raise HTTPException(status_code=500, detail=f"Missing {k} in .env")

        post_id = extract_post_id(url)
        if not post_id:
            raise HTTPException(status_code=400, detail="Invalid Reddit URL or post ID not found.")

        comments = fetch_reddit_comments(post_id)
        if not comments:
            return empty_response("No comments found.")

        all_texts = collect_all_texts(comments)
        sentiments = analyze_sentiments(all_texts)
        sentiment_counts = count_sentiments(sentiments)
        word_frequencies = categorize_words_by_sentiment(all_texts, sentiments)

        top = get_top_reddit_comments(comments)
        
        def convert_timestamp(ts):
            if isinstance(ts, (int, float)):
                return datetime.utcfromtimestamp(ts).isoformat()
            return ts if isinstance(ts, str) else datetime.utcnow().isoformat()

        top_comments_cleaned = []
        for c in top:
            cleaned_replies = [
                {
                    "text": r["text"],
                    "scores": r.get("scores"),
                    "publishedAt": convert_timestamp(r.get("publishedAt", 0))
                } for r in c.get("replies", [])
            ]
            top_comments_cleaned.append({
                "text": c["text"],
                "scores": c.get("scores"),
                "publishedAt": convert_timestamp(c.get("publishedAt", 0)),
                "replies": cleaned_replies
            })

        return {
            "sentiment_distribution": sentiment_counts,
            "frequent_words": word_frequencies,
            "top_comments": top_comments_cleaned
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =======================
# HELPERS
# =======================

def empty_response(msg):
    return {
        "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
        "frequent_words": {
            "positive_words": [],
            "neutral_words": [],
            "negative_words": []
        },
        "top_comments": [
            {"text": msg, "likes": 0, "scores": 0, "publishedAt": datetime.utcnow().isoformat(), "replies": []}
        ]
    }

def count_sentiments(sentiments):
    return {
        "positive": sentiments.count("positive"),
        "neutral": sentiments.count("neutral"),
        "negative": sentiments.count("negative"),
    }

def collect_all_texts(comments_list):
    texts = []
    for comment in comments_list:
        texts.append(comment["text"])
        if comment.get("replies"):
            texts.extend(collect_all_texts(comment["replies"]))
    return texts
