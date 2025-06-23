from heapq import nlargest
from datetime import datetime

def get_top_comments(comments, top_n=5):
    valid_comments = [
        c for c in comments 
        if isinstance(c, dict) and "text" in c and "likes" in c and "publishedAt" in c
    ]
    
    # Sort by timestamp descending (latest first), then by likes
    valid_comments.sort(
        key=lambda c: (
            datetime.strptime(c["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"),  # parse ISO timestamp
            c["likes"]
        ), 
        reverse=True
    )

    # Then pick top_n by likes (among latest ones)
    return nlargest(top_n, valid_comments, key=lambda c: c["likes"])
