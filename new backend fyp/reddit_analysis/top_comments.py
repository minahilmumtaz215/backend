from heapq import nlargest

def get_top_comments(comments, top_n=5, max_replies=5):
    # Filter comments that have required fields
    valid_comments = [c for c in comments if isinstance(c, dict) and "text" in c and "scores" in c]

    # Get top_n comments by likes
    top_comments = nlargest(top_n, valid_comments, key=lambda c: c["scores"])

    # For each top comment, limit replies to max_replies and remove nested replies inside each reply
    for comment in top_comments:
        if "replies" in comment:
            # Limit replies
            comment["replies"] = comment["replies"][:max_replies]

            # Remove nested replies inside each reply
            for reply in comment["replies"]:
                # Remove deeper nested replies if present
                reply["replies"] = []
        else:
            comment["replies"] = []

    return top_comments
