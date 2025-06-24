import torch
import nltk
import re
import multiprocessing
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
from nltk.corpus import stopwords
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# Direct model name
MODEL_DIR = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

# Use XLMRobertaTokenizer to avoid fast tokenizer issues
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Label mapping
reverse_label_map = {0: "negative", 1: "neutral", 2: "positive"}

# Download stopwords
nltk.download("stopwords")
ENGLISH_STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = set([
    "hai", "ka", "ki", "kya", "na", "toh", "bhi", "me", "ho", "raha", "rha",
    "not", "what", "who", "her", "his", "him", "you", "karo", "har", "aur", "u", "i", "im", "its",
    "ye", "ke", "se", "ko", "tha", "thi", "they", "them", "will", "shall", "been", "has", "have",
    "had", "is", "are", "was", "were", "do", "does", "did", "just", "would", "should", "could"
])
ALL_STOPWORDS = ENGLISH_STOPWORDS.union(CUSTOM_STOPWORDS)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    return [w for w in tokens if w not in ALL_STOPWORDS and len(w) > 2]

def batch_comments(texts, tokenizer, max_comments=64, max_tokens=512):
    batches, current_batch, current_tokens = [], [], 0
    token_lens = [len(tokenizer(text, truncation=True, max_length=512)["input_ids"]) for text in texts]

    for text, token_count in zip(texts, token_lens):
        if token_count > max_tokens:
            continue
        if (current_tokens + token_count > max_tokens) or (len(current_batch) >= max_comments):
            if current_batch:
                batches.append(current_batch)
            current_batch = [text]
            current_tokens = token_count
        else:
            current_batch.append(text)
            current_tokens += token_count

    if current_batch:
        batches.append(current_batch)
    return batches

def process_batch(batch):
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1).tolist()

    return [reverse_label_map[pred] for pred in predictions]

def analyze_sentiments(comments, max_comments_per_batch=64, max_tokens=480):
    batches = batch_comments(comments, tokenizer, max_comments_per_batch, max_tokens)
    all_sentiments = []

    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            all_sentiments.extend(future.result())

    return all_sentiments

def categorize_words_by_sentiment(texts, sentiments):
    word_sentiment_map = {"positive": Counter(), "neutral": Counter(), "negative": Counter()}

    for text, sentiment in zip(texts, sentiments):
        tokens = clean_text(text)
        if sentiment in word_sentiment_map:
            word_sentiment_map[sentiment].update(tokens)

    word_totals = Counter()
    word_to_sentiment_counts = {}

    for sentiment, counter in word_sentiment_map.items():
        for word, count in counter.items():
            word_totals[word] += count
            if word not in word_to_sentiment_counts:
                word_to_sentiment_counts[word] = {"positive": 0, "neutral": 0, "negative": 0}
            word_to_sentiment_counts[word][sentiment] += count

    final = {"positive_words": [], "neutral_words": [], "negative_words": []}
    for word, sentiment_counts in word_to_sentiment_counts.items():
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        if sentiment_counts[dominant_sentiment] >= 2 and sentiment_counts[dominant_sentiment] >= 0.6 * word_totals[word]:
            final[f"{dominant_sentiment}_words"].append({
                "word": word,
                "count": sentiment_counts[dominant_sentiment]
            })

    for key in final:
        final[key] = sorted(final[key], key=lambda x: x["count"], reverse=True)[:10]
        if not final[key]:
            final[key] = [{"word": key.split("_")[0].capitalize() + " words are not available", "count": 0}]
    
    return final
