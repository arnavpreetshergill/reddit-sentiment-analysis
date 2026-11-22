"""Different text preparation for sentiment and topics, preserving negation."""
import html
import re
from functools import lru_cache

URL = re.compile(r"https?://\S+|www\.\S+", re.I)
USER = re.compile(r"\b/?u/[A-Za-z0-9_-]+")
SPACE = re.compile(r"\s+")
MAX_CHARS = 8000


def sentiment_text(text, max_chars=MAX_CHARS):
    # Capitals, punctuation, emoji, emoticons and negation carry sentiment.
    text = html.unescape(str(text or ""))
    text = URL.sub(" ", text)
    text = USER.sub(" ", text)
    return SPACE.sub(" ", text).strip()[:max_chars]


def fingerprint(text):
    import hashlib
    return hashlib.sha256(SPACE.sub(" ", text).lower().strip().encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def vader_label(score):
    return 2 if score >= 0.05 else 0 if score <= -0.05 else 1


def score_text(text):
    score = vader().polarity_scores(text)["compound"]
    return vader_label(score), score
