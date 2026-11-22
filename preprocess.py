"""
Reddit Sentiment Analysis Engine — Data Preprocessing Pipeline
===============================================================
Phase 1: Data Cleaning
Phase 2: Text Preprocessing (NLP)
Phase 3: Feature Engineering

Reads from reddit_8million.db → writes to reddit_clean.db
"""

import html
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime, timezone

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SOURCE_DB = "reddit_8million.db"
CLEAN_DB  = "reddit_clean.db"

BATCH_SIZE = 50_000          # rows per processing batch

# Subreddit → category mapping (from 3monthdb.py)
CATEGORY_MAP = {
    # TECHNOLOGY
    "technology": "technology", "programming": "technology",
    "software": "technology",   "hardware": "technology",
    "linux": "technology",      "windows": "technology",
    "apple": "technology",      "Android": "technology",
    "gadgets": "technology",    "MachineLearning": "technology",
    # SCIENCE
    "datascience": "science",   "artificial": "science",
    "ChatGPT": "science",      "science": "science",
    "askscience": "science",   "Physics": "science",
    "chemistry": "science",    "biology": "science",
    # NEWS / GENERAL
    "worldnews": "news_general", "news": "news_general",
    "geopolitics": "news_general", "business": "news_general",
    "Economics": "news_general",   "AskReddit": "news_general",
    "NoStupidQuestions": "news_general",
}

# Case-insensitive lookup helper
_CATEGORY_LOWER = {k.lower(): v for k, v in CATEGORY_MAP.items()}

def get_category(subreddit: str) -> str:
    return _CATEGORY_LOWER.get(subreddit.lower(), "other")


# Reddit-specific stopwords to add on top of NLTK's defaults
REDDIT_STOPWORDS = {
    "edit", "update", "tldr", "tl", "dr", "imo", "imho",
    "iirc", "afaik", "eli5", "ama", "iama", "op", "til",
    "dae", "cmv", "ftfy", "ootl", "fwiw",
}

# Negation words to KEEP (critical for sentiment)
KEEP_WORDS = {
    "not", "no", "nor", "never", "neither", "nobody", "nothing",
    "nowhere", "hardly", "barely", "scarcely", "don", "doesn",
    "didn", "won", "wouldn", "couldn", "shouldn", "isn", "aren",
    "wasn", "weren", "hasn", "haven", "hadn", "mustn", "needn",
    "t",   # for contractions like "don't" → "don" + "t"
}


# ---------------------------------------------------------------------------
# Phase 2 — Text Cleaning Functions
# ---------------------------------------------------------------------------

# Compiled regex patterns for speed
_RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_USER = re.compile(r"/?(u|user)/[A-Za-z0-9_-]+")
_RE_SUB = re.compile(r"/?(r)/[A-Za-z0-9_-]+")
_RE_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RE_MARKDOWN_BOLD = re.compile(r"\*{1,3}(.*?)\*{1,3}")
_RE_MARKDOWN_STRIKE = re.compile(r"~~(.*?)~~")
_RE_CODE_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_RE_INLINE_CODE = re.compile(r"`([^`]*)`")
_RE_QUOTE = re.compile(r"^>+\s?", re.MULTILINE)
_RE_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_RE_WHITESPACE = re.compile(r"\s+")
_RE_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    """Apply all Phase 2 cleaning steps to raw text."""
    if not text:
        return ""

    # 1. Decode HTML entities
    text = html.unescape(text)

    # 2. Strip markdown
    text = _RE_CODE_BLOCK.sub(" ", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_MARKDOWN_LINK.sub(r"\1", text)
    text = _RE_MARKDOWN_BOLD.sub(r"\1", text)
    text = _RE_MARKDOWN_STRIKE.sub(r"\1", text)
    text = _RE_QUOTE.sub("", text)
    text = _RE_HEADING.sub("", text)

    # 3. Remove URLs, user/subreddit references
    text = _RE_URL.sub("", text)
    text = _RE_USER.sub("", text)
    text = _RE_SUB.sub("", text)

    # 4. Remove emojis
    text = _RE_EMOJI.sub("", text)

    # 5. Unicode normalization
    text = unicodedata.normalize("NFKD", text)

    # 6. Lowercase
    text = text.lower()

    # 7. Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


# ---------------------------------------------------------------------------
# Phase 1 + 2 + 3 — Full Pipeline
# ---------------------------------------------------------------------------

def create_clean_schema(con: sqlite3.Connection) -> None:
    """Create the clean database schema with all new columns."""
    con.executescript("""
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;

        CREATE TABLE IF NOT EXISTS posts_clean (
            id                TEXT PRIMARY KEY,
            subreddit         TEXT NOT NULL,
            category          TEXT NOT NULL,
            created_utc       INTEGER NOT NULL,
            date              TEXT NOT NULL,
            year              INTEGER NOT NULL,
            month             INTEGER NOT NULL,
            year_month        TEXT NOT NULL,
            day_of_week       INTEGER NOT NULL,
            hour_of_day       INTEGER NOT NULL,
            is_weekend        INTEGER NOT NULL,
            title_raw         TEXT NOT NULL,
            selftext_raw      TEXT NOT NULL,
            post_type         TEXT NOT NULL,
            has_url           INTEGER NOT NULL,
            text_clean        TEXT NOT NULL,
            text_length       INTEGER NOT NULL,
            word_count        INTEGER NOT NULL,
            score             INTEGER,
            score_bucket      TEXT,
            num_comments      INTEGER,
            is_nsfw           INTEGER,
            is_spoiler        INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_pc_subreddit  ON posts_clean(subreddit);
        CREATE INDEX IF NOT EXISTS idx_pc_category   ON posts_clean(category);
        CREATE INDEX IF NOT EXISTS idx_pc_year_month ON posts_clean(year_month);
        CREATE INDEX IF NOT EXISTS idx_pc_created    ON posts_clean(created_utc);

        CREATE TABLE IF NOT EXISTS comments_clean (
            id                TEXT PRIMARY KEY,
            subreddit         TEXT NOT NULL,
            category          TEXT NOT NULL,
            created_utc       INTEGER NOT NULL,
            date              TEXT NOT NULL,
            year              INTEGER NOT NULL,
            month             INTEGER NOT NULL,
            year_month        TEXT NOT NULL,
            day_of_week       INTEGER NOT NULL,
            hour_of_day       INTEGER NOT NULL,
            is_weekend        INTEGER NOT NULL,
            post_id           TEXT,
            parent_id         TEXT,
            body_raw          TEXT NOT NULL,
            text_clean        TEXT NOT NULL,
            text_length       INTEGER NOT NULL,
            word_count        INTEGER NOT NULL,
            score             INTEGER,
            score_bucket      TEXT,
            is_distinguished  INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_cc_subreddit  ON comments_clean(subreddit);
        CREATE INDEX IF NOT EXISTS idx_cc_category   ON comments_clean(category);
        CREATE INDEX IF NOT EXISTS idx_cc_year_month ON comments_clean(year_month);
        CREATE INDEX IF NOT EXISTS idx_cc_created    ON comments_clean(created_utc);
        CREATE INDEX IF NOT EXISTS idx_cc_post_id    ON comments_clean(post_id);

        CREATE TABLE IF NOT EXISTS preprocessing_meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
    """)
    con.commit()


def score_bucket(score) -> str | None:
    """Categorize a numeric score into engagement buckets."""
    if score is None:
        return None
    if score < 0:
        return "negative"
    if score == 0:
        return "zero"
    if score <= 10:
        return "low"
    if score <= 100:
        return "medium"
    if score <= 1000:
        return "high"
    return "viral"


def temporal_features(utc_ts: int) -> dict:
    """Derive all temporal features from a Unix timestamp."""
    dt = datetime.fromtimestamp(utc_ts, tz=timezone.utc)
    return {
        "date":        dt.strftime("%Y-%m-%d"),
        "year":        dt.year,
        "month":       dt.month,
        "year_month":  dt.strftime("%Y-%m"),
        "day_of_week": dt.weekday(),      # 0=Mon, 6=Sun
        "hour_of_day": dt.hour,
        "is_weekend":  1 if dt.weekday() >= 5 else 0,
    }


def strip_reddit_prefix(rid: str | None) -> str | None:
    """Remove t1_/t3_ prefixes from Reddit IDs."""
    if not rid:
        return None
    if rid.startswith(("t1_", "t3_")):
        return rid[3:]
    return rid


# ---------------------------------------------------------------------------
# Processing Functions
# ---------------------------------------------------------------------------

def process_posts(src: sqlite3.Connection, dst: sqlite3.Connection) -> dict:
    """Process all posts: clean → feature-engineer → write to clean DB."""
    cursor = src.execute("""
        SELECT id, subreddit, created_utc, title, selftext, url,
               score, num_comments, over_18, spoiler
        FROM posts
        WHERE selftext NOT IN ('[deleted]', '[removed]')
          AND title != ''
    """)

    stats = {"total_read": 0, "written": 0, "skipped_short": 0}
    batch = []

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        for row in rows:
            stats["total_read"] += 1
            rid, sub, created, title, selftext, url, sc, ncom, nsfw, spoil = row

            # Determine post type
            post_type = "link" if (not selftext or selftext == "") else "self"

            # Build unified text
            if post_type == "link":
                raw_text = title
            else:
                raw_text = f"{title}. {selftext}"

            # Clean text
            cleaned = clean_text(raw_text)
            words = cleaned.split()
            wc = len(words)

            # Filter out very short texts
            if wc < 3:
                stats["skipped_short"] += 1
                continue

            # Temporal features
            tf = temporal_features(created)

            # Has URL?
            has_url = 1 if (url and not url.startswith("https://www.reddit.com/r/")) else 0

            batch.append((
                rid, sub, get_category(sub), created,
                tf["date"], tf["year"], tf["month"], tf["year_month"],
                tf["day_of_week"], tf["hour_of_day"], tf["is_weekend"],
                title, selftext or "", post_type, has_url,
                cleaned, len(cleaned), wc,
                sc, score_bucket(sc), ncom, nsfw, spoil
            ))

            if len(batch) >= BATCH_SIZE:
                _flush_posts(dst, batch)
                stats["written"] += len(batch)
                batch.clear()
                _print_progress("Posts", stats["total_read"], stats["written"])

    if batch:
        _flush_posts(dst, batch)
        stats["written"] += len(batch)

    _print_progress("Posts", stats["total_read"], stats["written"], final=True)
    return stats


def process_comments(src: sqlite3.Connection, dst: sqlite3.Connection) -> dict:
    """Process all comments: clean → feature-engineer → write to clean DB."""
    cursor = src.execute("""
        SELECT id, subreddit, created_utc, body, link_id, parent_id,
               score, distinguished
        FROM comments
        WHERE body NOT IN ('[deleted]', '[removed]', '')
    """)

    stats = {"total_read": 0, "written": 0, "skipped_short": 0}
    batch = []

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        for row in rows:
            stats["total_read"] += 1
            rid, sub, created, body, link_id, parent_id, sc, dist = row

            # Clean text
            cleaned = clean_text(body)
            words = cleaned.split()
            wc = len(words)

            if wc < 3:
                stats["skipped_short"] += 1
                continue

            tf = temporal_features(created)

            batch.append((
                rid, sub, get_category(sub), created,
                tf["date"], tf["year"], tf["month"], tf["year_month"],
                tf["day_of_week"], tf["hour_of_day"], tf["is_weekend"],
                strip_reddit_prefix(link_id),
                strip_reddit_prefix(parent_id),
                body, cleaned, len(cleaned), wc,
                sc, score_bucket(sc),
                1 if dist else 0
            ))

            if len(batch) >= BATCH_SIZE:
                _flush_comments(dst, batch)
                stats["written"] += len(batch)
                batch.clear()
                _print_progress("Comments", stats["total_read"], stats["written"])

    if batch:
        _flush_comments(dst, batch)
        stats["written"] += len(batch)

    _print_progress("Comments", stats["total_read"], stats["written"], final=True)
    return stats


def _flush_posts(con: sqlite3.Connection, batch: list) -> None:
    con.executemany("""
        INSERT OR IGNORE INTO posts_clean VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, batch)
    con.commit()


def _flush_comments(con: sqlite3.Connection, batch: list) -> None:
    con.executemany("""
        INSERT OR IGNORE INTO comments_clean VALUES (
            ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
        )
    """, batch)
    con.commit()


def _print_progress(label: str, read: int, written: int, final: bool = False) -> None:
    end = "\n" if final else "\r"
    marker = "DONE" if final else " ..."
    print(f"  {marker} {label}: read {read:>10,} | written {written:>10,}", end=end, flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 60)
    print("  REDDIT PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"  Source:  {SOURCE_DB}")
    print(f"  Output:  {CLEAN_DB}")
    print()

    t0 = time.time()

    # Open source (read-only)
    src = sqlite3.connect(f"file:{SOURCE_DB}?mode=ro", uri=True)

    # Open/create destination
    dst = sqlite3.connect(CLEAN_DB)
    create_clean_schema(dst)

    # ---------- Posts ----------
    print("[Phase 1-3] Processing posts …")
    post_stats = process_posts(src, dst)

    # ---------- Comments ----------
    print("[Phase 1-3] Processing comments …")
    comment_stats = process_comments(src, dst)

    # ---------- Metadata ----------
    elapsed = time.time() - t0
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('processed_at', ?)",
                (datetime.now(timezone.utc).isoformat(),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('source_db', ?)", (SOURCE_DB,))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('posts_read', ?)", (str(post_stats["total_read"]),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('posts_written', ?)", (str(post_stats["written"]),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('posts_skipped_short', ?)", (str(post_stats["skipped_short"]),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('comments_read', ?)", (str(comment_stats["total_read"]),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('comments_written', ?)", (str(comment_stats["written"]),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('comments_skipped_short', ?)", (str(comment_stats["skipped_short"]),))
    dst.execute("INSERT OR REPLACE INTO preprocessing_meta VALUES ('elapsed_seconds', ?)", (f"{elapsed:.1f}",))
    dst.commit()

    src.close()
    dst.close()

    # ---------- Summary ----------
    print()
    print("=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Posts:    {post_stats['total_read']:>10,} read -> {post_stats['written']:>10,} written  ({post_stats['skipped_short']:,} too short)")
    print(f"  Comments: {comment_stats['total_read']:>10,} read -> {comment_stats['written']:>10,} written  ({comment_stats['skipped_short']:,} too short)")
    total_w = post_stats["written"] + comment_stats["written"]
    print(f"  Total clean records: {total_w:,}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output:  {CLEAN_DB}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
