"""SQLite contracts. The original archive is always opened read-only."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

TABLES = {"post": "posts_clean", "comment": "comments_clean"}


@contextmanager
def connect(path, readonly=True):
    path = Path(path).resolve()
    if readonly and not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    conn = sqlite3.connect(path.as_uri() + ("?mode=ro" if readonly else "?mode=rwc"), uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    if readonly:
        conn.execute("PRAGMA query_only=ON")
    try:
        yield conn
    finally:
        conn.close()


def source_info(path):
    p = Path(path).resolve()
    with connect(p) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not set(TABLES.values()) <= tables:
            raise ValueError("Use reddit_clean.db with posts_clean and comments_clean tables. Run preprocess.py for a raw archive.")
        counts = {}
        bounds = {}
        for kind, table in TABLES.items():
            r = c.execute(f"SELECT count(*), min(rowid), max(rowid), min(created_utc), max(created_utc) FROM {table}").fetchone()
            counts[kind] = r[0]
            bounds[kind] = list(r[1:])
        metadata = dict(c.execute("SELECT key,value FROM preprocessing_meta")) if "preprocessing_meta" in tables else {}
    return {"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
            "counts": counts, "bounds": bounds, "preprocessing": metadata}


def record_query(kind):
    table = TABLES[kind]
    raw = "title_raw || '. ' || selftext_raw" if kind == "post" else "body_raw"
    return (f"SELECT rowid AS source_rowid, id, subreddit, category, created_utc, year_month, "
            f"score, {raw} AS raw_text, text_clean FROM {table}")


def create_analysis(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with connect(path, False) as c:
        existing = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        allowed = {"metadata", "checkpoints", "predictions", "monthly", "annotations"}
        if existing - allowed:
            raise ValueError("The output path contains a different database. Choose a new analysis database path.")
        c.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checkpoints (kind TEXT PRIMARY KEY, last_rowid INTEGER NOT NULL, processed INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS predictions (
            kind TEXT NOT NULL, source_rowid INTEGER NOT NULL, reddit_id TEXT NOT NULL,
            subreddit TEXT NOT NULL, category TEXT NOT NULL, created_utc INTEGER NOT NULL,
            year_month TEXT NOT NULL, score INTEGER, compound REAL NOT NULL,
            vader INTEGER NOT NULL, logistic INTEGER NOT NULL, naive_bayes INTEGER NOT NULL,
            svm INTEGER NOT NULL, neural INTEGER NOT NULL,
            topic INTEGER NOT NULL, topic_probability REAL NOT NULL,
            truncated INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(kind,source_rowid)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS predictions_kind_created
            ON predictions(kind,created_utc DESC,source_rowid);
        CREATE TABLE IF NOT EXISTS monthly (
            kind TEXT NOT NULL, subreddit TEXT NOT NULL, category TEXT NOT NULL,
            year_month TEXT NOT NULL, model TEXT NOT NULL, sentiment INTEGER NOT NULL,
            topic INTEGER NOT NULL, n INTEGER NOT NULL,
            compound_sum REAL NOT NULL, score_sum REAL NOT NULL, score_n INTEGER NOT NULL,
            PRIMARY KEY(kind,subreddit,category,year_month,model,sentiment,topic)
        ) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS monthly_model_month ON monthly(model,year_month);
        CREATE TABLE IF NOT EXISTS annotations (
            kind TEXT NOT NULL, source_rowid INTEGER NOT NULL, label TEXT NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY(kind,source_rowid)
        );
        """)


def explorer_indexes(c):
    c.executescript("""
        CREATE INDEX IF NOT EXISTS predictions_month ON predictions(year_month,created_utc);
        CREATE INDEX IF NOT EXISTS predictions_sub_month ON predictions(subreddit,year_month,created_utc);
        CREATE INDEX IF NOT EXISTS predictions_topic_month ON predictions(topic,year_month);
        CREATE INDEX IF NOT EXISTS predictions_created ON predictions(created_utc DESC);
    """)


def put_meta(c, key, value):
    c.execute("INSERT INTO metadata VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
              (key, json.dumps(value)))


def metadata(path):
    with connect(path) as c:
        return {r[0]: json.loads(r[1]) for r in c.execute("SELECT key,value FROM metadata")}


def checkpoints(path):
    with connect(path) as c:
        return {r["kind"]: dict(r) for r in c.execute("SELECT * FROM checkpoints")}
