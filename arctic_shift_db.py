"""Build a local Reddit research database from Arctic Shift archive data.

This program intentionally does not call Reddit's API.  It retrieves historical
post and comment records from Arctic Shift's public archive endpoints and writes
only the fields needed for the sentiment-analysis project to SQLite.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_BASE_URL = "https://arctic-shift.photon-reddit.com"
SOURCE_NAME = "arctic_shift"
USER_AGENT = "reddit-sentiment-research-db/1.0 (academic collector)"

POST_FIELDS = (
    "id,subreddit,created_utc,retrieved_on,title,selftext,url,score,"
    "num_comments,over_18,spoiler"
)
COMMENT_FIELDS = (
    "id,subreddit,created_utc,retrieved_on,body,link_id,parent_id,score,"
    "distinguished"
)


class ArcticShiftError(RuntimeError):
    """Raised when Arctic Shift cannot provide a usable response."""


@dataclass(frozen=True)
class CollectionResult:
    pages: int
    records: int
    completed: bool


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS subreddits (
    name TEXT PRIMARY KEY,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    reddit_id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL REFERENCES subreddits(name),
    title TEXT NOT NULL DEFAULT '',
    selftext TEXT NOT NULL DEFAULT '',
    url TEXT,
    created_utc INTEGER NOT NULL,
    retrieved_on INTEGER,
    score INTEGER,
    num_comments INTEGER,
    is_nsfw INTEGER,
    is_spoiler INTEGER,
    first_ingested_at TEXT NOT NULL,
    last_ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_posts_subreddit_created
    ON posts(subreddit, created_utc);

CREATE TABLE IF NOT EXISTS comments (
    reddit_id TEXT PRIMARY KEY,
    subreddit TEXT NOT NULL REFERENCES subreddits(name),
    post_reddit_id TEXT,
    parent_reddit_id TEXT,
    parent_kind TEXT CHECK(parent_kind IN ('post', 'comment')),
    body TEXT NOT NULL DEFAULT '',
    created_utc INTEGER NOT NULL,
    retrieved_on INTEGER,
    score INTEGER,
    distinguished TEXT,
    first_ingested_at TEXT NOT NULL,
    last_ingested_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_comments_subreddit_created
    ON comments(subreddit, created_utc);
CREATE INDEX IF NOT EXISTS idx_comments_post
    ON comments(post_reddit_id);

CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed')),
    requested_after TEXT,
    requested_before TEXT,
    details TEXT
);

CREATE TABLE IF NOT EXISTS sync_checkpoints (
    source TEXT NOT NULL,
    resource TEXT NOT NULL CHECK(resource IN ('posts', 'comments')),
    subreddit TEXT NOT NULL,
    requested_after TEXT NOT NULL,
    requested_before TEXT NOT NULL,
    last_created_utc INTEGER,
    completed_at TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (source, resource, subreddit, requested_after, requested_before)
);

CREATE TABLE IF NOT EXISTS ingestion_errors (
    id INTEGER PRIMARY KEY,
    collection_run_id INTEGER NOT NULL REFERENCES collection_runs(id),
    resource TEXT NOT NULL,
    subreddit TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    message TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(bool(value))
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return 1
    if lowered in {"false", "0", "no", "n"}:
        return 0
    return None


def normalize_reddit_id(value: Any) -> str | None:
    """Store Reddit's base-36 ID without its optional t1_/t3_ prefix."""
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith(("t1_", "t3_")):
        return text[3:]
    return text or None


def parent_kind(value: Any) -> str | None:
    if not value:
        return None
    return "post" if str(value).startswith("t3_") else "comment"


def validate_subreddit(value: str) -> str:
    subreddit = value.strip().removeprefix("r/")
    if not subreddit or not subreddit.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError(
            "subreddit must contain only letters, numbers, underscores, or hyphens"
        )
    return subreddit


class ArcticShiftClient:
    def __init__(
        self,
        base_url: str = API_BASE_URL,
        timeout_seconds: float = 45.0,
        min_interval_seconds: float = 1.0,
        retries: int = 4,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.min_interval_seconds = min_interval_seconds
        self.retries = retries
        self._last_request_at = 0.0

    def search(
        self, resource: str, subreddit: str, after: str | int, before: str, limit: int
    ) -> list[dict[str, Any]]:
        if resource not in {"posts", "comments"}:
            raise ValueError("resource must be 'posts' or 'comments'")
        fields = POST_FIELDS if resource == "posts" else COMMENT_FIELDS
        params = {
            "subreddit": subreddit,
            "after": str(after),
            "before": before,
            "sort": "asc",
            "limit": str(limit),
            "fields": fields,
        }
        payload = self._get_json(f"/api/{resource}/search", params)
        records = payload.get("data")
        if not isinstance(records, list):
            raise ArcticShiftError("Arctic Shift response did not contain a 'data' list")
        if not all(isinstance(record, dict) for record in records):
            raise ArcticShiftError("Arctic Shift returned an invalid record")
        return records

    def _get_json(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        for attempt in range(self.retries + 1):
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < self.min_interval_seconds:
                time.sleep(self.min_interval_seconds - elapsed)

            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    self._last_request_at = time.monotonic()
                    body = response.read().decode("utf-8")
                    return json.loads(body)
            except HTTPError as error:
                self._last_request_at = time.monotonic()
                if error.code not in {429, 500, 502, 503, 504} or attempt == self.retries:
                    raise ArcticShiftError(f"Arctic Shift returned HTTP {error.code}") from error
                retry_after = as_int(error.headers.get("Retry-After")) if error.headers else None
                time.sleep(retry_after if retry_after is not None else 2**attempt)
            except (URLError, TimeoutError, json.JSONDecodeError) as error:
                self._last_request_at = time.monotonic()
                if attempt == self.retries:
                    raise ArcticShiftError(f"Arctic Shift request failed: {error}") from error
                time.sleep(2**attempt)

        raise AssertionError("retry loop should either return or raise")


class ResearchDatabase:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA_SQL)
        self.connection.commit()

    def start_run(self, after: str, before: str, subreddits: list[str]) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO collection_runs(source, started_at, status, requested_after,
                                        requested_before, details)
            VALUES (?, ?, 'running', ?, ?, ?)
            """,
            (SOURCE_NAME, utc_now(), after, before, json.dumps({"subreddits": subreddits})),
        )
        self.connection.commit()
        return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, details: dict[str, Any]) -> None:
        self.connection.execute(
            "UPDATE collection_runs SET finished_at = ?, status = ?, details = ? WHERE id = ?",
            (utc_now(), status, json.dumps(details), run_id),
        )
        self.connection.commit()

    def log_error(self, run_id: int, resource: str, subreddit: str, message: str) -> None:
        self.connection.execute(
            """
            INSERT INTO ingestion_errors(collection_run_id, resource, subreddit, occurred_at, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, resource, subreddit, utc_now(), message),
        )
        self.connection.commit()

    def checkpoint(
        self, resource: str, subreddit: str, requested_after: str, requested_before: str
    ) -> sqlite3.Row | None:
        return self.connection.execute(
            """
            SELECT last_created_utc, completed_at FROM sync_checkpoints
            WHERE source = ? AND resource = ? AND subreddit = ?
              AND requested_after = ? AND requested_before = ?
            """,
            (SOURCE_NAME, resource, subreddit, requested_after, requested_before),
        ).fetchone()

    def save_checkpoint(
        self,
        resource: str,
        subreddit: str,
        requested_after: str,
        requested_before: str,
        last_created_utc: int | None,
        completed: bool,
    ) -> None:
        completed_at = utc_now() if completed else None
        self.connection.execute(
            """
            INSERT INTO sync_checkpoints(
                source, resource, subreddit, requested_after, requested_before,
                last_created_utc, completed_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source, resource, subreddit, requested_after, requested_before)
            DO UPDATE SET
                last_created_utc = excluded.last_created_utc,
                completed_at = excluded.completed_at,
                updated_at = excluded.updated_at
            """,
            (
                SOURCE_NAME,
                resource,
                subreddit,
                requested_after,
                requested_before,
                last_created_utc,
                completed_at,
                utc_now(),
            ),
        )
        self.connection.commit()

    def upsert_posts(self, records: Iterable[dict[str, Any]]) -> int:
        now = utc_now()
        saved = 0
        with self.connection:
            for record in records:
                reddit_id = normalize_reddit_id(record.get("id"))
                subreddit = str(record.get("subreddit") or "").strip()
                created_utc = as_int(record.get("created_utc"))
                if not reddit_id or not subreddit or created_utc is None:
                    continue
                self._touch_subreddit(subreddit, now)
                self.connection.execute(
                    """
                    INSERT INTO posts(
                        reddit_id, subreddit, title, selftext, url, created_utc, retrieved_on,
                        score, num_comments, is_nsfw, is_spoiler, first_ingested_at, last_ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(reddit_id) DO UPDATE SET
                        subreddit = excluded.subreddit,
                        title = excluded.title,
                        selftext = excluded.selftext,
                        url = excluded.url,
                        created_utc = excluded.created_utc,
                        retrieved_on = excluded.retrieved_on,
                        score = excluded.score,
                        num_comments = excluded.num_comments,
                        is_nsfw = excluded.is_nsfw,
                        is_spoiler = excluded.is_spoiler,
                        last_ingested_at = excluded.last_ingested_at
                    """,
                    (
                        reddit_id,
                        subreddit,
                        str(record.get("title") or ""),
                        str(record.get("selftext") or ""),
                        record.get("url"),
                        created_utc,
                        as_int(record.get("retrieved_on")),
                        as_int(record.get("score")),
                        as_int(record.get("num_comments")),
                        as_bool_int(record.get("over_18")),
                        as_bool_int(record.get("spoiler")),
                        now,
                        now,
                    ),
                )
                saved += 1
        return saved

    def upsert_comments(self, records: Iterable[dict[str, Any]]) -> int:
        now = utc_now()
        saved = 0
        with self.connection:
            for record in records:
                reddit_id = normalize_reddit_id(record.get("id"))
                subreddit = str(record.get("subreddit") or "").strip()
                created_utc = as_int(record.get("created_utc"))
                if not reddit_id or not subreddit or created_utc is None:
                    continue
                self._touch_subreddit(subreddit, now)
                self.connection.execute(
                    """
                    INSERT INTO comments(
                        reddit_id, subreddit, post_reddit_id, parent_reddit_id, parent_kind,
                        body, created_utc, retrieved_on, score, distinguished,
                        first_ingested_at, last_ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(reddit_id) DO UPDATE SET
                        subreddit = excluded.subreddit,
                        post_reddit_id = excluded.post_reddit_id,
                        parent_reddit_id = excluded.parent_reddit_id,
                        parent_kind = excluded.parent_kind,
                        body = excluded.body,
                        created_utc = excluded.created_utc,
                        retrieved_on = excluded.retrieved_on,
                        score = excluded.score,
                        distinguished = excluded.distinguished,
                        last_ingested_at = excluded.last_ingested_at
                    """,
                    (
                        reddit_id,
                        subreddit,
                        normalize_reddit_id(record.get("link_id")),
                        normalize_reddit_id(record.get("parent_id")),
                        parent_kind(record.get("parent_id")),
                        str(record.get("body") or ""),
                        created_utc,
                        as_int(record.get("retrieved_on")),
                        as_int(record.get("score")),
                        record.get("distinguished"),
                        now,
                        now,
                    ),
                )
                saved += 1
        return saved

    def _touch_subreddit(self, subreddit: str, timestamp: str) -> None:
        self.connection.execute(
            """
            INSERT INTO subreddits(name, first_seen_at, last_seen_at) VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (subreddit, timestamp, timestamp),
        )


def collect_resource(
    database: ResearchDatabase,
    client: ArcticShiftClient,
    resource: str,
    subreddit: str,
    requested_after: str,
    requested_before: str,
    page_size: int,
    max_pages: int,
    resume: bool,
) -> CollectionResult:
    checkpoint = database.checkpoint(resource, subreddit, requested_after, requested_before)
    if resume and checkpoint and checkpoint["completed_at"]:
        return CollectionResult(pages=0, records=0, completed=True)

    cursor: str | int = requested_after
    if resume and checkpoint and checkpoint["last_created_utc"] is not None:
        cursor = int(checkpoint["last_created_utc"])

    pages = 0
    records_saved = 0
    while pages < max_pages:
        records = client.search(resource, subreddit, cursor, requested_before, page_size)
        pages += 1
        if not records:
            last_cursor = cursor if isinstance(cursor, int) else None
            database.save_checkpoint(
                resource, subreddit, requested_after, requested_before, last_cursor, completed=True
            )
            return CollectionResult(pages=pages, records=records_saved, completed=True)

        if resource == "posts":
            records_saved += database.upsert_posts(records)
        else:
            records_saved += database.upsert_comments(records)

        timestamps = [as_int(record.get("created_utc")) for record in records]
        last_timestamp = max(timestamp for timestamp in timestamps if timestamp is not None)
        if isinstance(cursor, int) and last_timestamp <= cursor:
            raise ArcticShiftError(
                "pagination made no timestamp progress; stop to avoid skipping archive data"
            )
        cursor = last_timestamp
        database.save_checkpoint(
            resource, subreddit, requested_after, requested_before, last_timestamp, completed=False
        )

    return CollectionResult(pages=pages, records=records_saved, completed=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="create the SQLite schema")
    init_db.add_argument("--database", default="reddit_archive.db", help="SQLite file path")

    collect = subparsers.add_parser("collect", help="collect Arctic Shift archive records")
    collect.add_argument("--database", default="reddit_archive.db", help="SQLite file path")
    collect.add_argument("--subreddit", action="append", type=validate_subreddit, required=True)
    collect.add_argument("--after", required=True, help="inclusive starting date/time, e.g. 2024-01-01")
    collect.add_argument("--before", required=True, help="exclusive ending date/time, e.g. 2024-02-01")
    collect.add_argument(
        "--resource",
        action="append",
        choices=("posts", "comments"),
        help="resource to collect; specify once or omit for both",
    )
    collect.add_argument("--page-size", type=int, default=100, help="1-100; default 100")
    collect.add_argument("--max-pages", type=int, default=100, help="maximum pages per subreddit/resource")
    collect.add_argument("--min-interval", type=float, default=1.0, help="minimum seconds between requests")
    collect.add_argument("--resume", action="store_true", help="continue a matching interrupted window")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init-db":
        database = ResearchDatabase(args.database)
        try:
            database.initialize()
            print(f"Database ready: {database.path.resolve()}")
            return 0
        finally:
            database.close()

    if not 1 <= args.page_size <= 100:
        raise SystemExit("--page-size must be between 1 and 100")
    if args.max_pages < 1:
        raise SystemExit("--max-pages must be at least 1")
    if args.min_interval < 0:
        raise SystemExit("--min-interval cannot be negative")

    resources = args.resource or ["posts", "comments"]
    database = ResearchDatabase(args.database)
    database.initialize()
    run_id = database.start_run(args.after, args.before, args.subreddit)
    client = ArcticShiftClient(min_interval_seconds=args.min_interval)
    summary: dict[str, Any] = {"pages": 0, "records": 0, "completed": 0, "partial": 0}
    failures = 0
    try:
        for resource in resources:
            for subreddit in args.subreddit:
                try:
                    result = collect_resource(
                        database,
                        client,
                        resource,
                        subreddit,
                        args.after,
                        args.before,
                        args.page_size,
                        args.max_pages,
                        args.resume,
                    )
                    summary["pages"] += result.pages
                    summary["records"] += result.records
                    summary["completed" if result.completed else "partial"] += 1
                    state = "complete" if result.completed else "partial; rerun with --resume"
                    print(
                        f"{resource} r/{subreddit}: {result.records} records, "
                        f"{result.pages} pages, {state}"
                    )
                except ArcticShiftError as error:
                    failures += 1
                    database.log_error(run_id, resource, subreddit, str(error))
                    print(f"{resource} r/{subreddit}: failed - {error}", file=sys.stderr)
        summary["failures"] = failures
        database.finish_run(run_id, "failed" if failures else "completed", summary)
        print(json.dumps(summary, sort_keys=True))
        return 1 if failures else 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
