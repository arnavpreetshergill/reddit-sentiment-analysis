import sqlite3
import requests
import time
import sys
from datetime import datetime, timezone, timedelta
from calendar import monthrange

# ============================================================
# CONFIGURATION
# ============================================================

BASE = "https://arctic-shift.photon-reddit.com"

DB = "reddit_8million.db"

# 4 years
START = "2022-08-18"
END = "2026-08-18"

# Target per subreddit PER MONTH
POST_TARGET = 1000
COMMENT_TARGET = 6000

# Arctic Shift:
# limit=auto can return up to 1000 depending on server capacity.
PAGE_SIZE = "auto"

# Minimum delay between successful requests.
# Start low; Arctic Shift will be slowed down automatically on errors.
WAIT = 0.25

TIMEOUT = 120

# Retry configuration
MAX_RETRIES = 7

# ============================================================
# 25 SUBREDDITS
# ============================================================

SUBREDDITS = [

    # TECHNOLOGY
    "technology",
    "programming",
    "software",
    "hardware",
    "linux",
    "windows",
    "apple",
    "android",
    "gadgets",
    "MachineLearning",

    # SCIENCE
    "datascience",
    "artificial",
    "ChatGPT",
    "science",
    "AskScience",
    "Physics",
    "chemistry",
    "biology",

    # NEWS / GENERAL
    "worldnews",
    "news",
    "geopolitics",
    "business",
    "economics",
    "AskReddit",
    "NoStupidQuestions",
]


# ============================================================
# FIELDS
# ============================================================

POST_FIELDS = (
    "id,subreddit,created_utc,retrieved_on,title,selftext,url,"
    "score,num_comments,over_18,spoiler"
)

COMMENT_FIELDS = (
    "id,subreddit,created_utc,retrieved_on,body,"
    "link_id,parent_id,score,distinguished"
)


# ============================================================
# HELPERS
# ============================================================

def now():
    return datetime.now(timezone.utc).isoformat()


def date_to_str(dt):
    return dt.strftime("%Y-%m-%d")


def get_month_windows(start_date, end_date):

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    current = start

    while current < end:

        year = current.year
        month = current.month

        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)

        month_end = min(next_month, end)

        yield (
            date_to_str(current),
            date_to_str(month_end)
        )

        current = next_month


# ============================================================
# DATABASE
# ============================================================

def setup_db():

    con = sqlite3.connect(DB)

    # Faster SQLite settings
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-100000")

    con.executescript("""

    CREATE TABLE IF NOT EXISTS posts (

        id TEXT PRIMARY KEY,

        subreddit TEXT,

        created_utc INTEGER,

        retrieved_on INTEGER,

        title TEXT,

        selftext TEXT,

        url TEXT,

        score INTEGER,

        num_comments INTEGER,

        over_18 INTEGER,

        spoiler INTEGER,

        ingested_at TEXT
    );


    CREATE INDEX IF NOT EXISTS idx_posts_subreddit
    ON posts(subreddit);


    CREATE INDEX IF NOT EXISTS idx_posts_created
    ON posts(created_utc);


    CREATE TABLE IF NOT EXISTS comments (

        id TEXT PRIMARY KEY,

        subreddit TEXT,

        created_utc INTEGER,

        retrieved_on INTEGER,

        body TEXT,

        link_id TEXT,

        parent_id TEXT,

        score INTEGER,

        distinguished TEXT,

        ingested_at TEXT
    );


    CREATE INDEX IF NOT EXISTS idx_comments_subreddit
    ON comments(subreddit);


    CREATE INDEX IF NOT EXISTS idx_comments_created
    ON comments(created_utc);


    CREATE INDEX IF NOT EXISTS idx_comments_post
    ON comments(link_id);


    CREATE TABLE IF NOT EXISTS progress (

        subreddit TEXT,

        resource TEXT,

        month_start TEXT,

        month_end TEXT,

        last_timestamp INTEGER,

        records_collected INTEGER DEFAULT 0,

        completed INTEGER DEFAULT 0,

        PRIMARY KEY(
            subreddit,
            resource,
            month_start,
            month_end
        )
    );

    """)

    con.commit()

    return con


# ============================================================
# PROGRESS
# ============================================================

def get_progress(
    con,
    subreddit,
    resource,
    month_start,
    month_end
):

    row = con.execute("""
        SELECT
            last_timestamp,
            records_collected,
            completed

        FROM progress

        WHERE
            subreddit=?
            AND resource=?
            AND month_start=?
            AND month_end=?

    """, (
        subreddit,
        resource,
        month_start,
        month_end
    )).fetchone()

    return row


def save_progress(
    con,
    subreddit,
    resource,
    month_start,
    month_end,
    timestamp,
    records,
    completed
):

    con.execute("""
        INSERT INTO progress (
            subreddit,
            resource,
            month_start,
            month_end,
            last_timestamp,
            records_collected,
            completed
        )

        VALUES (?, ?, ?, ?, ?, ?, ?)

        ON CONFLICT(
            subreddit,
            resource,
            month_start,
            month_end
        )

        DO UPDATE SET

            last_timestamp =
                excluded.last_timestamp,

            records_collected =
                excluded.records_collected,

            completed =
                excluded.completed

    """, (
        subreddit,
        resource,
        month_start,
        month_end,
        timestamp,
        records,
        completed
    ))

    con.commit()


# ============================================================
# FETCH
# ============================================================

session = requests.Session()

session.headers.update({
    "User-Agent":
        "Reddit-Sentiment-Research/1.0"
})


def fetch(
    resource,
    subreddit,
    after,
    before
):

    fields = (
        POST_FIELDS
        if resource == "posts"
        else COMMENT_FIELDS
    )

    url = f"{BASE}/api/{resource}/search"

    params = {

        "subreddit": subreddit,

        "after": after,

        "before": before,

        "sort": "asc",

        "limit": PAGE_SIZE,

        "fields": fields
    }


    for attempt in range(MAX_RETRIES):

        try:

            r = session.get(
                url,
                params=params,
                timeout=TIMEOUT
            )


            # -------------------------------
            # SUCCESS
            # -------------------------------

            if r.status_code == 200:

                data = r.json().get("data")

                if data is None:
                    return []

                return data


            # -------------------------------
            # RATE LIMIT / SERVER BUSY
            # -------------------------------

            if r.status_code in (
                422,
                429,
                500,
                502,
                503,
                504
            ):

                # Exponential backoff
                wait = min(
                    30,
                    2 ** attempt
                )

                print(
                    f"\n  HTTP {r.status_code}"
                    f" → waiting {wait}s"
                )

                time.sleep(wait)

                continue


            # -------------------------------
            # OTHER ERROR
            # -------------------------------

            print(
                f"\n  HTTP {r.status_code}: "
                f"{r.text[:200]}"
            )

            return []


        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ) as e:

            wait = min(
                30,
                2 ** attempt
            )

            print(
                f"\n  Network error: {e}"
                f" → waiting {wait}s"
            )

            time.sleep(wait)


        except Exception as e:

            print(
                f"\n  Request error: {e}"
            )

            time.sleep(3)


    print(
        "\n  FAILED after maximum retries"
    )

    return []


# ============================================================
# SAVE POSTS
# ============================================================

def save_posts(con, records):

    data = []

    timestamp = now()

    for x in records:

        if not x.get("id"):
            continue

        data.append((

            x.get("id"),

            x.get("subreddit"),

            x.get("created_utc"),

            x.get("retrieved_on"),

            x.get("title") or "",

            x.get("selftext") or "",

            x.get("url"),

            x.get("score"),

            x.get("num_comments"),

            int(bool(x.get("over_18")))
            if x.get("over_18") is not None
            else None,

            int(bool(x.get("spoiler")))
            if x.get("spoiler") is not None
            else None,

            timestamp
        ))


    con.executemany("""

        INSERT OR REPLACE INTO posts

        VALUES (
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?
        )

    """, data)


# ============================================================
# SAVE COMMENTS
# ============================================================

def save_comments(con, records):

    data = []

    timestamp = now()

    for x in records:

        if not x.get("id"):
            continue

        data.append((

            x.get("id"),

            x.get("subreddit"),

            x.get("created_utc"),

            x.get("retrieved_on"),

            x.get("body") or "",

            x.get("link_id"),

            x.get("parent_id"),

            x.get("score"),

            x.get("distinguished"),

            timestamp
        ))


    con.executemany("""

        INSERT OR REPLACE INTO comments

        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?
        )

    """, data)


# ============================================================
# COLLECT ONE MONTH
# ============================================================

def collect_month(
    con,
    subreddit,
    resource,
    month_start,
    month_end
):

    target = (
        POST_TARGET
        if resource == "posts"
        else COMMENT_TARGET
    )


    print()
    print(
        f"{resource.upper()} "
        f"r/{subreddit} "
        f"{month_start} → {month_end}"
    )


    progress = get_progress(
        con,
        subreddit,
        resource,
        month_start,
        month_end
    )


    # Already completed
    if progress and progress[2] == 1:

        print(
            f"  Already complete "
            f"({progress[1]:,})"
        )

        return progress[1]


    if progress:

        after = progress[0]

        total = progress[1]

        if after:

            print(
                f"  Resuming at "
                f"{after}"
            )

    else:

        after = month_start

        total = 0


    pages = 0


    while total < target:

        records = fetch(
            resource,
            subreddit,
            after,
            month_end
        )


        if not records:

            # No more data available
            save_progress(
                con,
                subreddit,
                resource,
                month_start,
                month_end,
                after if isinstance(after, int) else 0,
                total,
                1
            )

            print(
                f"  Month finished: "
                f"{total:,}/{target:,}"
            )

            return total


        # Don't exceed target
        remaining = target - total

        if len(records) > remaining:

            records = records[:remaining]


        # Save
        if resource == "posts":

            save_posts(
                con,
                records
            )

        else:

            save_comments(
                con,
                records
            )


        con.commit()


        total += len(records)

        pages += 1


        timestamps = [

            x.get("created_utc")

            for x in records

            if x.get("created_utc")
            is not None
        ]


        if not timestamps:

            print(
                "  No timestamps. "
                "Stopping."
            )

            break


        newest = max(timestamps)


        # Prevent pagination loop
        if (
            isinstance(after, int)
            and newest <= after
        ):

            print(
                "  Timestamp did not advance."
            )

            break


        after = newest


        save_progress(
            con,
            subreddit,
            resource,
            month_start,
            month_end,
            after,
            total,
            0
        )


        print(
            f"  page {pages:,} | "
            f"{total:,}/{target:,}"
        )


        # Small delay only after success
        time.sleep(WAIT)


    # Target reached
    save_progress(
        con,
        subreddit,
        resource,
        month_start,
        month_end,
        after if isinstance(after, int) else 0,
        total,
        1
    )


    print(
        f"  COMPLETE: {total:,}"
    )


    return total


# ============================================================
# MAIN
# ============================================================

def main():

    months = list(
        get_month_windows(
            START,
            END
        )
    )


    expected_posts = (
        len(SUBREDDITS)
        * len(months)
        * POST_TARGET
    )

    expected_comments = (
        len(SUBREDDITS)
        * len(months)
        * COMMENT_TARGET
    )

    expected_total = (
        expected_posts
        + expected_comments
    )


    print()
    print("==========================================")
    print(" REDDIT LARGE-SCALE RESEARCH DATABASE")
    print("==========================================")
    print()

    print(
        f"Period: "
        f"{START} → {END}"
    )

    print(
        f"Subreddits: "
        f"{len(SUBREDDITS)}"
    )

    print(
        f"Months: "
        f"{len(months)}"
    )

    print(
        f"Posts/month/subreddit: "
        f"{POST_TARGET:,}"
    )

    print(
        f"Comments/month/subreddit: "
        f"{COMMENT_TARGET:,}"
    )

    print()

    print(
        f"Expected posts: "
        f"{expected_posts:,}"
    )

    print(
        f"Expected comments: "
        f"{expected_comments:,}"
    )

    print(
        f"EXPECTED TOTAL: "
        f"{expected_total:,}"
    )

    print()

    print(
        f"Database: {DB}"
    )

    print(
        "Press Ctrl+C at any time to stop."
    )

    print(
        "Progress is saved after every page."
    )

    print()


    con = setup_db()


    total_collected = 0


    try:

        for subreddit_index, subreddit in enumerate(
            SUBREDDITS,
            1
        ):

            print()
            print(
                "########################################"
            )

            print(
                f"SUBREDDIT "
                f"{subreddit_index}/"
                f"{len(SUBREDDITS)}: "
                f"r/{subreddit}"
            )

            print(
                "########################################"
            )


            for month_index, (
                month_start,
                month_end
            ) in enumerate(months, 1):


                print()
                print(
                    f"MONTH "
                    f"{month_index}/"
                    f"{len(months)}"
                )


                for resource in (
                    "posts",
                    "comments"
                ):

                    try:

                        count = collect_month(
                            con,
                            subreddit,
                            resource,
                            month_start,
                            month_end
                        )

                        total_collected += count


                    except KeyboardInterrupt:

                        print()
                        print(
                            "Stopped by user."
                        )

                        print(
                            "Progress saved."
                        )

                        return


                    except Exception as e:

                        print(
                            f"ERROR: "
                            f"{resource} "
                            f"r/{subreddit} "
                            f"{month_start}: "
                            f"{e}",
                            file=sys.stderr
                        )


    finally:

        con.close()


    print()
    print("==========================================")
    print(" COLLECTION FINISHED")
    print("==========================================")

    print(
        f"Target records: "
        f"{expected_total:,}"
    )

    print(
        f"Run records processed: "
        f"{total_collected:,}"
    )

    print(
        f"Database: {DB}"
    )


if __name__ == "__main__":

    main()