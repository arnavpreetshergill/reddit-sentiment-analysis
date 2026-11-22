import tempfile
import unittest
from pathlib import Path

from arctic_shift_db import ResearchDatabase, normalize_reddit_id, parent_kind


class ResearchDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.database = ResearchDatabase(str(Path(self.tempdir.name) / "test.db"))
        self.database.initialize()

    def tearDown(self):
        self.database.close()
        self.tempdir.cleanup()

    def test_normalizes_prefixed_ids(self):
        self.assertEqual(normalize_reddit_id("t3_abc123"), "abc123")
        self.assertEqual(normalize_reddit_id("t1_def456"), "def456")
        self.assertEqual(parent_kind("t3_abc123"), "post")
        self.assertEqual(parent_kind("t1_def456"), "comment")

    def test_upserts_posts_and_comments(self):
        self.assertEqual(
            self.database.upsert_posts(
                [
                    {
                        "id": "t3_post1",
                        "subreddit": "MachineLearning",
                        "created_utc": 1_700_000_000,
                        "title": "A post",
                        "selftext": "Text",
                        "score": 2,
                        "num_comments": 1,
                    }
                ]
            ),
            1,
        )
        self.assertEqual(
            self.database.upsert_comments(
                [
                    {
                        "id": "t1_comment1",
                        "subreddit": "MachineLearning",
                        "created_utc": 1_700_000_001,
                        "link_id": "t3_post1",
                        "parent_id": "t3_post1",
                        "body": "A comment",
                        "score": 3,
                    }
                ]
            ),
            1,
        )
        post = self.database.connection.execute(
            "SELECT reddit_id, title FROM posts"
        ).fetchone()
        comment = self.database.connection.execute(
            "SELECT post_reddit_id, parent_kind, body FROM comments"
        ).fetchone()
        self.assertEqual(dict(post), {"reddit_id": "post1", "title": "A post"})
        self.assertEqual(
            dict(comment),
            {"post_reddit_id": "post1", "parent_kind": "post", "body": "A comment"},
        )


if __name__ == "__main__":
    unittest.main()
