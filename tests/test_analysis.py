"""Regression coverage for model inference, safe filters, trends and resumability."""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from reddit_insights.analytics import (aggregate, csv_bytes, emerging_topics, evidence_package,
                                       explore, save_annotation, timeline, totals, where_clause)
from reddit_insights.storage import connect, metadata, put_meta
from reddit_insights.text import score_text, sentiment_text, vader_label


class SentimentAndTrendTests(unittest.TestCase):
    def test_polarity_boundaries_and_negation(self):
        self.assertEqual([vader_label(x) for x in [-0.051, -0.05, 0, 0.049, 0.05]], [0, 0, 1, 1, 2])
        self.assertEqual(score_text("This is wonderful! I love it!")[0], 2)
        self.assertEqual(score_text("This is terrible and awful.")[0], 0)
        self.assertLess(score_text("This is not good.")[1], score_text("This is good.")[1])
        self.assertIn("GREAT!!!", sentiment_text("GREAT!!! https://example.com u/username"))

    def test_empty_filter_does_not_expand_scope_or_interpolate_sql(self):
        where, params = where_clause({"subreddit": ["x' OR 1=1 --"], "kind": [], "model": "vader"})
        self.assertNotIn("OR 1=1", where)
        self.assertIn("0", where)
        self.assertIn("x' OR 1=1 --", params)
        with self.assertRaises(ValueError):
            where_clause({"model": "vader;DROP TABLE monthly"})

    def test_missing_months_are_not_neutral_and_baseline_has_no_future_leak(self):
        rows = []
        for month, positive in [("2025-01", 60), ("2025-02", 60), ("2025-04", 60), ("2025-05", 10)]:
            for label, n in [(0, 100 - positive), (2, positive)]:
                rows.append({"year_month": month, "sentiment": label, "n": n})
        frame = pd.DataFrame(rows)
        result = timeline(frame, "2025-01", "2025-05")
        self.assertTrue(np.isnan(result.iloc[2].net))
        self.assertEqual(result.iloc[2].n, 0)
        self.assertAlmostEqual(result.iloc[4].baseline, 20)
        self.assertFalse(result.iloc[0].anomaly)

    def test_topic_growth_is_change_in_share_not_counts(self):
        frame = pd.DataFrame([
            {"topic": 0, "year_month": "2025-01", "n": 10},
            {"topic": 1, "year_month": "2025-01", "n": 10},
            {"topic": 0, "year_month": "2025-02", "n": 30},
            {"topic": 1, "year_month": "2025-02", "n": 70},
        ])
        result = emerging_topics(frame, "2025-01", "2025-02").set_index("topic")
        self.assertEqual(result.loc[0, "change_pp"], -20)

    def test_exports_neutralize_formulas(self):
        result = csv_bytes(pd.DataFrame({"text": ["=1+1", "normal", "  @formula"]})).decode("utf-8-sig")
        self.assertIn("'=1+1", result)
        self.assertIn("'  @formula", result)

    def test_evidence_package_includes_twelve_sanitized_excerpts(self):
        frame = pd.DataFrame([
            {"year_month": "2025-01", "sentiment": 2, "n": 1, "score_n": 0,
             "score_sum": 0, "subreddit": "science", "topic": 0},
        ])
        records = pd.DataFrame([
            {"evidence_id": f"comment:c{i}", "raw_text": "A useful https://example.com u/redditor excerpt",
             "label": "positive", "subreddit": "science"}
            for i in range(12)
        ])
        package = evidence_package(
            frame,
            {"model": "vader", "start": "2025-01", "end": "2025-01"},
            {"topics": [], "limitations": []},
            records,
        )
        self.assertEqual(len(package["evidence"]), 12)
        self.assertNotIn("https://", package["evidence"][0]["text"])
        self.assertNotIn("u/redditor", package["evidence"][0]["text"])


@unittest.skipUnless(Path("artifacts/bundle.joblib").exists(), "Train models to run saved-model integration checks")
class CorpusIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from reddit_insights.pipeline import analyze
        cls.temp = tempfile.TemporaryDirectory(prefix="reddit-analysis-")
        cls.source = Path(cls.temp.name) / "source.db"
        cls.output = Path(cls.temp.name) / "analysis.db"
        # Both kinds deliberately share the same IDs and rowids.
        with closing(sqlite3.connect(cls.source)) as c:
            c.execute("CREATE TABLE posts_clean (id TEXT,subreddit TEXT,category TEXT,created_utc INTEGER,year_month TEXT,score INTEGER,title_raw TEXT,selftext_raw TEXT,text_clean TEXT)")
            c.execute("CREATE TABLE comments_clean (id TEXT,subreddit TEXT,category TEXT,created_utc INTEGER,year_month TEXT,score INTEGER,body_raw TEXT,text_clean TEXT,post_id TEXT)")
            for i in range(12):
                text = ["This is wonderful and I love it!", "This is awful and terrible!", "The meeting starts on Monday."][i % 3]
                base = (f"same{i}", "science", "science", 1735689600 + i, "2025-01", i)
                c.execute("INSERT INTO posts_clean VALUES (?,?,?,?,?,?,?,?,?)", (*base, text, "", text.lower()))
                c.execute("INSERT INTO comments_clean VALUES (?,?,?,?,?,?,?,?,?)", (*base, text, text.lower(), f"same{i}"))
            c.commit()
        cls.source_before = cls.source.read_bytes()
        analyze(cls.source, "artifacts", cls.output, workers=1, batch_size=5, progress=lambda *a, **k: None)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_coverage_and_aggregate_conservation(self):
        with connect(self.output) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM predictions").fetchone()[0], 24)
            self.assertEqual(c.execute("SELECT count(*) FROM predictions WHERE topic_probability BETWEEN 0 AND 1").fetchone()[0], 24)
            sums = dict(c.execute("SELECT model,sum(n) FROM monthly GROUP BY model"))
            self.assertEqual(set(sums.values()), {24})
            self.assertEqual(len(sums), 5)
        self.assertEqual(self.source.read_bytes(), self.source_before)

    def test_resume_is_idempotent_and_version_changes_rejected(self):
        from reddit_insights.pipeline import analyze
        with connect(self.output, False) as c:
            with c:
                put_meta(c, "status", "interrupted")
        analyze(self.source, "artifacts", self.output, workers=1, progress=lambda *a, **k: None)
        self.assertEqual(totals(aggregate(self.output, {"model": "svm"}))["n"], 24)
        self.assertEqual(metadata(self.output)["status"], "complete")
        expected_hash = metadata(self.output)["model_hash"]
        try:
            with connect(self.output, False) as c:
                with c:
                    put_meta(c, "model_hash", "a-different-model-version")
            with self.assertRaisesRegex(ValueError, "Source or model version changed"):
                analyze(self.source, "artifacts", self.output, workers=1, progress=lambda *a, **k: None)
        finally:
            with connect(self.output, False) as c:
                with c:
                    put_meta(c, "model_hash", expected_hash)

    def test_explorer_search_annotation_and_empty_scope(self):
        rows = explore(self.output, self.source, {"model": "vader"}, search="wonderful")
        self.assertEqual(len(rows), 8)
        self.assertTrue((rows.label == "positive").all())
        self.assertTrue(explore(self.output, self.source, {"model": "vader", "kind": []}).empty)
        self.assertTrue(explore(self.output, self.source, {"model": "vader"}, search="%' OR 1=1 --").empty)
        first = rows.iloc[0]
        save_annotation(self.output, first.kind, int(first.source_rowid), "positive")
        save_annotation(self.output, first.kind, int(first.source_rowid), "neutral")
        with connect(self.output) as c:
            self.assertEqual(c.execute("SELECT count(*) FROM annotations").fetchone()[0], 1)
            self.assertEqual(c.execute("SELECT label FROM annotations").fetchone()[0], "neutral")

    def test_saved_models_handle_oov_and_long_text(self):
        import joblib
        from reddit_insights.modeling import predict_texts
        bundle = joblib.load("artifacts/bundle.joblib")
        values = predict_texts(bundle, ["xyzzyplughqwerty", "great " * 2000])
        self.assertEqual(values["topic"][0], -1)
        self.assertEqual(values["topic_probability"][0], 0)
        self.assertTrue(values["truncated"][1])
        self.assertEqual(set(bundle["models"]), {"logistic", "naive_bayes", "svm", "neural"})


if __name__ == "__main__":
    unittest.main()
