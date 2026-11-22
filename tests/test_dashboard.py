"""Exercise the actual dashboard pages and interactions against saved outputs."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


@unittest.skipUnless(Path("artifacts/manifest.json").exists() and Path("reddit_analysis.db").exists(),
                     "Run the corpus pipeline before dashboard integration checks")
class DashboardTests(unittest.TestCase):
    def test_pages_and_filter_changes(self):
        app = AppTest.from_file("app.py", default_timeout=60).run()
        self.assertFalse(app.exception)
        for page in ["Overview", "Sentiment & trends", "Topics", "Communities", "Explore records",
                     "Model laboratory", "AI assistant", "Data & pipeline"]:
            with self.subTest(page=page):
                app.radio[0].set_value(page).run()
                self.assertFalse(app.exception, [e.message for e in app.exception])
        app.radio[0].set_value("Overview").run()
        model = next(w for w in app.selectbox if w.label == "Sentiment model")
        model.set_value("svm")
        next(w for w in app.button if w.label == "Apply filters").click().run()
        self.assertFalse(app.exception)
        kinds = next(w for w in app.multiselect if w.label == "Record types")
        kinds.set_value([])
        next(w for w in app.button if w.label == "Apply filters").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("No analyzed records" in message.value for message in app.info))

    def test_real_time_model_predictions(self):
        app = AppTest.from_file("app.py", default_timeout=60).run()
        app.radio[0].set_value("Model laboratory").run()
        next(w for w in app.text_area if w.label == "Text to analyze").set_value("I love this amazing camera!")
        next(w for w in app.button if w.label == "Analyze with all models").click().run()
        self.assertFalse(app.exception, [e.message for e in app.exception])
        self.assertTrue(any("VADER compound:" in c.value for c in app.caption))


if __name__ == "__main__":
    unittest.main()
