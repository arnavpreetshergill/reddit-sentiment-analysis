"""Reproducible sentiment, topic and trend analysis for the Reddit archive."""

LABELS = ["negative", "neutral", "positive"]
MODEL_NAMES = {
    "vader": "VADER",
    "logistic": "Logistic regression",
    "naive_bayes": "Naive Bayes",
    "svm": "Linear SVM",
    "neural": "Neural network (MLP)",
}
