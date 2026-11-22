"""Train real supervised baselines and LDA from an unlabelled archive.

VADER labels are weak supervision, never human ground truth. A chronological,
text-deduplicated train/validation/test split prevents fitting on the holdout.
"""
from __future__ import annotations

import json
import time
import warnings
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from . import LABELS
from .storage import TABLES, connect, record_query, source_info
from .text import MAX_CHARS, fingerprint, score_text, sentiment_text

SEED = 42
TOPIC_STOPWORDS = sorted(set(ENGLISH_STOP_WORDS) | {
    "edit", "reddit", "subreddit", "deleted", "removed", "https", "www", "com",
    "like", "just", "really", "people", "think", "know", "don", "doesn", "didn",
    "ve", "ll", "im", "amp", "tldr", "yeah", "would", "could", "get", "got",
})


def write_json(path, payload):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def sample_corpus(source, size=60000, info=None):
    """Seeded proportional sampling via indexed rowids; no random sort of 6M rows.

    Random rowids are accepted only when present (no next-row bias for holes).
    Counts and achieved coverage are persisted in the training manifest.
    """
    info = info or source_info(source)
    rng = np.random.default_rng(SEED)
    total = sum(info["counts"].values())
    records = []
    with connect(source) as c:
        for kind in TABLES:
            count = info["counts"][kind]
            if not count:
                continue
            target = min(count, max(1, round(size * count / max(total, 1))))
            low, high = info["bounds"][kind][:2]
            selected = {}
            attempts = 0
            while len(selected) < target and attempts < 30:
                wanted = min(high - low + 1, max(1000, (target - len(selected)) * 2))
                ids = rng.integers(low, high + 1, size=wanted).tolist()
                for start in range(0, len(ids), 800):
                    chunk = ids[start:start + 800]
                    sql = record_query(kind) + " WHERE rowid IN (" + ",".join("?" for _ in chunk) + ")"
                    for row in c.execute(sql, chunk):
                        selected[row["source_rowid"]] = dict(row, kind=kind)
                attempts += 1
            values = list(selected.values())
            rng.shuffle(values)
            records.extend(values[:target])
    if len(records) < 300:
        raise ValueError("At least 300 usable records are required to train and evaluate the models.")
    df = pd.DataFrame(records)
    df["text"] = df.raw_text.map(sentiment_text)
    df["fingerprint"] = df.text.map(fingerprint)
    df = df[df.text.str.len() > 2].sort_values("created_utc").drop_duplicates("fingerprint")
    return df.reset_index(drop=True)


def metric_result(truth, predicted):
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)),
            "report": classification_report(truth, predicted, labels=[0, 1, 2], target_names=LABELS,
                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1, 2]).tolist(),
            "n": len(truth)}


def topic_coherence(model, validation_counts, top_n=10):
    """Mean pairwise document NPMI on unseen documents, range [-1, 1]."""
    binary = (validation_counts > 0).astype(np.float64).tocsc()
    n = binary.shape[0]
    scores = []
    for topic in model.components_:
        indices = topic.argsort()[-top_n:]
        x = binary[:, indices]
        df = np.asarray(x.sum(axis=0)).ravel()
        co = (x.T @ x).toarray()
        pairs = []
        for i in range(len(indices)):
            for j in range(i):
                if co[i, j] == 0 or not df[i] or not df[j]:
                    pairs.append(-1.0)
                else:
                    pij = co[i, j] / n
                    denom = -np.log(pij)
                    pairs.append(float(np.log(pij / ((df[i] / n) * (df[j] / n))) / denom) if denom > 0 else 0.0)
        scores.append(float(np.mean(pairs)))
    return float(np.mean(scores))


def train(source, artifacts, sample_size=60000, topic_candidates=(4, 6, 8), progress=print):
    artifacts = Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    if (artifacts / "bundle.joblib").exists():
        raise FileExistsError("Model artifacts already exist. Use a new --artifacts directory to train a new version.")
    start = time.monotonic()
    info = source_info(source)
    progress(f"Sampling {sample_size:,} records across both tables...", flush=True)
    df = sample_corpus(source, sample_size, info)
    progress(f"Generating VADER weak labels for {len(df):,} distinct texts...", flush=True)
    scored = [score_text(t) for t in df.text]
    df["label"] = [s[0] for s in scored]
    df["compound"] = [s[1] for s in scored]
    n = len(df)
    cut1, cut2 = int(n * 0.65), int(n * 0.82)
    tr, val, test = df.iloc[:cut1], df.iloc[cut1:cut2], df.iloc[cut2:]
    for name, split in [("train", tr), ("validation", val), ("test", test)]:
        if set(split.label.unique()) != {0, 1, 2}:
            raise ValueError(f"The {name} split does not contain all three sentiment classes. Increase --sample-size.")
    progress("Fitting TF-IDF on training data only (unigrams + bigrams, negation retained)...", flush=True)
    tfidf = TfidfVectorizer(max_features=24000, ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True, strip_accents="unicode", dtype=np.float32)
    xtr = tfidf.fit_transform(tr.text)
    xv, xt = tfidf.transform(val.text), tfidf.transform(test.text)
    configs = {
        "logistic": [LogisticRegression(C=c, max_iter=400, class_weight="balanced", random_state=SEED) for c in (0.5, 2.0)],
        "naive_bayes": [MultinomialNB(alpha=a) for a in (0.25, 1.0)],
        "svm": [LinearSVC(C=c, class_weight="balanced", random_state=SEED, max_iter=3000) for c in (0.5, 1.0)],
        "neural": [MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, batch_size=512,
                                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=7,
                                 random_state=SEED, alpha=0.001)],
    }
    models, results, errors = {}, {}, []
    with threadpool_limits(limits=2):
        for name, candidates in configs.items():
            progress(f"Training {name}...", flush=True)
            fitted = []
            t0 = time.monotonic()
            for estimator in candidates:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    kwargs = {"sample_weight": compute_sample_weight("balanced", tr.label)} if name == "neural" else {}
                    estimator.fit(xtr, tr.label, **kwargs)
                validation = metric_result(val.label, estimator.predict(xv))
                fitted.append((validation["macro_f1"], estimator, validation, [str(w.message) for w in caught]))
            _, best, validation, notes = max(fitted, key=lambda v: v[0])
            models[name] = best
            predicted = best.predict(xt)
            results[name] = {"test": metric_result(test.label, predicted), "validation": validation,
                             "training_seconds": round(time.monotonic() - t0, 2),
                             "parameters": best.get_params(), "warnings": notes}
            for row, prediction in zip(test.to_dict("records"), predicted):
                if row["label"] != prediction and sum(e["model"] == name for e in errors) < 25:
                    errors.append({"model": name, "text": row["text"][:1200], "weak_label": LABELS[row["label"]],
                                   "prediction": LABELS[int(prediction)], "kind": row["kind"], "id": row["id"]})
            progress(f"  {name}: held-out weak-label macro F1 {results[name]['test']['macro_f1']:.3f}", flush=True)
        progress("Training LDA candidates and measuring held-out topic coherence...", flush=True)
        topic_tr = tr.sample(min(len(tr), 20000), random_state=SEED)
        topic_val = val.sample(min(len(val), 4000), random_state=SEED)
        counts = CountVectorizer(max_features=10000, min_df=3, max_df=0.85,
                                 stop_words=TOPIC_STOPWORDS, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
        ctr = counts.fit_transform(topic_tr.text_clean)
        cv = counts.transform(topic_val.text_clean)
        candidates, topic_scores = [], []
        for k in topic_candidates:
            progress(f"  LDA: {k} topics...", flush=True)
            lda = LatentDirichletAllocation(n_components=k, learning_method="online", max_iter=8,
                                           batch_size=512, max_doc_update_iter=30, random_state=SEED, n_jobs=1)
            lda.fit(ctr)
            coherence = topic_coherence(lda, cv)
            topic_scores.append({"k": k, "npmi": coherence, "perplexity": float(lda.perplexity(cv))})
            candidates.append(lda)
        best_index = int(np.argmax([r["npmi"] for r in topic_scores]))
        lda = candidates[best_index]
    vocab = counts.get_feature_names_out()
    topics = []
    for i, weights in enumerate(lda.components_):
        ids = weights.argsort()[-12:][::-1]
        terms = vocab[ids].tolist()
        topics.append({"id": i, "name": " / ".join(terms[:3]), "words": terms,
                       "weights": (weights[ids] / weights.sum()).tolist()})
    selected = max(results, key=lambda name: results[name]["validation"]["macro_f1"])
    manifest = {
        "version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "library_versions": {name: version(name) for name in ["scikit-learn", "numpy", "scipy", "pandas", "joblib", "vaderSentiment"]},
        "label_source": "VADER weak supervision", "human_validated": False,
        "evaluation_note": "Metrics measure agreement with VADER on unseen later texts, NOT accuracy against human labels. VADER is the label generator and is not scored against itself.",
        "source": info, "sample_size": n, "requested_sample_size": sample_size,
        "sample_subreddits": sorted(df.subreddit.unique().tolist()),
        "sample_months": sorted(df.year_month.unique().tolist()),
        "split": {name: {"n": len(part), "from": int(part.created_utc.min()), "to": int(part.created_utc.max()),
                         "class_counts": {LABELS[int(k)]: int(v) for k, v in part.label.value_counts().items()}}
                  for name, part in [("train", tr), ("validation", val), ("test", test)]},
        "max_text_chars": MAX_CHARS, "default_model": "vader", "best_student": selected,
        "models": results, "topic_selection": topic_scores, "topics": topics,
        "topic_training_records": len(topic_tr), "topic_validation_records": len(topic_val),
        "training_seconds": round(time.monotonic() - start, 1),
        "limitations": ["No human-labelled test set is available.", "Sarcasm, mixed sentiment and non-English text can be misclassified.",
                        "The original collector used monthly quotas and ascending timestamps; this corpus is not a random sample of all Reddit activity.",
                        "Texts longer than 8,000 cleaned characters are truncated for inference; affected rows are flagged.",
                        "LDA assigns a dominant topic; documents may discuss several topics. Out-of-vocabulary texts are unassigned."]}
    bundle = {"tfidf": tfidf, "models": models, "counts": counts, "lda": lda, "topics": topics,
              "training_fingerprints": set(tr.fingerprint), "manifest": manifest}
    temp = artifacts / "bundle.joblib.tmp"
    joblib.dump(bundle, temp, compress=3)
    temp.replace(artifacts / "bundle.joblib")
    write_json(artifacts / "manifest.json", manifest)
    write_json(artifacts / "errors.json", errors)
    df[["kind", "id", "source_rowid", "created_utc", "label", "fingerprint"]].assign(
        split=["train"] * len(tr) + ["validation"] * len(val) + ["test"] * len(test)
    ).to_csv(artifacts / "training_sample.csv", index=False)
    progress(f"Training complete. Saved all four classifiers and LDA to {artifacts}.", flush=True)
    return manifest


def predict_texts(bundle, raw_texts, clean_texts=None):
    texts = [sentiment_text(t) for t in raw_texts]
    matrix = bundle["tfidf"].transform(texts)
    values = {name: model.predict(matrix).astype(int) for name, model in bundle["models"].items()}
    scored = [score_text(text) for text in texts]
    values["vader"] = np.array([s[0] for s in scored], dtype=int)
    values["compound"] = np.array([s[1] for s in scored])
    topic_counts = bundle["counts"].transform(clean_texts if clean_texts is not None else texts)
    probabilities = bundle["lda"].transform(topic_counts)
    topics = probabilities.argmax(axis=1).astype(int)
    strength = probabilities.max(axis=1)
    empty = np.asarray(topic_counts.sum(axis=1)).ravel() == 0
    topics[empty], strength[empty] = -1, 0
    values["topic"], values["topic_probability"] = topics, strength
    values["truncated"] = [len(sentiment_text(t, max_chars=len(str(t)) * 6 + 100)) > MAX_CHARS for t in raw_texts]
    return values
