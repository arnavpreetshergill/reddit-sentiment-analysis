"""Resumable full-corpus inference with bounded multiprocessing and atomic batches."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from . import MODEL_NAMES
from .modeling import metric_result, predict_texts, train, write_json
from .storage import TABLES, checkpoints, connect, create_analysis, explorer_indexes, metadata, put_meta, record_query, source_info
from .text import fingerprint, sentiment_text

_BUNDLE = None
_THREAD_LIMIT = None


def worker_init(bundle_path):
    global _BUNDLE, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _BUNDLE = joblib.load(bundle_path)


def predict_batch(payload):
    kind, records = payload
    values = predict_texts(_BUNDLE, [r["raw_text"] for r in records], [r["text_clean"] for r in records])
    rows = []
    groups = defaultdict(lambda: [0, 0.0, 0.0, 0])
    for i, record in enumerate(records):
        labels = [int(values[name][i]) for name in MODEL_NAMES]
        compound, topic = float(values["compound"][i]), int(values["topic"][i])
        rows.append((kind, record["source_rowid"], record["id"], record["subreddit"], record["category"],
                     record["created_utc"], record["year_month"], record["score"], compound, *labels,
                     topic, float(values["topic_probability"][i]), int(values["truncated"][i])))
        for name, label in zip(MODEL_NAMES, labels):
            key = (kind, record["subreddit"], record["category"], record["year_month"], name, label, topic)
            bucket = groups[key]
            bucket[0] += 1
            bucket[1] += compound
            bucket[2] += record["score"] if record["score"] is not None else 0
            bucket[3] += int(record["score"] is not None)
    return kind, records[-1]["source_rowid"], rows, [(*k, *v) for k, v in groups.items()]


@contextmanager
def exclusive_lock(path):
    """OS lock automatically released after a crash; lock file may safely persist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    if handle.tell() == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError("Another pipeline process is already using this artifacts directory.") from exc
    try:
        yield
    finally:
        handle.close()


def analyze(source, artifacts, output, workers=4, batch_size=2000, progress=print):
    source, artifacts, output = Path(source).resolve(), Path(artifacts).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("The output must be a separate database from the source archive.")
    bundle_path = artifacts / "bundle.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError("Train models first with python -m reddit_insights.pipeline train.")
    model_hash = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    info = source_info(source)
    create_analysis(output)
    existing = metadata(output)
    if existing and (existing.get("model_hash") != model_hash or existing.get("source") != info):
        raise ValueError("Source or model version changed. Use a new --output database to avoid mixing predictions.")
    if existing.get("status") == "complete":
        with connect(output, False) as out:
            explorer_indexes(out)
        progress("All source records already analyzed; nothing to repeat.", flush=True)
        return
    with connect(output, False) as out:
        with out:
            put_meta(out, "source", info)
            put_meta(out, "model_hash", model_hash)
            put_meta(out, "artifacts", str(artifacts))
            put_meta(out, "status", "running")
            put_meta(out, "started_at", existing.get("started_at", datetime.now(timezone.utc).isoformat()))
            put_meta(out, "pid", os.getpid())
            put_meta(out, "last_error", None)
        done = checkpoints(output)
        total = sum(info["counts"].values())
        processed = sum(r["processed"] for r in done.values())
        initial, started = processed, time.monotonic()
        progress(f"Analyzing all {total:,} records; resuming at {processed:,}. Workers: {workers}.", flush=True)

        def commit_result(result):
            nonlocal processed
            kind, last_id, rows, aggregates = result
            # Checkpoint, row predictions and aggregate increments commit together.
            # A crash can only repeat an uncommitted batch; never double count it.
            with out:
                out.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
                out.executemany("""INSERT INTO monthly VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(kind,subreddit,category,year_month,model,sentiment,topic)
                    DO UPDATE SET n=n+excluded.n, compound_sum=compound_sum+excluded.compound_sum,
                        score_sum=score_sum+excluded.score_sum,score_n=score_n+excluded.score_n""", aggregates)
                out.execute("""INSERT INTO checkpoints VALUES (?,?,?) ON CONFLICT(kind)
                    DO UPDATE SET last_rowid=excluded.last_rowid,processed=processed+excluded.processed""", (kind, last_id, len(rows)))
                processed += len(rows)
                elapsed = time.monotonic() - started
                rate = (processed - initial) / max(elapsed, 0.001)
                put_meta(out, "progress", {"processed": processed, "total": total, "records_per_second": round(rate, 1),
                                          "eta_seconds": round((total - processed) / max(rate, 1)),
                                          "updated_at": datetime.now(timezone.utc).isoformat()})
            progress(f"{processed:,}/{total:,} ({processed / max(total, 1):.1%}) | {rate:.0f} records/s", flush=True)

        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=worker_init, initargs=(str(bundle_path),)) as pool:
                for kind in TABLES:
                    cursor_id = done.get(kind, {}).get("last_rowid", 0)
                    pending = deque()
                    with connect(source) as src:
                        cursor = src.execute(record_query(kind) + " WHERE rowid > ? ORDER BY rowid", (cursor_id,))
                        while rows := cursor.fetchmany(batch_size):
                            pending.append(pool.submit(predict_batch, (kind, [dict(r) for r in rows])))
                            if len(pending) >= workers * 2:
                                commit_result(pending.popleft().result())
                        while pending:
                            commit_result(pending.popleft().result())
            progress("Creating explorer indexes and verifying row counts...", flush=True)
            explorer_indexes(out)
            actual = out.execute("SELECT count(*) FROM predictions").fetchone()[0]
            if actual != total:
                raise RuntimeError(f"Coverage check failed: {actual:,} predictions for {total:,} source rows.")
            for model in MODEL_NAMES:
                n = out.execute("SELECT sum(n) FROM monthly WHERE model=?", (model,)).fetchone()[0]
                if n != total:
                    raise RuntimeError(f"Aggregate mismatch for {model}: {n} != {total}")
            with out:
                put_meta(out, "status", "complete")
                put_meta(out, "completed_at", datetime.now(timezone.utc).isoformat())
                put_meta(out, "truncated_records", out.execute("SELECT sum(truncated) FROM predictions").fetchone()[0])
            out.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            progress(f"Complete: {total:,} records, all five sentiment models, LDA and monthly trends.", flush=True)
        except BaseException as exc:
            with out:
                put_meta(out, "status", "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
                put_meta(out, "last_error", f"{type(exc).__name__}: {exc}")
            raise


def evaluate_annotations(source, output, artifacts):
    """Independent human-label evaluation; training-text overlap is excluded."""
    bundle = joblib.load(Path(artifacts) / "bundle.joblib")
    with connect(output) as c:
        labels = [dict(r) for r in c.execute("SELECT * FROM annotations")]
    rows, skipped = [], 0
    with connect(source) as c:
        for row in labels:
            text = c.execute(record_query(row["kind"]) + " WHERE rowid=?", (row["source_rowid"],)).fetchone()
            if not text:
                continue
            if fingerprint(sentiment_text(text["raw_text"])) in bundle["training_fingerprints"]:
                skipped += 1
                continue
            rows.append((text["raw_text"], text["text_clean"], row["label"]))
    if not rows:
        raise ValueError("No independent human labels available. Annotate unseen records in the dashboard first.")
    from . import LABELS
    truth = [LABELS.index(row[2]) for row in rows]
    with threadpool_limits(limits=2):
        predictions = predict_texts(bundle, [r[0] for r in rows], [r[1] for r in rows])
    result = {"label_source": "human annotations", "n": len(rows), "excluded_training_overlap": skipped,
              "class_counts": {label: truth.count(i) for i, label in enumerate(LABELS)},
              "models": {name: metric_result(truth, predictions[name]) for name in MODEL_NAMES},
              "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "note": "User-selected annotations are not a representative random test set."}
    write_json(Path(artifacts) / "human_evaluation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "analyze", "run", "status", "evaluate"])
    parser.add_argument("--source", default="reddit_clean.db")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="reddit_analysis.db")
    parser.add_argument("--sample-size", type=int, default=60000)
    parser.add_argument("--topics", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--batch-size", type=int, default=2000)
    args = parser.parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.sample_size < 300 or any(k < 2 for k in args.topics):
        parser.error("Workers and batches must be positive; samples >= 300 and topic counts >= 2.")
    if args.command == "status":
        print(json.dumps(metadata(args.output), indent=2))
        return
    with exclusive_lock(Path(args.artifacts) / "pipeline.lock"):
        if args.command == "train" or (args.command == "run" and not (Path(args.artifacts) / "bundle.joblib").exists()):
            train(args.source, args.artifacts, args.sample_size, tuple(args.topics))
        if args.command in ("run", "analyze"):
            analyze(args.source, args.artifacts, args.output, args.workers, args.batch_size)
        if args.command == "evaluate":
            print(json.dumps(evaluate_annotations(args.source, args.output, args.artifacts), indent=2))


if __name__ == "__main__":
    main()
