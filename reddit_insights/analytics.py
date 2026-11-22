"""Parameterized, read-only dashboard queries and transparent trend formulas."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import LABELS, MODEL_NAMES
from .storage import TABLES, connect, record_query
from .text import sentiment_text


def where_clause(filters, alias="", include_model=True):
    prefix = f"{alias}." if alias else ""
    clauses, params = [], []
    if include_model:
        model = filters.get("model", "vader")
        if model not in MODEL_NAMES:
            raise ValueError("Unknown sentiment model")
        clauses.append(prefix + "model=?")
        params.append(model)
    for key in ("subreddit", "category", "kind", "topic"):
        values = filters.get(key)
        if values is not None:
            if not values:
                clauses.append("0")
            else:
                clauses.append(prefix + key + " IN (" + ",".join("?" for _ in values) + ")")
                params.extend(values)
    if filters.get("start"):
        clauses.append(prefix + "year_month>=?")
        params.append(filters["start"])
    if filters.get("end"):
        clauses.append(prefix + "year_month<=?")
        params.append(filters["end"])
    return " AND ".join(clauses) or "1", params


def aggregate(path, filters):
    clause, params = where_clause(filters)
    with connect(path) as c:
        return pd.read_sql_query(f"SELECT * FROM monthly WHERE {clause}", c, params=params)


def totals(frame):
    if frame.empty:
        return {"n": 0, "negative": 0, "neutral": 0, "positive": 0, "net": 0, "avg_score": None}
    labels = frame.groupby("sentiment").n.sum()
    n = int(frame.n.sum())
    result = {label: int(labels.get(i, 0)) for i, label in enumerate(LABELS)}
    score_n = int(frame.score_n.sum())
    return {"n": n, **result, "net": 100 * (result["positive"] - result["negative"]) / n,
            "avg_score": float(frame.score_sum.sum() / score_n) if score_n else None}


def sentiment_breakdown(frame, group):
    if frame.empty:
        return pd.DataFrame(columns=[group, "negative", "neutral", "positive", "n", "net"])
    pivot = frame.groupby([group, "sentiment"]).n.sum().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    pivot.columns = LABELS
    pivot["n"] = pivot.sum(axis=1)
    pivot["net"] = 100 * (pivot.positive - pivot.negative) / pivot.n
    for label in LABELS:
        pivot[label + "_pct"] = 100 * pivot[label] / pivot.n
    return pivot.reset_index()


def timeline(frame, start, end, min_records=30):
    counts = sentiment_breakdown(frame, "year_month")
    months = pd.DataFrame({"year_month": pd.period_range(start, end, freq="M").astype(str)})
    data = months.merge(counts, on="year_month", how="left")
    for col in [*LABELS, "n"]:
        data[col] = data[col].fillna(0).astype(int)
    for label in LABELS:
        data[label + "_pct"] = np.where(data.n > 0, 100 * data[label] / data.n, np.nan)
    data["net"] = np.where(data.n > 0, 100 * (data.positive - data.negative) / data.n, np.nan)
    data["eligible"] = data.n >= min_records
    signal = data.net.where(data.eligible)
    data["rolling_net"] = signal.rolling(3, min_periods=2).mean()
    baseline = signal.shift(1).rolling(6, min_periods=3)
    data["baseline"] = baseline.mean()
    deviation = baseline.std(ddof=0)
    data["z_score"] = (signal - data.baseline) / deviation.replace(0, np.nan)
    data["change_pp"] = signal.diff()
    data["anomaly"] = data.eligible & data.z_score.abs().ge(2) & data.change_pp.abs().ge(5)
    return data


def emerging_topics(frame, previous, current):
    if frame.empty:
        return pd.DataFrame()
    part = frame[frame.year_month.isin([previous, current])]
    pivot = part.groupby(["topic", "year_month"]).n.sum().unstack(fill_value=0)
    pivot = pivot.reindex(columns=[previous, current], fill_value=0)
    prior_total, current_total = pivot[previous].sum(), pivot[current].sum()
    if not prior_total or not current_total:
        return pd.DataFrame()
    result = pd.DataFrame({"topic": pivot.index, "previous_n": pivot[previous].values, "current_n": pivot[current].values})
    result["previous_share"] = 100 * result.previous_n / prior_total
    result["current_share"] = 100 * result.current_n / current_total
    result["change_pp"] = result.current_share - result.previous_share
    return result.sort_values("change_pp", ascending=False)


def explore(output, source, filters, search="", sentiment=None, min_score=None, page=0, page_size=40):
    model = filters.get("model", "vader")
    if model not in MODEL_NAMES:
        raise ValueError("Unknown model")
    clause, params = where_clause(filters, "p", include_model=False)
    # Give the chronological index timestamp bounds as well as month labels.
    # This avoids scanning newer years when exploring an early month.
    if filters.get("start"):
        clause += " AND p.created_utc>=?"
        params.append(int(pd.Timestamp(filters["start"], tz="UTC").timestamp()))
    if filters.get("end"):
        clause += " AND p.created_utc<?"
        end = pd.Timestamp(filters["end"], tz="UTC") + pd.offsets.MonthBegin(1)
        params.append(int(end.timestamp()))
    if sentiment is not None:
        clause += f" AND p.{model}=?"
        params.append(int(sentiment))
    if min_score is not None:
        clause += " AND p.score>=?"
        params.append(int(min_score))
    with connect(output) as c:
        c.execute("ATTACH DATABASE ? AS archive", (Path(source).resolve().as_uri() + "?mode=ro",))
        pieces, all_params = [], []
        for kind, table in TABLES.items():
            raw = "s.title_raw || '. ' || s.selftext_raw" if kind == "post" else "s.body_raw"
            post_id = "s.id" if kind == "post" else "s.post_id"
            sql = f"""SELECT p.*, {raw} AS raw_text, s.text_clean, {post_id} AS post_id,
                p.{model} AS sentiment FROM predictions p INDEXED BY predictions_kind_created
                JOIN archive.{table} s ON s.rowid=p.source_rowid
                WHERE p.kind=? AND {clause}"""
            values = [kind, *params]
            if search:
                escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                sql += " AND s.text_clean LIKE ? ESCAPE '\\'"
                values.append("%" + escaped.lower() + "%")
            # Limit each sorted branch before the union and global ordering.
            # At most 2*(offset+page_size) rows enter the final merge, while
            # predicates (including text search) still apply to the whole scope.
            sql += " ORDER BY p.created_utc DESC,p.source_rowid LIMIT ?"
            values.append((max(0, page) + 1) * page_size)
            pieces.append("SELECT * FROM (" + sql + ")")
            all_params.extend(values)
        sql = "SELECT * FROM (" + " UNION ALL ".join(pieces) + ") ORDER BY created_utc DESC,kind,source_rowid LIMIT ? OFFSET ?"
        rows = pd.read_sql_query(sql, c, params=[*all_params, page_size, max(0, page) * page_size])
    if not rows.empty:
        rows["label"] = rows.sentiment.map(dict(enumerate(LABELS)))
        rows["date"] = pd.to_datetime(rows.created_utc, unit="s", utc=True)
        rows["evidence_id"] = rows.kind + ":" + rows.reddit_id
    return rows


def save_annotation(path, kind, source_rowid, label):
    if kind not in TABLES or label not in LABELS:
        raise ValueError("Invalid annotation")
    with connect(path, False) as c:
        if not c.execute("SELECT 1 FROM predictions WHERE kind=? AND source_rowid=?", (kind, source_rowid)).fetchone():
            raise ValueError("Record does not exist in the analysis")
        with c:
            c.execute("""INSERT INTO annotations VALUES (?,?,?,?) ON CONFLICT(kind,source_rowid)
                DO UPDATE SET label=excluded.label,updated_at=excluded.updated_at""",
                      (kind, int(source_rowid), label, datetime.now(timezone.utc).isoformat()))


def evidence_package(frame, filters, manifest, records=None):
    trend = timeline(frame, filters["start"], filters["end"])
    data = {"filters": filters, "overview": totals(frame),
            "trends": trend[["year_month", "n", "net", "change_pp", "anomaly"]].tail(24).replace({np.nan: None}).to_dict("records"),
            "communities": sentiment_breakdown(frame, "subreddit").sort_values("n", ascending=False).head(15).to_dict("records"),
            "topics": manifest["topics"], "method": "VADER weak supervision; no human-labelled benchmark",
            "limitations": manifest["limitations"], "evidence": []}
    if records is not None and not records.empty:
        data["evidence"] = [{"id": r.evidence_id, "text": sentiment_text(r.raw_text)[:900],
                             "sentiment": r.label, "subreddit": r.subreddit} for r in records.itertuples()]
    return data


def csv_bytes(frame):
    """Avoid spreadsheet formula execution when exporting untrusted Reddit text."""
    copy = frame.copy()
    for col in copy.select_dtypes(include=["object", "str", "string"]).columns:
        copy[col] = copy[col].map(lambda value: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value)
    return copy.to_csv(index=False).encode("utf-8-sig")
