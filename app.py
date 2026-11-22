"""Reddit Insight Studio: real corpus analytics, model evidence and local inference."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from threadpoolctl import threadpool_limits

from reddit_insights import LABELS, MODEL_NAMES
from reddit_insights.analytics import (aggregate, csv_bytes, emerging_topics, evidence_package, explore,
                                       save_annotation, sentiment_breakdown, timeline, totals)
from reddit_insights.assistant import explain
from reddit_insights.jobs import is_running, log_tail, start_pipeline
from reddit_insights.modeling import predict_texts
from reddit_insights.pipeline import evaluate_annotations
from reddit_insights.storage import connect, metadata

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get("REDDIT_SOURCE_DB", ROOT / "reddit_clean.db")).resolve()
OUTPUT = Path(os.environ.get("REDDIT_ANALYSIS_DB", ROOT / "reddit_analysis.db")).resolve()
ARTIFACTS = Path(os.environ.get("REDDIT_ARTIFACTS", ROOT / "artifacts")).resolve()
COLORS = {"positive": "#27856A", "neutral": "#8B9AA5", "negative": "#CE614F"}
PAGES = ["Overview", "Sentiment & trends", "Topics", "Communities", "Explore records", "Model laboratory", "AI assistant", "Data & pipeline"]

st.set_page_config(page_title="Reddit Insight Studio", page_icon=":material/forum:", layout="wide")


@st.cache_data(ttl=10, max_entries=4)
def load_meta(path):
    return metadata(path)


@st.cache_data(ttl=30, max_entries=48)
def load_aggregate(path, filters_json, revision):
    return aggregate(path, json.loads(filters_json))


@st.cache_resource(max_entries=2)
def load_bundle(path, stamp):
    return joblib.load(path)


def read_json(name):
    path = ARTIFACTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def chart(fig, key, selectable=False):
    fig.update_layout(margin=dict(l=8, r=8, t=30, b=8), height=350,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.2, x=0), font=dict(size=12))
    if selectable:
        return st.plotly_chart(fig, width="stretch", key=key, on_select="rerun", selection_mode="points")
    st.plotly_chart(fig, width="stretch", key=key)


def topic_names(manifest):
    return {-1: "Unassigned", **{t["id"]: t["name"] for t in manifest["topics"]}}


def download(frame, name, label="Download CSV"):
    st.download_button(label, csv_bytes(frame), name + ".csv", "text/csv", icon=":material/download:", key=name)


def show_records(filters, key, search="", sentiment=None, minimum=None, page=0, size=40):
    try:
        rows = explore(OUTPUT, SOURCE, filters, search, sentiment, minimum, page, size)
    except ValueError as exc:
        st.info(str(exc))
        return
    if rows.empty:
        st.info("No records match these filters.")
        return
    view = rows[["date", "subreddit", "kind", "label", "score", "raw_text"]].rename(columns={"raw_text": "Text", "label": "Sentiment"})
    event = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row", key=key,
                         column_config={"Text": st.column_config.TextColumn(width="large"),
                                        "date": st.column_config.DatetimeColumn("Date (UTC)", format="YYYY-MM-DD HH:mm")})
    st.caption("Select a row to inspect its full text, model predictions, and topic.")
    if event.selection.rows:
        row = rows.iloc[event.selection.rows[0]]
        with st.container(border=True):
            st.subheader(f"r/{row.subreddit} · {row.kind}")
            st.text(row.raw_text)
            st.caption(f"Evidence ID: {row.evidence_id}")
            if row.truncated:
                st.warning("Only the first 8,000 characters after light cleaning were used for sentiment inference.")
            cols = st.columns(5)
            for col, (model, label) in zip(cols, MODEL_NAMES.items()):
                col.metric(label, LABELS[int(row[model])])
            st.caption(f"VADER compound: {row.compound:.3f} · Dominant topic: {int(row.topic)} · Topic probability: {row.topic_probability:.2f}")
            with st.expander("Cleaned text"):
                st.text(row.text_clean)
            with st.form(key + "_annotation"):
                human = st.selectbox("Your sentiment label", LABELS, index=1)
                if st.form_submit_button("Save human annotation"):
                    save_annotation(OUTPUT, row.kind, int(row.source_rowid), human)
                    st.success("Annotation saved. Evaluate independent annotations in Model laboratory.")
    download(view, key + "_records", "Export this page")


def overview(frame, filters, manifest):
    st.caption("CORPUS OBSERVATORY")
    st.title("What is Reddit talking about?")
    st.write("Follow sentiment, discover discussion themes, and trace every insight back to the conversation.")
    summary = totals(frame)
    series = timeline(frame, filters["start"], filters["end"])
    with st.container(horizontal=True):
        st.metric("Analyzed records", f"{summary['n']:,}", border=True, chart_data=series.n.tolist(), chart_type="bar")
        for label in ["positive", "neutral", "negative"]:
            st.metric(label.title() + " share", f"{100 * summary[label] / max(summary['n'], 1):.1f}%", border=True)
        st.metric("Net sentiment", f"{summary['net']:+.1f} pp", border=True,
                  help="Positive share minus negative share. Range: -100 to +100 percentage points.")
    left, right = st.columns([2, 1])
    with left, st.container(border=True):
        st.subheader("Sentiment through time")
        melted = series.melt(id_vars="year_month", value_vars=[label + "_pct" for label in LABELS], value_name="Share (%)", var_name="Sentiment")
        melted["Sentiment"] = melted.Sentiment.str.replace("_pct", "")
        fig = px.line(melted, x="year_month", y="Share (%)", color="Sentiment", color_discrete_map=COLORS,
                      labels={"year_month": "Month"})
        chart(fig, "overview_trends")
    with right, st.container(border=True):
        st.subheader("The overall mood")
        chart(px.pie(names=LABELS, values=[summary[l] for l in LABELS], hole=0.7,
                     color=LABELS, color_discrete_map=COLORS), "overview_mood")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Most represented communities")
        communities = sentiment_breakdown(frame, "subreddit").sort_values("n").tail(10)
        chart(px.bar(communities, x="n", y="subreddit", orientation="h", color="net",
                     color_continuous_scale=[COLORS["negative"], COLORS["neutral"], COLORS["positive"]],
                     range_color=[-100, 100], labels={"n": "Collected records", "subreddit": "", "net": "Net sentiment"}), "overview_communities")
    with right, st.container(border=True):
        st.subheader("Discussion themes")
        topics = frame.groupby("topic", as_index=False).n.sum().sort_values("n").tail(8)
        topics["Theme"] = topics.topic.map(topic_names(manifest))
        chart(px.bar(topics, x="n", y="Theme", orientation="h", color_discrete_sequence=["#6486B2"], labels={"n": "Collected records"}), "overview_topics")
    st.caption("Historical archive · Monthly collection quotas · Sentiment reflects text tone, not factual accuracy or population opinion.")


def trends(frame, filters, manifest):
    st.title("Sentiment & trends")
    st.write("Compare sentiment shares across months and inspect unusually large changes.")
    minimum = st.slider("Minimum records per month for trend alerts", 10, 500, 30, 10)
    series = timeline(frame, filters["start"], filters["end"], minimum)
    mode = st.segmented_control("Chart", ["Sentiment share", "Record count", "Net sentiment"], default="Sentiment share")
    if mode == "Net sentiment":
        fig = go.Figure()
        fig.add_scatter(x=series.year_month, y=series.net, name="Net sentiment", mode="lines+markers", line_color="#27856A")
        fig.add_scatter(x=series.year_month, y=series.rolling_net, name="3-month mean", line=dict(color="#D5A34C", dash="dash"))
        fig.update_yaxes(title="Positive minus negative share (pp)")
    else:
        columns = LABELS if mode == "Record count" else [l + "_pct" for l in LABELS]
        data = series.melt(id_vars="year_month", value_vars=columns, var_name="Sentiment", value_name="Value")
        data.Sentiment = data.Sentiment.str.replace("_pct", "")
        fig = px.line(data, x="year_month", y="Value", color="Sentiment", markers=True, color_discrete_map=COLORS,
                      labels={"year_month": "Month", "Value": "Collected records" if mode == "Record count" else "Share (%)"})
    selection = chart(fig, "trend_chart", selectable=True)
    st.caption("Click a month to inspect its records. Gaps mean no analyzed records; they are not neutral sentiment.")
    points = selection.selection.points
    if points:
        month = str(points[-1]["x"])[:7]
        st.subheader(f"Conversations in {month}")
        show_records({**filters, "start": month, "end": month}, "trend_drilldown", size=20)
    anomalies = series[series.anomaly]
    st.subheader("Changes worth investigating")
    if anomalies.empty:
        st.info("No months exceed the alert thresholds for these filters.")
    else:
        st.dataframe(anomalies[["year_month", "n", "net", "change_pp", "z_score"]], hide_index=True)
    st.caption("Alerts require at least 3 preceding observed months, an absolute z-score of 2 against the prior 6 months, and a change of at least 5 percentage points. These are descriptive flags, not significance tests or causal explanations. Boundary months may be partial.")
    download(series, "sentiment_trends")


def topics_page(frame, filters, manifest):
    st.title("Topic atlas")
    st.write("LDA discovers co-occurring words. Topic names below are descriptive keyword labels.")
    names = topic_names(manifest)
    grouped = sentiment_breakdown(frame, "topic")
    grouped["Theme"] = grouped.topic.map(names)
    chart(px.bar(grouped, x="Theme", y=LABELS, barmode="stack", color_discrete_map=COLORS,
                 labels={"value": "Records", "variable": "Sentiment"}), "topic_sentiments")
    topic_time = frame.groupby(["year_month", "topic"], as_index=False).n.sum()
    topic_time["Share (%)"] = 100 * topic_time.n / topic_time.groupby("year_month").n.transform("sum")
    topic_time["Theme"] = topic_time.topic.map(names)
    chart(px.line(topic_time, x="year_month", y="Share (%)", color="Theme",
                  labels={"year_month": "Month"}), "topic_timeline")
    months = sorted(frame.year_month.unique())
    if len(months) > 1:
        left, right = st.columns(2)
        previous = left.selectbox("Compare from", months[:-1], index=len(months) - 2)
        current = right.selectbox("Compare to", [m for m in months if m > previous], index=len([m for m in months if m > previous]) - 1)
        growth = emerging_topics(frame, previous, current)
        if not growth.empty:
            growth["Theme"] = growth.topic.map(names)
            st.subheader("Change in share of discussion")
            st.dataframe(growth[["Theme", "previous_n", "current_n", "previous_share", "current_share", "change_pp"]], hide_index=True,
                         column_config={"change_pp": st.column_config.NumberColumn("Change (pp)", format="%+.2f"),
                                        "previous_share": st.column_config.NumberColumn("Previous (%)", format="%.2f"),
                                        "current_share": st.column_config.NumberColumn("Current (%)", format="%.2f")})
            st.caption("Share changes are relative to the filtered, collected corpus. Check Data & pipeline for partial boundary months.")
    chosen = st.selectbox("Inspect a topic", list(names), format_func=lambda k: f"{k + 1}. {names[k]}" if k >= 0 else names[k])
    if chosen >= 0:
        detail = next(t for t in manifest["topics"] if t["id"] == chosen)
        st.write(" · ".join(detail["words"]))
        chart(px.bar(x=detail["words"], y=detail["weights"], labels={"x": "Keyword", "y": "Topic word probability"}), "topic_keywords")
    show_records({**filters, "topic": [chosen]}, "topic_examples", size=20)


def communities_page(frame, filters, manifest):
    st.title("Community comparison")
    breakdown = sentiment_breakdown(frame, "subreddit").sort_values("n", ascending=False)
    chosen = st.multiselect("Compare communities", breakdown.subreddit.tolist(), default=breakdown.subreddit.head(5).tolist())
    selected = breakdown[breakdown.subreddit.isin(chosen)]
    if selected.empty:
        st.info("Select at least one community.")
        return
    melted = selected.melt(id_vars="subreddit", value_vars=[l + "_pct" for l in LABELS], var_name="Sentiment", value_name="Share (%)")
    melted.Sentiment = melted.Sentiment.str.replace("_pct", "")
    chart(px.bar(melted, x="subreddit", y="Share (%)", color="Sentiment", color_discrete_map=COLORS, barmode="group"), "community_shares")
    subframe = frame[frame.subreddit.isin(chosen)]
    time = subframe.groupby(["subreddit", "year_month", "sentiment"], as_index=False).n.sum()
    time["signed"] = time.n * time.sentiment.map({0: -1, 1: 0, 2: 1})
    time = time.groupby(["subreddit", "year_month"], as_index=False)[["signed", "n"]].sum()
    time["Net sentiment"] = 100 * time.signed / time.n
    chart(px.line(time, x="year_month", y="Net sentiment", color="subreddit", markers=True), "community_net")
    st.dataframe(selected, hide_index=True)
    engagement = subframe.groupby("subreddit", as_index=False)[["score_sum", "score_n"]].sum()
    engagement["Mean archived score"] = engagement.score_sum / engagement.score_n.replace(0, float("nan"))
    chart(px.bar(engagement, x="subreddit", y="Mean archived score"), "community_engagement")
    st.caption("Mean score uses records with a non-missing archived score. Scores are heavily skewed and are not current vote counts.")
    download(selected, "community_comparison")


def explorer_page(frame, filters, manifest):
    st.title("Explore the conversations")
    with st.form("record_search"):
        a, b, c, d = st.columns([3, 1, 1, 1])
        query = a.text_input("Search cleaned text", max_chars=200, placeholder="battery life, artificial intelligence…")
        label = b.selectbox("Sentiment", ["All", *LABELS])
        score = c.number_input("Minimum score", value=-100000, step=10)
        page = d.number_input("Page", min_value=1, max_value=10000, value=1)
        st.form_submit_button("Search", type="primary")
    st.caption("40 records per page, newest first. Text searches work best with a narrow date range or subreddit.")
    show_records(filters, "explorer", query, None if label == "All" else LABELS.index(label), score, page - 1)


def model_page(frame, filters, manifest):
    st.title("Model laboratory")
    st.warning(manifest["evaluation_note"])
    evaluation = [{"Model": MODEL_NAMES[name], "Accuracy (weak labels)": r["test"]["accuracy"],
                   "Macro precision": r["test"]["report"]["macro avg"]["precision"],
                   "Macro recall": r["test"]["report"]["macro avg"]["recall"],
                   "Macro F1": r["test"]["macro_f1"], "Training seconds": r["training_seconds"]}
                  for name, r in manifest["models"].items()]
    table = pd.DataFrame(evaluation)
    st.dataframe(table, hide_index=True)
    st.caption(f"VADER generates the labels and is not benchmarked against itself. Training sample: {manifest['sample_size']:,} distinct texts. Split: 65% training / 17% validation / 18% later test records. Best student by validation macro F1: {MODEL_NAMES[manifest['best_student']]}.")
    selected = st.selectbox("Inspect classifier", list(manifest["models"]), format_func=MODEL_NAMES.get)
    report = manifest["models"][selected]
    left, right = st.columns(2)
    with left:
        chart(px.imshow(report["test"]["confusion_matrix"], x=LABELS, y=LABELS, text_auto=True,
                        labels={"x": "Prediction", "y": "VADER weak label", "color": "Records"}, color_continuous_scale="Greens"), "confusion")
    with right:
        st.subheader("Class-by-class results")
        st.dataframe(pd.DataFrame({l: report["test"]["report"][l] for l in LABELS}).T)
        for note in report["warnings"]:
            st.info(note)
    with st.expander("Disagreements on the held-out test set"):
        errors = read_json("errors.json") or []
        st.dataframe(pd.DataFrame([e for e in errors if e["model"] == selected]), hide_index=True)
    st.subheader("Topic model selection")
    scores = pd.DataFrame(manifest["topic_selection"])
    left, right = st.columns(2)
    with left:
        chart(px.line(scores, x="k", y="npmi", markers=True, labels={"k": "Topics", "npmi": "Held-out NPMI coherence"}), "coherence")
    with right:
        chart(px.line(scores, x="k", y="perplexity", markers=True, labels={"k": "Topics", "perplexity": "Held-out perplexity (lower is better)"}), "perplexity")
    st.caption(f"Selected {len(manifest['topics'])} topics by highest validation NPMI. LDA uses word counts with stopword removal; classifiers use TF-IDF with negation preserved. The neural network is a scikit-learn MLP with 64 and 32 hidden units, trained on CPU.")
    st.subheader("Try your own text")
    with st.form("predict_text"):
        text = st.text_area("Text to analyze", placeholder="The camera is great, but the battery is terrible.", max_chars=20000)
        submitted = st.form_submit_button("Analyze with all models", type="primary")
    if submitted and text.strip():
        bundle = load_bundle(str(ARTIFACTS / "bundle.joblib"), (ARTIFACTS / "bundle.joblib").stat().st_mtime_ns)
        with threadpool_limits(limits=2):
            predictions = predict_texts(bundle, [text])
        st.dataframe(pd.DataFrame({"Model": list(MODEL_NAMES.values()),
                                   "Prediction": [LABELS[int(predictions[name][0])] for name in MODEL_NAMES]}), hide_index=True)
        st.caption(f"VADER compound: {predictions['compound'][0]:.3f} · Topic: {topic_names(manifest)[int(predictions['topic'][0])]} · Topic probability: {predictions['topic_probability'][0]:.2f}")
        if predictions["truncated"][0]:
            st.info("Text was truncated to 8,000 characters for inference.")
    st.subheader("Human evaluation")
    st.write("Save labels in Explore records, then evaluate them here. Training texts are excluded automatically.")
    if st.button("Evaluate saved human annotations"):
        try:
            with st.spinner("Evaluating independent annotations..."):
                result = evaluate_annotations(SOURCE, OUTPUT, ARTIFACTS)
            st.success(f"Evaluated {result['n']} independently annotated records.")
        except ValueError as exc:
            st.info(str(exc))
    human = read_json("human_evaluation.json")
    if human:
        st.caption(f"{human['n']} human-labelled examples · {human['excluded_training_overlap']} overlapping training texts excluded · {human['note']}")
        st.dataframe(pd.DataFrame([{"Model": MODEL_NAMES[k], "Accuracy": r["accuracy"], "Macro F1": r["macro_f1"]} for k, r in human["models"].items()]), hide_index=True)
    download(table, "model_comparison")


def secret(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def assistant_page(frame, filters, manifest):
    st.title("Ask the archive")
    st.write("Get an explanation grounded in the active filters and computed statistics.")
    key, model = secret("GEMINI_API_KEY"), secret("GEMINI_MODEL", "gemini-3.5-flash")
    summary = totals(frame)
    st.info(f"Current evidence: {summary['n']:,} records · {filters['start']} to {filters['end']} · {MODEL_NAMES[filters['model']]}")
    if not key:
        st.warning("The AI assistant needs an API key. All analytics and models work without one.")
        st.code('GEMINI_API_KEY = "your-key"\nGEMINI_MODEL = "gemini-3.5-flash"', language="toml")
        st.caption("Place these settings in .streamlit/secrets.toml. The example file is included; never commit the real key.")
    with st.form("ask_archive"):
        question = st.text_area("Question", placeholder="How did negative sentiment change over the selected period?", max_chars=2000)
        submitted = st.form_submit_button("Generate explanation", type="primary", disabled=not bool(key))
    if submitted:
        try:
            with st.spinner("Preparing evidence and generating an explanation..."):
                package = evidence_package(frame, filters, manifest)
                package["coverage_status"] = load_meta(str(OUTPUT)).get("status")
                import hashlib
                request_hash = hashlib.sha256(json.dumps([model, question, package], sort_keys=True).encode()).hexdigest()
                cache = st.session_state.setdefault("ai_cache", {})
                if request_hash in cache:
                    result = cache[request_hash]
                else:
                    result = explain(question, package, key, model)
                    if len(cache) >= 12:
                        cache.pop(next(iter(cache)))
                    cache[request_hash] = result
                st.session_state["ai_answer"] = (json.dumps(filters, sort_keys=True), result)
        except Exception as exc:
            # API messages can contain request details; show a safe, actionable error.
            st.error(str(exc) if isinstance(exc, ValueError) else f"The API request failed ({type(exc).__name__}). Check the key, model access, connection, and account quota.")
    saved = st.session_state.get("ai_answer")
    if saved and saved[0] == json.dumps(filters, sort_keys=True):
        result = saved[1]
        with st.chat_message("assistant"):
            st.write(result["answer"])
            for finding in result["findings"]:
                st.write(finding["text"])
                st.caption("Evidence: " + ", ".join(finding["references"]))
            for limitation in result["limitations"]:
                st.caption(limitation)
            if result["follow_up_questions"]:
                st.write("Suggested questions")
                for follow_up in result["follow_up_questions"]:
                    st.write("• " + follow_up)


def pipeline_page(meta, manifest):
    st.title("Data & pipeline")
    st.write("Training artifacts, archive coverage, and resumable full-corpus analysis.")
    running = is_running(ARTIFACTS)
    if meta:
        progress = meta.get("progress", {})
        total = sum(meta["source"]["counts"].values())
        completed = progress.get("processed", 0)
        st.progress(min(completed / max(total, 1), 1), text=f"{completed:,} / {total:,} records · {meta.get('status', 'pending')}")
        if running and progress.get("records_per_second"):
            st.caption(f"{progress['records_per_second']:,.0f} records/sec · Estimated remaining: {progress['eta_seconds'] / 60:.0f} minutes")
        if meta.get("last_error"):
            st.error(meta["last_error"])
    else:
        st.info("Train the models and analyze the archive to populate the dashboard.")
    if st.button("Start / resume full pipeline", type="primary", disabled=running or meta.get("status") == "complete"):
        try:
            pid = start_pipeline(ROOT, SOURCE, ARTIFACTS, OUTPUT)
            st.success(f"Pipeline started (process {pid}). You can leave this page while it runs.")
        except (OSError, RuntimeError) as exc:
            st.error(str(exc))
    if running:
        st.info("The pipeline is running in the background. Refresh to update progress.")
    if st.button("Refresh pipeline status"):
        st.cache_data.clear()
        st.rerun()
    with st.expander("Processing log"):
        st.code(log_tail(ARTIFACTS), language=None)
    st.code(".venv\\Scripts\\python.exe -m reddit_insights.pipeline run", language="powershell")
    if manifest:
        with st.expander("Training provenance and settings"):
            st.json({k: manifest[k] for k in ["created_at", "seed", "sample_size", "label_source", "split", "training_seconds", "topic_selection"]})
        for limitation in manifest["limitations"]:
            st.caption("• " + limitation)
    if meta:
        source = meta["source"]
        prep = source.get("preprocessing", {})
        st.subheader("Cleaning and data quality")
        st.dataframe(pd.DataFrame([{"Type": kind, "Clean records": n,
                                   "After placeholder removal": int(prep.get(kind + "s_read", n)),
                                   "Short records removed": int(prep.get(kind + "s_skipped_short", 0))}
                                  for kind, n in source["counts"].items()]), hide_index=True)
        times = [b for b in source["bounds"].values() if b[2] is not None]
        start, end = min(b[2] for b in times), max(b[3] for b in times)
        st.caption(f"Archive coverage (UTC): {pd.to_datetime(start, unit='s', utc=True)} to {pd.to_datetime(end, unit='s', utc=True)}. First and last months are partial.")
        base_filters = {"model": "vader"}
        coverage = load_aggregate(str(OUTPUT), json.dumps(base_filters), meta.get("progress", {}).get("processed", 0))
        if not coverage.empty:
            coverage = coverage.groupby(["subreddit", "year_month"]).n.sum().unstack(fill_value=0)
            chart(px.imshow(coverage, aspect="auto", color_continuous_scale="Greens", labels={"color": "Analyzed records", "x": "Month", "y": "Subreddit"}), "coverage")
        st.caption(f"Records truncated for sentiment inference: {meta.get('truncated_records', 'available after completion')}")


def main():
    manifest = read_json("manifest.json")
    meta = load_meta(str(OUTPUT)) if OUTPUT.exists() else {}
    with st.sidebar:
        st.title(":material/forum: Reddit Insights")
        st.caption("DISCUSSION ANALYSIS STUDIO")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
    if page == "Data & pipeline" or not manifest or not meta:
        pipeline_page(meta, manifest)
        return
    source = meta["source"]
    if Path(source["path"]).resolve() != SOURCE:
        st.error("The analysis database was built from a different source. Set REDDIT_SOURCE_DB to its recorded source path.")
        return
    bounds = [b for b in source["bounds"].values() if b[2] is not None]
    first = pd.to_datetime(min(b[2] for b in bounds), unit="s", utc=True).strftime("%Y-%m")
    last = pd.to_datetime(max(b[3] for b in bounds), unit="s", utc=True).strftime("%Y-%m")
    months = pd.period_range(first, last, freq="M").astype(str).tolist()
    with st.sidebar, st.form("global_filters", border=False):
        st.subheader("Scope your analysis")
        model = st.selectbox("Sentiment model", list(MODEL_NAMES), format_func=MODEL_NAMES.get)
        date_range = st.select_slider("Month range (UTC)", options=months, value=(first, last))
        kinds = st.multiselect("Record types", ["post", "comment"], default=["post", "comment"])
        categories = st.multiselect("Domains", ["technology", "science", "news_general", "other"],
                                   default=["technology", "science", "news_general", "other"])
        subs = st.multiselect("Subreddits (empty = all)", manifest["sample_subreddits"])
        st.form_submit_button("Apply filters", type="primary", width="stretch")
    filters = {"model": model, "start": date_range[0], "end": date_range[1], "kind": kinds, "category": categories}
    if subs:
        filters["subreddit"] = subs
    if meta.get("status") != "complete":
        progress = meta.get("progress", {})
        st.warning(f"Partial analysis: {progress.get('processed', 0):,} of {sum(source['counts'].values()):,} records processed. Trends may be biased until processing finishes.")
    frame = load_aggregate(str(OUTPUT), json.dumps(filters, sort_keys=True), meta.get("progress", {}).get("processed", 0))
    st.sidebar.caption(f"{MODEL_NAMES[model]} · {date_range[0]} — {date_range[1]}")
    st.sidebar.caption("Trained classifiers use VADER-generated labels.")
    if frame.empty:
        st.info("No analyzed records match these filters. Broaden the scope or check Data & pipeline.")
        return
    render = {"Overview": overview, "Sentiment & trends": trends, "Topics": topics_page,
              "Communities": communities_page, "Explore records": explorer_page,
              "Model laboratory": model_page, "AI assistant": assistant_page}
    render[page](frame, filters, manifest)


if __name__ == "__main__":
    main()
