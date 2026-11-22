"""
Reddit Sentiment Analysis Engine — Visualization Pipeline (Polished)
===================================================================
Generates 14 clean, non-overlapping static PNG charts from reddit_clean.db.
All charts use a consistent dark theme with vibrant accent colors.
"""

import os
import sqlite3
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CLEAN_DB   = "reddit_clean.db"
RAW_DB     = "reddit_8million.db"
OUTPUT_DIR = "visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Dark Theme Setup
# ---------------------------------------------------------------------------

BG_DARK     = "#0d1117"
BG_CARD     = "#161b22"
TEXT_COLOR   = "#e6edf3"
TEXT_MUTED   = "#8b949e"
GRID_COLOR   = "#21262d"
ACCENT_1     = "#58a6ff"   # blue
ACCENT_2     = "#f78166"   # orange
ACCENT_3     = "#7ee787"   # green
ACCENT_4     = "#d2a8ff"   # purple
ACCENT_5     = "#ff7b72"   # red
ACCENT_6     = "#ffa657"   # amber
ACCENT_7     = "#79c0ff"   # light blue

CATEGORY_COLORS = {
    "technology":   "#58a6ff",
    "science":      "#7ee787",
    "news_general": "#f78166",
}


def apply_dark_theme():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor":   BG_CARD,
        "axes.edgecolor":   GRID_COLOR,
        "axes.labelcolor":  TEXT_COLOR,
        "text.color":       TEXT_COLOR,
        "xtick.color":      TEXT_MUTED,
        "ytick.color":      TEXT_MUTED,
        "grid.color":       GRID_COLOR,
        "grid.alpha":       0.4,
        "legend.facecolor": BG_CARD,
        "legend.edgecolor": GRID_COLOR,
        "legend.labelcolor": TEXT_COLOR,
        "font.family":      "sans-serif",
        "font.size":        11,
        "axes.titlesize":   14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "savefig.facecolor": BG_DARK,
        "savefig.edgecolor": BG_DARK,
    })

apply_dark_theme()


def save_fig(fig, name: str):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print(f"  [OK] Saved {path}")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def query_clean(sql: str, params=()) -> pd.DataFrame:
    con = sqlite3.connect(CLEAN_DB)
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    return df


def query_raw(sql: str, params=()) -> pd.DataFrame:
    con = sqlite3.connect(RAW_DB)
    df = pd.read_sql_query(sql, con, params=params)
    con.close()
    return df


# ===================================================================
# V1: Dataset Composition Bar Chart
# ===================================================================

def v1_dataset_composition():
    print("[V1] Dataset Composition …")

    posts = query_clean("SELECT subreddit, COUNT(*) AS posts FROM posts_clean GROUP BY subreddit")
    comments = query_clean("SELECT subreddit, COUNT(*) AS comments FROM comments_clean GROUP BY subreddit")
    df = posts.merge(comments, on="subreddit", how="outer").fillna(0)
    df["total"] = df["posts"] + df["comments"]
    df = df.sort_values("total", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(df))
    ax.barh(y, df["comments"], color=ACCENT_1, label="Comments", edgecolor="none")
    ax.barh(y, df["posts"], left=df["comments"], color=ACCENT_3, label="Posts", edgecolor="none")
    ax.set_yticks(y)
    ax.set_yticklabels(df["subreddit"], fontsize=9)
    ax.set_xlabel("Record Count", labelpad=10)
    ax.set_title("V1 · Dataset Composition by Subreddit", pad=15)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    save_fig(fig, "V01_dataset_composition")


# ===================================================================
# V2: Data Cleaning Funnel
# ===================================================================

def v2_cleaning_funnel():
    print("[V2] Data Cleaning Funnel …")

    raw_posts    = query_raw("SELECT COUNT(*) AS n FROM posts").iloc[0, 0]
    raw_comments = query_raw("SELECT COUNT(*) AS n FROM comments").iloc[0, 0]
    raw_total    = raw_posts + raw_comments

    del_posts = query_raw("SELECT COUNT(*) AS n FROM posts WHERE selftext IN ('[deleted]','[removed]')").iloc[0, 0]
    del_comments = query_raw("SELECT COUNT(*) AS n FROM comments WHERE body IN ('[deleted]','[removed]','')").iloc[0, 0]
    after_del = raw_total - del_posts - del_comments

    clean_posts    = query_clean("SELECT COUNT(*) AS n FROM posts_clean").iloc[0, 0]
    clean_comments = query_clean("SELECT COUNT(*) AS n FROM comments_clean").iloc[0, 0]
    clean_total    = clean_posts + clean_comments

    stages = ["Raw Dataset", "After Removing\nDeleted/Removed", "After Text Cleaning\n& Short Filtering"]
    values = [raw_total, after_del, clean_total]
    colors = [ACCENT_4, ACCENT_1, ACCENT_3]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(stages, values, color=colors, width=0.5, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + raw_total * 0.02,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=12, fontweight="bold", color=TEXT_COLOR)

    for i in range(1, len(values)):
        drop = values[i-1] - values[i]
        pct = drop / values[i-1] * 100
        mid_x = i - 0.5
        mid_y = (values[i-1] + values[i]) / 2
        ax.annotate(f"-{drop:,.0f}\n({pct:.1f}% dropped)",
                    xy=(mid_x, mid_y), fontsize=9, ha="center", va="center",
                    color=ACCENT_5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_CARD, edgecolor=ACCENT_5, alpha=0.9))

    ax.set_ylabel("Record Count", labelpad=10)
    ax.set_title("V2 · Data Cleaning Funnel", pad=15)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.set_ylim(0, raw_total * 1.18)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    save_fig(fig, "V02_cleaning_funnel")


# ===================================================================
# V3: Category Distribution Donut
# ===================================================================

def v3_category_donut():
    print("[V3] Category Distribution …")

    df = query_clean("""
        SELECT category, COUNT(*) AS n FROM (
            SELECT category FROM posts_clean
            UNION ALL
            SELECT category FROM comments_clean
        ) GROUP BY category
    """)

    labels_map = {"technology": "Technology", "science": "Science", "news_general": "News & General"}
    df["label"] = df["category"].map(labels_map)
    colors = [CATEGORY_COLORS.get(c, ACCENT_4) for c in df["category"]]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        df["n"], labels=df["label"], autopct="%1.1f%%",
        colors=colors, pctdistance=0.75, startangle=90,
        wedgeprops={"width": 0.4, "edgecolor": BG_DARK, "linewidth": 2},
        textprops={"fontsize": 12, "color": TEXT_COLOR, "fontweight": "bold"},
    )
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    total = df["n"].sum()
    ax.text(0, 0, f"{total:,.0f}\nclean records", ha="center", va="center",
            fontsize=15, fontweight="bold", color=TEXT_COLOR)

    ax.set_title("V3 · Records by Category", pad=20)
    fig.tight_layout()
    save_fig(fig, "V03_category_donut")


# ===================================================================
# V4: Monthly Volume Time Series
# ===================================================================

def v4_monthly_timeseries():
    print("[V4] Monthly Time Series …")

    df = query_clean("""
        SELECT year_month, category, COUNT(*) AS n FROM (
            SELECT year_month, category FROM posts_clean
            UNION ALL
            SELECT year_month, category FROM comments_clean
        ) GROUP BY year_month, category
        ORDER BY year_month
    """)

    pivot = df.pivot(index="year_month", columns="category", values="n").fillna(0)
    pivot["Total"] = pivot.sum(axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(len(pivot))

    for cat, color in CATEGORY_COLORS.items():
        if cat in pivot.columns:
            label = {"technology": "Technology", "science": "Science", "news_general": "News & General"}[cat]
            ax.plot(x, pivot[cat], color=color, linewidth=2, label=label, alpha=0.9)

    ax.plot(x, pivot["Total"], color=TEXT_COLOR, linewidth=2.5, label="Total Corpus", linestyle="--", alpha=0.7)

    labels = list(pivot.index)
    step = max(1, len(labels) // 12)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=45, ha="right", fontsize=9)

    ax.set_ylabel("Records per Month", labelpad=10)
    ax.set_title("V4 · Monthly Record Volume by Category (2022–2026)", pad=15)
    ax.legend(loc="upper left", framealpha=0.85, fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    save_fig(fig, "V04_monthly_timeseries")


# ===================================================================
# V5: Day-of-Week × Hour-of-Day Heatmap
# ===================================================================

def v5_activity_heatmap():
    print("[V5] Activity Heatmap …")

    df = query_clean("""
        SELECT day_of_week, hour_of_day, COUNT(*) AS n FROM (
            SELECT day_of_week, hour_of_day FROM posts_clean
            UNION ALL
            SELECT day_of_week, hour_of_day FROM comments_clean
        ) GROUP BY day_of_week, hour_of_day
    """)

    pivot = df.pivot(index="day_of_week", columns="hour_of_day", values="n").fillna(0)
    pivot = pivot.reindex(range(7)).fillna(0)
    pivot = pivot.reindex(columns=range(24)).fillna(0)

    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    sns.heatmap(
        pivot, ax=ax, cmap="YlOrRd", linewidths=0.5, linecolor=BG_DARK,
        cbar_kws={"label": "Total Records", "shrink": 0.8},
        xticklabels=[f"{h:02d}:00" for h in range(24)],
        yticklabels=day_labels,
    )
    ax.set_xlabel("Hour of Day (UTC)", labelpad=10)
    ax.set_ylabel("Day of Week", labelpad=10)
    ax.set_title("V5 · Activity Heatmap by Day & Hour (UTC)", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)
    fig.tight_layout()

    save_fig(fig, "V05_activity_heatmap")


# ===================================================================
# V6: Year-over-Year Monthly Comparison
# ===================================================================

def v6_yoy_comparison():
    print("[V6] Year-over-Year …")

    df = query_clean("""
        SELECT year, month, COUNT(*) AS n FROM (
            SELECT year, month FROM posts_clean
            UNION ALL
            SELECT year, month FROM comments_clean
        ) GROUP BY year, month
    """)

    years = sorted(df["year"].unique())
    year_colors = dict(zip(years, [ACCENT_1, ACCENT_3, ACCENT_2, ACCENT_4, ACCENT_7]))

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.8 / len(years)

    for i, yr in enumerate(years):
        subset = df[df["year"] == yr].sort_values("month")
        positions = subset["month"] - 1 + i * bar_width - (len(years) * bar_width / 2) + bar_width / 2
        ax.bar(positions, subset["n"], width=bar_width, label=str(yr),
               color=year_colors.get(yr, ACCENT_4), edgecolor="none", alpha=0.85)

    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=10)
    ax.set_ylabel("Records", labelpad=10)
    ax.set_title("V6 · Year-over-Year Monthly Record Volume", pad=15)
    ax.legend(title="Year", framealpha=0.85, loc="upper right")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    save_fig(fig, "V06_yoy_comparison")


# ===================================================================
# V7: Text Length Distribution
# ===================================================================

def v7_text_length():
    print("[V7] Text Length Distribution …")

    posts_len = query_clean("SELECT text_length FROM posts_clean WHERE text_length < 5000")
    comments_len = query_clean("SELECT text_length FROM comments_clean WHERE text_length < 5000")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].hist(posts_len["text_length"], bins=80, color=ACCENT_1, edgecolor="none", alpha=0.85)
    axes[0].set_title("Posts — Cleaned Character Length", fontsize=12)
    axes[0].set_xlabel("Character Count", labelpad=8)
    axes[0].set_ylabel("Count (log scale)", labelpad=8)
    axes[0].set_yscale("log")
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].hist(comments_len["text_length"], bins=80, color=ACCENT_3, edgecolor="none", alpha=0.85)
    axes[1].set_title("Comments — Cleaned Character Length", fontsize=12)
    axes[1].set_xlabel("Character Count", labelpad=8)
    axes[1].set_ylabel("Count (log scale)", labelpad=8)
    axes[1].set_yscale("log")
    axes[1].grid(axis="y", alpha=0.2)

    fig.suptitle("V7 · Cleaned Text Length Distribution (Log Scale)", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()

    save_fig(fig, "V07_text_length_distribution")


# ===================================================================
# V8: Word Cloud per Category
# ===================================================================

def v8_word_clouds():
    print("[V8] Word Clouds …")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cat_labels = {"technology": "Technology", "science": "Science", "news_general": "News & General"}
    cat_cmaps  = {"technology": "Blues", "science": "Greens", "news_general": "Oranges"}

    for ax, (cat, label) in zip(axes, cat_labels.items()):
        df = query_clean(
            "SELECT text_clean FROM comments_clean WHERE category = ? ORDER BY RANDOM() LIMIT 40000",
            (cat,)
        )
        text = " ".join(df["text_clean"].tolist())

        wc = WordCloud(
            width=800, height=500,
            background_color=BG_CARD,
            colormap=cat_cmaps[cat],
            max_words=100,
            min_font_size=10,
            max_font_size=80,
            prefer_horizontal=0.7,
        ).generate(text)

        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(label, fontsize=14, pad=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle("V8 · Frequent Corpus Words by Subreddit Category", y=1.02, fontsize=16, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "V08_word_clouds")


# ===================================================================
# V9: Top N-grams
# ===================================================================

def v9_top_ngrams():
    print("[V9] Top N-grams …")

    df = query_clean("SELECT text_clean FROM comments_clean ORDER BY RANDOM() LIMIT 80000")

    uni_counter = Counter()
    bi_counter  = Counter()
    for text in df["text_clean"]:
        tokens = text.split()
        uni_counter.update(tokens)
        bi_counter.update(zip(tokens, tokens[1:]))

    uni_counter = {k: v for k, v in uni_counter.items() if len(k) > 2}

    top_uni = sorted(uni_counter.items(), key=lambda x: x[1], reverse=True)[:20]
    top_bi  = sorted(bi_counter.items(), key=lambda x: x[1], reverse=True)[:20]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    words_u, counts_u = zip(*reversed(top_uni))
    axes[0].barh(range(len(words_u)), counts_u, color=ACCENT_1, edgecolor="none")
    axes[0].set_yticks(range(len(words_u)))
    axes[0].set_yticklabels(words_u, fontsize=10)
    axes[0].set_title("Top 20 Unigrams", fontsize=13)
    axes[0].set_xlabel("Frequency", labelpad=8)
    axes[0].grid(axis="x", alpha=0.2)

    bigram_labels = [f"{a} {b}" for (a, b) in [x[0] for x in reversed(top_bi)]]
    counts_b = [x[1] for x in reversed(top_bi)]
    axes[1].barh(range(len(bigram_labels)), counts_b, color=ACCENT_2, edgecolor="none")
    axes[1].set_yticks(range(len(bigram_labels)))
    axes[1].set_yticklabels(bigram_labels, fontsize=10)
    axes[1].set_title("Top 20 Bigrams", fontsize=13)
    axes[1].set_xlabel("Frequency", labelpad=8)
    axes[1].grid(axis="x", alpha=0.2)

    fig.suptitle("V9 · Most Frequent Unigrams and Bigrams (Sampled Corpus)", y=1.02, fontsize=15, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "V09_top_ngrams")


# ===================================================================
# V10: Score Distribution (Fixed: Log/Symlog scale & Staggered Non-Overlapping Text)
# ===================================================================

def v10_score_distribution():
    print("[V10] Score Distribution (Fixed non-overlapping) …")

    df = query_clean("SELECT score FROM posts_clean WHERE score IS NOT NULL AND score >= 0")
    scores = df["score"].values

    p50 = float(np.percentile(scores, 50))
    p90 = float(np.percentile(scores, 90))
    p99 = float(np.percentile(scores, 99))
    p_max = float(scores.max())

    fig, ax = plt.subplots(figsize=(11, 6))

    bins = np.logspace(0, np.log10(max(p_max, 10)), 100)
    bins = np.insert(bins, 0, -0.5)

    n, bins_out, patches = ax.hist(scores, bins=bins, color=ACCENT_4, edgecolor="none", alpha=0.85)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("log")

    ax.set_xlabel("Post Score (symlog scale: 0 to 50,000+)", labelpad=10)
    ax.set_ylabel("Post Count (log scale)", labelpad=10)
    ax.set_title("V10 · Post Score Distribution & Percentiles", pad=15)
    ax.grid(True, which="both", alpha=0.15)

    percentile_configs = [
        (50, p50, 0.70, ACCENT_3),
        (90, p90, 0.45, ACCENT_6),
        (99, p99, 0.20, ACCENT_5),
    ]

    max_y = ax.get_ylim()[1]

    for pct, val, y_pct, color in percentile_configs:
        ax.axvline(val, color=color, linestyle="--", linewidth=1.8, alpha=0.95, label=f"P{pct} = {val:,.0f}")

        x_text = max(val * 1.25, 0.2 if val == 0 else val)
        y_text = 10 ** (np.log10(max_y) * y_pct)

        ax.annotate(
            f"  P{pct} = {val:,.0f}",
            xy=(val, y_text),
            xytext=(x_text, y_text),
            color=color,
            fontsize=10,
            fontweight="bold",
            va="center",
            arrowprops=dict(arrowstyle="->", color=color, lw=1, alpha=0.7) if val > 0 else None,
            bbox=dict(boxstyle="round,pad=0.2", facecolor=BG_CARD, edgecolor=color, alpha=0.85)
        )

    summary_text = (
        f"Score Metrics:\n"
        f"• Median (P50): {p50:,.0f}\n"
        f"• P90: {p90:,.0f}\n"
        f"• P99: {p99:,.0f}\n"
        f"• Max: {p_max:,.0f}"
    )
    ax.text(0.97, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=TEXT_COLOR,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=BG_CARD, edgecolor=GRID_COLOR, alpha=0.9))

    ax.legend(loc="upper left", framealpha=0.85, fontsize=10)
    fig.tight_layout()

    save_fig(fig, "V10_score_distribution")


# ===================================================================
# V11: Score vs Comment Count Scatter
# ===================================================================

def v11_score_vs_comments():
    print("[V11] Score vs Comments …")

    df = query_clean("""
        SELECT score, num_comments, category FROM posts_clean
        WHERE score IS NOT NULL AND num_comments IS NOT NULL
          AND score > 0 AND num_comments > 0
        ORDER BY RANDOM() LIMIT 25000
    """)

    fig, ax = plt.subplots(figsize=(10, 7))
    for cat, color in CATEGORY_COLORS.items():
        subset = df[df["category"] == cat]
        label = {"technology": "Technology", "science": "Science", "news_general": "News & General"}[cat]
        ax.scatter(subset["score"], subset["num_comments"],
                   c=color, alpha=0.3, s=10, label=label, edgecolors="none")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Post Score (log scale)", labelpad=10)
    ax.set_ylabel("Number of Comments (log scale)", labelpad=10)
    ax.set_title("V11 · Post Score vs. Comment Count Correlation", pad=15)
    ax.legend(framealpha=0.85, markerscale=2.5, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    save_fig(fig, "V11_score_vs_comments")


# ===================================================================
# V12: Engagement by Subreddit Box Plots
# ===================================================================

def v12_engagement_boxplots():
    print("[V12] Engagement Box Plots …")

    df = query_clean("""
        SELECT subreddit, score FROM posts_clean
        WHERE score IS NOT NULL AND score BETWEEN 0 AND 2000
    """)

    medians = df.groupby("subreddit")["score"].median().sort_values()
    order = medians.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 8.5))
    sns.boxplot(
        data=df, y="subreddit", x="score", order=order,
        color=ACCENT_1, fliersize=1, linewidth=0.8,
        flierprops={"alpha": 0.15, "color": TEXT_MUTED},
        medianprops={"color": ACCENT_3, "linewidth": 2},
        boxprops={"edgecolor": ACCENT_1, "alpha": 0.7},
        whiskerprops={"color": TEXT_MUTED},
        capprops={"color": TEXT_MUTED},
        ax=ax
    )
    ax.set_xlabel("Post Score (truncated at 2,000 for visibility)", labelpad=10)
    ax.set_ylabel("", labelpad=0)
    ax.set_title("V12 · Post Score Distribution by Subreddit", pad=15)
    ax.grid(axis="x", alpha=0.2)
    plt.yticks(fontsize=9)
    fig.tight_layout()

    save_fig(fig, "V12_engagement_boxplots")


# ===================================================================
# V13: Missing/Deleted Data Stacked Bar
# ===================================================================

def v13_data_quality():
    print("[V13] Data Quality …")

    raw_posts = query_raw("SELECT subreddit, COUNT(*) AS total FROM posts GROUP BY subreddit")
    del_posts = query_raw("""
        SELECT subreddit, COUNT(*) AS deleted FROM posts
        WHERE selftext IN ('[deleted]','[removed]')
        GROUP BY subreddit
    """)
    empty_posts = query_raw("""
        SELECT subreddit, COUNT(*) AS empty FROM posts
        WHERE selftext = '' AND title != ''
        GROUP BY subreddit
    """)

    df = raw_posts.merge(del_posts, on="subreddit", how="left").merge(empty_posts, on="subreddit", how="left")
    df = df.fillna(0)
    df["clean"] = df["total"] - df["deleted"] - df["empty"]
    df["pct_clean"]   = df["clean"]   / df["total"] * 100
    df["pct_deleted"] = df["deleted"] / df["total"] * 100
    df["pct_empty"]   = df["empty"]   / df["total"] * 100
    df = df.sort_values("pct_clean", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(df))
    ax.barh(y, df["pct_clean"],   color=ACCENT_3, label="Clean (Self-text)", edgecolor="none")
    ax.barh(y, df["pct_empty"],   left=df["pct_clean"], color=ACCENT_1, label="Link Posts (Title only)", edgecolor="none")
    ax.barh(y, df["pct_deleted"], left=df["pct_clean"] + df["pct_empty"], color=ACCENT_5, label="Deleted / Removed", edgecolor="none")

    ax.set_yticks(y)
    ax.set_yticklabels(df["subreddit"], fontsize=9)
    ax.set_xlabel("Percentage of Posts", labelpad=10)
    ax.set_xlim(0, 100)
    ax.set_title("V13 · Post Data Quality & Type Composition by Subreddit", pad=15)
    ax.legend(loc="lower right", framealpha=0.85)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()

    save_fig(fig, "V13_data_quality")


# ===================================================================
# V14: Collection Coverage Heatmap
# ===================================================================

def v14_coverage_heatmap():
    print("[V14] Coverage Heatmap …")

    df = query_clean("""
        SELECT subreddit, year_month, COUNT(*) AS n FROM (
            SELECT subreddit, year_month FROM posts_clean
            UNION ALL
            SELECT subreddit, year_month FROM comments_clean
        ) GROUP BY subreddit, year_month
    """)

    pivot = df.pivot(index="subreddit", columns="year_month", values="n").fillna(0)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False)
    pivot = pivot.drop(columns="total")

    fig, ax = plt.subplots(figsize=(20, 8.5))
    sns.heatmap(
        pivot, ax=ax, cmap="viridis", linewidths=0.3, linecolor=BG_DARK,
        cbar_kws={"label": "Records per Month", "shrink": 0.6},
        xticklabels=True, yticklabels=True,
    )
    xticks = ax.get_xticks()
    xlabels = [t.get_text() for t in ax.get_xticklabels()]
    ax.set_xticks(xticks[::2])
    ax.set_xticklabels(xlabels[::2], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    ax.set_xlabel("Month (2022-08 to 2026-08)", labelpad=10)
    ax.set_ylabel("Subreddit", labelpad=10)
    ax.set_title("V14 · Collection Coverage: Records per Subreddit per Month", pad=15)
    fig.tight_layout()

    save_fig(fig, "V14_coverage_heatmap")


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    print("=" * 60)
    print("  VISUALIZATION PIPELINE (POLISHED & NON-OVERLAPPING)")
    print("=" * 60)
    print(f"  Source: {CLEAN_DB}")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    if not os.path.exists(CLEAN_DB):
        print(f"ERROR: {CLEAN_DB} not found. Run preprocess.py first.")
        return 1

    generators = [
        v1_dataset_composition,
        v2_cleaning_funnel,
        v3_category_donut,
        v4_monthly_timeseries,
        v5_activity_heatmap,
        v6_yoy_comparison,
        v7_text_length,
        v8_word_clouds,
        v9_top_ngrams,
        v10_score_distribution,
        v11_score_vs_comments,
        v12_engagement_boxplots,
        v13_data_quality,
        v14_coverage_heatmap,
    ]

    for gen in generators:
        try:
            gen()
        except Exception as e:
            print(f"  [ERR] {gen.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 60)
    print("  ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print(f"  Output directory: {OUTPUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
