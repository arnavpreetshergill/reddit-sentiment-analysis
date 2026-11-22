================================================================================
REDDIT SENTIMENT ANALYSIS ENGINE — VISUALIZATION EXPLANATION GUIDE
================================================================================
Dataset Scope: 8,152,751 raw records -> 6,786,093 clean records
Time Period:   August 18, 2022 to August 17, 2026 (4 full years)
Subreddits:    25 subreddits across Technology, Science, and News/General categories
Output Dir:    visualizations/

This guide provides detailed explanations for all 14 generated progress report
visualizations, including what each chart represents, key data insights, and
its relevance to downstream sentiment modeling.

--------------------------------------------------------------------------------
CATEGORY 1: DATASET OVERVIEW
--------------------------------------------------------------------------------

V01 · Dataset Composition by Subreddit
File: visualizations/V01_dataset_composition.png
• What it shows: Stacked horizontal bar chart showing the count of posts (green)
  and comments (blue) collected across all 25 subreddits.
• How to read it: Subreddits are ordered from lowest total volume (bottom) to
  highest (top). Comments make up ~89% of the dataset, while posts make up ~11%.
• Key Insights: High-volume communities (r/AskReddit, r/NoStupidQuestions,
  r/news, r/worldnews) have ~294k comments each, whereas smaller subreddits
  (r/gadgets, r/datascience) have fewer posts due to lower natural posting activity.
• Report Relevance: Demonstrates dataset scale and balance across communities.

V02 · Data Cleaning Funnel
File: visualizations/V02_cleaning_funnel.png
• What it shows: Waterfall/funnel chart illustrating the 3 data cleaning stages:
  1. Raw Dataset (8,152,751 total records)
  2. After Removing Deleted/Removed (7,113,526 records; -12.7% dropped)
  3. After Text Cleaning & Short Filtering (6,786,093 records; -4.6% dropped)
• How to read it: Each step displays the exact record count remaining and the
  percentage dropped.
• Key Insights: 1,039,225 deleted/removed entries ([deleted], [removed]) contained
  no usable text and were dropped. Another 327,433 records had < 3 words after
  cleaning (mostly single emojis, punctuation, or empty strings).
• Report Relevance: Proves rigorous data quality preprocessing before analysis.

V03 · Records by Category
File: visualizations/V03_category_donut.png
• What it shows: Donut chart displaying the distribution of clean records across
  3 primary domain categories:
  1. Technology (10 subreddits): ~38.5%
  2. News & General (7 subreddits): ~32.8%
  3. Science (8 subreddits): ~28.7%
• How to read it: Center shows the total clean corpus size (6,786,093 records).
• Key Insights: Balanced dataset representation across tech, science, and news/
  general discussion domains without any single domain dominating.
• Report Relevance: Justifies cross-domain sentiment comparison capability.

--------------------------------------------------------------------------------
CATEGORY 2: TEMPORAL ANALYSIS
--------------------------------------------------------------------------------

V04 · Monthly Record Volume by Category (2022–2026)
File: visualizations/V04_monthly_timeseries.png
• What it shows: Multi-line time series plot showing monthly record volume over
  48 months for Technology, Science, News/General, and Total Corpus.
• How to read it: X-axis represents time (YYYY-MM); Y-axis shows record count.
• Key Insights: Consistent, stable collection velocity (~140k–150k records/month
  total) across the 4-year collection window, demonstrating data collection
  reliability without artificial spikes or missing months.
• Report Relevance: Verifies longitudinal consistency for temporal sentiment trends.

V05 · Activity Heatmap by Day of Week & Hour (UTC)
File: visualizations/V05_activity_heatmap.png
• What it shows: 7x24 color matrix showing post/comment activity intensity by day
  of the week (Mon–Sun) and hour of the day (00:00–23:00 UTC).
• How to read it: Darker red cells indicate peak activity hours; lighter yellow
  cells indicate lower activity hours.
• Key Insights: Peak Reddit activity occurs between 14:00 UTC and 22:00 UTC
  (corresponding to US morning through evening hours), with weekdays (Tue–Thu)
  showing slightly higher volume than weekends.
• Report Relevance: Useful for hourly sentiment pulse and peak engagement modeling.

V06 · Year-over-Year Monthly Record Volume
File: visualizations/V06_yoy_comparison.png
• What it shows: Grouped bar chart comparing monthly activity across different years
  (2022 to 2026) for each month (Jan–Dec).
• How to read it: Each month cluster has side-by-side bars for available years.
• Key Insights: Captures seasonal stability year over year. Partial years (2022
  starts in Aug; 2026 ends in Aug) reflect expected collection window boundaries.
• Report Relevance: Validates seasonality controls for time-series modeling.

--------------------------------------------------------------------------------
CATEGORY 3: TEXT & CONTENT FEATURES
--------------------------------------------------------------------------------

V07 · Cleaned Text Length Distribution (Log Scale)
File: visualizations/V07_text_length_distribution.png
• What it shows: Dual-panel histogram (log scale) showing the character length
  distribution of cleaned text for Posts (left) and Comments (right).
• How to read it: X-axis represents character count (0 to 5,000); Y-axis is log-scale.
• Key Insights:
  - Posts have a bi-modal length distribution: short titles (~68 chars) for link
    posts, and longer self-text descriptions (median ~142 chars).
  - Comments have a smooth right-skewed distribution centered around ~217 chars,
    with long-form comments extending up to 5,000 characters.
• Report Relevance: Informs sequence length truncation choices (e.g. max_length=256
  or 512 tokens) for transformer tokenizers in sentiment analysis.

V08 · Frequent Corpus Words by Subreddit Category
File: visualizations/V08_word_clouds.png
• What it shows: 3 side-by-side word clouds highlighting top non-stopword terms in
  Technology (Blue), Science (Green), and News & General (Orange).
• How to read it: Word size corresponds to term frequency in cleaned comments.
• Key Insights:
  - Technology: "app", "system", "device", "code", "ai", "user", "build", "data"
  - Science: "study", "research", "cell", "energy", "brain", "space", "model"
  - News/General: "people", "country", "government", "market", "policy", "state"
• Report Relevance: Confirms domain vocabulary distinctness for sub-models.

V09 · Most Frequent Unigrams and Bigrams
File: visualizations/V09_top_ngrams.png
• What it shows: Horizontal bar charts displaying the 20 most common single words
  (unigrams) and 2-word phrases (bigrams) extracted from a 80,000 comment sample.
• How to read it: Bars ordered by corpus frequency.
• Key Insights: Top unigrams include "people", "like", "think", "time", "make".
  Top bigrams include "first time", "years ago", "make sense", "looks like", "even if".
• Report Relevance: Highlights candidate features for n-gram baseline models.

--------------------------------------------------------------------------------
CATEGORY 4: ENGAGEMENT ANALYSIS
--------------------------------------------------------------------------------

V10 · Post Score Distribution & Percentiles (Symlog Scale)
File: visualizations/V10_score_distribution.png
• What it shows: Log-histogram of post scores (upvotes - downvotes) overlaid with
  percentile markers (P50, P90, P99) and a summary metrics box.
• How to read it: Uses a symlog x-axis scale to clearly display scores ranging
  from 0 to 48,912 without label overlap.
• Key Insights:
  - P50 (Median) = 1: 50% of posts get a score of <= 1.
  - P90 = 176: Only top 10% of posts exceed 176 upvotes.
  - P99 = 4,636: Only top 1% achieve viral reach (4,636+ upvotes).
  - Max = 48,912: Peak post score.
• Report Relevance: Proves that raw scores are heavily skewed and must be log-
  transformed or binned into score buckets for machine learning models.

V11 · Post Score vs. Comment Count Correlation
File: visualizations/V11_score_vs_comments.png
• What it shows: Log-log scatter plot of post score (x-axis) vs. number of comments
  (y-axis) for 25,000 sampled posts, color-coded by category.
• How to read it: Both axes use logarithmic scaling.
• Key Insights: Strong positive log-linear correlation between post score and
  comment volume (high-scoring posts attract significantly more comments).
  Controversial posts appear in top-left (low score, high comment count).
• Report Relevance: Helps identify controversial vs. consensus viral posts.

V12 · Post Score Distribution by Subreddit
File: visualizations/V12_engagement_boxplots.png
• What it shows: Horizontal box plot comparing post score distributions across all
  25 subreddits, ordered by median score.
• How to read it: Box = middle 50% (IQR); green line = median; x-axis truncated at
  2,000 for visibility.
• Key Insights: High community variance:
  - Mass subreddits (r/worldnews, r/news, r/AskReddit) have high median scores (100–500+).
  - Technical subreddits (r/programming, r/datascience, r/software) have low median
    scores (0–20), even though text quality is high.
• Report Relevance: Demonstrates the need to normalize engagement metrics relative
  to each subreddit's baseline.

--------------------------------------------------------------------------------
CATEGORY 5: DATA QUALITY & COVERAGE
--------------------------------------------------------------------------------

V13 · Post Data Quality & Type Composition by Subreddit
File: visualizations/V13_data_quality.png
• What it shows: 100% stacked bar chart showing the composition of posts in each
  subreddit: Clean self-text (green), Link-only posts (blue), and Deleted/Removed (red).
• How to read it: Ordered by percentage of clean self-text posts.
• Key Insights: Subreddits like r/AskReddit and r/NoStupidQuestions are nearly 100%
  clean text posts. Subreddits like r/gadgets and r/news contain a high percentage
  of link-only posts (title present, no body text).
• Report Relevance: Explains why title text must be used as primary text for link posts.

V14 · Collection Coverage: Records per Subreddit per Month
File: visualizations/V14_coverage_heatmap.png
• What it shows: 25x49 heatmap showing record counts collected for every subreddit
  in every month from August 2022 to August 2026.
• How to read it: Y-axis lists subreddits (sorted by total volume); X-axis lists
  months. Darker purple/yellow cells indicate solid collection density.
• Key Insights: Complete grid coverage with zero missing cells across all 49 months,
  confirming no data collection gaps or missing periods.
• Report Relevance: Confirms dataset completeness for progress report submission.

================================================================================
SUMMARY
================================================================================
All 14 visualizations are generated directly from reddit_clean.db via visualize.py
and saved in PNG format in visualizations/ for embedding in reports and slides.
