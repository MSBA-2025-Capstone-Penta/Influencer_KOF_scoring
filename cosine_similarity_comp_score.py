# app.py
import io
import re
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from typing import Union, cast
from scipy.sparse import csr_matrix

# Initial render
st.set_page_config(page_title="Tweet ↔ Media Matcher",
                   page_icon="🔎", layout="wide")
st.title("🔎 Influencer Scoring for Tweet & News Articles")

# --- Safe import block: show error if a dependency is missing ---
try:
    import re
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import plotly.express as px
except Exception as e:
    st.error(f"Import failed: {e}")
    st.info("Try installing missing packages:\n\n"
            "`pip install pandas numpy scikit-learn plotly openpyxl`")
    st.stop()

st.caption("Modules loaded ✅")

# --------------------------
# Helper functions
# --------------------------


def clean_text(text: str) -> str:
    import re as _re
    return _re.sub(r"\W+", " ", str(text).lower()).strip()


def smart_read_csv(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue() if hasattr(
        uploaded_file, "getvalue") else uploaded_file.read()
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be", "utf-32"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, encoding_errors="strict",
                               engine="python", sep=None, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer.api import from_bytes
        best = from_bytes(raw).best()
        if best and best.encoding:
            return pd.read_csv(io.BytesIO(raw), encoding=best.encoding, encoding_errors="replace",
                               engine="python", sep=None, on_bad_lines="skip")
    except Exception:
        pass
    return pd.read_csv(io.BytesIO(raw), encoding="latin-1", encoding_errors="replace",
                       engine="python", sep=None, on_bad_lines="skip")


@st.cache_data(show_spinner=False)
def read_tweets_file(file, sheet_name="tweets") -> pd.DataFrame:
    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        return smart_read_csv(file)
    return pd.read_excel(file, sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def read_media_file(file) -> pd.DataFrame:
    name = getattr(file, "name", "")
    if name.lower().endswith(".csv"):
        return smart_read_csv(file)
    return pd.read_excel(file)

# Build a regex that matches whole tokens only (not inside longer words)


def build_keyword_regex(keywords) -> str | None:
    """
    Returns an OR-joined regex where each keyword is wrapped in custom
    'token boundaries' so 'ACA' won't match 'tobaccas' or 'ACADEMY'.
    We block letters/digits/underscore on both sides.
    """
    pats = []
    for k in keywords:
        k = k.strip()
        if not k:
            continue
        pats.append(rf"(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])")
    return "|".join(pats) if pats else None


def tfidf_cosine_matches(tweet_subset: pd.DataFrame,
                         article_subset: pd.DataFrame,
                         similarity_cutoff: float = 0.30,
                         max_lag_days: int = 2,
                         *,
                         min_lead_minutes: int = 30) -> pd.DataFrame:
    """Return tweet→article pairs that meet the similarity cutoff within the lag window
    AND where the tweet occurs at least `min_lead_minutes` before the article."""
    article_subset = article_subset.copy()
    tweet_subset = tweet_subset.copy()

    article_subset["clean_body"] = article_subset["Body"].apply(clean_text)
    tweet_subset["clean_text"] = tweet_subset["text"].apply(clean_text)

    article_subset.reset_index(drop=True, inplace=True)
    tweet_subset.reset_index(drop=True, inplace=True)

    # Robustness: drop empty rows; early return consistent columns
    article_subset = article_subset[article_subset["clean_body"].str.strip().ne(
        "")]
    tweet_subset = tweet_subset[tweet_subset["clean_text"].str.strip().ne("")]
    if article_subset.empty or tweet_subset.empty:
        return pd.DataFrame(columns=[
            "Cosine Similarity", "Article Headline", "Article Date",
            "Tweet Date", "Tweet Text", "KOF", "Target"
        ])

    # ✅ Fix: re-reset indices so positions align with TF-IDF slices and sim matrix
    article_subset = article_subset.reset_index(drop=True)
    tweet_subset = tweet_subset.reset_index(drop=True)

    combined_corpus = pd.concat(
        [article_subset["clean_body"], tweet_subset["clean_text"]], ignore_index=True)
    vectorizer = TfidfVectorizer(stop_words="english")

    # Pylance-safe: cast to csr_matrix so slicing [:, :] is recognized
    _X = vectorizer.fit_transform(combined_corpus)
    tfidf_matrix = cast(csr_matrix, _X)

    n_articles = len(article_subset)
    tfidf_articles = tfidf_matrix[:n_articles, :]
    tfidf_tweets = tfidf_matrix[n_articles:, :]

    sim = cosine_similarity(tfidf_articles, tfidf_tweets)

    article_times = pd.to_datetime(
        article_subset["Load Time"], errors="coerce")
    tweet_times = pd.to_datetime(tweet_subset["created_at"], errors="coerce")

    lead_delta = pd.Timedelta(minutes=int(min_lead_minutes))

    matches = []
    for i, a_time in enumerate(article_times):
        if pd.isna(a_time):
            continue
        window_start = a_time - pd.Timedelta(days=max_lag_days)
        window_end = a_time

        idx = tweet_times[(tweet_times >= window_start) &
                          (tweet_times <= window_end)].index
        for j in idx:
            t_time = tweet_times.loc[j]
            if pd.isna(t_time):
                continue
            if not (t_time <= a_time - lead_delta):
                continue

            score = float(sim[i, j])
            if score >= similarity_cutoff:
                matches.append({
                    "Cosine Similarity": round(score, 3),
                    "Article Headline": article_subset.loc[i, "Headline"],
                    "Article Date": a_time,
                    "Tweet Date": t_time,
                    "Tweet Text": tweet_subset.loc[j, "text"],
                    "KOF": tweet_subset.loc[j, "kof"] if "kof" in tweet_subset.columns else None,
                    "Target": tweet_subset.loc[j, "target"] if "target" in tweet_subset.columns else None
                })
    return pd.DataFrame(matches)


# --------------------------
# Sidebar UI
# --------------------------
with st.sidebar:
    st.header("1) Upload your files")
    tweets_file = st.file_uploader(
        "Tweets (.xlsx or .csv)", type=["xlsx", "csv"])
    media_file = st.file_uploader(
        "Media (.csv or .xlsx)",  type=["csv", "xlsx"])

    st.header("2) Topic Filter")
    default_keywords = [
        "health", "medicare", "abortion", "healthcare", "mental health", "reproductive",
        "insurance", "medicaid", "ACA", "coverage", "medical", "health provider",
        "co-pay", "health premium", "deductible", "uninsured", "preexisting condition",
        "primary care", "telehealth", "Affordable Care Act"
    ]
    kw_text = st.text_area("Health keywords", value="\n".join(
        default_keywords), height=200)
    keywords = [k.strip() for k in kw_text.splitlines() if k.strip()]

    # FIX: word-boundary matching to avoid false positives like 'tobaccas' for 'ACA'
    regex = build_keyword_regex(keywords)

    # Matching knobs
    sim_cutoff = st.slider("Similarity cutoff", 0.0, 1.0, 0.30, 0.01)
    lag_days = st.slider(
        "Tweets must be within N days before article", 0, 7, 2)
    min_lead_minutes = st.number_input(
        "Require tweet to lead article by at least (minutes)",
        min_value=0, max_value=720, value=30, step=5
    )

    # Top-K for leaderboards
    top_k = st.slider(
        "Overall Leaderboard Top K influencers",
        min_value=5, max_value=50, value=20, step=1,
        help="Controls how many rows show up in the composite tables/plots and leaderboards."
    )

    # Composite scoring – weights only
    with st.expander("Composite Score Weights", expanded=False):
        st.caption(
            "Tune weights for engagement signals used in the composite influence score.")
        w_retweets = st.number_input("Weight: Retweets", 0.0, 10.0, 2.0, 0.5)
        w_quotes = st.number_input("Weight: Quotes",   0.0, 10.0, 2.0, 0.5)
        w_faves = st.number_input("Weight: Favorites", 0.0, 10.0, 1.0, 0.5)
        w_replies = st.number_input("Weight: Replies",  0.0, 10.0, 2.0, 0.5)

    # Dedupe strategy
    dedupe_option = st.selectbox(
        "Deduplicate matches",
        ("None", "Per article–tweet (keep max similarity)",
         "Per article–tweet (collapse targets)"),
        help="Choose how to handle repeated pairs where one tweet matches the same article multiple times."
    )

    st.caption("Columns expected by default:")
    st.code("Tweets: created_at, text, (optional: kof, target, retweets, quotes, favorites, replies)\n"
            "Media: Load Time, Headline, Body")

    run_btn = st.button("Run Matching & Scoring",
                        type="primary", use_container_width=True)

# Placeholders
status_ph = st.empty()
table_ph = st.empty()
chart_ph = st.empty()

score_header_ph = st.empty()
score_table_ph = st.empty()
score_chart_ph = st.empty()

# --------------------------
# Main action
# --------------------------
if run_btn:
    if not tweets_file or not media_file:
        st.error("Please upload both a Tweets file and a Media file.")
        st.stop()

    # Read files
    try:
        with st.spinner("Reading files…"):
            tweets_df = read_tweets_file(tweets_file, sheet_name="tweets")
            media_df = read_media_file(media_file)
    except UnicodeDecodeError:
        st.error("Could not read files due to a text-encoding issue.")
        st.info(
            "Tip: Re-save your CSV as CSV UTF-8 or upload the original .xlsx instead.")
        st.stop()
    except Exception as e:
        st.error(f"Could not read files: {e}")
        st.stop()

    # Validate columns
    required_tweet_cols = {"created_at", "text"}
    required_media_cols = {"Load Time", "Headline", "Body"}
    if not required_tweet_cols.issubset(set(tweets_df.columns)) or not required_media_cols.issubset(set(media_df.columns)):
        st.error("Missing required columns.\n\nTweets need: created_at, text (optional: kof, target, retweets, quotes, favorites, replies)\nMedia need: Load Time, Headline, Body")
        st.stop()

    # Coerce datetimes & drop nulls
    tweets_df["created_at"] = pd.to_datetime(
        tweets_df["created_at"], errors="coerce")
    media_df["Load Time"] = pd.to_datetime(
        media_df["Load Time"],   errors="coerce")
    media_df = media_df.dropna(subset=["Body", "Load Time"])
    tweets_df = tweets_df.dropna(subset=["text", "created_at"])

    # Keyword filtering (uses boundary-safe regex)
    if regex:
        tweets_df = tweets_df[tweets_df["text"].fillna(
            "").str.contains(regex, case=False, regex=True)]
        media_df = media_df[media_df["Body"].fillna(
            "").str.contains(regex, case=False, regex=True)]

    # Align tweet date range to media range (+ buffer)
    if len(media_df):
        min_article_date = media_df["Load Time"].min().normalize()
        max_article_date = media_df["Load Time"].max()
        tweets_df = tweets_df[
            (tweets_df["created_at"] >= (min_article_date - pd.Timedelta(days=lag_days))) &
            (tweets_df["created_at"] <= max_article_date)
        ]

    # --------------------------
    # Composite influence scoring (per target)
    # --------------------------
    influencer_counts = None
    score_header_ph.subheader("🏆 Composite Influence Scores (Health topics)")
    if "target" not in tweets_df.columns:
        score_header_ph.info(
            "No `target` column found in tweets — cannot compute influencer scores. Add a `target` field to group by influencer.")
    else:
        for col in ["retweets", "quotes", "favorites", "replies"]:
            if col not in tweets_df.columns:
                tweets_df[col] = 0
        for col in ["retweets", "quotes", "favorites", "replies"]:
            tweets_df[col] = pd.to_numeric(
                tweets_df[col], errors="coerce").fillna(0)

        influencer_counts = (
            tweets_df.groupby("target", dropna=True)
            .agg({"text": "count", "retweets": "sum", "quotes": "sum", "favorites": "sum", "replies": "sum"})
            .rename(columns={"text": "number_health_tweets"})
            .sort_values("number_health_tweets", ascending=False)
        )

        if influencer_counts.empty:
            score_header_ph.info(
                "No tweets matched the current keyword/date filters for computing scores.")
        else:
            influencer_counts["influence_score"] = (
                influencer_counts["retweets"] * float(w_retweets) +
                influencer_counts["quotes"] * float(w_quotes) +
                influencer_counts["favorites"] * float(w_faves) +
                influencer_counts["replies"] * float(w_replies)
            )
            min_s = influencer_counts["influence_score"].min()
            max_s = influencer_counts["influence_score"].max()
            if max_s > min_s:
                influencer_counts["normalized_score"] = (
                    influencer_counts["influence_score"] - min_s) / (max_s - min_s) * 100.0
            else:
                influencer_counts["normalized_score"] = 100.0

            top_influencers = influencer_counts.sort_values(
                "influence_score", ascending=False).head(int(top_k)).reset_index()

            # Table
            score_table_ph.dataframe(
                top_influencers[["target", "normalized_score", "number_health_tweets",
                                 "retweets", "quotes", "favorites", "replies"]],
                use_container_width=True, hide_index=True
            )

            # Bar
            fig_scores = px.bar(
                top_influencers.sort_values("normalized_score"),
                x="normalized_score", y="target", orientation="h",
                title="Top Influencers by Composite Score",
                labels={
                    "normalized_score": "Composite Influence Score (0–100)", "target": "Influencer (target)"}
            )
            fig_scores.update_layout(height=550, xaxis_range=[0, 100])
            score_chart_ph.plotly_chart(fig_scores, use_container_width=True)

    # --------------------------
    # Matching (TF-IDF cosine)
    # --------------------------
    keep_tweet_cols = ["created_at", "text"] + \
        [c for c in ["kof", "target"] if c in tweets_df.columns]
    tweet_subset = tweets_df[keep_tweet_cols].copy()
    article_subset = media_df[["Load Time", "Headline", "Body"]].copy()

    status_ph.info("Vectorizing & matching…")
    try:
        results_df = tfidf_cosine_matches(
            tweet_subset=tweet_subset,
            article_subset=article_subset,
            similarity_cutoff=sim_cutoff,
            max_lag_days=lag_days,
            min_lead_minutes=int(min_lead_minutes)
        )
    except Exception as e:
        status_ph.error(f"Matching failed: {e}")
        st.stop()

    if results_df.empty:
        status_ph.warning(
            "No matches found with the current filters/threshold. Try lowering the cutoff or widening the day window.")
        st.stop()

    # Deduplication strategy
    if dedupe_option == "Per article–tweet (keep max similarity)":
        results_df = (
            results_df.sort_values("Cosine Similarity", ascending=False)
                      .drop_duplicates(subset=["Article Headline", "Tweet Text", "Tweet Date"])
        )
    elif dedupe_option == "Per article–tweet (collapse targets)":
        results_df = (
            results_df.groupby(["Article Headline", "Article Date",
                               "Tweet Date", "Tweet Text", "KOF"], dropna=False)
            .agg({"Cosine Similarity": "max",
                  "Target": lambda s: ", ".join(sorted({str(x) for x in s.dropna()})) or None})
            .reset_index()
            .sort_values("Cosine Similarity", ascending=False)
        )

    # Top 10 preview
    top10 = results_df.sort_values(
        "Cosine Similarity", ascending=False).head(10).reset_index(drop=True)
    status_ph.success(
        f"Done! {len(results_df)} matches found. Showing Top 10:")
    top10_display = top10.copy()
    top10_display["Tweet (truncated)"] = top10_display["Tweet Text"].str.slice(
        0, 120) + "…"

    cols = ["Cosine Similarity", "Article Headline",
            "Article Date", "Tweet Date", "Tweet (truncated)"]
    for optional in ["KOF", "Target"]:
        if optional in top10_display.columns:
            cols.append(optional)
    table_ph.dataframe(top10_display[cols],
                       use_container_width=True, hide_index=True)

    y_label = "Target" if "Target" in top10_display.columns else "Tweet (truncated)"
    plot_df = top10_display.assign(
        Y=top10_display["Target"] if "Target" in top10_display.columns else top10_display["Tweet (truncated)"])
    fig = px.bar(
        plot_df.sort_values("Cosine Similarity"),
        x="Cosine Similarity", y="Y", orientation="h",
        title="Top 10 Tweet–Article Matches by Cosine Similarity",
    )
    fig.update_layout(yaxis_title=y_label,
                      xaxis_title="Cosine Similarity", height=550)
    chart_ph.plotly_chart(fig, use_container_width=True)

    # --- Influence candidates (matches × engagement)
    st.subheader("🔗 Influence candidates: Matches × Engagement")
    if "Target" not in results_df.columns:
        st.info(
            "No `Target` field in matches. Ensure tweets include a `target` column to group by influencer.")
        summary = pd.DataFrame()
    else:
        per_target_matches = (
            results_df.dropna(subset=["Target"])
            .groupby("Target")
            .agg(matches=("Cosine Similarity", "count"),
                 avg_similarity=("Cosine Similarity", "mean"),
                 first_match=("Article Date", "min"),
                 last_match=("Article Date", "max"))
            .reset_index()
            .rename(columns={"Target": "target"})
        )
        if isinstance(influencer_counts, pd.DataFrame) and "influence_score" in influencer_counts.columns:
            summary = (
                per_target_matches.merge(
                    influencer_counts.reset_index(
                    )[["target", "number_health_tweets", "influence_score", "normalized_score"]],
                    on="target", how="outer"
                )
                .fillna({"matches": 0, "avg_similarity": 0})
            ).sort_values(["matches", "normalized_score"], ascending=[False, False])
        else:
            summary = per_target_matches.copy()
            summary["number_health_tweets"] = np.nan
            summary["influence_score"] = np.nan
            summary["normalized_score"] = np.nan

    if not summary.empty:
        st.dataframe(
            summary[["target", "matches", "avg_similarity", "normalized_score",
                     "number_health_tweets", "influence_score", "first_match", "last_match"]],
            use_container_width=True, hide_index=True
        )

        # --- Overall Leaderboard (combined score + Pareto front)
        st.markdown("### 🏁 Overall Leaderboard")

        def _minmax(s: Union[pd.Series, np.ndarray]) -> pd.Series:
            ser = s if isinstance(s, pd.Series) else pd.Series(s)
            ser = pd.to_numeric(ser, errors="coerce").fillna(0.0)
            lo, hi = float(ser.min()), float(ser.max())
            return (ser - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=ser.index)

        summary["norm_matches"] = _minmax(
            summary["matches"].astype(float).pipe(np.log1p))
        summary["norm_engagement"] = summary["normalized_score"].fillna(
            0) / 100.0
        summary["norm_similarity"] = _minmax(
            summary["avg_similarity"].astype(float).fillna(0))

        half_life_days = 30
        last_ts = pd.to_datetime(
            summary["last_match"], errors="coerce", utc=True)
        age_days = (pd.Timestamp.now(tz="UTC").normalize() -
                    last_ts.dt.normalize()).dt.days
        age_days = age_days.fillna(
            age_days.max() if age_days.notna().any() else 0)
        summary["recency_score"] = np.exp(-np.log(2)
                                          * age_days / half_life_days)

        wM, wE, wS, wR = 0.50, 0.40, 0.10, 0.00
        wsum = wM + wE + wS + wR
        wM, wE, wS, wR = [w/wsum for w in (wM, wE, wS, wR)]
        summary["overall_score"] = 100 * (
            wM * summary["norm_matches"] +
            wE * summary["norm_engagement"] +
            wS * summary["norm_similarity"] +
            wR * summary["recency_score"]
        )

        pts = summary[["norm_matches", "norm_engagement"]].to_numpy()
        dominates = (pts[:, None, :] >= pts[None, :, :]).all(
            axis=2) & (pts[:, None, :] > pts[None, :, :]).any(axis=2)
        summary["pareto_front"] = ~dominates.any(axis=0)

        leaderboard = (
            summary.sort_values(["overall_score", "matches", "avg_similarity"], ascending=[
                                False, False, False])
            .reset_index(drop=True)
        )
        leaderboard_top = leaderboard.head(int(top_k))

        st.dataframe(
            leaderboard_top[["target", "overall_score", "matches", "avg_similarity",
                             "normalized_score", "number_health_tweets",
                             "pareto_front", "first_match", "last_match"]],
            use_container_width=True, hide_index=True
        )

        fig_rank = px.scatter(
            leaderboard_top,
            x="normalized_score", y="matches",
            color="overall_score", size="number_health_tweets",
            hover_name="target",
            labels={
                "normalized_score": "Composite Score (0–100)", "matches": "# of tweet→article matches"},
            title=f"Overall Rank: Top {int(top_k)} by Combined Score (size = #health tweets)",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

        if len(leaderboard_top):
            best_row = leaderboard_top.iloc[0]
            st.success(
                f"**Best overall (current weights):** {best_row['target']}  "
                f"(Overall Score: {best_row['overall_score']:.1f}, "
                f"Matches: {int(best_row['matches'])}, "
                f"Avg Similarity: {best_row['avg_similarity']:.3f}, "
                f"Engagement Composite: {best_row['normalized_score']:.1f})"
            )

    # ------------------------------------------------------------------
    # NEW: KOF Leaderboard (originators & spread)
    # ------------------------------------------------------------------
    st.markdown("### 📰 KOF Leaderboard (originators & spread)")
    if "KOF" in results_df.columns and results_df["KOF"].notna().any():
        kof_summary = (
            results_df.dropna(subset=["KOF"])
            .groupby("KOF", dropna=True)
            .agg(matches=("Cosine Similarity", "count"),
                 unique_targets=("Target", lambda s: s.dropna().nunique()),
                 avg_similarity=("Cosine Similarity", "mean"),
                 first_match=("Article Date", "min"),
                 last_match=("Article Date", "max"))
            .reset_index()
        )

        if "kof" in tweets_df.columns:
            tmp = tweets_df.copy()
            for c in ["retweets", "quotes", "favorites", "replies"]:
                if c not in tmp.columns:
                    tmp[c] = 0
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)
            echo = (
                tmp.dropna(subset=["kof"])
                .groupby("kof", dropna=True)
                .agg(echo_tweets=("text", "count"),
                     retweets=("retweets", "sum"),
                     quotes=("quotes", "sum"),
                     favorites=("favorites", "sum"),
                     replies=("replies", "sum"))
                .reset_index()
            )
            kof_summary = kof_summary.merge(
                echo, left_on="KOF", right_on="kof", how="left").drop(columns=["kof"])
            kof_summary[["retweets", "quotes", "favorites", "replies"]] = kof_summary[[
                "retweets", "quotes", "favorites", "replies"]].fillna(0)
            kof_summary["kof_influence_score"] = (
                kof_summary["retweets"] * float(w_retweets)
                + kof_summary["quotes"] * float(w_quotes)
                + kof_summary["favorites"] * float(w_faves)
                + kof_summary["replies"] * float(w_replies)
            )
        else:
            kof_summary["echo_tweets"] = np.nan
            kof_summary["kof_influence_score"] = np.nan

        def _minmax_k(s: Union[pd.Series, np.ndarray]) -> pd.Series:
            ser = s if isinstance(s, pd.Series) else pd.Series(s)
            ser = pd.to_numeric(ser, errors="coerce").fillna(0.0)
            lo, hi = float(ser.min()), float(ser.max())
            return (ser - lo) / (hi - lo) if hi > lo else pd.Series(0.0, index=ser.index)

        kof_summary["norm_matches"] = _minmax_k(
            kof_summary["matches"].astype(float).pipe(np.log1p))
        kof_summary["norm_spread"] = _minmax_k(
            kof_summary["unique_targets"].astype(float))
        kof_summary["norm_similarity"] = _minmax_k(
            kof_summary["avg_similarity"].astype(float))
        kof_summary["norm_engagement"] = _minmax_k(
            kof_summary["kof_influence_score"].astype(float))

        last_ts_k = pd.to_datetime(
            kof_summary["last_match"], errors="coerce", utc=True)
        age_days_k = (pd.Timestamp.now(tz="UTC").normalize() -
                      last_ts_k.dt.normalize()).dt.days
        age_days_k = age_days_k.fillna(
            age_days_k.max() if age_days_k.notna().any() else 0)
        kof_summary["recency_score"] = np.exp(-np.log(2) * age_days_k / 30)

        wM, wE, wT, wS, wR = 0.35, 0.35, 0.20, 0.10, 0.00
        wsum = wM + wE + wT + wS + wR
        wM, wE, wT, wS, wR = [w / wsum for w in (wM, wE, wT, wS, wR)]
        kof_summary["overall_kof_score"] = 100 * (
            wM * kof_summary["norm_matches"]
            + wE * kof_summary["norm_engagement"]
            + wT * kof_summary["norm_spread"]
            + wS * kof_summary["norm_similarity"]
            + wR * kof_summary["recency_score"]
        )

        kof_board = kof_summary.sort_values(["overall_kof_score", "matches", "unique_targets"], ascending=[
                                            False, False, False]).reset_index(drop=True)
        kof_board_top = kof_board.head(int(top_k))

        st.dataframe(
            kof_board_top[["KOF", "overall_kof_score", "matches", "unique_targets", "avg_similarity",
                           "echo_tweets", "kof_influence_score", "first_match", "last_match"]],
            use_container_width=True, hide_index=True,
        )

        fig_kof = px.scatter(
            kof_board_top,
            x="kof_influence_score", y="matches",
            size="unique_targets", color="overall_kof_score",
            hover_name="KOF",
            labels={
                "kof_influence_score": "Engagement via Targets (weighted)", "matches": "# of tweet→article matches"},
            title=f"KOF Rank: Top {int(top_k)} originators (size = unique Targets)",
        )
        st.plotly_chart(fig_kof, use_container_width=True)

        if len(kof_board_top):
            best_kof = kof_board_top.iloc[0]
            st.info(
                f"**Top KOF:** {best_kof['KOF']}  "
                f"(Overall KOF Score: {best_kof['overall_kof_score']:.1f}, "
                f"Matches: {int(best_kof['matches'])}, "
                f"Unique Targets: {int(best_kof['unique_targets'])}, "
                f"Avg Similarity: {best_kof['avg_similarity']:.3f})"
            )
    else:
        st.info("No `KOF` values found in matches — KOF leaderboard not computed.")

    # Download full results
    st.download_button(
        "Download full matches as CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="tweet_article_matches.csv",
        mime="text/csv",
        use_container_width=True
    )
