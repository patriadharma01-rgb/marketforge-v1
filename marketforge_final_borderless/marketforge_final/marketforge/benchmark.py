from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def benchmark_competitors(results_df: pd.DataFrame) -> pd.DataFrame:
    df_comps = results_df.copy()
    paid_only = (
        df_comps[df_comps["price"] > 0]
        if "price" in df_comps.columns
        else pd.DataFrame()
    )
    metrics: list[tuple[str, float]] = []

    metrics.append(("Comparable Games", len(df_comps)))

    if not paid_only.empty:
        metrics.extend(
            [
                ("Minimum Paid Price", paid_only["price"].min()),
                ("25th Percentile Paid Price", paid_only["price"].quantile(0.25)),
                ("Median Paid Price", paid_only["price"].median()),
                ("75th Percentile Paid Price", paid_only["price"].quantile(0.75)),
                ("Maximum Paid Price", paid_only["price"].max()),
            ]
        )
    else:
        metrics.extend(
            [
                ("Minimum Paid Price", np.nan),
                ("25th Percentile Paid Price", np.nan),
                ("Median Paid Price", np.nan),
                ("75th Percentile Paid Price", np.nan),
                ("Maximum Paid Price", np.nan),
            ]
        )

    if "price" in df_comps.columns:
        known_prices = df_comps["price"].dropna()
        free_share = known_prices.eq(0).mean() * 100 if len(known_prices) else np.nan
        metrics.append(("Free Game Share %", free_share))
    else:
        metrics.append(("Free Game Share %", np.nan))

    def add_quantiles(column: str, label: str):
        if column in df_comps.columns:
            series = df_comps[column]
            metrics.extend(
                [
                    (f"25th Percentile {label}", series.quantile(0.25)),
                    (f"Median {label}", series.median()),
                    (f"75th Percentile {label}", series.quantile(0.75)),
                ]
            )
        else:
            metrics.extend(
                [
                    (f"25th Percentile {label}", np.nan),
                    (f"Median {label}", np.nan),
                    (f"75th Percentile {label}", np.nan),
                ]
            )

    if "owners_midpoint" in df_comps.columns:
        series = df_comps["owners_midpoint"]
        metrics.extend(
            [
                ("Minimum Owners Midpoint", series.min()),
                ("25th Percentile Owners Midpoint", series.quantile(0.25)),
                ("Median Owners Midpoint", series.median()),
                ("75th Percentile Owners Midpoint", series.quantile(0.75)),
                ("Maximum Owners Midpoint", series.max()),
            ]
        )
    else:
        metrics.extend(
            [
                ("Minimum Owners Midpoint", np.nan),
                ("25th Percentile Owners Midpoint", np.nan),
                ("Median Owners Midpoint", np.nan),
                ("75th Percentile Owners Midpoint", np.nan),
                ("Maximum Owners Midpoint", np.nan),
            ]
        )

    add_quantiles("positive_review_pct", "Positive Review %")
    add_quantiles("total_reviews", "Review Count")

    if "peak_ccu" in df_comps.columns:
        series = df_comps["peak_ccu"]
        metrics.extend(
            [
                ("25th Percentile Recent CCU", series.quantile(0.25)),
                ("Median Recent CCU", series.median()),
                ("75th Percentile Recent CCU", series.quantile(0.75)),
                ("Maximum Recent CCU", series.max()),
            ]
        )
    else:
        metrics.extend(
            [
                ("25th Percentile Recent CCU", np.nan),
                ("Median Recent CCU", np.nan),
                ("75th Percentile Recent CCU", np.nan),
                ("Maximum Recent CCU", np.nan),
            ]
        )

    if "average_playtime_forever" in df_comps.columns:
        avg_h = df_comps["average_playtime_forever"] / 60.0
        metrics.extend(
            [
                ("25th Percentile Average Playtime Hours", avg_h.quantile(0.25)),
                ("Median Average Playtime Hours", avg_h.median()),
                ("75th Percentile Average Playtime Hours", avg_h.quantile(0.75)),
            ]
        )
    else:
        metrics.extend(
            [
                ("25th Percentile Average Playtime Hours", np.nan),
                ("Median Average Playtime Hours", np.nan),
                ("75th Percentile Average Playtime Hours", np.nan),
            ]
        )

    if "median_playtime_forever" in df_comps.columns:
        med_h = df_comps["median_playtime_forever"] / 60.0
        metrics.extend(
            [
                ("25th Percentile Typical Playtime Hours", med_h.quantile(0.25)),
                ("Median Typical Playtime Hours", med_h.median()),
                ("75th Percentile Typical Playtime Hours", med_h.quantile(0.75)),
            ]
        )
    else:
        metrics.extend(
            [
                ("25th Percentile Typical Playtime Hours", np.nan),
                ("Median Typical Playtime Hours", np.nan),
                ("75th Percentile Typical Playtime Hours", np.nan),
            ]
        )

    if "release_year" in df_comps.columns:
        series = df_comps["release_year"]
        metrics.extend(
            [
                ("Minimum Release Year", series.min()),
                ("Median Release Year", series.median()),
                ("Maximum Release Year", series.max()),
            ]
        )
    else:
        metrics.extend(
            [
                ("Minimum Release Year", np.nan),
                ("Median Release Year", np.nan),
                ("Maximum Release Year", np.nan),
            ]
        )

    return pd.DataFrame(metrics, columns=["Metric", "Value"])


def get_market_metadata_profile(
    results_df: pd.DataFrame, top_n: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_games = len(results_df)

    all_tags: list[str] = []
    if "tags" in results_df.columns:
        for values in results_df["tags"].dropna():
            if isinstance(values, list):
                all_tags.extend(list(set(values)))

    tag_counts = Counter(all_tags).most_common(top_n)
    tag_profile = pd.DataFrame(
        [
            {
                "Tag": tag,
                "Games": count,
                "Competitor Share %": (count / total_games) * 100 if total_games else 0.0,
            }
            for tag, count in tag_counts
        ],
        columns=["Tag", "Games", "Competitor Share %"],
    )

    all_genres: list[str] = []
    if "genres" in results_df.columns:
        for values in results_df["genres"].dropna():
            if isinstance(values, list):
                all_genres.extend(list(set(values)))

    genre_counts = Counter(all_genres).most_common(top_n)
    genre_profile = pd.DataFrame(
        [
            {
                "Genre": genre,
                "Games": count,
                "Competitor Share %": (count / total_games) * 100 if total_games else 0.0,
            }
            for genre, count in genre_counts
        ],
        columns=["Genre", "Games", "Competitor Share %"],
    )

    return tag_profile, genre_profile


def build_release_year_profile(results_df: pd.DataFrame) -> pd.DataFrame:
    profile = results_df.copy()
    if "release_year" in profile.columns:
        profile = profile[profile["release_year"].notna()].copy()
        profile["release_year"] = profile["release_year"].astype(int)

    profile["average_playtime_hours"] = (
        profile["average_playtime_forever"] / 60.0
        if "average_playtime_forever" in profile.columns
        else np.nan
    )
    profile["typical_playtime_hours"] = (
        profile["median_playtime_forever"] / 60.0
        if "median_playtime_forever" in profile.columns
        else np.nan
    )
    if "peak_ccu" not in profile.columns:
        profile["peak_ccu"] = np.nan

    if profile.empty or "release_year" not in profile.columns:
        return pd.DataFrame(
            columns=[
                "Release Year",
                "Comparable Games",
                "Median Recent CCU",
                "Median Average Playtime Hours",
                "Median Typical Playtime Hours",
            ]
        )

    grouped = (
        profile.groupby("release_year")
        .agg(
            Comparable_Games=("appID", "count"),
            Median_Recent_CCU=("peak_ccu", "median"),
            Median_Average_Playtime_Hours=("average_playtime_hours", "median"),
            Median_Typical_Playtime_Hours=("typical_playtime_hours", "median"),
        )
        .reset_index()
    )
    grouped.columns = [
        "Release Year",
        "Comparable Games",
        "Median Recent CCU",
        "Median Average Playtime Hours",
        "Median Typical Playtime Hours",
    ]
    return grouped.sort_values("Release Year").reset_index(drop=True)
