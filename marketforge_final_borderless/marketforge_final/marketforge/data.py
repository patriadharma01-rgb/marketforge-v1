from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd

DATASET_ID = "Z02Z/steam-games-dataset"
DATASET_SPLIT = "train"

CANDIDATE_COLS = [
    "appID",
    "name",
    "release_date",
    "estimated_owners",
    "peak_ccu",
    "price",
    "short_description",
    "detailed_description",
    "developers",
    "publishers",
    "genres",
    "categories",
    "tags",
    "supported_languages",
    "positive",
    "negative",
    "average_playtime_forever",
    "median_playtime_forever",
]

LIST_COLS = [
    "genres",
    "tags",
    "categories",
    "supported_languages",
    "developers",
    "publishers",
]

NUMERIC_COLS = [
    "price",
    "peak_ccu",
    "positive",
    "negative",
    "average_playtime_forever",
    "median_playtime_forever",
]


def clean_and_parse_list(value) -> list:
    """Convert list-like source values into clean Python lists."""
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return list(value)
    if isinstance(value, dict):
        return list(value.keys())

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return list(parsed)
        if isinstance(parsed, dict):
            return list(parsed.keys())
    except (ValueError, SyntaxError):
        pass

    return [item.strip() for item in text.split(",") if item.strip()]


def parse_owners_range(value) -> tuple[float, float, float]:
    """Parse SteamSpy ownership bands while preserving missingness."""
    try:
        if pd.isna(value):
            return np.nan, np.nan, np.nan
    except (TypeError, ValueError):
        pass

    text = str(value).replace(",", "").replace(" ", "")
    if "-" in text:
        parts = text.split("-")
        try:
            lower = int(parts[0])
            upper = int(parts[1])
            return lower, upper, (lower + upper) / 2
        except (ValueError, IndexError):
            return np.nan, np.nan, np.nan

    try:
        single = int(text)
        return single, single, single
    except ValueError:
        return np.nan, np.nan, np.nan


def prepare_games(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the canonical MarketForge games table.

    Rules preserved from the research notebook:
    - one row per appID;
    - missing price stays missing, only an actual zero is free;
    - invalid/missing owners stay NaN;
    - zero-review titles have NaN positive-review percentage;
    - playtime remains stored in source minutes until presentation/benchmarking.
    """
    available_cols = [col for col in CANDIDATE_COLS if col in raw_df.columns]
    games = raw_df[available_cols].copy()

    if "appID" in games.columns:
        games = games.drop_duplicates(subset="appID", keep="first")

    for col in LIST_COLS:
        if col in games.columns:
            games[col] = games[col].apply(clean_and_parse_list)

    for col in NUMERIC_COLS:
        if col in games.columns:
            games[col] = pd.to_numeric(games[col], errors="coerce")

    if "release_date" in games.columns:
        games["release_date"] = pd.to_datetime(games["release_date"], errors="coerce")
        games["release_year"] = games["release_date"].dt.year

    if "price" in games.columns:
        games["is_free"] = games["price"].eq(0)

    if "estimated_owners" in games.columns:
        parsed = games["estimated_owners"].apply(parse_owners_range)
        games["owners_min"] = [x[0] for x in parsed]
        games["owners_max"] = [x[1] for x in parsed]
        games["owners_midpoint"] = [x[2] for x in parsed]

    if {"positive", "negative"}.issubset(games.columns):
        games["total_reviews"] = games["positive"] + games["negative"]
        games["positive_review_pct"] = np.where(
            games["total_reviews"] > 0,
            (games["positive"] / games["total_reviews"]) * 100,
            np.nan,
        )

    if "release_year" in games.columns:
        current_year = pd.Timestamp.today().year
        games["game_age"] = current_year - games["release_year"]

    return games.reset_index(drop=True)


def load_games(dataset_id: str = DATASET_ID, split: str = DATASET_SPLIT) -> pd.DataFrame:
    """Download the public Steam dataset from Hugging Face and prepare it."""
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split=split)
    return prepare_games(dataset.to_pandas())


def get_unique_metadata(games: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return sorted unique Steam tags and genres."""
    tags: set[str] = set()
    genres: set[str] = set()

    if "tags" in games.columns:
        for values in games["tags"].dropna():
            if isinstance(values, list):
                tags.update(str(v).strip() for v in values if str(v).strip())

    if "genres" in games.columns:
        for values in games["genres"].dropna():
            if isinstance(values, list):
                genres.update(str(v).strip() for v in values if str(v).strip())

    return sorted(tags, key=str.lower), sorted(genres, key=str.lower)


def search_games_database(games: pd.DataFrame, query: str, limit: int = 50) -> pd.DataFrame:
    """Search title, AppID, tag or genre; cap output at 50 rows."""
    query = str(query or "").strip()
    if len(query) < 2:
        return pd.DataFrame()

    limit = max(1, min(int(limit), 50))
    needle = query.casefold()

    name_mask = (
        games.get("name", pd.Series("", index=games.index))
        .astype(str)
        .str.casefold()
        .str.contains(needle, regex=False, na=False)
    )
    appid_mask = (
        games.get("appID", pd.Series("", index=games.index))
        .astype(str)
        .str.casefold()
        .str.contains(needle, regex=False, na=False)
    )

    if "tags" in games.columns:
        tags_mask = games["tags"].apply(
            lambda values: any(needle in str(v).casefold() for v in values)
            if isinstance(values, list)
            else False
        )
    else:
        tags_mask = pd.Series(False, index=games.index)

    if "genres" in games.columns:
        genres_mask = games["genres"].apply(
            lambda values: any(needle in str(v).casefold() for v in values)
            if isinstance(values, list)
            else False
        )
    else:
        genres_mask = pd.Series(False, index=games.index)

    filtered = games[name_mask | appid_mask | tags_mask | genres_mask].head(limit).copy()
    preferred = [
        "appID",
        "name",
        "release_year",
        "price",
        "estimated_owners",
        "owners_midpoint",
        "positive_review_pct",
        "total_reviews",
        "peak_ccu",
        "tags",
        "genres",
        "short_description",
    ]
    return filtered[[c for c in preferred if c in filtered.columns]].reset_index(drop=True)
