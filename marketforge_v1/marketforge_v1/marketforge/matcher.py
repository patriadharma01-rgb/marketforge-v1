from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = [values]
    return [str(x).strip().lower() for x in values if str(x).strip()]


def coverage_score(game_values, target_values: list[str]) -> float:
    if not target_values or not isinstance(game_values, list):
        return 0.0
    game_set = {str(x).strip().lower() for x in game_values}
    target_set = set(target_values)
    return len(game_set & target_set) / len(target_set)


@dataclass
class MatcherModel:
    matching_games: pd.DataFrame
    tfidf: TfidfVectorizer
    game_matrix: object

    @classmethod
    def from_games(cls, games: pd.DataFrame) -> "MatcherModel":
        matching_games = games.copy()
        matching_games = matching_games[matching_games["name"].notna()].copy()

        short_description = matching_games.get(
            "short_description",
            pd.Series("", index=matching_games.index),
        )
        matching_games["short_description"] = short_description.fillna("").astype(str)
        matching_games["tags"] = matching_games.get(
            "tags", pd.Series([[]] * len(matching_games), index=matching_games.index)
        ).apply(lambda x: x if isinstance(x, list) else [])
        matching_games["genres"] = matching_games.get(
            "genres", pd.Series([[]] * len(matching_games), index=matching_games.index)
        ).apply(lambda x: x if isinstance(x, list) else [])

        matching_games["matching_text"] = matching_games["short_description"]

        tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
        )
        game_matrix = tfidf.fit_transform(matching_games["matching_text"])

        return cls(
            matching_games=matching_games.reset_index(drop=True),
            tfidf=tfidf,
            game_matrix=game_matrix,
        )

    def find_competitors(
        self,
        concept: str,
        target_tags=None,
        target_genres=None,
        top_n: int = 10,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Return ranked competitors and the effective normalized signal weights."""
        target_tags = normalize_list(target_tags)
        target_genres = normalize_list(target_genres)

        has_tags = len(target_tags) > 0
        has_description = concept is not None and bool(str(concept).strip())
        has_genres = len(target_genres) > 0

        if not has_tags:
            raise ValueError("At least one target tag is required for competitor matching.")

        active_weights = {"tag": 0.50}
        if has_description:
            active_weights["text"] = 0.35
        if has_genres:
            active_weights["genre"] = 0.15

        total = sum(active_weights.values())
        normalized = {key: value / total for key, value in active_weights.items()}
        effective_weights = {
            "tag_weight": normalized.get("tag", 0.0),
            "text_weight": normalized.get("text", 0.0),
            "genre_weight": normalized.get("genre", 0.0),
        }

        results = self.matching_games.copy()

        if has_description:
            concept_vector = self.tfidf.transform([str(concept)])
            results["text_similarity"] = cosine_similarity(
                concept_vector, self.game_matrix
            ).flatten()
        else:
            results["text_similarity"] = 0.0

        results["tag_similarity"] = results["tags"].apply(
            lambda x: coverage_score(x, target_tags)
        )

        if has_genres:
            results["genre_similarity"] = results["genres"].apply(
                lambda x: coverage_score(x, target_genres)
            )
        else:
            results["genre_similarity"] = 0.0

        results["match_score"] = (
            effective_weights["tag_weight"] * results["tag_similarity"]
            + effective_weights["text_weight"] * results["text_similarity"]
            + effective_weights["genre_weight"] * results["genre_similarity"]
        )
        results["match_score_pct"] = results["match_score"] * 100

        results = results.sort_values(
            by=[
                "match_score",
                "tag_similarity",
                "text_similarity",
                "genre_similarity",
                "appID",
            ],
            ascending=[False, False, False, False, True],
        )

        output_cols = [
            col
            for col in [
                "appID",
                "name",
                "match_score_pct",
                "text_similarity",
                "tag_similarity",
                "genre_similarity",
                "genres",
                "tags",
                "price",
                "is_free",
                "estimated_owners",
                "owners_midpoint",
                "peak_ccu",
                "total_reviews",
                "positive_review_pct",
                "average_playtime_forever",
                "median_playtime_forever",
                "release_year",
                "short_description",
            ]
            if col in results.columns
        ]

        return results[output_cols].head(int(top_n)).reset_index(drop=True), effective_weights
