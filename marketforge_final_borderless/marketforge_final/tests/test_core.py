import numpy as np
import pandas as pd

from marketforge.benchmark import benchmark_competitors, build_release_year_profile
from marketforge.data import prepare_games
from marketforge.geography import MOCK_GEO_TEMPLATES, build_mock_geo_data, calculate_geographic_profile
from marketforge.matcher import MatcherModel
from marketforge.scenario import analyze_market, compare_competitor_sets


def raw_sample():
    return pd.DataFrame(
        {
            "appID": [1, 2, 3, 4, 5, 6],
            "name": ["A", "B", "C", "D", "E", "F"],
            "release_date": ["2024-01-01", "2023-01-01", "2022-01-01", "2021-01-01", "2020-01-01", "2019-01-01"],
            "estimated_owners": ["1000 - 3000", None, "0 - 20000", "5000 - 10000", "100 - 200", "300 - 700"],
            "peak_ccu": [100, 20, 40, 5, 12, 8],
            "price": [9.99, 0, np.nan, 19.99, 4.99, 14.99],
            "short_description": [
                "cozy fantasy potion dungeon adventure",
                "cozy fantasy shop management adventure",
                "fantasy dungeon roguelike adventure",
                "strategy fantasy management adventure",
                "cozy crafting fantasy adventure",
                "dungeon fantasy role playing adventure",
            ],
            "genres": [
                ["RPG", "Simulation"],
                ["Simulation"],
                ["RPG"],
                ["Strategy"],
                ["Simulation"],
                ["RPG"],
            ],
            "tags": [
                ["Cozy", "RPG", "Management", "Dungeon Crawler", "Fantasy"],
                ["Cozy", "Management", "Fantasy"],
                ["RPG", "Roguelike", "Dungeon Crawler", "Fantasy"],
                ["Strategy", "Management", "Fantasy"],
                ["Cozy", "Crafting", "Fantasy"],
                ["RPG", "Dungeon Crawler", "Fantasy"],
            ],
            "categories": [[]] * 6,
            "supported_languages": [["English"]] * 6,
            "developers": [["Dev"]] * 6,
            "publishers": [["Pub"]] * 6,
            "positive": [90, 0, 70, 20, 30, 10],
            "negative": [10, 0, 30, 5, 10, 10],
            "average_playtime_forever": [600, 120, 300, 60, 180, 240],
            "median_playtime_forever": [300, 60, 180, 30, 120, 150],
        }
    )


def test_cleaning_preserves_missingness():
    games = prepare_games(raw_sample())
    assert np.isnan(games.loc[2, "price"])
    assert bool(games.loc[1, "is_free"])
    assert not bool(games.loc[2, "is_free"])
    assert np.isnan(games.loc[1, "owners_midpoint"])
    assert np.isnan(games.loc[1, "positive_review_pct"])


def test_matcher_weights_and_required_tags():
    games = prepare_games(raw_sample())
    model = MatcherModel.from_games(games)
    results, weights = model.find_competitors(
        concept="cozy fantasy potion dungeon adventure",
        target_tags=["Cozy", "RPG", "Fantasy"],
        target_genres=["RPG"],
        top_n=5,
    )
    assert len(results) == 5
    assert weights == {"tag_weight": 0.5, "text_weight": 0.35, "genre_weight": 0.15}

    _, no_text = model.find_competitors(
        concept="",
        target_tags=["Cozy", "RPG"],
        target_genres=["RPG"],
        top_n=5,
    )
    assert no_text["text_weight"] == 0
    assert np.isclose(no_text["tag_weight"] + no_text["genre_weight"], 1.0)


def test_benchmark_and_release_profile():
    games = prepare_games(raw_sample())
    model = MatcherModel.from_games(games)
    results, _ = model.find_competitors("fantasy adventure", ["Fantasy"], ["RPG"], 5)
    results["Rank"] = np.arange(1, len(results) + 1)
    bench = benchmark_competitors(results)
    assert "Median Paid Price" in set(bench["Metric"])
    profile = build_release_year_profile(results)
    assert set(["Release Year", "Comparable Games"]).issubset(profile.columns)


def test_mock_geography_is_deterministic_and_sums_to_100():
    assert len(MOCK_GEO_TEMPLATES) == 8
    assert all(sum(template.values()) == 100 for template in MOCK_GEO_TEMPLATES)

    games = prepare_games(raw_sample())
    subset = games[["appID", "name"]].head(3)
    first = build_mock_geo_data(subset)
    second = build_mock_geo_data(subset)
    pd.testing.assert_frame_equal(first, second)
    sums = first.groupby("appID")["Player Share %"].sum()
    assert (sums == 100).all()


def test_scenario_and_geo_profile():
    games = prepare_games(raw_sample())
    model = MatcherModel.from_games(games)
    analysis = analyze_market(
        model,
        concept="cozy fantasy dungeon adventure",
        target_tags=["Cozy", "RPG", "Fantasy"],
        target_genres=["RPG"],
        scenario_tag="Roguelike",
        top_n=5,
    )
    assert len(analysis["base"]["results"]) <= 5
    assert len(analysis["comparison"]["results"]) <= 5
    assert np.isclose(analysis["geo"]["Base"].sum(), 100.0)
    assert np.isclose(analysis["geo"]["Scenario"].sum(), 100.0)


def test_rank_change_direction():
    base = {"results": pd.DataFrame({"appID": [1, 2], "name": ["A", "B"], "Rank": [2, 1], "match_score_pct": [50, 60]})}
    comparison = {"results": pd.DataFrame({"appID": [1, 3], "name": ["A", "C"], "Rank": [1, 2], "match_score_pct": [70, 40]})}
    movement = compare_competitor_sets(base, comparison)
    a = movement[movement["appID"] == 1].iloc[0]
    assert a["Rank Change"] == 1  # moved from rank 2 to rank 1
    assert a["Status"] == "Stayed"
