from __future__ import annotations

import numpy as np
import pandas as pd

from .benchmark import benchmark_competitors, build_release_year_profile, get_market_metadata_profile
from .geography import build_mock_geo_data, calculate_geographic_profile, compare_geo_profiles
from .matcher import MatcherModel


def run_game_scenario(
    model: MatcherModel,
    scenario_name: str,
    concept: str,
    target_tags,
    target_genres=None,
    top_n: int = 10,
) -> dict:
    results, effective_weights = model.find_competitors(
        concept=concept,
        target_tags=target_tags,
        target_genres=target_genres,
        top_n=top_n,
    )
    results = results.copy()
    results["Rank"] = np.arange(1, len(results) + 1)
    tag_profile, genre_profile = get_market_metadata_profile(results, top_n=10)

    return {
        "scenario_name": scenario_name,
        "results": results,
        "benchmark": benchmark_competitors(results),
        "tag_profile": tag_profile,
        "genre_profile": genre_profile,
        "release_year_profile": build_release_year_profile(results),
        "effective_weights": effective_weights.copy(),
    }


def compare_competitor_sets(base: dict, comparison: dict) -> pd.DataFrame:
    df_base = base["results"].copy().set_index("appID")
    df_comp = comparison["results"].copy().set_index("appID")
    appids = set(df_base.index).union(set(df_comp.index))
    records = []

    for appid in appids:
        in_base = appid in df_base.index
        in_comp = appid in df_comp.index
        name = df_base.loc[appid, "name"] if in_base else df_comp.loc[appid, "name"]
        base_rank = df_base.loc[appid, "Rank"] if in_base else np.nan
        comp_rank = df_comp.loc[appid, "Rank"] if in_comp else np.nan
        base_score = df_base.loc[appid, "match_score_pct"] if in_base else np.nan
        comp_score = df_comp.loc[appid, "match_score_pct"] if in_comp else np.nan

        if in_base and in_comp:
            status = "Stayed"
            rank_change = base_rank - comp_rank
        elif in_comp:
            status = "Entered"
            rank_change = np.nan
        else:
            status = "Exited"
            rank_change = np.nan

        records.append(
            {
                "appID": appid,
                "name": name,
                "Base Rank": base_rank,
                "Scenario Rank": comp_rank,
                "Rank Change": rank_change,
                "Base Match Score": base_score,
                "Scenario Match Score": comp_score,
                "Status": status,
            }
        )

    return (
        pd.DataFrame(records)
        .sort_values(["Scenario Rank", "Base Rank"], na_position="last")
        .reset_index(drop=True)
    )


def compare_scenario_benchmarks(base: dict, comparison: dict) -> pd.DataFrame:
    base_bench = base["benchmark"].copy().set_index("Metric")
    comp_bench = comparison["benchmark"].copy().set_index("Metric")
    common = base_bench.index.intersection(comp_bench.index)
    records = []
    for metric in common:
        base_value = base_bench.loc[metric, "Value"]
        scenario_value = comp_bench.loc[metric, "Value"]
        records.append(
            {
                "Metric": metric,
                "Base": base_value,
                "Scenario": scenario_value,
                "Difference": scenario_value - base_value,
            }
        )
    return pd.DataFrame(records)


def analyze_market(
    model: MatcherModel,
    concept: str,
    target_tags: list[str],
    target_genres: list[str] | None = None,
    scenario_tag: str | None = None,
    top_n: int = 10,
) -> dict:
    if not target_tags:
        raise ValueError("Please select at least one required tag.")
    if top_n < 5 or top_n > 25:
        raise ValueError("Top-N must be between 5 and 25.")

    scenario_tag = str(scenario_tag or "").strip()
    if scenario_tag and scenario_tag.casefold() in {x.casefold() for x in target_tags}:
        raise ValueError("The scenario tag is already part of the Base concept.")

    base = run_game_scenario(
        model,
        "Base Concept",
        concept,
        target_tags,
        target_genres,
        top_n,
    )
    base_geo = calculate_geographic_profile(base, build_mock_geo_data(base["results"]))

    if not scenario_tag:
        geo = base_geo.rename(columns={"Weighted Comparable Share %": "Base"}).copy()
        geo["Scenario"] = geo["Base"]
        geo["Difference"] = 0.0
        return {
            "base": base,
            "comparison": None,
            "movement": pd.DataFrame(),
            "benchmark_comparison": pd.DataFrame(),
            "overlap": {"shared_count": 0, "overlap_pct": 0.0, "entered_count": 0, "exited_count": 0},
            "entered": pd.DataFrame(),
            "exited": pd.DataFrame(),
            "geo": geo,
        }

    scenario_tags = list(target_tags) + [scenario_tag]
    comparison = run_game_scenario(
        model,
        f"Add {scenario_tag}",
        concept,
        scenario_tags,
        target_genres,
        top_n,
    )
    movement = compare_competitor_sets(base, comparison)
    benchmark_comparison = compare_scenario_benchmarks(base, comparison)

    base_ids = set(base["results"]["appID"])
    comp_ids = set(comparison["results"]["appID"])
    shared = base_ids & comp_ids
    entered_ids = comp_ids - base_ids
    exited_ids = base_ids - comp_ids

    overlap = {
        "shared_count": len(shared),
        "overlap_pct": (len(shared) / len(base["results"]) * 100) if len(base["results"]) else 0.0,
        "entered_count": len(entered_ids),
        "exited_count": len(exited_ids),
    }

    entered = comparison["results"][comparison["results"]["appID"].isin(entered_ids)].copy()
    entered["Scenario Rank"] = entered["Rank"]
    entered["Scenario Match Score"] = entered["match_score_pct"]
    entered = entered[
        [c for c in ["appID", "name", "Scenario Rank", "Scenario Match Score", "tags", "genres"] if c in entered.columns]
    ].sort_values("Scenario Rank")

    exited = base["results"][base["results"]["appID"].isin(exited_ids)].copy()
    exited["Base Rank"] = exited["Rank"]
    exited["Base Match Score"] = exited["match_score_pct"]
    exited = exited[
        [c for c in ["appID", "name", "Base Rank", "Base Match Score", "tags", "genres"] if c in exited.columns]
    ].sort_values("Base Rank")

    comp_geo = calculate_geographic_profile(
        comparison, build_mock_geo_data(comparison["results"])
    )
    geo = compare_geo_profiles(base_geo, comp_geo)

    return {
        "base": base,
        "comparison": comparison,
        "movement": movement,
        "benchmark_comparison": benchmark_comparison,
        "overlap": overlap,
        "entered": entered.reset_index(drop=True),
        "exited": exited.reset_index(drop=True),
        "geo": geo,
    }
