from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

MOCK_GEO_TEMPLATES = [
    {"United States": 32, "China": 18, "Japan": 12, "Germany": 8, "United Kingdom": 7, "Brazil": 6, "Canada": 5, "South Korea": 4, "Other": 8},
    {"United States": 24, "China": 15, "Japan": 26, "Germany": 6, "United Kingdom": 5, "Brazil": 4, "Canada": 4, "South Korea": 8, "Other": 8},
    {"United States": 25, "China": 25, "Japan": 7, "Germany": 9, "United Kingdom": 6, "Brazil": 8, "Canada": 5, "South Korea": 5, "Other": 10},
    {"United States": 30, "China": 8, "Japan": 6, "Germany": 14, "United Kingdom": 12, "Brazil": 5, "Canada": 8, "South Korea": 4, "Other": 13},
    {"United States": 20, "China": 30, "Japan": 12, "Germany": 5, "United Kingdom": 4, "Brazil": 7, "Canada": 4, "South Korea": 9, "Other": 9},
    {"United States": 28, "China": 10, "Japan": 9, "Germany": 8, "United Kingdom": 7, "Brazil": 14, "Canada": 6, "South Korea": 5, "Other": 13},
    {"United States": 35, "China": 12, "Japan": 8, "Germany": 7, "United Kingdom": 8, "Brazil": 5, "Canada": 10, "South Korea": 3, "Other": 12},
    {"United States": 22, "China": 18, "Japan": 18, "Germany": 8, "United Kingdom": 6, "Brazil": 6, "Canada": 5, "South Korea": 10, "Other": 7},
]

COUNTRY_ISO3 = {
    "United States": "USA",
    "China": "CHN",
    "Japan": "JPN",
    "Germany": "DEU",
    "United Kingdom": "GBR",
    "Brazil": "BRA",
    "Canada": "CAN",
    "South Korea": "KOR",
}


def build_mock_geo_data(game_df: pd.DataFrame) -> pd.DataFrame:
    if "appID" not in game_df.columns:
        raise ValueError("The input DataFrame must contain an 'appID' column.")

    cols = ["appID"] + (["name"] if "name" in game_df.columns else [])
    unique_games = game_df[cols].drop_duplicates(subset=["appID"])
    records: list[dict] = []

    for _, row in unique_games.iterrows():
        app_id = str(row["appID"])
        game_name = row.get("name", "Unknown Game")
        digest = hashlib.sha256(app_id.encode("utf-8")).hexdigest()
        template_index = int(digest[:8], 16) % len(MOCK_GEO_TEMPLATES)
        template = MOCK_GEO_TEMPLATES[template_index]

        for country, share in template.items():
            records.append(
                {
                    "appID": row["appID"],
                    "name": game_name,
                    "Country": country,
                    "Player Share %": share,
                    "Data Source": "MOCK / SYNTHETIC",
                    "Is Mock Data": True,
                    "Mock Template": template_index + 1,
                }
            )

    return pd.DataFrame(records)


def calculate_geographic_profile(scenario_dict: dict, mock_geography: pd.DataFrame) -> pd.DataFrame:
    comp_df = scenario_dict["results"].copy()
    comp_df["weight"] = (
        comp_df["match_score_pct"] if "match_score_pct" in comp_df.columns else 1.0
    )

    scenario_appids = comp_df["appID"].unique()
    geo_sub = mock_geography[mock_geography["appID"].isin(scenario_appids)].copy()
    merged = geo_sub.merge(comp_df[["appID", "weight"]], on="appID", how="inner")

    if merged.empty:
        return pd.DataFrame(
            columns=["Country", "Weighted Comparable Share %", "Data Source", "Is Mock Data"]
        )

    if np.isclose(merged["weight"].sum(), 0):
        merged["weight"] = 1.0

    merged["Weighted_Share"] = merged["Player Share %"] * merged["weight"]
    aggregated = (
        merged.groupby("Country")
        .agg(
            Sum_Weighted_Share=("Weighted_Share", "sum"),
            Sum_Weights=("weight", "sum"),
        )
        .reset_index()
    )
    aggregated["Weighted Comparable Share %"] = (
        aggregated["Sum_Weighted_Share"] / aggregated["Sum_Weights"]
    )
    aggregated["Data Source"] = "MOCK / SYNTHETIC"
    aggregated["Is Mock Data"] = True
    return aggregated[
        ["Country", "Weighted Comparable Share %", "Data Source", "Is Mock Data"]
    ]


def compare_geo_profiles(base_geo: pd.DataFrame, comp_geo: pd.DataFrame) -> pd.DataFrame:
    base = base_geo.set_index("Country")
    comp = comp_geo.set_index("Country")
    countries = base.index.intersection(comp.index)
    records = []
    for country in countries:
        base_share = base.loc[country, "Weighted Comparable Share %"]
        comp_share = comp.loc[country, "Weighted Comparable Share %"]
        records.append(
            {
                "Country": country,
                "Base": base_share,
                "Scenario": comp_share,
                "Difference": comp_share - base_share,
                "Data Source": "MOCK / SYNTHETIC",
                "Is Mock Data": True,
            }
        )
    return pd.DataFrame(records)
