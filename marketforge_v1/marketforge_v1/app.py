from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from marketforge.data import get_unique_metadata, load_games, search_games_database
from marketforge.geography import COUNTRY_ISO3
from marketforge.matcher import MatcherModel
from marketforge.scenario import analyze_market


st.set_page_config(
    page_title="MarketForge — Game Concept Market Simulator",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #0a0d14; color: #f4f7fb; }
    [data-testid="stSidebar"] { background: #111722; border-right: 1px solid #283041; }
    .block-container { max-width: 1500px; padding-top: 1.8rem; padding-bottom: 4rem; }
    .mf-kicker { color:#7fa6ff; font-size:.78rem; font-weight:800; letter-spacing:.13em; text-transform:uppercase; }
    .mf-title { font-size:2.45rem; font-weight:850; letter-spacing:-.045em; margin:.15rem 0 0; }
    .mf-subtitle { color:#9aa6b6; margin-top:.3rem; margin-bottom:1rem; }
    .mf-note { padding:.85rem 1rem; border-radius:10px; border-left:3px solid #5a8cff; background:#121a2b; color:#b9c3d2; margin:.4rem 0 1rem; }
    .mf-warning { padding:.9rem 1rem; border-radius:10px; border-left:3px solid #f5b84b; background:#241b0d; color:#f4d7a1; margin:.4rem 0 1rem; }
    .mf-small { color:#8d98a8; font-size:.82rem; }
    div[data-testid="stMetric"] { background:#121722; border:1px solid #293142; padding:.85rem; border-radius:12px; }
    div[data-testid="stDataFrame"] { border:1px solid #293142; border-radius:10px; overflow:hidden; }
    hr { border-color:#293142 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_CONCEPT = (
    "A cozy fantasy game where the player runs a potion shop during the day "
    "and explores dangerous dungeons at night to collect ingredients and fight monsters."
)
DEFAULT_TAGS = ["Cozy", "RPG", "Management", "Dungeon Crawler", "Fantasy"]
DEFAULT_GENRES = ["RPG", "Simulation"]


@st.cache_resource(show_spinner=False)
def load_assets():
    games = load_games()
    matcher = MatcherModel.from_games(games)
    tags, genres = get_unique_metadata(games)
    return games, matcher, tags, genres


@st.cache_data(show_spinner=False)
def cached_search(query: str, limit: int = 50):
    games, _, _, _ = load_assets()
    return search_games_database(games, query, limit)


def finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def fmt_number(value, decimals: int = 0) -> str:
    if not finite(value):
        return "—"
    return f"{float(value):,.{decimals}f}"


def fmt_price(value) -> str:
    if not finite(value):
        return "—"
    if float(value) == 0:
        return "Free"
    return f"${float(value):,.2f}"


def fmt_pct(value) -> str:
    return "—" if not finite(value) else f"{float(value):.1f}%"


def fmt_pp(value) -> str:
    return "—" if not finite(value) else f"{float(value):+.1f} pp"


def fmt_owners(value) -> str:
    if not finite(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K".replace(".0K", "K")
    return f"{value:,.0f}"


def fmt_hours(value) -> str:
    return "—" if not finite(value) else f"{float(value):.1f} h"


def benchmark_dict(df: pd.DataFrame) -> dict:
    return dict(zip(df["Metric"], df["Value"])) if not df.empty else {}


def format_metric(metric: str, value) -> str:
    name = metric.lower()
    if "price" in name:
        return fmt_price(value)
    if "free game share" in name or "positive review" in name:
        return fmt_pct(value)
    if "owners" in name:
        return fmt_owners(value)
    if "playtime" in name:
        return fmt_hours(value)
    if "release year" in name or "recent ccu" in name or "review count" in name or "comparable games" in name:
        return fmt_number(value, 0)
    return fmt_number(value, 2)


def format_difference(metric: str, value) -> str:
    if not finite(value):
        return "—"
    name = metric.lower()
    value = float(value)
    if "free game share" in name or "positive review" in name:
        return fmt_pp(value)
    if "price" in name:
        return f"{value:+.2f}"
    if "owners" in name:
        return ("+" if value > 0 else "") + fmt_owners(value)
    if "playtime" in name:
        return f"{value:+.1f} h"
    return f"{value:+,.2f}"


def prepare_competitor_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["Match Score %"] = out["match_score_pct"]
    out["Tag Similarity %"] = out["tag_similarity"] * 100
    out["Text Similarity %"] = out["text_similarity"] * 100
    out["Genre Similarity %"] = out["genre_similarity"] * 100
    out["Price"] = out["price"].apply(fmt_price)
    out["Owners Midpoint"] = out["owners_midpoint"].apply(fmt_owners)
    out["Recent CCU"] = out["peak_ccu"].apply(lambda x: fmt_number(x, 0))
    out["Positive Review %"] = out["positive_review_pct"].apply(fmt_pct)
    out["Release Year"] = out["release_year"].apply(lambda x: fmt_number(x, 0))
    out["Tags"] = out["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
    out["Genres"] = out["genres"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
    return out[
        [
            "Rank",
            "name",
            "Match Score %",
            "Tag Similarity %",
            "Text Similarity %",
            "Genre Similarity %",
            "Price",
            "Owners Midpoint",
            "Recent CCU",
            "Positive Review %",
            "Release Year",
            "Tags",
            "Genres",
        ]
    ].rename(columns={"name": "Game"})


def render_metric_grid(items: list[tuple[str, str]], columns: int = 3):
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (label, value) in zip(cols, items[start : start + columns]):
            col.metric(label, value)


def render_geo_map(geo: pd.DataFrame, mode: str):
    map_df = geo[geo["Country"] != "Other"].copy()
    map_df["ISO3"] = map_df["Country"].map(COUNTRY_ISO3)
    map_df = map_df[map_df["ISO3"].notna()].copy()

    if mode == "Base":
        z_col = "Base"
        colorscale = [[0, "#4b3510"], [0.45, "#b57919"], [1, "#ffd66b"]]
        zmin, zmax = 0, max(40, float(map_df[z_col].max() or 0))
        colorbar_title = "Share %"
    elif mode == "Scenario":
        z_col = "Scenario"
        colorscale = [[0, "#4b3510"], [0.45, "#b57919"], [1, "#ffd66b"]]
        zmin, zmax = 0, max(40, float(map_df[z_col].max() or 0))
        colorbar_title = "Share %"
    else:
        z_col = "Difference"
        colorscale = [[0, "#d34f68"], [0.5, "#353b48"], [1, "#45c98a"]]
        bound = max(1.0, float(np.nanmax(np.abs(map_df[z_col].to_numpy(dtype=float)))))
        zmin, zmax = -bound, bound
        colorbar_title = "pp"

    custom = np.stack(
        [
            map_df["Country"],
            map_df["Base"],
            map_df["Scenario"],
            map_df["Difference"],
        ],
        axis=-1,
    )

    fig = go.Figure(
        go.Choropleth(
            locations=map_df["ISO3"],
            z=map_df[z_col],
            locationmode="ISO-3",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            marker_line_color="#0a0d14",
            marker_line_width=0.6,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Base: %{customdata[1]:.1f}%<br>"
                "Scenario: %{customdata[2]:.1f}%<br>"
                "Difference: %{customdata[3]:+.1f} pp<extra></extra>"
            ),
            colorbar=dict(title=colorbar_title, thickness=12),
        )
    )
    fig.update_geos(
        projection_type="natural earth",
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="#303747",
        showland=True,
        landcolor="#222936",
        bgcolor="#0a0d14",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#0a0d14",
        plot_bgcolor="#0a0d14",
        height=520,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ---------- Load model ----------
st.markdown('<div class="mf-kicker">Gaming market analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="mf-title">MarketForge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="mf-subtitle">Stress-test a game concept against real Steam comparables, market benchmarks, scenario shifts, and a clearly-labelled synthetic geography prototype.</div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading Steam dataset and building the TF-IDF market index…"):
        games, matcher, all_tags, all_genres = load_assets()
except Exception as exc:
    st.error("MarketForge could not initialize the Steam dataset/model.")
    st.exception(exc)
    st.stop()

st.caption(
    f"{len(games):,} cleaned Steam titles • {len(all_tags):,} available tags • {len(all_genres):,} genres"
)

# ---------- Controls ----------
with st.sidebar:
    st.markdown("## Configure concept")
    concept = st.text_area("Game description", DEFAULT_CONCEPT, height=150)

    default_tags = [x for x in DEFAULT_TAGS if x in all_tags]
    default_genres = [x for x in DEFAULT_GENRES if x in all_genres]

    selected_tags = st.multiselect(
        "Required tags",
        options=all_tags,
        default=default_tags,
        help="Required tags establish the main comparable market.",
    )
    selected_genres = st.multiselect(
        "Optional genres",
        options=all_genres,
        default=default_genres,
        help="Genres add category context. Leave blank to dynamically reweight the model.",
    )

    scenario_candidates = [tag for tag in all_tags if tag not in selected_tags]
    scenario_options = ["— No scenario comparison —"] + scenario_candidates
    default_scenario_index = (
        scenario_options.index("Roguelike") if "Roguelike" in scenario_options else 0
    )
    scenario_choice = st.selectbox(
        "Scenario tag",
        scenario_options,
        index=default_scenario_index,
        help="Adds one tag while holding description and genres constant.",
    )
    scenario_tag = None if scenario_choice.startswith("—") else scenario_choice

    top_n = st.select_slider("Top comparable games", options=[5, 10, 15, 20, 25], value=10)
    run_clicked = st.button("Run Market Analysis", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Tags are required. Description and genres are optional and trigger dynamic weight normalization when omitted.")

if run_clicked:
    if not selected_tags:
        st.sidebar.error("Select at least one required tag.")
    else:
        try:
            with st.spinner("Matching competitors and building market profiles…"):
                st.session_state["marketforge_analysis"] = analyze_market(
                    matcher,
                    concept=concept,
                    target_tags=selected_tags,
                    target_genres=selected_genres,
                    scenario_tag=scenario_tag,
                    top_n=top_n,
                )
        except Exception as exc:
            st.error(str(exc))

analysis = st.session_state.get("marketforge_analysis")
base = analysis["base"] if analysis is not None else None
comparison = analysis["comparison"] if analysis is not None else None

if analysis is None:
    st.markdown(
        '<div class="mf-note">Configure the concept on the left and click <b>Run Market Analysis</b>. You can still use the Steam Database tab before running an analysis.</div>',
        unsafe_allow_html=True,
    )

competitors_tab, benchmark_tab, scenario_tab, geography_tab, database_tab = st.tabs(
    ["Competitors", "Market Benchmark", "Scenario", "Geography", "Steam Database"]
)

# ---------- Competitors ----------
with competitors_tab:
    if base is None:
        st.info("Run Market Analysis to populate competitors.")
    else:
        st.markdown(
            '<div class="mf-note"><b>Match Score</b> is a relative ranking score, not a probability of commercial success. Required tags establish the market; description refines ranking; genres add category context.</div>',
            unsafe_allow_html=True,
        )
        weights = base["effective_weights"]
        render_metric_grid(
            [
                ("Tag weight", f"{weights['tag_weight'] * 100:.1f}%"),
                ("Description weight", f"{weights['text_weight'] * 100:.1f}%"),
                ("Genre weight", f"{weights['genre_weight'] * 100:.1f}%"),
            ]
        )
        st.markdown("### Comparable Steam games")
        st.dataframe(
            prepare_competitor_table(base["results"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Match Score %": st.column_config.NumberColumn(format="%.1f%%"),
                "Tag Similarity %": st.column_config.NumberColumn(format="%.1f%%"),
                "Text Similarity %": st.column_config.NumberColumn(format="%.1f%%"),
                "Genre Similarity %": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

# ---------- Benchmark ----------
with benchmark_tab:
    if base is None:
        st.info("Run Market Analysis to populate market benchmarks.")
    else:
        st.markdown(
            '<div class="mf-note">Headline medians reduce sensitivity to extreme outliers in hit-driven gaming distributions. Recent CCU is treated as a current/recent activity signal; playtime measures are cumulative observed traits.</div>',
            unsafe_allow_html=True,
        )
        bench = benchmark_dict(base["benchmark"])
        headline = [
            ("Median paid price", fmt_price(bench.get("Median Paid Price"))),
            ("Free game share", fmt_pct(bench.get("Free Game Share %"))),
            ("Median owners midpoint", fmt_owners(bench.get("Median Owners Midpoint"))),
            ("Median positive reviews", fmt_pct(bench.get("Median Positive Review %"))),
            ("Median review count", fmt_number(bench.get("Median Review Count"), 0)),
            ("Median Recent CCU", fmt_number(bench.get("Median Recent CCU"), 0)),
            ("Median avg playtime", fmt_hours(bench.get("Median Average Playtime Hours"))),
            ("Median typical playtime", fmt_hours(bench.get("Median Typical Playtime Hours"))),
            ("Median release year", fmt_number(bench.get("Median Release Year"), 0)),
        ]
        render_metric_grid(headline)

        st.markdown("### P25 / Median / P75 distribution")
        distribution_specs = [
            ("Paid Price", "25th Percentile Paid Price", "Median Paid Price", "75th Percentile Paid Price"),
            ("Owners Midpoint", "25th Percentile Owners Midpoint", "Median Owners Midpoint", "75th Percentile Owners Midpoint"),
            ("Positive Review %", "25th Percentile Positive Review %", "Median Positive Review %", "75th Percentile Positive Review %"),
            ("Review Count", "25th Percentile Review Count", "Median Review Count", "75th Percentile Review Count"),
            ("Recent CCU", "25th Percentile Recent CCU", "Median Recent CCU", "75th Percentile Recent CCU"),
            ("Average Playtime Hours", "25th Percentile Average Playtime Hours", "Median Average Playtime Hours", "75th Percentile Average Playtime Hours"),
            ("Typical Playtime Hours", "25th Percentile Typical Playtime Hours", "Median Typical Playtime Hours", "75th Percentile Typical Playtime Hours"),
        ]
        distribution = pd.DataFrame(
            [
                {
                    "Metric": label,
                    "P25": format_metric(p25_name, bench.get(p25_name)),
                    "Median": format_metric(med_name, bench.get(med_name)),
                    "P75": format_metric(p75_name, bench.get(p75_name)),
                }
                for label, p25_name, med_name, p75_name in distribution_specs
            ]
        )
        st.dataframe(distribution, use_container_width=True, hide_index=True)

        st.markdown("### Release-year cohort profile")
        st.caption("This compares release cohorts using currently available Recent CCU and lifetime playtime characteristics. It is not year-over-year growth.")
        year_df = base["release_year_profile"].copy()
        if year_df.empty:
            st.info("No release-year profile is available for this cohort.")
        else:
            st.dataframe(year_df, use_container_width=True, hide_index=True)
            chart_df = year_df.set_index("Release Year")
            st.line_chart(
                chart_df[["Median Average Playtime Hours", "Median Typical Playtime Hours"]],
                use_container_width=True,
            )

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Most common tags")
            st.dataframe(base["tag_profile"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### Most common genres")
            st.dataframe(base["genre_profile"], use_container_width=True, hide_index=True)

# ---------- Scenario ----------
with scenario_tab:
    if analysis is None:
        st.info("Run Market Analysis to populate scenario sensitivity.")
    else:
        st.markdown(
            '<div class="mf-note">Scenario differences describe changes in the matched historical comparable cohort. They do <b>not</b> measure causal uplift or predict sales.</div>',
            unsafe_allow_html=True,
        )
        if comparison is None:
            st.info("No scenario tag was selected. Choose one in the sidebar and rerun the analysis.")
        else:
            overlap = analysis["overlap"]
            render_metric_grid(
                [
                    ("Shared competitors", fmt_number(overlap["shared_count"], 0)),
                    ("Competitor overlap", fmt_pct(overlap["overlap_pct"])),
                    ("Entered competitors", fmt_number(overlap["entered_count"], 0)),
                    ("Exited competitors", fmt_number(overlap["exited_count"], 0)),
                ],
                columns=4,
            )

            st.markdown("### Benchmark shift")
            cmp = analysis["benchmark_comparison"].copy()
            if not cmp.empty:
                formatted_cmp = pd.DataFrame(
                    {
                        "Metric": cmp["Metric"],
                        "Base": [format_metric(m, v) for m, v in zip(cmp["Metric"], cmp["Base"])],
                        "Scenario": [format_metric(m, v) for m, v in zip(cmp["Metric"], cmp["Scenario"])],
                        "Difference": [format_difference(m, v) for m, v in zip(cmp["Metric"], cmp["Difference"])],
                    }
                )
                st.dataframe(formatted_cmp, use_container_width=True, hide_index=True)

            st.markdown("### Competitor movement")
            movement = analysis["movement"].copy()
            if not movement.empty:
                for col in ["Base Match Score", "Scenario Match Score"]:
                    movement[col] = movement[col].apply(fmt_pct)
                st.dataframe(movement, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Entered competitors")
                entered = analysis["entered"].copy()
                if entered.empty:
                    st.caption("No entered competitors.")
                else:
                    entered["Scenario Match Score"] = entered["Scenario Match Score"].apply(fmt_pct)
                    entered["tags"] = entered["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
                    entered["genres"] = entered["genres"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
                    st.dataframe(entered, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("### Exited competitors")
                exited = analysis["exited"].copy()
                if exited.empty:
                    st.caption("No exited competitors.")
                else:
                    exited["Base Match Score"] = exited["Base Match Score"].apply(fmt_pct)
                    exited["tags"] = exited["tags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
                    exited["genres"] = exited["genres"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
                    st.dataframe(exited, use_container_width=True, hide_index=True)

# ---------- Geography ----------
with geography_tab:
    if analysis is None:
        st.info("Run Market Analysis to populate the geography prototype.")
    else:
        st.markdown(
            '<div class="mf-warning"><b>⚠ MOCK / SYNTHETIC GEOGRAPHY</b><br>Geographic values are synthetic demonstration data and are not observed Steam player demographics. This module demonstrates how real game-level geographic data could later connect through Steam AppID.</div>',
            unsafe_allow_html=True,
        )
        geo = analysis["geo"].copy()
        mode = st.radio("Map mode", ["Base", "Scenario", "Difference"], horizontal=True)
        render_geo_map(geo, mode)
        other = geo[geo["Country"] == "Other"]
        if not other.empty:
            row = other.iloc[0]
            render_metric_grid(
                [
                    ("Other markets — Base", fmt_pct(row["Base"])),
                    ("Other markets — Scenario", fmt_pct(row["Scenario"])),
                    ("Other markets — Difference", fmt_pp(row["Difference"])),
                ]
            )

# ---------- Database ----------
with database_tab:
    st.markdown(
        '<div class="mf-note">Search the cleaned Steam registry by title, AppID, tag, or genre. Results are capped at 50 rows.</div>',
        unsafe_allow_html=True,
    )
    q_col, b_col = st.columns([5, 1])
    query = q_col.text_input("Search Steam registry", placeholder="e.g. Portal, roguelike, 620")
    search_clicked = b_col.button("Search", use_container_width=True)
    if search_clicked:
        if len(query.strip()) < 2:
            st.warning("Enter at least two characters.")
        else:
            st.session_state["db_results"] = cached_search(query, 50)

    db_results = st.session_state.get("db_results")
    if isinstance(db_results, pd.DataFrame):
        if db_results.empty:
            st.info("No matching Steam games found.")
        else:
            display_db = db_results.copy()
            if "price" in display_db.columns:
                display_db["price"] = display_db["price"].apply(fmt_price)
            if "owners_midpoint" in display_db.columns:
                display_db["owners_midpoint"] = display_db["owners_midpoint"].apply(fmt_owners)
            if "positive_review_pct" in display_db.columns:
                display_db["positive_review_pct"] = display_db["positive_review_pct"].apply(fmt_pct)
            for col in ["tags", "genres"]:
                if col in display_db.columns:
                    display_db[col] = display_db[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else "—")
            st.caption(f"{len(display_db)} result(s)")
            st.dataframe(display_db, use_container_width=True, hide_index=True)
