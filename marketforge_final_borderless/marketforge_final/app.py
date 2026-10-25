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
    initial_sidebar_state="collapsed",
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
    if any(token in name for token in ["release year", "recent ccu", "review count", "comparable games"]):
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


def list_to_text(value) -> str:
    return ", ".join(value) if isinstance(value, list) and value else "—"


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
    out["Tags"] = out["tags"].apply(list_to_text)
    out["Genres"] = out["genres"].apply(list_to_text)
    return out[
        [
            "Rank", "name", "Match Score %", "Tag Similarity %", "Text Similarity %",
            "Genre Similarity %", "Price", "Owners Midpoint", "Recent CCU",
            "Positive Review %", "Release Year", "Tags", "Genres",
        ]
    ].rename(columns={"name": "Game"})


def render_metric_grid(items: list[tuple[str, str]], columns: int = 4):
    for start in range(0, len(items), columns):
        cols = st.columns(columns)
        for col, (label, value) in zip(cols, items[start : start + columns]):
            col.metric(label, value)


def render_globe(geo: pd.DataFrame, mode: str):
    map_df = geo[geo["Country"] != "Other"].copy()
    map_df["ISO3"] = map_df["Country"].map(COUNTRY_ISO3)
    map_df = map_df[map_df["ISO3"].notna()].copy()
    if map_df.empty:
        st.info("No geographic data is available for this cohort.")
        return

    if mode == "Base":
        z_col = "Base"
        colorscale = [[0, "#332407"], [0.25, "#775013"], [0.5, "#b77618"], [0.75, "#f0c04e"], [1, "#ffe48f"]]
        zmin, zmax, colorbar_title = 0, max(35, float(map_df[z_col].max() or 0)), "Share %"
    elif mode == "Scenario":
        z_col = "Scenario"
        colorscale = [[0, "#332407"], [0.25, "#775013"], [0.5, "#b77618"], [0.75, "#f0c04e"], [1, "#ffe48f"]]
        zmin, zmax, colorbar_title = 0, max(35, float(map_df[z_col].max() or 0)), "Share %"
    else:
        z_col = "Difference"
        colorscale = [[0, "#c45768"], [0.36, "#7d4c59"], [0.5, "#2d3342"], [0.64, "#2e8763"], [1, "#5be29a"]]
        bound = max(1.0, float(np.nanmax(np.abs(map_df[z_col].to_numpy(dtype=float)))))
        zmin, zmax, colorbar_title = -bound, bound, "pp"

    custom = np.stack([map_df["Country"], map_df["Base"], map_df["Scenario"], map_df["Difference"]], axis=-1)
    fig = go.Figure(
        go.Choropleth(
            locations=map_df["ISO3"],
            z=map_df[z_col],
            locationmode="ISO-3",
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            marker_line_color="#0b1019",
            marker_line_width=0.6,
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Base: %{customdata[1]:.1f}%<br>"
                "Scenario: %{customdata[2]:.1f}%<br>Difference: %{customdata[3]:+.1f} pp<extra></extra>"
            ),
            colorbar=dict(title=colorbar_title, thickness=12),
        )
    )
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=10, lat=18),
        showframe=False,
        showcoastlines=False,
        showcountries=True,
        countrycolor="#30394a",
        showland=True,
        landcolor="#1d2431",
        showocean=True,
        oceancolor="#08111d",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=570,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dbe7f7",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


def render_weight_donut(weights: dict):
    fig = go.Figure(
        go.Pie(
            labels=["Tags", "Description", "Genres"],
            values=[weights.get("tag_weight", 0) * 100, weights.get("text_weight", 0) * 100, weights.get("genre_weight", 0) * 100],
            hole=0.7,
            marker=dict(colors=["#79a7ff", "#57e5ff", "#58f0b0"]),
            sort=False,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=10),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#dbe7f7",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_release_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No release-year profile is available for this cohort.")
        return
    chart = df.sort_values("Release Year")
    fig = go.Figure()
    fig.add_scatter(
        x=chart["Release Year"], y=chart["Median Recent CCU"], mode="lines+markers",
        name="Median Recent CCU", line=dict(color="#79a7ff", width=3), marker=dict(size=8),
    )
    fig.add_scatter(
        x=chart["Release Year"], y=chart["Median Average Playtime Hours"], mode="lines+markers",
        name="Median Avg Playtime (h)", line=dict(color="#57e5ff", width=3), marker=dict(size=8), yaxis="y2",
    )
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=25, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dbe7f7",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(gridcolor="rgba(255,255,255,.08)"),
        yaxis=dict(title="Median Recent CCU", gridcolor="rgba(255,255,255,.08)"),
        yaxis2=dict(title="Median Avg Playtime (h)", overlaying="y", side="right", showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def inject_theme(analysis_mode: bool):
    bg = """
        radial-gradient(1200px 760px at 0% -8%, rgba(104,133,255,.26), transparent 52%),
        radial-gradient(840px 520px at 100% 0%, rgba(76,231,255,.16), transparent 44%),
        radial-gradient(760px 420px at 52% 24%, rgba(173,107,255,.14), transparent 54%),
        radial-gradient(700px 360px at 14% 82%, rgba(79,241,210,.12), transparent 50%),
        linear-gradient(180deg,#07111d 0%, #0a1322 38%, #090f1b 100%)
    """ if analysis_mode else """
        radial-gradient(1100px 760px at -5% -10%, rgba(122,162,255,.22), transparent 54%),
        radial-gradient(900px 620px at 100% 0%, rgba(83,223,255,.18), transparent 48%),
        radial-gradient(720px 460px at 50% 26%, rgba(170,140,255,.12), transparent 52%),
        linear-gradient(180deg,#fbfdff 0%, #f4f9ff 34%, #eef5ff 100%)
    """
    text = "#f2f7ff" if analysis_mode else "#10192a"
    muted = "#9eb1ca" if analysis_mode else "#617089"
    panel = "rgba(14,22,38,.74)" if analysis_mode else "rgba(255,255,255,.72)"
    input_bg = "rgba(255,255,255,.06)" if analysis_mode else "rgba(255,255,255,.80)"
    input_text = "#eff5ff" if analysis_mode else "#132238"

    st.markdown(
        f"""
        <style>
        html {{ scroll-behavior:smooth; }}
        .stApp {{
            color:{text};
            background:{bg};
            background-attachment:fixed;
            transition:background 1.25s ease, color .8s ease;
        }}
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none !important; }}
        .block-container {{ max-width:1480px; padding-top:1.1rem; padding-bottom:5rem; }}
        .stApp:before {{
            content:""; position:fixed; inset:0; pointer-events:none; z-index:0;
            background: repeating-linear-gradient(90deg, rgba(32,66,120,.035) 0 1px, transparent 1px 100px),
                        repeating-linear-gradient(0deg, rgba(32,66,120,.028) 0 1px, transparent 1px 100px);
            opacity:{'.20' if analysis_mode else '.55'};
        }}
        .block-container > div {{ position:relative; z-index:1; }}
        .mf-kicker {{ color:#6f8eff; font-size:.92rem; font-weight:900; letter-spacing:.20em; text-transform:uppercase; }}
        .mf-title {{
            font-family:"Arial Black","Helvetica Neue",Inter,system-ui,sans-serif;
            font-size:clamp(4.2rem,9vw,7.8rem); line-height:.86; font-weight:900;
            letter-spacing:-.075em; text-transform:uppercase; margin:.45rem 0 1.1rem;
        }}
        .mf-subtitle {{ color:{muted}; font-size:1.35rem; line-height:1.62; max-width:900px; }}
        .mf-badges {{ display:flex; flex-wrap:wrap; gap:.65rem; margin-top:1.4rem; }}
        .mf-badge {{
            padding:.68rem .95rem; border-radius:999px; font-size:.95rem; font-weight:650;
            background:linear-gradient(135deg, rgba(122,162,255,.13), rgba(83,223,255,.07));
            box-shadow:0 10px 32px rgba(72,100,170,.08);
        }}
        .mf-section {{ font-size:clamp(2.25rem,5vw,3.7rem); font-weight:900; letter-spacing:-.055em; line-height:1.02; margin:0 0 .65rem; }}
        .mf-section-copy {{ color:{muted}; font-size:1.15rem; line-height:1.65; max-width:820px; margin-bottom:1.15rem; }}
        .mf-step {{ color:#6f8eff; font-size:.82rem; letter-spacing:.16em; font-weight:900; text-transform:uppercase; margin-bottom:.65rem; }}
        .mf-note {{ color:{muted}; font-size:.98rem; line-height:1.6; margin:.55rem 0 1.1rem; }}
        .mf-warning {{
            padding:1rem 1.15rem; border-radius:18px; font-size:.95rem; line-height:1.55;
            background:{'linear-gradient(135deg, rgba(68,46,12,.78), rgba(39,31,19,.58))' if analysis_mode else 'linear-gradient(135deg, rgba(255,196,104,.16), rgba(255,196,104,.06))'};
            color:{'#f7ddb0' if analysis_mode else '#7e5b15'}; box-shadow:0 18px 50px rgba(0,0,0,.08);
        }}
        div[data-testid="stVerticalBlock"]:has(.mf-slide-anchor) {{
            min-height:calc(100vh - 3rem); display:flex; flex-direction:column; justify-content:center; padding:2.5rem 0;
        }}
        div[data-testid="stVerticalBlock"]:has(.mf-result-anchor) {{ padding-top:1rem; }}
        div[data-testid="stMetric"] {{
            background:{panel}; border:0 !important; padding:1.05rem 1.1rem; border-radius:20px;
            box-shadow:{'0 24px 60px rgba(0,0,0,.26)' if analysis_mode else '0 24px 70px rgba(23,44,89,.09)'};
            backdrop-filter:blur(18px);
        }}
        div[data-testid="stMetric"] label {{ color:{muted} !important; font-size:.9rem !important; font-weight:750 !important; }}
        div[data-testid="stMetricValue"] {{ font-size:2.0rem !important; letter-spacing:-.04em; font-weight:850 !important; }}
        div[data-testid="stDataFrame"] {{ border:0 !important; border-radius:20px; overflow:hidden; box-shadow:0 22px 60px rgba(0,0,0,.10); }}
        [data-testid="stForm"] {{ border:0 !important; padding:0 !important; }}
        [data-testid="stTextArea"] textarea,
        [data-testid="stSelectbox"] > div > div,
        [data-testid="stMultiSelect"] > div > div,
        [data-testid="stTextInput"] input {{
            background:{input_bg} !important; color:{input_text} !important; border:0 !important; border-radius:18px !important;
            box-shadow:0 16px 40px rgba(40,60,105,.08) !important; font-size:1.05rem !important;
        }}
        [data-testid="stMultiSelect"] span {{ font-size:.96rem !important; }}
        div[data-baseweb="select"] > div {{ border:0 !important; }}
        .stButton > button, [data-testid="stFormSubmitButton"] > button {{
            border:0 !important; border-radius:17px !important; min-height:3.25rem; padding:.75rem 1.25rem;
            font-size:1.02rem; font-weight:850; color:white;
            background:linear-gradient(135deg,#6f93ff,#53dfff) !important;
            box-shadow:0 18px 42px rgba(83,223,255,.20); transition:transform .2s ease, box-shadow .2s ease;
        }}
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {{ transform:translateY(-1px); box-shadow:0 24px 52px rgba(83,223,255,.27); }}
        .stTabs [data-baseweb="tab-list"] {{ background:rgba(255,255,255,{'.045' if analysis_mode else '.52'}); border:0; border-radius:18px; padding:.35rem; gap:.25rem; }}
        .stTabs [data-baseweb="tab"] {{ border-radius:13px; min-height:3rem; font-size:1rem; font-weight:750; color:{muted}; }}
        .stTabs [aria-selected="true"] {{ background:linear-gradient(135deg,rgba(122,162,255,.28),rgba(83,223,255,.14)); color:{text}; }}
        hr {{ border-color:transparent !important; }}
        p, li {{ font-size:1rem; }}
        @media(max-width:760px) {{
            .block-container {{ padding-left:1rem; padding-right:1rem; }}
            .mf-subtitle {{ font-size:1.08rem; }}
            div[data-testid="stVerticalBlock"]:has(.mf-slide-anchor) {{ min-height:auto; padding:3rem 0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(games_count: int, tag_count: int, genre_count: int, analysis_mode: bool):
    mode_copy = (
        "The concept has moved from ideation into analysis. Explore the matched Steam cohort, benchmark shifts, and synthetic geographic signal below."
        if analysis_mode
        else "Build the concept one decision at a time. MarketForge keeps ideation bright and spacious, then shifts into a darker analytical environment only after the model runs."
    )
    st.markdown('<div class="mf-kicker">Game concept market simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="mf-title">MarketForge</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="mf-subtitle">{mode_copy}</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="mf-badges">
            <span class="mf-badge"><b>{games_count:,}</b>&nbsp; cleaned Steam titles</span>
            <span class="mf-badge"><b>{tag_count:,}</b>&nbsp; tags</span>
            <span class="mf-badge"><b>{genre_count:,}</b>&nbsp; genres</span>
            <span class="mf-badge">50 / 35 / 15 canonical weighting</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Load data/model before rendering form options.
analysis = st.session_state.get("marketforge_analysis")
analysis_mode = analysis is not None
inject_theme(analysis_mode)

try:
    with st.spinner("Loading Steam market data and matching index…"):
        games, matcher, all_tags, all_genres = load_assets()
except Exception as exc:
    st.error("MarketForge could not initialize the Steam dataset/model.")
    st.exception(exc)
    st.stop()

render_hero(len(games), len(all_tags), len(all_genres), analysis_mode)


if not analysis_mode:
    default_tags = [x for x in DEFAULT_TAGS if x in all_tags]
    default_genres = [x for x in DEFAULT_GENRES if x in all_genres]

    with st.form("marketforge_concept_form", clear_on_submit=False):
        with st.container():
            st.markdown('<div class="mf-slide-anchor"></div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-step">01 · Core fantasy</div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-section">Define the game idea.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="mf-section-copy">Start with the player fantasy and gameplay loop. Description is optional in the matcher, but when supplied it refines competitor ranking through TF-IDF similarity.</div>',
                unsafe_allow_html=True,
            )
            concept = st.text_area(
                "Game description",
                value=st.session_state.get("draft_concept", DEFAULT_CONCEPT),
                height=220,
                label_visibility="collapsed",
            )
            st.markdown('<div class="mf-note">Describe what the player does, the core loop, tone, and distinguishing mechanics. Avoid writing marketing copy.</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="mf-slide-anchor"></div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-step">02 · Market anchors</div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-section">Choose the required tags.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="mf-section-copy">Tags establish the primary comparable market and carry the largest canonical weight. At least one tag is required.</div>',
                unsafe_allow_html=True,
            )
            selected_tags = st.multiselect(
                "Required tags",
                options=all_tags,
                default=st.session_state.get("draft_tags", default_tags),
                help="Required tags establish the main comparable market.",
            )
            st.markdown('<div class="mf-note">Canonical base weight: <b>50%</b>. The score is normalized dynamically when optional inputs are omitted.</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="mf-slide-anchor"></div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-step">03 · Context & sensitivity</div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-section">Refine the market and choose a scenario.</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="mf-section-copy">Genres add category context. The scenario tag asks a single positioning question: what changes in the matched cohort if this tag is added while the rest of the concept stays constant?</div>',
                unsafe_allow_html=True,
            )
            c1, c2 = st.columns(2, gap="large")
            with c1:
                selected_genres = st.multiselect(
                    "Optional genres",
                    options=all_genres,
                    default=st.session_state.get("draft_genres", default_genres),
                )
            with c2:
                scenario_candidates = [tag for tag in all_tags if tag not in selected_tags]
                scenario_options = ["— No scenario comparison —"] + scenario_candidates
                stored_scenario = st.session_state.get("draft_scenario", "Roguelike")
                scenario_index = scenario_options.index(stored_scenario) if stored_scenario in scenario_options else 0
                scenario_choice = st.selectbox("Scenario tag", scenario_options, index=scenario_index)
                scenario_tag = None if scenario_choice.startswith("—") else scenario_choice
            top_n = st.select_slider(
                "Top comparable games",
                options=[5, 10, 15, 20, 25],
                value=st.session_state.get("draft_top_n", 10),
            )
            st.markdown('<div class="mf-note">Scenario results are sensitivity analysis only. They do not claim that adding a tag causes higher sales, owners, reviews, or CCU.</div>', unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="mf-slide-anchor"></div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-step">04 · Final review</div>', unsafe_allow_html=True)
            st.markdown('<div class="mf-section">Ready to enter analysis mode?</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="mf-section-copy">This is intentionally the first point where the analysis action appears. Review the concept, then run the real matching and benchmark pipeline.</div>',
                unsafe_allow_html=True,
            )
            summary_cols = st.columns(4)
            summary_cols[0].metric("Required tags", len(selected_tags))
            summary_cols[1].metric("Optional genres", len(selected_genres))
            summary_cols[2].metric("Scenario", scenario_tag or "None")
            summary_cols[3].metric("Comparable cohort", f"Top {top_n}")
            st.markdown(
                '<div class="mf-warning"><b>Geography remains MOCK / SYNTHETIC.</b> It demonstrates the future geographic layer and never affects matching, ranking, or benchmarks.</div>',
                unsafe_allow_html=True,
            )
            submitted = st.form_submit_button("Analyze concept", use_container_width=True, type="primary")

    if submitted:
        if not selected_tags:
            st.error("Select at least one required tag before analysis.")
        else:
            try:
                with st.spinner("Matching competitors, benchmarking the cohort, and building scenario sensitivity…"):
                    result = analyze_market(
                        matcher,
                        concept=concept,
                        target_tags=selected_tags,
                        target_genres=selected_genres,
                        scenario_tag=scenario_tag,
                        top_n=top_n,
                    )
                st.session_state["marketforge_analysis"] = result
                st.session_state["marketforge_inputs"] = {
                    "concept": concept,
                    "tags": selected_tags,
                    "genres": selected_genres,
                    "scenario_tag": scenario_tag,
                    "top_n": top_n,
                }
                st.session_state["draft_concept"] = concept
                st.session_state["draft_tags"] = selected_tags
                st.session_state["draft_genres"] = selected_genres
                st.session_state["draft_scenario"] = scenario_tag or "— No scenario comparison —"
                st.session_state["draft_top_n"] = top_n
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

else:
    inputs = st.session_state.get("marketforge_inputs", {})
    base = analysis["base"]
    comparison = analysis["comparison"]
    bench = benchmark_dict(base["benchmark"])
    top_row = base["results"].iloc[0] if not base["results"].empty else None
    overlap = analysis["overlap"]

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown('<div class="mf-result-anchor"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="mf-note">Current concept · {", ".join(inputs.get("tags", [])) or "No tags"} · Scenario: {inputs.get("scenario_tag") or "None"}</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Edit concept", use_container_width=True):
            st.session_state.pop("marketforge_analysis", None)
            st.rerun()

    render_metric_grid(
        [
            ("Top comparable", top_row["name"] if top_row is not None else "—"),
            ("Best match score", fmt_pct(top_row["match_score_pct"] if top_row is not None else None)),
            ("Median owners midpoint", fmt_owners(bench.get("Median Owners Midpoint"))),
            ("Competitor overlap", fmt_pct(overlap.get("overlap_pct")) if comparison is not None else "—"),
        ],
        columns=4,
    )

    st.markdown('<div class="mf-step" style="margin-top:2.6rem">Global market footprint</div>', unsafe_allow_html=True)
    st.markdown('<div class="mf-section" style="font-size:2.8rem">Where does the matched cohort lean?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mf-warning"><b>⚠ MOCK / SYNTHETIC GEOGRAPHY</b><br>Country shares below are deterministic demo data keyed by Steam AppID. They are not observed Steam player demographics and do not feed the matcher.</div>',
        unsafe_allow_html=True,
    )
    map_mode = st.radio("Map mode", ["Base", "Scenario", "Difference"], horizontal=True, label_visibility="collapsed")
    map_col, market_col = st.columns([1.65, 0.7], gap="large")
    with map_col:
        render_globe(analysis["geo"].copy(), map_mode)
    with market_col:
        geo = analysis["geo"].copy()
        key = map_mode if map_mode in ["Base", "Scenario"] else "Difference"
        leaderboard = geo[geo["Country"] != "Other"][["Country", key]].copy()
        if key == "Difference":
            leaderboard = leaderboard.sort_values(key, ascending=False)
            leaderboard[key] = leaderboard[key].apply(fmt_pp)
        else:
            leaderboard = leaderboard.sort_values(key, ascending=False)
            leaderboard[key] = leaderboard[key].apply(fmt_pct)
        st.markdown("### Leading markets")
        st.dataframe(leaderboard, use_container_width=True, hide_index=True)
        other = geo[geo["Country"] == "Other"]
        if not other.empty:
            row = other.iloc[0]
            st.metric("Other markets", fmt_pct(row[key]) if key != "Difference" else fmt_pp(row[key]))

    competitor_tab, benchmark_tab, scenario_tab, geography_tab, database_tab = st.tabs(
        ["Competitors", "Market Benchmark", "Scenario", "Geography", "Steam Database"]
    )

    with competitor_tab:
        st.markdown('<div class="mf-note"><b>Match Score</b> is a relative ranking score, not a probability of commercial success.</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([1.6, 0.65], gap="large")
        with c1:
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
        with c2:
            st.markdown("### Matching recipe")
            render_weight_donut(base["effective_weights"])
            st.caption("Required tags anchor the comparable market. Description refines ranking. Genre adds category context.")

    with benchmark_tab:
        st.markdown('<div class="mf-note">Headline medians reduce sensitivity to extreme outliers. Recent CCU is a current/recent activity signal; playtime is cumulative.</div>', unsafe_allow_html=True)
        render_metric_grid(
            [
                ("Median paid price", fmt_price(bench.get("Median Paid Price"))),
                ("Free game share", fmt_pct(bench.get("Free Game Share %"))),
                ("Median owners midpoint", fmt_owners(bench.get("Median Owners Midpoint"))),
                ("Median positive reviews", fmt_pct(bench.get("Median Positive Review %"))),
                ("Median review count", fmt_number(bench.get("Median Review Count"), 0)),
                ("Median Recent CCU", fmt_number(bench.get("Median Recent CCU"), 0)),
                ("Median avg playtime", fmt_hours(bench.get("Median Average Playtime Hours"))),
                ("Median release year", fmt_number(bench.get("Median Release Year"), 0)),
            ],
            columns=4,
        )
        st.markdown("### Release-year cohort profile")
        st.caption("This is a cohort comparison using currently available metrics, not year-over-year growth.")
        render_release_chart(base["release_year_profile"].copy())
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Most common tags")
            st.dataframe(base["tag_profile"], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("### Most common genres")
            st.dataframe(base["genre_profile"], use_container_width=True, hide_index=True)

    with scenario_tab:
        st.markdown('<div class="mf-note">Scenario differences describe a changed historical comparable cohort. They do <b>not</b> measure causal uplift.</div>', unsafe_allow_html=True)
        if comparison is None:
            st.info("No scenario tag was selected for this run.")
        else:
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
                formatted_cmp = pd.DataFrame({
                    "Metric": cmp["Metric"],
                    "Base": [format_metric(m, v) for m, v in zip(cmp["Metric"], cmp["Base"])],
                    "Scenario": [format_metric(m, v) for m, v in zip(cmp["Metric"], cmp["Scenario"])],
                    "Difference": [format_difference(m, v) for m, v in zip(cmp["Metric"], cmp["Difference"])],
                })
                st.dataframe(formatted_cmp, use_container_width=True, hide_index=True)

            st.markdown("### Competitor movement")
            movement = analysis["movement"].copy()
            if not movement.empty:
                for col in ["Base Match Score", "Scenario Match Score"]:
                    movement[col] = movement[col].apply(fmt_pct)
                st.dataframe(movement, use_container_width=True, hide_index=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Entered")
                entered = analysis["entered"].copy()
                if entered.empty:
                    st.caption("No entered competitors.")
                else:
                    entered["Scenario Match Score"] = entered["Scenario Match Score"].apply(fmt_pct)
                    entered["tags"] = entered["tags"].apply(list_to_text)
                    entered["genres"] = entered["genres"].apply(list_to_text)
                    st.dataframe(entered, use_container_width=True, hide_index=True)
            with c2:
                st.markdown("### Exited")
                exited = analysis["exited"].copy()
                if exited.empty:
                    st.caption("No exited competitors.")
                else:
                    exited["Base Match Score"] = exited["Base Match Score"].apply(fmt_pct)
                    exited["tags"] = exited["tags"].apply(list_to_text)
                    exited["genres"] = exited["genres"].apply(list_to_text)
                    st.dataframe(exited, use_container_width=True, hide_index=True)

    with geography_tab:
        st.markdown('<div class="mf-warning"><b>MOCK / SYNTHETIC DEMO DATA.</b> This section is intentionally separated from observed Steam measures.</div>', unsafe_allow_html=True)
        detail_mode = st.radio("Detailed map mode", ["Base", "Scenario", "Difference"], horizontal=True)
        render_globe(analysis["geo"].copy(), detail_mode)
        st.dataframe(analysis["geo"], use_container_width=True, hide_index=True)

    with database_tab:
        st.markdown('<div class="mf-note">Search the cleaned Steam registry by title, AppID, tag, or genre. Results are capped at 50 rows.</div>', unsafe_allow_html=True)
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
                        display_db[col] = display_db[col].apply(list_to_text)
                st.caption(f"{len(display_db)} result(s)")
                st.dataframe(display_db, use_container_width=True, hide_index=True)
