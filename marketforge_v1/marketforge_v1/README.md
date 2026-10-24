# MarketForge V1

**Game Concept Market Simulator**

MarketForge stress-tests a game concept against real Steam comparables before development. The application takes a concept description, required tags, optional genres, and an optional scenario tag, then builds a comparable cohort and surfaces market benchmarks, competitor movement, and a clearly labelled synthetic geography prototype.

## What V1 includes

- **Competitor matching** — required tags + optional description + optional genre.
- **Dynamic matching weights** — base weights are 50% Tags / 35% Description / 15% Genre and normalize automatically when optional inputs are omitted.
- **Comparable market benchmarking** — medians plus P25 / P75 for pricing, ownership midpoint proxy, reviews, Recent CCU, and playtime.
- **Release-year cohort profile** — cohort comparison, not YoY growth.
- **Scenario sensitivity** — add one tag and inspect overlap, rank movement, entered/exited competitors, and benchmark shifts.
- **Synthetic geography prototype** — deterministic mock country shares connected through Steam AppID. This is explicitly not observed Steam demographic data.
- **Steam registry search** — search title, AppID, tag, or genre, capped at 50 rows.

## Analytical interpretation

MarketForge does **not** predict sales or success probability. `match_score_pct` is a relative ranking score. `owners_midpoint` is a SteamSpy ownership-band proxy, not exact unit sales. `peak_ccu` is presented as a **Recent CCU/current activity signal**, not lifetime success. Scenario changes show how the historical comparable cohort changes when positioning changes; they are not causal uplift estimates.

## Run locally

Python 3.11+ is recommended.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or after installing requirements:

```bash
./run_local.sh
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or double-click/run:

```text
run_local.bat
```

## First startup

The first run downloads the public `Z02Z/steam-games-dataset` dataset from Hugging Face and builds the TF-IDF index. Streamlit caches the prepared dataset/model for the running process, so later interactions are faster.

## Project structure

```text
marketforge_v1/
├── app.py
├── marketforge/
│   ├── data.py
│   ├── matcher.py
│   ├── benchmark.py
│   ├── scenario.py
│   └── geography.py
├── tests/
├── requirements.txt
└── .streamlit/config.toml
```

## Deploy

A straightforward next step is to put this folder in a GitHub repository and deploy `app.py` on Streamlit Community Cloud. The app needs outbound access to Hugging Face on first startup. If a hosted environment has tight memory limits, the next optimization would be to persist a pre-cleaned dataset and/or serialized TF-IDF artifacts rather than rebuilding them at boot.

## Data and methodology note

The core methodology is ported from the validated MarketForge research notebook. The standalone app separates the analytical engine from the presentation layer so the UI no longer depends on Google Colab JavaScript callbacks.
