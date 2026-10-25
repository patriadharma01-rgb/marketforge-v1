# MarketForge V1 — Premium Streamlit Build

**Game Concept Market Simulator**

MarketForge stress-tests an unreleased game concept against real Steam comparables before development. This build combines the validated analytical engine with the approved borderless premium UI direction.

## Experience design

The app deliberately uses two visual states:

- **Ideation mode — bright / borderless.** The user moves through four full-height input sections: core concept, required tags, genre + scenario context, and final review. The Analyze action only appears in the final review section.
- **Analysis mode — dark / analytical.** After the real model completes, the interface shifts to a darker data environment with headline KPIs, an orthographic 3D-style world globe, and tabbed detail views.
- **Edit concept** returns the user to the bright setup experience while preserving the previous inputs as defaults.

The visual layer is intentionally separated from the analytical methodology. UI changes do not alter the canonical matching, benchmarking, scenario, or geography rules.

## Analytical engine

- **Competitor matching** — required tags + optional description + optional genre.
- **Dynamic matching weights** — canonical base weights are 50% Tags / 35% Description / 15% Genre and normalize automatically when optional inputs are omitted.
- **Comparable market benchmarking** — medians plus percentile context for pricing, ownership midpoint proxy, reviews, Recent CCU, and playtime.
- **Release-year cohort profile** — cohort comparison, not YoY growth.
- **Scenario sensitivity** — adds one tag while holding description and genres constant; compares overlap, rank movement, entered/exited competitors, and benchmark shifts.
- **Synthetic geography prototype** — deterministic mock country shares keyed through Steam AppID. It never affects matching or benchmarks and is prominently labelled as synthetic.
- **Steam registry search** — title, AppID, tag, or genre search, capped at 50 rows.

## Interpretation guardrails

MarketForge does **not** predict sales or success probability. `match_score_pct` is a relative ranking score. `owners_midpoint` is a SteamSpy ownership-band proxy, not exact unit sales. `peak_ccu` is presented as a **Recent CCU/current activity signal**, not lifetime success. Scenario changes describe a changed historical comparable cohort and are not causal uplift estimates.

## Run locally

Python 3.11+ is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Or run `run_local.bat` after installing the dependencies.

### macOS / Linux

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Or run `./run_local.sh` after installing the dependencies.

## First startup

The first run downloads the public `Z02Z/steam-games-dataset` dataset from Hugging Face and builds the TF-IDF index. Streamlit caches the prepared dataset/model for the process, so later analyses are faster.

## Project structure

```text
marketforge_final/
├── app.py
├── marketforge/
│   ├── data.py
│   ├── matcher.py
│   ├── benchmark.py
│   ├── scenario.py
│   └── geography.py
├── tests/
│   └── test_core.py
├── requirements.txt
├── run_local.bat
├── run_local.sh
└── .streamlit/
    └── config.toml
```

## Validation

Run:

```bash
python -m py_compile app.py marketforge/*.py
python -m pytest -q
```

The current package passes all six core analytical tests.

## Deploy on Streamlit Community Cloud

1. Upload this folder's contents to the root of the GitHub repository.
2. In Streamlit Community Cloud select the repository and `main` branch.
3. Set the main file path to `app.py`.
4. Deploy.

The hosted app needs outbound access to Hugging Face on first startup. If memory becomes the limiting factor, optimize loading/persistence rather than changing the canonical matching methodology.
