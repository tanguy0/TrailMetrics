# TrailMetrics

Personal-use Python project for simulating running metrics from watch data, organized as a small DDD-flavored codebase.

Today's only usecase is **personalized GAP (Gradient Adjusted Pace) modeling**: pulling a runner's Strava trail-run history and fitting two GAP models to compare against reference curves (Balanced Runner, Kilian Jornet).

## Repository layout

```
TrailMetrics/
├── pyproject.toml                 # installable package (pip install -e .)
├── requirements.txt
├── src/                           # source code, split into DDD layers
│   ├── domain/                    # pure business logic + ports (interfaces)
│   │   ├── models/                # dataclasses (entities / value objects)
│   │   │   ├── activity.py        # ActivityStream
│   │   │   └── gap.py             # ProcessedStream, DownsampledDataset, GapCurve
│   │   ├── gap/                   # GAP-specific domain services
│   │   │   ├── base.py            # GapModel ABC
│   │   │   ├── preprocessing.py   # StreamPreprocessor ABC + DefaultStreamPreprocessor
│   │   │   ├── efficiency_model.py# Strava-style bucketed efficiency model
│   │   │   ├── xgboost_model.py   # XGBoost GAP regressor
│   │   │   ├── plotting.py        # plot_gap_curves helper
│   │   │   └── reference_curves.py# Balanced Runner + Kilian Jornet reference curves
│   │   ├── progress/                # long-term-progress domain services
│   │   │   ├── models.py           # ActivityProgress + PR distances / gradient bands
│   │   │   ├── records.py          # sliding-window best efforts + record progression
│   │   │   ├── aggregates.py       # season overlays, avg gradient, gradient map
│   │   │   └── plotting.py         # Plotly (clickable) figures, Trail/Earthy theme
│   │   └── ports/
│   │       └── activity_stream_source.py  # ActivityStreamSource ABC
│   ├── infrastructure/            # concrete adapters implementing domain ports
│   │   └── strava/
│   │       └── strava_client.py   # StravaClient(ActivityStreamSource)
│   ├── usecases/                  # orchestrate domain + infrastructure to fulfill one task
│   │   ├── base.py                # UseCase ABC
│   │   ├── fetch_athlete_history.py        # fetch max history once (cached by the app)
│   │   ├── simulate_personalized_gap_model.py  # runs on pre-fetched streams (or fetches)
│   │   └── analyze_long_term_progress.py    # per-activity summaries for season trends
│   └── utils.py                   # tiny time-format helpers
├── .streamlit/
│   └── config.toml                # Trail/Earthy UI theme
├── app/
│   ├── Home.py                    # landing page: Strava connect + one-shot data load
│   ├── _helpers.py                # shared UI helpers (sys.path, PNG download button)
│   └── pages/                     # auto-listed in the sidebar
│       ├── 1_Personalized_GAP_Simulator.py  # analysis; unlocks once data is loaded
│       └── 2_How_it_works.py      # explanations
└── notebook/                      # exploratory notebooks (kept for research/iteration)
    └── gap/
        ├── full-flow.ipynb        # end-to-end personalized GAP flow
        └── experimental.ipynb     # low-level preprocessing exploration
```

### Layer responsibilities

- **`domain/`** — pure logic and abstract ports. No I/O, no framework.
  - `domain/models/` holds the dataclasses (entities, value objects) — the *data shapes* shared across services.
  - `domain/<topic>/` holds the domain services (`StreamPreprocessor`, `GapModel`, ...).
  - `domain/ports/` holds the interfaces (`ActivityStreamSource`) that infrastructure must implement.
  Where DDD's "domain" and "core" usually overlap, we keep them merged here to avoid splitting one concept across two folders.
- **`infrastructure/`** — adapters that implement domain ports (e.g. `StravaClient` implements `ActivityStreamSource`). This is the only place that knows about external services or persistence.
- **`usecases/`** — one class per user-facing task. Receives dependencies (a stream source, optionally a preprocessor) and orchestrates them. `SimulatePersonalizedGapModel.execute(...)` is the only usecase today.
- **`app/`** — the Streamlit front-end. Imports usecases directly (no HTTP layer for now); an `api/` layer (e.g. FastAPI) will be added if/when a remote consumer appears.

## Setup

```bash
# 1. Create + activate a virtual env
python -m venv .venv
source .venv/bin/activate

# 2. Install the project in editable mode (so `from src...` works everywhere)
pip install -e .
pip install -r requirements.txt

# 3. Configure Strava credentials (do NOT commit these)
export STRAVA_CLIENT_ID=...
export STRAVA_CLIENT_SECRET=...
```

## Running the Streamlit app

```bash
streamlit run app/Home.py
```

The flow is **load once, analyse many**:

1. On the **Home** page: enter your name, provide Strava credentials, authorize,
   then click **Load my data**. The app fetches the maximum history available
   (all run types) and caches the raw streams in the Streamlit session, then
   reports the date range fetched (oldest → most recent session).
2. Each analysis lives in its own file under `app/pages/` and shows up
   automatically in the sidebar. Analyses are **locked until data is loaded**.
   Their inputs — date range, session types, model parameters — are filters
   applied to the in-memory data, so you can tweak and re-run without
   re-fetching from Strava.

Today there's one analysis:

- **Personalized GAP Simulator** &mdash; date range + session-type filters, both
  GAP models, intensity-stratified panels, and download buttons under every figure.

A separate **How it works** page holds the usage explanations.

To add a new analysis, drop a `app/pages/<N>_<Name>.py` file in the folder. Use
`add_repo_root_to_path()` from `app/_helpers.py` to keep `from src...` imports
working, gate on `"athlete_streams" in st.session_state`, and pass
`streams=st.session_state["athlete_streams"]` into your usecase input.

### Theme

Colors live in one place — [`src/domain/gap/theme.py`](src/domain/gap/theme.py)
for the plots and [`.streamlit/config.toml`](.streamlit/config.toml) for the UI
chrome — so the app and figures stay coherent (Trail / Earthy palette).

## Running the notebooks

```bash
jupyter notebook notebook/gap/full-flow.ipynb
```

The notebooks expect the same `STRAVA_CLIENT_ID` / `STRAVA_CLIENT_SECRET` env vars.

## Adding a new usecase

1. If new data shapes are needed, add dataclasses under `src/domain/models/`.
2. If new domain logic is needed, add it under `src/domain/<topic>/`.
3. If a new external dependency is needed, add a port under `src/domain/ports/` and an adapter under `src/infrastructure/`.
4. Create a new file under `src/usecases/` with a class inheriting `UseCase` and implementing `.execute(params)`.
5. Wire it into a new `app/pages/<N>_<Name>.py` page (and, eventually, an `api/` route).
