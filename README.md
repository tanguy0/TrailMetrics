# TrailMetrics

A data-science workbench for running data. The athlete builds the analyses: pick a
data source, then add the plots you want over it.

The organising idea is that **an analysis is data, not code**:

```
PageSpec ──< PanelSpec ──< PlotSpec
                └── DataSourceSpec
```

* a **panel** has exactly one data source and as many plots as you like;
* a **data source** is either a hand-picked set of activities, one time window, or
  several named windows compared side by side;
* a **plot** is a registry entry with a declarative parameter schema, so its form —
  including sub-parameters that only appear when relevant — is generated, never written.

Because an analysis is a serializable document, "the athlete built this" and "the app
ships this to everyone" are the same mechanism. The three **default analyses**
(Personalized GAP Simulator, Race Comparator, Long-Term Progress) are `PageSpec`s
assembled in [`src/dashboards/`](src/dashboards/) from the same panels and plots
anyone gets, then **stored per athlete** on first use. They are edited in place like
any other analysis; the only thing that marks them is that they cannot be deleted.

They were briefly generated per request and served read-only, on the theory that they
were examples to duplicate. That failed the Race Comparator outright — it *is* a
hand-picked set of workouts, and a read-only page cannot be given one — which is the
clearest argument for the current model: if a shipped analysis needs something the
builder cannot express, the builder is missing a feature.

## Architecture

Four abstractions carry the whole design.

**1. Specs** — [`src/domain/spec/`](src/domain/spec/). Pure, JSON-serializable
dataclasses. Parameter schemas include their conditional logic as a serializable
`Condition` tree, so the browser evaluates the same visibility rules as Python
instead of re-implementing them.

**2. Three data levels** — [`src/domain/dataset/`](src/domain/dataset/). Each plot
declares which it consumes, and the resolver builds only that, lazily and memoized:

| level | shape | feeds |
|---|---|---|
| `ACTIVITY` | tidy frame, one row per (group, activity) | trends, records, tables, scatter |
| `STREAM` | per-second series for one activity | within-activity signal traces |
| `SPLIT` | samples pooled across a group | GAP model fitting |

The activity feature table is the centrepiece. Columns are *raw sums* — `distance_m`,
`moving_s` — never averages, because [`metrics.py`](src/domain/dataset/metrics.py)
derives averages as ratios (Σ distance ÷ Σ time) that re-aggregate correctly over any
time bin. A mean of per-activity paces silently over-weights short runs; expressing it
as a ratio metric makes that bug unwritable.

**3. Plot registry** — [`src/domain/plots/`](src/domain/plots/). A plot type is a key,
a parameter schema, a data level, and a pure `compute`. Nothing else. It never renders
and never reads request state.

**4. Chart IR** — [`src/domain/charts/ir.py`](src/domain/charts/ir.py). `compute`
returns traces, axes, tables, prose and images as data, not figures. One renderer
draws it ([Python](src/domain/charts/plotly.py) for notebooks,
[TypeScript](web/components/ChartView.tsx) for the web app), so palette, duration axes
and hover styling are defined once and every new plot type inherits them.

The IR round-trips: `PlotOutput.to_dict()` / `from_dict()` are inverses, which is what
lets a computed output be *stored* — see "Computed once" below.

### Extension points

* **A new quantity to plot** → add a column in
  [`features.py`](src/domain/dataset/features.py) and an entry in
  [`metrics.py`](src/domain/dataset/metrics.py). It is immediately available in every
  metric-taking plot, at every granularity, in every chart form, with no frontend change.
* **A new plot type** → one module in [`src/domain/plots/`](src/domain/plots/) plus a
  `register(...)` call. `/registry` picks it up and the UI renders its form.

Panel content that is *not* data goes through the same door: `text_block` and
`image_block` are ordinary registry entries with `requires_data=False`, so a page can
carry its own commentary and images while being edited, stored and reordered by
exactly the same machinery as a chart. A page is a document, so it has to be able to
say what it found.

### Computed once

One plot type fits models rather than aggregating rows, and per-second data behind it
makes that slow. Two mechanisms keep it off the reader's critical path:

* **Outputs are persisted** in `plot_outputs`, keyed by a hash of the render signature
  — plot type, coerced parameters, source spec, *resolved activity ids*, body mass,
  language. Because the activity ids are in the key, importing a run invalidates by
  construction: the same page yields a different key and the stale row is never read.
  A cold worker then serves a fitted GAP page in milliseconds instead of refitting it.
* **`POST /precompute`** fills that cache in the background, rendering the athlete's
  stored analyses exactly as a browser would, skipping panels with no expensive plot
  in them. The web app fires it on connect, so the GAP analysis opens already drawn.
  It is safe to call every time — a pass over a full cache is a read.

`Recompute` on a page (`refresh: true` on `/render`) ignores both caches, refits, and
overwrites. That is the escape hatch instead of a scheme for guessing when a cached
fit has gone stale.

## Repository layout

```
TrailMetrics/
├── src/
│   ├── domain/                     # pure logic, no I/O, no framework
│   │   ├── models/                 # ActivityStream, GapCurve, …
│   │   ├── spec/                   # PageSpec / PanelSpec / PlotSpec / params
│   │   ├── dataset/                # feature table, metric registry, binning, resolver
│   │   ├── charts/                 # chart IR + the Plotly renderer
│   │   ├── plots/                  # the plot registry (one module per type)
│   │   ├── gap/ races/ progress/   # the analytics primitives
│   │   └── ports/                  # interfaces infrastructure must implement
│   ├── infrastructure/
│   │   ├── strava/                 # API client + OAuth/refresh
│   │   ├── postgres/               # schema.sql + repositories
│   │   └── storage/                # stream blobs (Supabase Storage / local disk)
│   ├── usecases/                   # resolve_panel_data, render_page, sync, …
│   └── dashboards/                 # the default analyses, as PageSpecs
├── api/                            # FastAPI compute service
├── web/                            # Next.js app (Vercel)
└── notebook/                       # exploratory notebooks
```

## Running it locally

Three processes: Postgres, the compute API, the web app.

```bash
# 1. Postgres
docker run -d --name tm-pg -e POSTGRES_PASSWORD=tm -e POSTGRES_DB=trailmetrics \
  -p 55432:5432 postgres:16-alpine

# 2. Compute API
pip install -r requirements-compute.txt -r requirements-api.txt
cp .env.example .env.local && python -m api.keys   # paste the three secrets into .env.local
set -a && . ./.env.local && set +a
uvicorn api.main:app --reload --port 8000

# 3. Web app
cd web && npm install
cp .env.example .env.local            # TRAILMETRICS_SERVICE_TOKEN must match the API's
npm run dev                           # http://localhost:3000
```

The schema is applied automatically on API startup (every statement is
`if not exists`). With `SUPABASE_URL` unset, per-second streams are written to
`LOCAL_STREAM_ROOT` on disk, so nothing but Postgres is needed to run the whole thing.

`GET /health` reports what is configured and names any missing environment variable —
the fastest way to diagnose a half-configured setup.

### Deployment

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). The shape is Next.js on **Vercel** →
FastAPI on **Railway** (or Render) → **Supabase** for Postgres and Storage.

Worth knowing up front: **Streamlit cannot run on Vercel** — it needs a long-lived
server with websockets. That is why the web UI is a separate Next.js app rather than a
port of the Streamlit pages.

### The Streamlit app is gone

An earlier version of this project was a Streamlit app with three hard-coded pages.
It was removed once the stack above was verified end to end, because it was a second
UI over the same analytics and the two would drift. Deleting it also retired ~2,800
lines that existed only to serve it — `progress/aggregates.py`, `progress/plotting.py`,
`progress/seasons.py`, `races/plotting.py` and two use cases — all of which are
superseded by the metric registry, the chart IR and the single generic renderer.

The analytics primitives it shared (`gap/`, `races/metrics.py`, `races/smoothing.py`,
`progress/models.py`, `progress/records.py`) are unchanged and still used.

## Notes on the data

* **Power is stored per kilogram.** The model `P = m·v·Cr·factor(gradient)` is linear
  in body mass, so a stored row is valid for any weight and changing yours rescales
  the whole history instantly instead of invalidating it.
* **Running power is always ours; cycling power prefers the meter.** A crank power
  meter is a real measurement, so a ride uses Strava's watts when they exist and the
  aero model otherwise. A watch's *running* power is not a measurement — it is that
  vendor's own undocumented model — so a run always uses the model above, and
  power-to-HR stays comparable across watches and across a whole history.
* **Stored feature rows are recomputed only on a `FEATURE_VERSION` bump.** Nothing is
  re-derived at read time except the body-mass scaling above, so any change to
  `build_activity_features` that alters a stored number has to bump that constant
  (`src/domain/dataset/features.py`) or existing athletes keep their old values
  forever while new ones get the new ones. The bump also invalidates cached renders,
  since `plot_signature` includes it — otherwise a rebuilt row would still be drawn
  from `plot_outputs` as it was before.
* **A bump is served from stored streams, not from Strava.** A feature row is a pure
  function of the per-second arrays (already in object storage) and body mass, so
  `refeaturize_athlete_activities` rebuilds a whole history locally: no rate limit,
  and it works for athletes whose Strava token is long gone. Every sync runs it
  before touching the API, and `python -m api.refeaturize` does the whole population
  in one pass after a deploy (`--dry-run` counts first). Strava is needed only where
  a blob is missing or predates a stream the featurizer now reads — that bound is
  `MIN_LOCAL_REBUILD_VERSION`, which any bump that *adds* a stream must raise.
* **Relative Effort is reported, not computed.** Strava derives its training-load
  score from the athlete's own heart-rate zones, which its API does not expose — so
  `relative_effort` is read off the activity *listing* rather than from the streams.
  That listing is walked by every sync anyway, which is why
  `set_relative_efforts` can refresh the whole history for free, and why a column
  added after an import backfills on the next sync instead of needing a re-import.
* **Strava returns no email address**, under any scope. The app asks for one at
  `/welcome`, immediately after the first sign-in, and stores it on `athletes.email`.
* **Activities without per-second streams** (manual entries, activities Strava won't
  serve) still get a feature row from the activity summary, so they count in volume
  trends. Plots that need full traces skip them *and say how many they skipped*.
* **The GAP preprocessor reads raw altitude** and discards any 10-second split whose
  gradient changes sign. Barometric traces are smooth enough for this; with GPS-grade
  altitude jitter (σ ≈ 0.4 m) it discards nearly every split. Worth revisiting if GAP
  curves ever come back empty on real data.
* **Both GAP models need heart rate**, and it is the constraint that decides whether a
  curve exists at all. The efficiency model normalises every gradient bucket by the
  median efficiency (`HR / speed`) of the flat band, and the auto-learning model can
  only learn an adjustment where a climbing section shares a heart rate with a flat
  one. So an activity recorded without an HR strap contributes nothing, and the
  preprocessor drops it — a year with no HR data produces no curve, and the plot says
  so rather than drawing an empty axis. This is also why the GAP analysis includes
  road runs: they are where the flat reference samples come from.
* **Road runs are included in the GAP analysis on purpose.** See the comment on
  `_DEFAULT_SPORTS` in [`gap_simulator.py`](src/dashboards/gap_simulator.py) — excluding
  them starves the flat reference both models calibrate against.

## Running the notebooks

```bash
jupyter notebook notebook/gap/full-flow.ipynb
```

They expect `STRAVA_CLIENT_ID` / `STRAVA_CLIENT_SECRET` and use the domain layer
directly.
