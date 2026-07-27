# TrailMetrics

A data-science workbench for running data. The user builds the pages: pick a data
source, then add the plots you want over it.

The organising idea is that **a page is data, not code**:

```
PageSpec ──< PanelSpec ──< PlotSpec
                └── DataSourceSpec
```

* a **panel** has exactly one data source and as many plots as you like;
* a **data source** is either a hand-picked set of activities, one time window, or
  several named windows compared side by side;
* a **plot** is a registry entry with a declarative parameter schema, so its form —
  including sub-parameters that only appear when relevant — is generated, never written.

Because a page is a serializable document, "the user built this", "the app ships this
as an example" and "this lives in a database" are all the same mechanism. The three
example pages (Personalized GAP Simulator, Race Comparator, Long-Term Progress) are
`PageSpec`s assembled in [`src/dashboards/`](src/dashboards/) from the same panels and
plots a user gets. If an example needs something the builder cannot express, the
builder is missing a feature.

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
returns traces, axes and tables as data, not figures. One renderer draws it
([Python](src/domain/charts/plotly.py) for notebooks,
[TypeScript](web/components/ChartView.tsx) for the web app), so palette, duration axes
and hover styling are defined once and every new plot type inherits them.

### Extension points

* **A new quantity to plot** → add a column in
  [`features.py`](src/domain/dataset/features.py) and an entry in
  [`metrics.py`](src/domain/dataset/metrics.py). It is immediately available in every
  metric-taking plot, at every granularity, in every chart form, with no frontend change.
* **A new plot type** → one module in [`src/domain/plots/`](src/domain/plots/) plus a
  `register(...)` call. `/registry` picks it up and the UI renders its form.

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
│   └── dashboards/                 # the built-in example PageSpecs
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
cp .env.example .env && python -m api.keys   # paste the three secrets into .env
set -a && . ./.env && set +a
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

* **Power is stored per kilogram.** The model `P = m·v·(Cr + g·s)` is linear in body
  mass, so a stored row is valid for any weight and changing yours rescales the whole
  history instantly instead of invalidating it.
* **Activities without per-second streams** (manual entries, activities Strava won't
  serve) still get a feature row from the activity summary, so they count in volume
  trends. Plots that need full traces skip them *and say how many they skipped*.
* **The GAP preprocessor reads raw altitude** and discards any 10-second split whose
  gradient changes sign. Barometric traces are smooth enough for this; with GPS-grade
  altitude jitter (σ ≈ 0.4 m) it discards nearly every split. Worth revisiting if GAP
  curves ever come back empty on real data.

## Running the notebooks

```bash
jupyter notebook notebook/gap/full-flow.ipynb
```

They expect `STRAVA_CLIENT_ID` / `STRAVA_CLIENT_SECRET` and use the domain layer
directly.
