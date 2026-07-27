-- TrailMetrics schema (Supabase Postgres).
--
-- Design notes worth keeping in mind:
--
-- * The heavy per-second arrays are NOT here. Only one small feature row per
--   activity, which is what every activity-level plot reads. Raw streams live in
--   object storage, referenced by `stream_object`.
--
-- * `features` is JSONB on purpose. It holds the *generated* column families —
--   time per gradient band, best effort per PR distance — whose membership is
--   defined in Python (GRADIENT_BANDS, PR_DISTANCES). Adding a PR distance or a
--   band should not require a migration, and all aggregation happens in pandas
--   after loading, so nothing is lost by not having them as columns.
--
-- * Power is stored per kilogram (`avg_power_w_per_kg`). The power model is
--   linear in body mass, so a row stays valid when the athlete's weight changes —
--   the API multiplies through on read instead of recomputing history.
--
-- * Access goes through the API with the service role, so there are no RLS
--   policies: every query is already scoped by athlete_id server-side. If you
--   ever expose PostgREST directly to browsers, add RLS before doing so.

create extension if not exists pgcrypto;

-- --- Accounts --------------------------------------------------------------

create table if not exists athletes (
    id            bigint primary key,           -- Strava athlete id
    firstname     text not null default '',
    lastname      text not null default '',
    profile_url   text,
    weight_kg     double precision,             -- enables the power metrics
    created_at    timestamptz not null default now(),
    updated_at    timestamptz not null default now()
);

-- Strava tokens, encrypted application-side (Fernet) before they get here.
create table if not exists strava_credentials (
    athlete_id        bigint primary key references athletes(id) on delete cascade,
    access_token_enc  bytea not null,
    refresh_token_enc bytea not null,
    expires_at        timestamptz not null,
    scope             text not null default '',
    updated_at        timestamptz not null default now()
);

-- Import progress, polled by the UI during the first (long) sync.
create table if not exists sync_state (
    athlete_id      bigint primary key references athletes(id) on delete cascade,
    status          text not null default 'idle',   -- idle | running | done | error
    done            integer not null default 0,
    total           integer not null default 0,
    message         text not null default '',
    last_synced_at  timestamptz,
    updated_at      timestamptz not null default now()
);

-- --- Activities ------------------------------------------------------------

create table if not exists activities (
    athlete_id          bigint not null references athletes(id) on delete cascade,
    activity_id         bigint not null,
    start_date          timestamptz not null,
    sport_type          text not null,
    has_streams         boolean not null default false,

    -- Raw sums, never averages: pace and gradient are derived as ratios at query
    -- time so they re-aggregate correctly over any time bin.
    distance_m          double precision,
    elevation_gain_m    double precision,
    moving_s            double precision,
    elapsed_s           double precision,
    gap_distance_m      double precision,
    avg_hr              double precision,
    max_hr              double precision,
    avg_power_w_per_kg  double precision,
    power_to_hr_per_kg  double precision,

    -- time_<band> and best_<distance> families; see the note at the top.
    features            jsonb not null default '{}'::jsonb,

    -- Bumped when the feature computation changes, so a backfill can find stale
    -- rows without wiping the table.
    feature_version     integer not null default 1,
    -- Path in object storage, null when the activity has no per-second data.
    stream_object       text,
    updated_at          timestamptz not null default now(),

    primary key (athlete_id, activity_id)
);

-- Selection queries are always "this athlete, ordered by date".
create index if not exists activities_athlete_date_idx
    on activities (athlete_id, start_date);
create index if not exists activities_athlete_sport_idx
    on activities (athlete_id, sport_type);
create index if not exists activities_stale_idx
    on activities (athlete_id, feature_version);

-- --- Pages -----------------------------------------------------------------

-- A page is a document: the spec is the whole thing, stored as JSONB. `name` is
-- duplicated out of the spec so the page list is one cheap query.
create table if not exists pages (
    id              text primary key,
    athlete_id      bigint not null references athletes(id) on delete cascade,
    name            text not null,
    description     text not null default '',
    icon            text not null default '',
    spec            jsonb not null,
    schema_version  integer not null default 1,
    created_at      timestamptz not null default now(),
    updated_at      timestamptz not null default now()
);

create index if not exists pages_athlete_updated_idx
    on pages (athlete_id, updated_at desc);
