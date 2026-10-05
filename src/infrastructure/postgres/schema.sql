-- TAGG schema (Supabase Postgres).
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
-- * Access goes through the API with the service role, and every query is
--   already scoped by athlete_id server-side. RLS is enabled below with no
--   permissive policies anyway, purely as a backstop: the service role (and a
--   local superuser) bypass it regardless, but it means an anon/authenticated
--   Supabase key — or a future PostgREST exposure — gets nothing instead of
--   everything.

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

-- Strava exposes neither of these, so the athlete types them in once. Birthdate
-- rather than age: an age column is wrong within a year of being written.
alter table athletes add column if not exists birthdate date;
alter table athletes add column if not exists height_cm double precision;

-- Strava's API does not return an email address under any scope, so this is asked
-- for once, right after the first sign-in (see api/routers/auth.py). Nullable
-- because every existing account predates it; the app gates on it being set rather
-- than on the column being NOT NULL, which would lock those accounts out.
alter table athletes add column if not exists email text;

-- Self-reported training zones and VMA pace. Purely a reference the athlete
-- writes down alongside their profile — nothing in the app reads these back
-- into a calculation, so there is no validation beyond "it's a number" and no
-- consequence to leaving them blank.
alter table athletes add column if not exists hr_zone1_end integer;
alter table athletes add column if not exists hr_zone2_end integer;
alter table athletes add column if not exists hr_zone3_end integer;
alter table athletes add column if not exists hr_zone4_end integer;
alter table athletes add column if not exists hr_max integer;
alter table athletes add column if not exists vma_pace_s_per_km double precision;

-- The UI language the athlete has chosen, read by every endpoint that returns
-- translated text (see api/deps.py's `language` dependency). Defaults to
-- English for every existing and new account; unlike the self-reported fields
-- above there is no meaningful "unset" state, so this is NOT NULL rather than
-- gated on presence.
alter table athletes add column if not exists lang text not null default 'en';

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

-- Strava's Relative Effort (its `suffer_score`) — the training-load score for the
-- session. Added rather than baked into the create above so an existing deployment
-- converges without a migration step. It is *reported*, not computed: Strava derives
-- it from the athlete's own heart-rate zones, which the API does not expose, so it
-- is refreshed from the activity listing on every sync (see
-- `set_relative_efforts`) rather than recomputed from stored streams.
alter table activities add column if not exists relative_effort double precision;

-- The route as a Google-encoded polyline, from Strava's activity summary.
-- Deliberately NOT part of the feature row: it is metadata for drawing a map, not
-- a quantity anything aggregates over, so it stays out of the numeric frame the
-- plots read. Nullable — older imports predate it, and indoor runs have no route.
alter table activities add column if not exists summary_polyline text;

-- Power, beyond running's existing `avg_power_w_per_kg`: real meter data wins
-- when Strava has it, so it gets its own absolute column rather than reusing the
-- per-kg one. Running is the exception and never uses it — a watch's running
-- power is a per-vendor model rather than a measurement, so a run always uses
-- ours (see build_activity_features) and leaves this column null. Cycling's
-- modelled estimate (src/domain/cycling/power.py) is also absolute rather than
-- per-kg: unlike running's, its aero term is a fixed population constant, not
-- proportional to rider mass, so "compute once at 1 kg and rescale on read"
-- would corrupt it — it is instead computed once against whatever weight was on
-- file at sync time, like every other cached feature. `power_source` says which
-- of the three (measured / running's per-kg model / cycling's absolute model)
-- produced the number, for a UI badge; it is not a `_SCALAR_COLUMNS` entry (those
-- all run through a float coercion), so it is written like `summary_polyline`.
alter table activities add column if not exists avg_power_w_measured double precision;
alter table activities add column if not exists avg_power_w_modelled double precision;
alter table activities add column if not exists power_source text;

-- Power-to-HR pairs with avg_power_w_measured the same way power_to_hr_per_kg
-- pairs with avg_power_w_per_kg: an already-absolute ratio (real watts ÷ HR),
-- so it needs no weight to use, unlike the running-modelled figure.
alter table activities add column if not exists power_to_hr_measured double precision;

-- Athlete-entered, not synced from Strava — never touched by upsert_rows/_to_params,
-- so a resync can never overwrite what the athlete typed. rpe is 1-10; feeling is
-- one of 'faible' / 'ok' / 'fort'.
alter table activities add column if not exists rpe smallint;
alter table activities add column if not exists feeling text;

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

-- Which default analysis this row is, or NULL for one the athlete created.
--
-- Every athlete is seeded with the three analyses the product ships. They are normal
-- pages — edited in place, saved like any other — and this column buys them exactly
-- one property: they cannot be deleted. Denormalized out of the spec so seeding can
-- ask "does this athlete already have the GAP one?" in one indexed query.
alter table pages add column if not exists builtin_key text;

-- Idempotent seeding, enforced by the database rather than by a careful caller: two
-- concurrent first-page-loads cannot produce two GAP analyses.
create unique index if not exists pages_athlete_builtin_key_idx
    on pages (athlete_id, builtin_key) where builtin_key is not null;

-- --- Computed plot outputs -------------------------------------------------

-- A finished plot output, keyed by everything that could change it.
--
-- The app already memoizes outputs in the API process. That is enough for editing
-- a page, and useless for the expensive ones: a GAP curve is a model fit over
-- per-second data, and losing it on every deploy means the athlete waits for it
-- again. Persisting it turns "wait for the fit" into "the page opens drawn".
--
-- `signature` is a hash of the render signature (plot type, coerced parameters,
-- source spec, resolved activity ids, body mass, language) — see
-- `src.usecases.render_page.plot_signature`. Because the resolved activity ids are
-- in it, importing a new run invalidates by construction: the same page produces a
-- different key and the stale row is simply never read again. `created_at` is what
-- the sweeper uses to retire those.
create table if not exists plot_outputs (
    athlete_id   bigint not null references athletes(id) on delete cascade,
    signature    text not null,
    plot_type    text not null default '',
    payload      jsonb not null,
    created_at   timestamptz not null default now(),

    primary key (athlete_id, signature)
);

create index if not exists plot_outputs_athlete_created_idx
    on plot_outputs (athlete_id, created_at desc);

-- --- Background precomputation --------------------------------------------

-- Progress of a background compute pass, polled by the UI exactly like `sync_state`.
-- One row per (athlete, kind); `kind` is the job name, currently only 'builtin'
-- (fill the cache for the built-in example pages).
create table if not exists precompute_jobs (
    athlete_id   bigint not null references athletes(id) on delete cascade,
    kind         text not null,
    status       text not null default 'idle',   -- idle | running | done | error
    done         integer not null default 0,
    total        integer not null default 0,
    message      text not null default '',
    finished_at  timestamptz,
    updated_at   timestamptz not null default now(),

    primary key (athlete_id, kind)
);

-- --- Training diary ----------------------------------------------------

-- A planned workout or planned goal: a text cell an athlete (or, later, their
-- coach) puts on a day of the training calendar. `kind` is the only thing that
-- distinguishes a goal from a workout — both are title + body — so they share one
-- table rather than two identical schemas.
create table if not exists planned_items (
    id          text primary key,
    athlete_id  bigint not null references athletes(id) on delete cascade,
    kind        text not null,              -- 'workout' | 'goal'
    date        date not null,
    title       text not null default '',
    body        text not null default '',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Only meaningful for a goal: a secondary goal keeps the goal's colour but shaded,
-- so a diary with several goals can still say which one is the main target.
alter table planned_items add column if not exists importance text not null default 'primary';

-- Optional multi-day span for a note. NULL means "just `date`" — true for every
-- workout and goal (both are inherently one day) and for any note that predates
-- this column. The API reads it as `coalesce(end_date, date)` everywhere rather
-- than backfilling it.
alter table planned_items add column if not exists end_date date;

-- The calendar always queries "this athlete, this date range".
create index if not exists planned_items_athlete_date_idx
    on planned_items (athlete_id, date);

-- --- Activity comments -------------------------------------------------------

-- Free-text notes an athlete attaches to one of their own completed activities.
-- Multiple per activity, each independently editable and deletable. `activity_id`
-- is not foreign-keyed to `activities` (nothing else in this schema references
-- that composite-keyed table either) — a comment on an activity later re-synced
-- or removed is harmless orphaned text, not a referential-integrity problem.
create table if not exists activity_comments (
    id          text primary key,
    athlete_id  bigint not null references athletes(id) on delete cascade,
    activity_id bigint not null,
    body        text not null,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- The detail card always queries "this athlete, this activity, oldest first".
create index if not exists activity_comments_athlete_activity_idx
    on activity_comments (athlete_id, activity_id, created_at);

-- --- Uploaded images -------------------------------------------------------

-- Images an athlete puts in a panel. Bytes live in Postgres rather than in the
-- stream bucket: they are small, there are few of them, and keeping them here means
-- the feature works identically on a local Postgres and on Supabase with no bucket
-- to configure and no signed-URL handling. `MAX_ASSET_BYTES` in api/routers/assets.py
-- is what keeps that assumption true.
create table if not exists assets (
    id           text primary key,
    athlete_id   bigint not null references athletes(id) on delete cascade,
    filename     text not null default '',
    content_type text not null,
    byte_size    integer not null,
    data         bytea not null,
    created_at   timestamptz not null default now()
);

create index if not exists assets_athlete_created_idx
    on assets (athlete_id, created_at desc);

-- --- Blog -------------------------------------------------------------------

-- Written by the single "master" operator account (see api/deps.py's
-- `require_master`), read by anyone — the blog is public, unlike the rest of the
-- app. The PDF and its rasterized pages live in object storage (see
-- src/domain/ports/blog_media.py); this row is just the metadata and `pdf_path`,
-- the prefix under which those objects are found.
create table if not exists blog_posts (
    id           text primary key,
    slug         text not null unique,
    title        text not null default '',
    body_text    text not null default '',
    pdf_path     text not null default '',
    page_count   integer not null default 0,
    published    boolean not null default true,
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now()
);

create index if not exists blog_posts_published_created_idx
    on blog_posts (published, created_at desc);

-- --- Race plans -------------------------------------------------------------

-- A saved "Plan de course": its inputs, not its result. The plan is recomputed on
-- every open (it is fast), so a plan on a personal GAP curve follows the athlete's
-- newer runs rather than freezing the curve it was first built with. The GPX is
-- gzipped in place, like an asset: a course is a few hundred KB of repetitive XML,
-- and keeping it here needs no bucket. `distance_m` / `elevation_gain_m` are
-- denormalized from the course only so the list can show them without a parse.
create table if not exists race_plans (
    id                text primary key,
    athlete_id        bigint not null references athletes(id) on delete cascade,
    title             text not null default '',
    gpx_name          text not null default '',
    gpx_gz            bytea not null,
    params            jsonb not null,
    distance_m        double precision,
    elevation_gain_m  double precision,
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create index if not exists race_plans_athlete_updated_idx
    on race_plans (athlete_id, updated_at desc);

-- --- Accounts (v2, design/specs/auth.md) -------------------------------------

-- The identity is an email + password account; the Strava athlete becomes an
-- optional attachment of it (`athletes.account_id`). What comes from Strava stays
-- keyed by `athlete_id`; what comes from the account (saved plans, estimated
-- zones, coaching requests) is keyed by `account_id`. Nothing from Strava is
-- written into `accounts`.
create extension if not exists citext;

create table if not exists accounts (
    id                uuid primary key default gen_random_uuid(),
    email             citext unique not null,
    password_hash     text not null,              -- argon2id, parameters encoded in the hash
    role              text not null default 'athlete'
                      check (role in ('athlete', 'coach', 'master')),
    lang              text not null default 'en',
    email_verified_at timestamptz,                -- set by a verification link or a completed reset
    created_at        timestamptz not null default now(),
    last_login_at     timestamptz
);

alter table athletes add column if not exists account_id uuid
    references accounts(id) on delete set null;
create unique index if not exists athletes_account_id_idx on athletes (account_id);

-- Opaque session tokens, stored hashed: a database dump cannot be replayed as a
-- cookie. One row per signed-in device, so a session can be revoked on its own
-- (sign out), all together (sign out everywhere, password reset).
create table if not exists sessions (
    id           uuid primary key default gen_random_uuid(),
    account_id   uuid not null references accounts(id) on delete cascade,
    token_hash   bytea not null unique,           -- sha256 of the token, never the token
    created_at   timestamptz not null default now(),
    expires_at   timestamptz not null,
    last_seen_at timestamptz,
    user_agent   text
);

create index if not exists sessions_account_idx on sessions (account_id);

create table if not exists password_resets (
    token_hash  bytea primary key,
    account_id  uuid not null references accounts(id) on delete cascade,
    expires_at  timestamptz not null,
    used_at     timestamptz
);

-- Proof that the account holds its address. Gates what must not go to whoever
-- merely typed an email first — today, the `master` role of MASTER_EMAIL.
create table if not exists email_verifications (
    token_hash  bytea primary key,
    account_id  uuid not null references accounts(id) on delete cascade,
    expires_at  timestamptz not null,
    used_at     timestamptz
);

-- Fixed-window counters for the auth endpoints. In the database rather than in
-- memory so the limit holds across replicas and restarts; no Redis at this scale.
create table if not exists login_attempts (
    key          text not null,                   -- 'login-email:<email>', 'login-ip:<ip>', ...
    window_start timestamptz not null,
    count        integer not null default 0,
    primary key (key, window_start)
);

-- Saved plans belong to the account (design/tagg/access.md: saving needs an
-- account, not Strava). `athlete_id` stays for plans saved before accounts; they
-- move over when that athlete is attached (`link_athlete`).
alter table race_plans add column if not exists account_id uuid
    references accounts(id) on delete cascade;
alter table race_plans alter column athlete_id drop not null;
create index if not exists race_plans_account_updated_idx
    on race_plans (account_id, updated_at desc);

-- The list's thumbnail (route + elevation profile, downsampled), computed on
-- save. Null for plans saved before it existed: the list fills those in once.
alter table race_plans add column if not exists preview jsonb;

-- Level assessments (design/specs/level.md). Inputs and result as JSON: the
-- three tests take different inputs, and the result is the LevelEstimate the
-- tool showed — kept as shown, not recomputed, so "estimated on …" stays true.
create table if not exists level_estimates (
    id          uuid primary key default gen_random_uuid(),
    account_id  uuid not null references accounts(id) on delete cascade,
    method      text not null,
    inputs      jsonb not null,
    result      jsonb not null,
    created_at  timestamptz not null default now()
);

create index if not exists level_estimates_account_created_idx
    on level_estimates (account_id, created_at desc);

-- --- Coaching (v2, design/specs/coaching.md) ---------------------------------

-- A request to be coached. One pending per account (partial unique index); a
-- decided request stays as a record — declining never deletes.
create table if not exists coaching_requests (
    id          uuid primary key default gen_random_uuid(),
    account_id  uuid not null references accounts(id) on delete cascade,
    message     text not null default '',
    phone       text,                               -- as typed
    phone_e164  text,                               -- normalized when possible
    contact     text not null check (contact in ('email', 'phone')),
    status      text not null default 'pending'
                check (status in ('pending', 'accepted', 'declined', 'withdrawn')),
    created_at  timestamptz not null default now(),
    decided_at  timestamptz,
    decided_by  uuid references accounts(id) on delete set null
);

create unique index if not exists coaching_requests_one_pending_idx
    on coaching_requests (account_id) where status = 'pending';

-- Who coaches whom. `is_coached` is "a row exists with athlete_id = me"; the
-- rail's athlete switcher and the view-as check both read this table.
create table if not exists coaching (
    coach_id    uuid not null references accounts(id) on delete cascade,
    athlete_id  uuid not null references accounts(id) on delete cascade,
    since       timestamptz not null default now(),
    primary key (coach_id, athlete_id)
);

create index if not exists coaching_athlete_idx on coaching (athlete_id);

-- --- Row-level security -----------------------------------------------------

-- No policies defined: this is a default-deny backstop for any role other than
-- the service role / a local superuser, both of which bypass RLS outright. See
-- the note at the top of this file.
alter table athletes enable row level security;
alter table strava_credentials enable row level security;
alter table sync_state enable row level security;
alter table activities enable row level security;
alter table pages enable row level security;
alter table plot_outputs enable row level security;
alter table precompute_jobs enable row level security;
alter table planned_items enable row level security;
alter table assets enable row level security;
alter table activity_comments enable row level security;
alter table blog_posts enable row level security;
alter table race_plans enable row level security;
alter table accounts enable row level security;
alter table sessions enable row level security;
alter table password_resets enable row level security;
alter table email_verifications enable row level security;
alter table level_estimates enable row level security;
alter table coaching_requests enable row level security;
alter table coaching enable row level security;
alter table login_attempts enable row level security;
