# Deployment

```
Vercel (Next.js UI)
   │  browser → /api/proxy/* on the same origin (first-party cookie, no CORS)
   ▼
Railway or Render (FastAPI + pandas/scipy/XGBoost)
   │
   ▼
Supabase — Postgres (athletes, feature rows, pages) + Storage (per-second streams)
```

Two things determine this shape:

* **Streamlit cannot run on Vercel**, and Vercel is a poor host for this Python
  dependency set — XGBoost + scipy + pandas + scikit-learn sits at or over the 250 MB
  serverless bundle limit, with a cold start on every model fit. So the UI is a
  Next.js app on Vercel and the compute is a long-lived container elsewhere.
* **The OAuth callback lands on the web app, not the API.** A cookie set by the API's
  domain would be third-party — increasingly blocked outright — and would force
  credentialed CORS. Landing on Next.js keeps the session cookie first-party.

---

## 1. Supabase

1. Create a project. From **Connect → ORMs/psql**, take a **pooler** URI — not the
   direct `db.<ref>.supabase.co` one, which resolves to IPv6 only unless you pay for
   the IPv4 add-on, so it fails from most laptops and many hosts.

   Of the two pooler modes, prefer the **session** pooler (port `5432` on the
   `pooler.supabase.com` host) for the API container: it behaves like a normal
   Postgres connection, and this app holds a small pool of long-lived ones rather
   than opening a connection per request. The **transaction** pooler (port `6543`)
   also works — [`pool.py`](../src/infrastructure/postgres/pool.py) disables
   server-side prepared statements so it does not break there.

   The username is `postgres.<project-ref>`, and any special character in the
   password must be percent-encoded in the URI (`@` → `%40`).
2. From **Project settings → API**, copy the project URL and the **service role** key.
   That key bypasses row-level security, so it must only ever live on the API.
3. Nothing else. The API applies [`schema.sql`](../src/infrastructure/postgres/schema.sql)
   on startup and creates the Storage bucket if it is missing.

There are no RLS policies, because every query is already scoped by `athlete_id`
server-side and the browser never talks to Supabase. **If you later expose PostgREST
to browsers, add RLS first.**

## 2. Strava API application

At <https://www.strava.com/settings/api>:

* **Authorization Callback Domain** — the *web app's* host, with no scheme or path:
  `your-app.vercel.app` (and `localhost` for local work). This is the most common
  setup mistake: it is the Next.js host, not the API's.
* Note the client ID and secret.

The app requests `read` and `activity:read_all`. The `activity:read_all` scope is
required to see private and followers-only activities; plain `activity:read` silently
returns only public ones.

## 3. Compute API (Railway)

Deploy this repository. [`railway.json`](../railway.json) selects the Dockerfile and
points the health check at `/health`. For Render, [`render.yaml`](../render.yaml) does
the same.

Generate the secrets:

```bash
python -m api.keys
```

Environment variables:

| Variable | Value |
|---|---|
| `DATABASE_URL` | Supabase pooler connection string |
| `SUPABASE_URL` | `https://<project>.supabase.co` |
| `SUPABASE_SERVICE_KEY` | service role key |
| `SUPABASE_BUCKET` | `activity-streams` |
| `STRAVA_CLIENT_ID` / `STRAVA_CLIENT_SECRET` | from step 2 |
| `ENCRYPTION_KEY` | generated — Fernet key encrypting Strava tokens at rest |
| `SERVICE_TOKEN` | generated — **must match the web app's** |
| `WEB_APP_URL` | `https://taggcoaching.vercel.app` |
| `MASTER_EMAIL` | the operator's email — becomes `master` once proven, shown as the contact without mail |
| `MAIL_*` | **leave unset** for now — see *Mail* below |

Roles live in the database (`accounts.role`): `master` (blog + coach), `coach`
(browses other athletes via the sidebar switcher — their data, pages and training
diary, never their Strava connection), `athlete`. Every account starts as
`athlete`. The account registered with `MASTER_EMAIL` becomes `master` once it
proves it holds the address — the verification link sent at sign-up, or a
completed password reset (which also reclaims the address from anyone who
registered it first). Any other role — and the master, while no mail is
configured — is set by hand, either in the Supabase SQL editor:

```sql
update accounts set role = 'master' where email = 'tanguy.blervacque@gmail.com';
update accounts set role = 'coach' where email = 'someone@example.com';
```

or from a shell in the API container (`railway ssh`):

```bash
python -m api.roles set someone@example.com coach
python -m api.roles show someone@example.com
```

### Mail

The deployment runs **without mail**: `MAIL_FROM` is unset, so nothing is sent.
No verification link at sign-up (the Home reminder stays hidden), no coaching
notification, and "Forgot your password?" shows "write to `MASTER_EMAIL`"
instead of a form. A reset is then done by hand: check that the person writing
does own the address, then send them a link —

```bash
python -m api.roles reset-link someone@example.com   # in a `railway ssh` shell
```

— valid 30 minutes, single use, and it signs out every device, exactly like
the emailed one. Completing it also proves the address (so it is how the
master account can verify itself, too).

Why not just send from Gmail: the web app lives on `taggcoaching.vercel.app`,
whose DNS is Vercel's, so no mail service can be authorized to send for it
(SPF/DKIM/DMARC, required by Gmail, Yahoo and Microsoft); and Railway blocks
outbound SMTP below the Pro plan. Turning mail on later needs no code: buy a
domain (e.g. `taggcoaching.com`, ~10 $/year), verify `mail.<domain>` at Resend
(free tier), then set `MAIL_FROM=TAGG <no-reply@mail.<domain>>` and
`MAIL_RESEND_API_KEY`. Resend is an HTTPS API, so the SMTP block does not apply.

Sessions are opaque tokens stored hashed in `sessions` (30 days, sliding). The
JWT sessions of earlier versions are no longer read: after this deploy everyone
signs in again — existing users create an account, connect Strava, and get their
history back through the attachment.

Leave `DEV_MODE` unset. It bypasses authentication and only takes effect when
explicitly set together with `DEV_ATHLETE_ID`, but there is no reason for it in a
deployed environment.

Check `GET /health`: `missing_config` must be empty.

**Sizing.** One worker per container — a request can hold a large pandas frame and a
fitted model, and the per-athlete caches are per process. Scale with replicas, not
workers. 1 GB is comfortable; 512 MB works with the knobs below turned down.

Three environment variables are the memory ceiling. They have defaults in the
Dockerfile sized for a 1 GB instance, and they are what to change when the
container is being OOM-killed — before reaching for a bigger plan.

| Variable | Default | What it bounds |
|---|---|---|
| `MAX_CONCURRENT_REQUESTS` | `4` | CPU-bound handlers running at once. Handlers are sync, so they run in anyio's threadpool, whose own default is 40 — sized for I/O, not for a render that holds a decoded history. Requests past this queue rather than run. |
| `MEMO_BUDGET_MB` | `192` | Bytes of decoded streams, derived series and fitted models kept warm across requests, for **all** athletes together (see `api/memo.py`). Evicting is always safe — entries are memoized pure computations — so this can be cut hard. |
| `MALLOC_ARENA_MAX` | `2` | glibc arenas. Left at its default a threaded process fragments across many arenas and its RSS never returns, so the kill lands on a later, smaller request than the one that caused the spike. |

`GET /health` reports `memory.rss_bytes` alongside what the memo is holding, which
is the fastest way to tell a real leak from a cache doing its job.

On 512 MB, start from `MAX_CONCURRENT_REQUESTS=2` and `MEMO_BUDGET_MB=64`.

## 4. Web app (Vercel)

Import the repository with **root directory `web`**. Vercel detects Next.js.

| Variable | Value |
|---|---|
| `TRAILMETRICS_API_URL` | the Railway/Render service URL |
| `TRAILMETRICS_SERVICE_TOKEN` | same value as the API's `SERVICE_TOKEN` |
| `NEXT_PUBLIC_APP_URL` | `https://your-app.vercel.app` |
| `NEXT_PUBLIC_LANG` | `fr` or `en` |
| `NEXT_PUBLIC_MAP_TILE_URL` | optional — route-map tile URL template |
| `NEXT_PUBLIC_MAP_TILE_ATTRIBUTION` | optional — attribution for that tile host |

The two map variables are optional: unset, route maps fall back to
OpenStreetMap's own tile servers. That is within their [tile usage
policy](https://operations.osmfoundation.org/policies/tiles/) at hobby scale
only — the policy rules out apps with real traffic, and osm.org answers
traffic it objects to with a placeholder image in place of the map. Set both
variables to a provider with a plan (MapTiler, Stadia, Thunderforest, …) before
the app has an audience.

`TRAILMETRICS_API_URL` and `TRAILMETRICS_SERVICE_TOKEN` are read only in server
components and route handlers, so neither reaches the browser bundle. Do not rename
them with a `NEXT_PUBLIC_` prefix — that would publish the service token.

Once both are deployed, set `WEB_APP_URL` on the API to the real Vercel URL
(`https://taggcoaching.vercel.app`) and redeploy it. The same URL goes in
`NEXT_PUBLIC_APP_URL` on Vercel (build-time: redeploy after changing it) and its
host in the Strava app's *Authorization Callback Domain*.

## 5. First run

1. Open the app and **create your account** with `MASTER_EMAIL`, then make it
   `master` (SQL editor or `api.roles`, see above) — without mail there is no
   verification link to do it for you.
2. **Connect Strava** from Home: an athlete who signed in before accounts gets
   their history and saved plans back.
3. **Import my activities.** This is the slow part: Strava allows 100 requests per 15
   minutes, so a long history takes a while. It runs in the background, writes rows in
   batches, and is resumable — re-running continues where it stopped, because
   already-stored activities are skipped.
4. Set your weight to unlock the power and power-to-heart-rate metrics.

## Operational notes

**Rotating `ENCRYPTION_KEY`** does not lose activity data, but every athlete has to
reconnect Strava: stored tokens become undecryptable and the API treats that as
"not connected".

**Changing the feature computation** — bump `FEATURE_VERSION` in
[`activity_repository.py`](../src/infrastructure/postgres/activity_repository.py). Rows
at an older version stop counting as "known", so the next import recomputes them.
`POST /activities/sync {"force": true}` forces a full recompute.

**Background sync and restarts.** The sync runs as a FastAPI background task and
reports progress into the `sync_state` table rather than memory, so progress survives a
redeploy — but an in-flight sync is killed by one. Re-running finishes the job. If you
grow past that, this is the piece to move to a real queue.

**Costs.** Streams compress to roughly 20 kB per hour of activity, so a 500-activity
history is about 10 MB of Storage and a few hundred small Postgres rows. The container
is the only meaningful cost.
