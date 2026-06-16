"""TrailMetrics home page — connect to Strava and load your history once.

Analyses live on their own pages and only unlock once the data is loaded here.
The fetched streams stay in session memory until the app is closed, so you can
play with analysis parameters without ever re-fetching.
"""

from _helpers import add_repo_root_to_path, inject_theme_css, render_run_loader

add_repo_root_to_path()

import json
import os
from pathlib import Path

# Credentials are entered in the UI rather than the environment, so silence
# stravalib's env-var warnings before it is imported.
os.environ.setdefault("SILENCE_TOKEN_WARNINGS", "true")

from datetime import date, datetime, time, timedelta

import streamlit as st
from stravalib import Client

from src.infrastructure.strava.strava_client import StravaClient
from src.usecases.fetch_athlete_history import (
    FetchAthleteHistory,
    FetchAthleteHistoryInput,
)

st.set_page_config(page_title="TrailMetrics", page_icon="🏔️", layout="wide")
inject_theme_css()

st.title("🏔️ TrailMetrics")
st.subheader("Connect Strava, load your history, then explore your running data")

st.markdown(
    """
    Set up the connection below and load your activity history **once**. After
    that, open any analysis from the **left sidebar** — they all reuse this data,
    so you can tweak parameters freely without re-fetching.

    New here? See **How it works** in the sidebar.
    """
)

st.divider()

# --- 1. Connect Strava -----------------------------------------------------
# Strava auth comes first on purpose: authorizing fully reloads the page (a new
# Streamlit session that wipes st.session_state), so anything typed *before* it
# would be lost. Credentials are cached to a local file so they survive that
# reload; the runner details below are entered *after* it, so they're safe.
_CREDS_CACHE = Path.home() / ".trailmetrics_strava.json"


def _load_cached_creds() -> tuple[str, str]:
    try:
        data = json.loads(_CREDS_CACHE.read_text())
        return data.get("client_id", ""), data.get("client_secret", "")
    except Exception:
        return "", ""


def _save_cached_creds(client_id: str, client_secret: str) -> None:
    try:
        _CREDS_CACHE.write_text(
            json.dumps({"client_id": client_id, "client_secret": client_secret})
        )
        _CREDS_CACHE.chmod(0o600)  # readable only by the owner
    except Exception:
        pass


_cached_id, _cached_secret = _load_cached_creds()

st.header("1. Connect Strava")
col_id, col_secret = st.columns(2)
with col_id:
    client_id = st.text_input(
        "STRAVA_CLIENT_ID",
        value=os.environ.get("STRAVA_CLIENT_ID") or _cached_id,
        help="Your Strava application client ID.",
    )
with col_secret:
    client_secret = st.text_input(
        "STRAVA_CLIENT_SECRET",
        value=os.environ.get("STRAVA_CLIENT_SECRET") or _cached_secret,
        type="password",
    )

# Persist as soon as both are present, so they're available after the redirect.
if client_id and client_secret:
    _save_cached_creds(client_id, client_secret)

# Strava redirects back to this app with `?code=...`; the code is picked up
# automatically — no copy-paste. Strava only validates the callback *domain*, so
# any path/port works; the app's "Authorization Callback Domain" must match
# REDIRECT_URI's host (localhost).
REDIRECT_URI = os.environ.get("STRAVA_REDIRECT_URI", "http://localhost:8501")

# When Strava sends us back with ?code=..., exchange it once for a token, then
# clear the URL so a refresh can't reuse a now-spent code. Credentials come from
# the inputs above (env or cache file), which survive the page reload.
incoming_code = st.query_params.get("code")
if incoming_code and "access_token" not in st.session_state:
    if client_id and client_secret:
        try:
            token_response = Client().exchange_code_for_token(
                client_id=int(client_id),
                client_secret=client_secret,
                code=incoming_code,
            )
            st.session_state["access_token"] = token_response["access_token"]
        except Exception as e:
            st.error(f"Token exchange failed: {e}")
    else:
        st.error("Missing credentials — re-enter STRAVA_CLIENT_ID / SECRET above.")
    st.query_params.clear()

if "access_token" in st.session_state:
    st.success("✅ Connected to Strava.")
elif client_id and client_secret:
    auth_url = Client().authorization_url(
        client_id=int(client_id),
        redirect_uri=REDIRECT_URI,
    )
    # Plain anchor with target="_self" so the auth happens in the *same* tab
    # (st.link_button always opens a new one, leaving a stale page behind).
    st.markdown(
        f'<a href="{auth_url}" target="_self" style="display:inline-block;'
        "padding:0.55rem 1.1rem;background:#fc4c02;color:#fff;border-radius:0.5rem;"
        'text-decoration:none;font-weight:600;">Connect with Strava</a>',
        unsafe_allow_html=True,
    )
else:
    st.info("Enter your STRAVA_CLIENT_ID and STRAVA_CLIENT_SECRET above to connect.")

# Nothing below is actionable until Strava is connected, so hide the rest of the
# setup (runner details, data loading) behind a successful connection.
if "access_token" not in st.session_state:
    st.stop()

st.divider()

# --- 2. Runner -------------------------------------------------------------
# Entered after the Strava connect (and its page reload), so it isn't wiped.
st.header("2. Runner")
runner_name = st.text_input("Your name", value=st.session_state.get("runner_name", ""))
if runner_name:
    st.session_state["runner_name"] = runner_name

runner_weight = st.number_input(
    "Your weight (kg)",
    min_value=30.0,
    max_value=200.0,
    value=st.session_state.get("runner_weight_kg"),
    step=0.5,
    help="Used to estimate running power on the Race Comparator. Leave empty to skip.",
)
if runner_weight:
    st.session_state["runner_weight_kg"] = runner_weight

st.divider()

# --- 3. Load data ----------------------------------------------------------
st.header("3. Load your data")
st.caption(
    "Fetches every run (all types) from the date below up to today. This can "
    "take a while the first time — it only runs once per session."
)

default_from = date.today() - timedelta(days=365 * 2)
fetch_from_date = st.date_input(
    "Fetch data back to",
    value=default_from,
    max_value=date.today(),
    help="The oldest date to fetch initially. Older activities are ignored.",
)

load = st.button(
    "Load my data",
    type="primary",
    disabled=("access_token" not in st.session_state) or not runner_name,
)

if load:
    loader = st.empty()
    # Indeterminate trail while we list activities (length unknown yet).
    render_run_loader(loader, "Scouting your activities on Strava…", frac=None)

    def _on_progress(done: int, total: int) -> None:
        frac = (done / total) if total else 1.0
        render_run_loader(loader, f"Fetching activity {done} of {total}…", frac=frac)

    stravalib_client = Client(access_token=st.session_state["access_token"])
    stream_source = StravaClient(stravalib_client)
    usecase = FetchAthleteHistory(stream_source=stream_source)
    result = usecase.execute(
        FetchAthleteHistoryInput(
            sport_types=["TrailRun", "Run"],
            from_date=datetime.combine(fetch_from_date, time.min),
            to_date=datetime.combine(date.today(), time.max),
            max_activities=None,
            verbose=False,
        ),
        progress_callback=_on_progress,
    )
    loader.empty()

    st.session_state["athlete_streams"] = result.streams
    st.session_state["fetch_oldest_date"] = result.oldest_date
    st.session_state["fetch_newest_date"] = result.newest_date
    st.session_state["fetch_activity_count"] = result.activity_count

# --- Status ----------------------------------------------------------------
if "athlete_streams" in st.session_state:
    count = st.session_state.get("fetch_activity_count", 0)
    oldest = st.session_state.get("fetch_oldest_date")
    newest = st.session_state.get("fetch_newest_date")

    st.success(f"✅ Loaded {count} activities — analyses are unlocked.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Activities", count)
    c2.metric(
        "Oldest session",
        oldest.strftime("%Y-%m-%d") if isinstance(oldest, datetime) else "—",
    )
    c3.metric(
        "Most recent session",
        newest.strftime("%Y-%m-%d") if isinstance(newest, datetime) else "—",
    )
    st.info("Open an analysis from the **left sidebar** to get started.")
else:
    st.warning("No data loaded yet. Complete steps 1–4 to unlock the analyses.")
