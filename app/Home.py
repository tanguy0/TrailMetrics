"""TrailMetrics home page — connect to Strava and load your history once.

Analyses live on their own pages and only unlock once the data is loaded here.
The fetched streams stay in session memory until the app is closed, so you can
play with analysis parameters without ever re-fetching.
"""

from _helpers import add_repo_root_to_path, inject_theme_css, render_run_loader

add_repo_root_to_path()

import os
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

# --- 1. Runner -------------------------------------------------------------
st.header("1. Runner")
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

# --- 2. Strava credentials -------------------------------------------------
st.header("2. Strava credentials")
col_id, col_secret = st.columns(2)
with col_id:
    client_id = st.text_input(
        "STRAVA_CLIENT_ID",
        value=os.environ.get("STRAVA_CLIENT_ID", ""),
        help="Your Strava application client ID.",
    )
with col_secret:
    client_secret = st.text_input(
        "STRAVA_CLIENT_SECRET",
        value=os.environ.get("STRAVA_CLIENT_SECRET", ""),
        type="password",
    )

# --- 3. Authorize ----------------------------------------------------------
st.header("3. Authorize")
if st.button("Generate authorization URL"):
    if not client_id:
        st.error("Set STRAVA_CLIENT_ID first.")
    else:
        tmp_client = Client()
        st.session_state["auth_url"] = tmp_client.authorization_url(
            client_id=int(client_id),
            redirect_uri="http://localhost:5000/authorization",
        )

if "auth_url" in st.session_state:
    st.markdown(f"[Open Strava authorization]({st.session_state['auth_url']})")
    st.caption("After authorizing, copy the `code=...` value from the redirect URL.")

auth_code = st.text_input("Authorization code (one-shot)", value="")

if st.button("Exchange code for token"):
    if not (client_id and client_secret and auth_code):
        st.error("client_id, client_secret and auth_code are all required.")
    else:
        try:
            token_response = Client().exchange_code_for_token(
                client_id=int(client_id),
                client_secret=client_secret,
                code=auth_code,
            )
            st.session_state["access_token"] = token_response["access_token"]
            st.success("Token stored in session.")
        except Exception as e:
            st.error(f"Token exchange failed: {e}")

st.divider()

# --- 4. Load data ----------------------------------------------------------
st.header("4. Load your data")
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
