"""How TrailMetrics works — explanations kept off the main page."""

from _helpers import add_repo_root_to_path, inject_theme_css

add_repo_root_to_path()

import streamlit as st

st.set_page_config(page_title="How it works", page_icon="❓", layout="wide")
inject_theme_css()

st.title("❓ How TrailMetrics works")

st.markdown(
    """
    TrailMetrics is a personal lab for running-data simulations. The flow is
    designed around **loading your data once** and then exploring it freely.

    ### 1. Connect & load (Home page)
    1. Provide your Strava `client_id` / `client_secret`, then click
       **Connect with Strava** — you'll be redirected to authorize and brought
       straight back with a token (no codes to copy). Credentials are remembered
       for next time.
    2. Enter your name and weight.
    3. Click **Load my data**. TrailMetrics fetches the maximum history
       available (all run types) and keeps it in memory for the whole session.

    Once loaded, you'll see how many activities were fetched and the date range
    they span (oldest → most recent session).

    ### 2. Run an analysis (sidebar pages)
    Each analysis lives on its own page and unlocks only after data is loaded.
    All of its inputs — **date range**, **session types**, model parameters,
    intensity ranges — are *filters applied to the data already in memory*. That
    means you can tweak any parameter and re-run instantly, without re-fetching
    from Strava.

    - The selectable date range is bounded by your fetched history
      (oldest session → today).

    ### Available analyses
    - **Personalized GAP Simulator** — builds personalized GAP (Gradient
      Adjusted Pace) curves from your history and compares them against the
      *Balanced Runner* and *Kilian Jornet* reference curves, including
      intensity-stratified panels. Every figure has a **Download data** button
      (PNG + CSV bundled in a ZIP).

    ### Notes
    - Your Strava token lives only in this browser session and is never stored
      on disk.
    - Closing the app clears the loaded data; you'll load it again next time.
    """
)
