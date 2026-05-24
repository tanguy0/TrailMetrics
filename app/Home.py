"""TrailMetrics home page."""

from _helpers import add_repo_root_to_path

add_repo_root_to_path()

import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

st.title("TrailMetrics")
st.subheader("Simulate and analyze your running data")

st.markdown(
    """
    **TrailMetrics** is a personal-use lab for running-data simulations. Pick an
    analysis from the sidebar to get started. Each analysis is self-contained,
    has its own parameters, and lets you download the resulting graphs.

    ### Available analyses

    - **Personalized GAP Simulator** &mdash; build personalized GAP (Gradient
      Adjusted Pace) curves from your Strava history and compare them against
      reference curves.

    More analyses will be added here as they ship.

    ---

    ### How it works

    1. Authorize the app to read your Strava activities (one-shot OAuth in the
       sidebar of each analysis).
    2. Choose the parameters for the analysis.
    3. Run the simulation, inspect the graphs, and download what you need.
    """
)

st.info("Open an analysis from the **left sidebar** to begin.")
