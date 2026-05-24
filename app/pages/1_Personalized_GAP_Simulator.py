"""Personalized GAP simulator analysis page."""

from _helpers import add_repo_root_to_path, render_figure_with_download

add_repo_root_to_path()

import os
from datetime import date, datetime, time, timedelta

import streamlit as st
from stravalib import Client

from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.plotting import plot_gap_curves
from src.domain.gap.reference_curves import balanced_runner, kilian_jornet
from src.infrastructure.strava.strava_client import StravaClient
from src.usecases.simulate_personalized_gap_model import (
    SimulatePersonalizedGapModel,
    SimulatePersonalizedGapModelInput,
)

st.set_page_config(page_title="Personalized GAP Simulator", layout="wide")
st.title("Personalized GAP Simulator")

st.markdown("""
    Build personalized GAP (Gradient Adjusted Pace) curves from your Strava
    history and compare them against reference curves.
    """)


with st.sidebar:
    st.header("1. Runner")
    runner_name = st.text_input(
        "Your name", value=st.session_state.get("runner_name", "")
    )
    if runner_name:
        st.session_state["runner_name"] = runner_name

    st.header("2. Strava credentials")
    client_id = st.text_input(
        "STRAVA_CLIENT_ID",
        value=os.environ.get("STRAVA_CLIENT_ID", ""),
        help="Your Strava application client ID.",
    )
    client_secret = st.text_input(
        "STRAVA_CLIENT_SECRET",
        value=os.environ.get("STRAVA_CLIENT_SECRET", ""),
        type="password",
    )

    st.header("3. Authorize")
    if st.button("Generate authorization URL"):
        if not client_id:
            st.error("Set STRAVA_CLIENT_ID first.")
        else:
            tmp_client = Client()
            auth_url = tmp_client.authorization_url(
                client_id=int(client_id),
                redirect_uri="http://localhost:5000/authorization",
            )
            st.session_state["auth_url"] = auth_url

    if "auth_url" in st.session_state:
        st.markdown(f"[Open Strava authorization]({st.session_state['auth_url']})")
        st.caption(
            "After authorizing, copy the `code=...` value from the redirect URL."
        )

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

    st.header("4. Date range")
    default_to = date.today()
    default_from = default_to - timedelta(days=365)
    from_date_input = st.date_input("From date", value=default_from)
    to_date_input = st.date_input("To date", value=default_to)

    st.header("5. Simulation parameters")
    sport_types = st.multiselect(
        "Sport types", ["TrailRun", "Run"], default=["TrailRun"]
    )
    split_min_time = st.number_input("Split min time (seconds)", min_value=1, value=10)
    hr_tolerance = st.number_input("HR tolerance (bpm)", min_value=1, value=3)
    efficiency_min_samples = st.number_input(
        "Efficiency model: min samples per bucket", min_value=10, value=250
    )
    efficiency_subset_min_samples = st.number_input(
        "Efficiency model (per-intensity slice): min samples per bucket",
        min_value=5,
        value=50,
        help="Lower than the full-dataset value because each HR slice has fewer points.",
    )
    xgb_bin_width = st.number_input("Bin width (m/km)", min_value=1, value=20)

    st.header("6. Intensity ranges")
    low_low = st.number_input("Low intensity: min HR", value=120)
    low_high = st.number_input("Low intensity: max HR", value=150)
    high_low = st.number_input("High intensity: min HR", value=160)
    high_high = st.number_input("High intensity: max HR", value=190)

    run = st.button(
        "Run simulation",
        type="primary",
        disabled=("access_token" not in st.session_state) or not runner_name,
    )


def _to_datetime_start(d: date) -> datetime:
    return datetime.combine(d, time.min)


def _to_datetime_end(d: date) -> datetime:
    return datetime.combine(d, time.max)


if run:
    with st.spinner("Fetching activities and fitting models..."):
        stravalib_client = Client(access_token=st.session_state["access_token"])
        stream_source = StravaClient(stravalib_client)

        usecase = SimulatePersonalizedGapModel(stream_source=stream_source)
        params = SimulatePersonalizedGapModelInput(
            sport_types=sport_types,
            from_date=_to_datetime_start(from_date_input),
            to_date=_to_datetime_end(to_date_input),
            max_activities=1000,
            split_min_time=float(split_min_time),
            hr_tolerance=float(hr_tolerance),
            efficiency_min_samples_per_bucket=int(efficiency_min_samples),
            xgboost_bin_width=float(xgb_bin_width),
            include_reference_curves=True,
            verbose=False,
        )
        result = usecase.execute(params)

    display_curves = {}
    for original_name, curve in result.gap_curves.items():
        if original_name == "Efficiency Model":
            new_name = f"{runner_name} (Efficiency model)"
        elif original_name == "XGBoost Model":
            new_name = f"{runner_name} (Auto-Learning model)"
        else:
            new_name = original_name
        display_curves[new_name] = curve

    st.success(
        f"Simulation complete on {len(result.dataset.speed)} downsampled splits."
    )

    st.subheader("GAP curves")
    fig = plot_gap_curves(display_curves)
    render_figure_with_download(
        fig, display_curves, base_filename="gap_curves", key="dl-gap-curves"
    )

    st.subheader("Intensity-stratified GAP curves")
    low_range = (float(low_low), float(low_high))
    high_range = (float(high_low), float(high_high))

    col_eff, col_xgb = st.columns(2)

    smoother = usecase.smoother

    with col_eff:
        st.markdown(f"**{runner_name} (Efficiency model)**")
        eff_curves = {}
        try:
            eff_low = smoother.smooth(
                EfficiencyGapModel(
                    min_samples_per_bucket=int(efficiency_subset_min_samples)
                )
                .fit_on_subset(result.dataset, heartrate_range=low_range)
                .gap_curve()
            )
            eff_low.color = "lime"
            eff_curves["Low Intensity"] = eff_low
        except Exception as e:
            st.info(f"Low intensity efficiency curve unavailable: {e}")
        try:
            eff_high = smoother.smooth(
                EfficiencyGapModel(
                    min_samples_per_bucket=int(efficiency_subset_min_samples)
                )
                .fit_on_subset(result.dataset, heartrate_range=high_range)
                .gap_curve()
            )
            eff_high.color = "green"
            eff_curves["High Intensity"] = eff_high
        except Exception as e:
            st.info(f"High intensity efficiency curve unavailable: {e}")
        if eff_curves:
            fig_eff = plot_gap_curves(eff_curves)
            render_figure_with_download(
                fig_eff,
                eff_curves,
                base_filename="efficiency_by_intensity",
                key="dl-eff-intensity",
            )

    with col_xgb:
        st.markdown(f"**{runner_name} (Auto-Learning model)**")
        xgb_curves = {}
        try:
            xgb_low = smoother.smooth(
                result.xgboost_model.gap_curve(
                    heartrate_range=low_range, bin_width=float(xgb_bin_width)
                )
            )
            xgb_low.color = "lime"
            xgb_curves["Low Intensity"] = xgb_low
        except Exception as e:
            st.info(f"Low intensity auto-learning curve unavailable: {e}")
        try:
            xgb_high = smoother.smooth(
                result.xgboost_model.gap_curve(
                    heartrate_range=high_range, bin_width=float(xgb_bin_width)
                )
            )
            xgb_high.color = "green"
            xgb_curves["High Intensity"] = xgb_high
        except Exception as e:
            st.info(f"High intensity auto-learning curve unavailable: {e}")
        if xgb_curves:
            fig_xgb = plot_gap_curves(xgb_curves)
            render_figure_with_download(
                fig_xgb,
                xgb_curves,
                base_filename="auto_learning_by_intensity",
                key="dl-xgb-intensity",
            )
else:
    st.info(
        "Enter your name, authorize with Strava in the sidebar, then click 'Run simulation'."
    )
