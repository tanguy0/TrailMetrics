"""Personalized GAP simulator — runs on the data loaded from the Home page.

All inputs here are filters/parameters applied to the already-fetched history,
so the simulation re-runs without ever hitting Strava again.
"""

from _helpers import (
    add_repo_root_to_path,
    inject_theme_css,
    render_figure_with_download,
    render_run_loader,
)

add_repo_root_to_path()

from datetime import date, datetime, time

import streamlit as st

from src.domain.gap import theme
from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.plotting import plot_gap_curves
from src.usecases.simulate_personalized_gap_model import (
    SimulatePersonalizedGapModel,
    SimulatePersonalizedGapModelInput,
)

st.set_page_config(page_title="Personalized GAP Simulator", layout="wide")
inject_theme_css()
st.title("Personalized GAP Simulator")

# --- Gate: data must be loaded on the Home page first ----------------------
if "athlete_streams" not in st.session_state:
    st.warning(
        "No data loaded yet. Go to the **Home** page, connect Strava and click "
        "**Load my data** to unlock this analysis."
    )
    st.stop()

streams = st.session_state["athlete_streams"]
runner_name = st.session_state.get("runner_name", "You")

st.markdown(
    """
    Build personalized GAP (Gradient Adjusted Pace) curves from your loaded
    history and compare them against reference curves. Adjust the filters and
    parameters in the sidebar, then re-run — no re-fetching required.
    """
)

# Date bounds come from what was actually fetched.
oldest = st.session_state.get("fetch_oldest_date")
newest = st.session_state.get("fetch_newest_date")
min_date = oldest.date() if isinstance(oldest, datetime) else date(2010, 1, 1)
max_date = newest.date() if isinstance(newest, datetime) else date.today()


with st.sidebar:
    st.header("Filters")
    sport_types = st.multiselect(
        "Session types", ["TrailRun", "Run"], default=["TrailRun"]
    )
    from_date_input = st.date_input(
        "From date", value=min_date, min_value=min_date, max_value=max_date
    )
    to_date_input = st.date_input(
        "To date", value=max_date, min_value=min_date, max_value=max_date
    )

    st.header("Simulation parameters")
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

    st.header("Intensity ranges")
    low_low = st.number_input("Low intensity: min HR", value=120)
    low_high = st.number_input("Low intensity: max HR", value=150)
    high_low = st.number_input("High intensity: min HR", value=160)
    high_high = st.number_input("High intensity: max HR", value=190)

    run = st.button("Run simulation", type="primary", disabled=not sport_types)


def _to_datetime_start(d: date) -> datetime:
    return datetime.combine(d, time.min)


def _to_datetime_end(d: date) -> datetime:
    return datetime.combine(d, time.max)


if run:
    loader = st.empty()
    render_run_loader(loader, "Filtering activities and fitting models…", frac=None)

    usecase = SimulatePersonalizedGapModel()
    params = SimulatePersonalizedGapModelInput(
        streams=streams,
        sport_types=sport_types,
        from_date=_to_datetime_start(from_date_input),
        to_date=_to_datetime_end(to_date_input),
        split_min_time=float(split_min_time),
        hr_tolerance=float(hr_tolerance),
        efficiency_min_samples_per_bucket=int(efficiency_min_samples),
        xgboost_bin_width=float(xgb_bin_width),
        include_reference_curves=True,
        verbose=False,
    )
    result = usecase.execute(params)
    loader.empty()

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
            eff_low.color = theme.LOW_INTENSITY
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
            eff_high.color = theme.HIGH_INTENSITY
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
            xgb_low.color = theme.LOW_INTENSITY
            xgb_curves["Low Intensity"] = xgb_low
        except Exception as e:
            st.info(f"Low intensity auto-learning curve unavailable: {e}")
        try:
            xgb_high = smoother.smooth(
                result.xgboost_model.gap_curve(
                    heartrate_range=high_range, bin_width=float(xgb_bin_width)
                )
            )
            xgb_high.color = theme.HIGH_INTENSITY
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
    st.info("Set your filters and parameters in the sidebar, then click 'Run simulation'.")
