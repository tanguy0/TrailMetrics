"""Long-Term Progress — season-over-season trends across the whole history.

Unlike the other analyses, this page always uses *every* loaded activity (Run +
Trail Run). It surfaces five views: the evolution of personal records per
distance, cumulative annual mileage and elevation gain, the average gradient per
season, and a gradient map of where time is spent. The heavy per-activity work
(best efforts, gradient classification) runs once and is cached in the session;
the controls below only re-aggregate those cached summaries.
"""

from _helpers import add_repo_root_to_path, get_lang, inject_theme_css, t

add_repo_root_to_path()

from datetime import datetime, time as time_of_day
from typing import List

import pandas as pd
import streamlit as st

from src.domain.models.activity import ActivityStream
from src.domain.progress import aggregates
from src.domain.progress import plotting
from src.domain.progress.models import PR_DISTANCES
from src.usecases.analyze_long_term_progress import (
    AnalyzeLongTermProgress,
    AnalyzeLongTermProgressInput,
)

st.set_page_config(page_title=t("page.ltp.title"), layout="wide")
inject_theme_css()
st.title(t("page.ltp.title"))

# --- Gate: data must be loaded on the Home page first ----------------------
if "athlete_streams" not in st.session_state:
    st.warning(t("gate.no_data"))
    st.stop()

streams: List[ActivityStream] = st.session_state["athlete_streams"]
lang = get_lang()

st.markdown(t("ltp.intro"))


# --- Heavy compute, cached per loaded dataset ------------------------------

def _activity_progress(streams: List[ActivityStream], mass_kg):
    """Compute (and cache) the per-activity summaries for the current history.

    The weight is part of the cache key: changing it on the Home page must
    recompute the (weight-dependent) power-to-HR metric.
    """
    signature = (len(streams), tuple(s.activity_id for s in streams), mass_kg)
    cached = st.session_state.get("ltp_cache")
    if cached and cached[0] == signature:
        return cached[1]
    with st.spinner(t("ltp.computing")):
        result = AnalyzeLongTermProgress().execute(
            AnalyzeLongTermProgressInput(streams=streams, mass_kg=mass_kg)
        )
    st.session_state["ltp_cache"] = (signature, result.activities)
    return result.activities


mass_kg = st.session_state.get("runner_weight_kg")
activities = _activity_progress(streams, mass_kg)
if not activities:
    st.info(t("ltp.no_data"))
    st.stop()


# --- Formatting helpers ----------------------------------------------------

def _fmt_hms(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _fmt_pace(seconds_per_km: float) -> str:
    minutes, secs = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{secs:02d}/km"


# --- 1. Evolution of personal records --------------------------------------
st.divider()
st.header(t("ltp.section.records"))
st.caption(t("ltp.section.records.help"))

metric = st.radio(
    t("ltp.records.metric_label"),
    options=["pace", "time"],
    horizontal=True,
    format_func=lambda c: t(f"ltp.records.metric.{c}"),
)

progressions = aggregates.pr_progressions(activities)
records = aggregates.current_records(activities)

col_plot, col_table = st.columns([3, 1])
with col_plot:
    if any(progressions.values()):
        fig = plotting.plot_pr_progression(
            progressions, as_pace=(metric == "pace"), lang=lang
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("ltp.records.none"))
with col_table:
    meters_by_label = {label: m for label, m in PR_DISTANCES}
    rows = []
    for label, _ in PR_DISTANCES:
        rec = records.get(label)
        if rec is None:
            rows.append({
                t("ltp.records.col.distance"): label,
                t("ltp.records.col.record"): "—",
                t("ltp.records.col.pace"): "—",
                t("ltp.records.col.date"): "—",
            })
        else:
            rec_date, rec_time = rec
            pace = rec_time / (meters_by_label[label] / 1000.0)
            rows.append({
                t("ltp.records.col.distance"): label,
                t("ltp.records.col.record"): _fmt_hms(rec_time),
                t("ltp.records.col.pace"): _fmt_pace(pace),
                t("ltp.records.col.date"): rec_date.strftime("%Y-%m-%d"),
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


# --- 2. Evolution of annual mileage ----------------------------------------
st.divider()
st.header(t("ltp.section.mileage"))

mileage = aggregates.annual_cumulative(activities, "distance_m", scale=0.001)
col_plot, col_table = st.columns([3, 1])
with col_plot:
    fig = plotting.plot_annual_cumulative(
        mileage,
        title=t("plot.ltp.mileage.title"),
        y_title=t("plot.ltp.mileage.y"),
        unit="km",
        lang=lang,
    )
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        pd.DataFrame([
            {
                t("ltp.col.season"): str(cy.year),
                t("ltp.mileage.col.total"): f"{cy.total:,.0f} km",
            }
            for cy in mileage
        ]),
        hide_index=True,
        width="stretch",
    )


# --- 3. Evolution of annual elevation gain ---------------------------------
st.divider()
st.header(t("ltp.section.elevation"))

elevation = aggregates.annual_cumulative(activities, "elevation_gain_m", scale=1.0)
col_plot, col_table = st.columns([3, 1])
with col_plot:
    fig = plotting.plot_annual_cumulative(
        elevation,
        title=t("plot.ltp.elevation.title"),
        y_title=t("plot.ltp.elevation.y"),
        unit="m",
        lang=lang,
    )
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        pd.DataFrame([
            {
                t("ltp.col.season"): str(cy.year),
                t("ltp.elevation.col.total"): f"{cy.total:,.0f} m",
            }
            for cy in elevation
        ]),
        hide_index=True,
        width="stretch",
    )


# --- 4. Evolution of average gradient per season ---------------------------
st.divider()
st.header(t("ltp.section.gradient"))
st.caption(t("ltp.section.gradient.help"))

grad_bin = st.radio(
    t("ltp.bin_label"),
    options=["week", "month"],
    horizontal=True,
    format_func=lambda c: t(f"ltp.bin.{c}"),
    key="ltp_grad_bin",
)

gradient = aggregates.avg_gradient_series(activities, granularity=grad_bin)
col_plot, col_table = st.columns([3, 1])
with col_plot:
    fig = plotting.plot_avg_gradient(gradient, lang=lang)
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        pd.DataFrame([
            {
                t("ltp.col.season"): str(gy.year),
                t("ltp.gradient.col.avg"): f"{gy.season_avg:.1f} %",
            }
            for gy in gradient
        ]),
        hide_index=True,
        width="stretch",
    )


# --- 5. Gradient map -------------------------------------------------------
st.divider()
st.header(t("ltp.section.gradient_map"))
st.caption(t("ltp.gradient_map.help"))

all_dates = sorted(a.date.date() for a in activities)
min_date, max_date = all_dates[0], all_dates[-1]

ctrl_range, ctrl_bin = st.columns([2, 1])
with ctrl_range:
    date_range = st.date_input(
        t("ltp.gradient_map.range_label"),
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
with ctrl_bin:
    map_bin = st.radio(
        t("ltp.bin_label"),
        options=["week", "month"],
        horizontal=True,
        format_func=lambda c: t(f"ltp.bin.{c}"),
        key="ltp_map_bin",
    )

# st.date_input returns a single date until both ends are picked.
if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    from_dt = datetime.combine(date_range[0], time_of_day.min)
    to_dt = datetime.combine(date_range[1], time_of_day.max)
else:
    from_dt = datetime.combine(min_date, time_of_day.min)
    to_dt = datetime.combine(max_date, time_of_day.max)

gmap = aggregates.gradient_map(
    activities, from_date=from_dt, to_date=to_dt, granularity=map_bin
)
if gmap.x:
    st.plotly_chart(plotting.plot_gradient_map(gmap, lang=lang), use_container_width=True)
else:
    st.info(t("ltp.no_data"))


# --- 6. Evolution of power-to-HR -------------------------------------------
st.divider()
st.header(t("ltp.section.power_hr"))
st.caption(t("ltp.section.power_hr.help"))

power_hr = aggregates.power_hr_weekly(activities)
if power_hr:
    st.plotly_chart(plotting.plot_power_hr(power_hr, lang=lang), use_container_width=True)
elif mass_kg is None:
    st.info(t("races.weight_needed"))
else:
    st.info(t("ltp.no_data"))
