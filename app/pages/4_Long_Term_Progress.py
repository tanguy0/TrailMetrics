"""Long-Term Progress — season-over-season trends across the whole history.

Unlike the other analyses, this page always uses *every* loaded activity (Run +
Trail Run). It surfaces the evolution of personal records per distance, annual
mileage and elevation gain, the average gradient, a gradient map of where time is
spent, and power-to-HR efficiency. "Seasons" are user-defined periods (calendar
years by default); the heavy per-activity work runs once and is cached, and the
controls below only re-aggregate those cached summaries against the seasons.
"""

from _helpers import add_repo_root_to_path, get_lang, inject_theme_css, t

add_repo_root_to_path()

from datetime import date, datetime, time as time_of_day
from typing import List, Tuple

import pandas as pd
import streamlit as st

from src.domain.models.activity import ActivityStream
from src.domain.progress import aggregates
from src.domain.progress import plotting
from src.domain.progress import seasons as seasons_mod
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


# --- UI helpers ------------------------------------------------------------

_GRANULARITIES = ["day", "week", "month", "quarter"]


def _scale_and_granularity(key: str, *, with_cumulative: bool = False):
    """Render the scale + granularity (+ cumulative) controls for a season plot."""
    cols = st.columns(3 if with_cumulative else 2)
    scale = cols[0].radio(
        t("ltp.scale_label"),
        options=["overlay", "continuous"],
        format_func=lambda c: t(f"ltp.scale.{c}"),
        horizontal=True,
        key=f"{key}_scale",
    )
    granularity = cols[1].radio(
        t("ltp.gran_label"),
        options=_GRANULARITIES,
        index=1,  # week
        format_func=lambda c: t(f"ltp.gran.{c}"),
        horizontal=True,
        key=f"{key}_gran",
    )
    cumulative = False
    if with_cumulative:
        cumulative = (
            cols[2].radio(
                t("ltp.view_label"),
                options=["cumulative", "periodic"],
                format_func=lambda c: t(f"ltp.view.{c}"),
                horizontal=True,
                key=f"{key}_view",
            )
            == "cumulative"
        )
    return scale, granularity, cumulative


def _parse_seasons(df: pd.DataFrame) -> Tuple[List[seasons_mod.Season], bool]:
    """Turn the season editor rows into :class:`Season`s; flag any invalid row."""
    parsed: List[seasons_mod.Season] = []
    invalid = False
    for _, row in df.iterrows():
        name, start, end = row.get("name"), row.get("start"), row.get("end")
        if pd.isna(start) or pd.isna(end) or not str(name or "").strip():
            if not (pd.isna(start) and pd.isna(end) and not str(name or "").strip()):
                invalid = True  # partially filled row
            continue
        start_d = pd.Timestamp(start).date()
        end_d = pd.Timestamp(end).date()
        if start_d > end_d:
            invalid = True
            continue
        parsed.append(seasons_mod.Season(name=str(name).strip(), start=start_d, end=end_d))
    return parsed, invalid


# --- 0. Season definition --------------------------------------------------
st.divider()
st.header(t("ltp.seasons.header"))
st.caption(t("ltp.seasons.help"))

_default_seasons = seasons_mod.calendar_year_seasons(activities)
if "ltp_seasons_df" not in st.session_state:
    st.session_state["ltp_seasons_df"] = pd.DataFrame(
        [{"name": s.name, "start": s.start, "end": s.end} for s in _default_seasons]
    )

edited_seasons = st.data_editor(
    st.session_state["ltp_seasons_df"],
    num_rows="dynamic",
    hide_index=True,
    width="stretch",
    column_config={
        "name": st.column_config.TextColumn(t("ltp.seasons.col.name")),
        "start": st.column_config.DateColumn(t("ltp.seasons.col.start")),
        "end": st.column_config.DateColumn(t("ltp.seasons.col.end")),
    },
    key="ltp_seasons_editor",
)

seasons, _seasons_invalid = _parse_seasons(edited_seasons)
if _seasons_invalid:
    st.warning(t("ltp.seasons.invalid"))

_overlaps = seasons_mod.find_overlaps(seasons)
if _overlaps:
    st.warning(
        t("ltp.seasons.overlap").format(
            pairs=", ".join(f"{a} ⇄ {b}" for a, b in _overlaps)
        )
    )

if not seasons:
    seasons = _default_seasons
    st.info(t("ltp.seasons.empty_fallback"))

unassigned = t("ltp.season.unassigned")
x_max_months = max(
    (seasons_mod.season_length_months(s) for s in seasons), default=None
)


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
last_activity_date = max(a.date for a in activities)

col_plot, col_table = st.columns([3, 1])
with col_plot:
    if any(progressions.values()):
        fig = plotting.plot_pr_progression(
            progressions,
            as_pace=(metric == "pace"),
            lang=lang,
            end_date=last_activity_date,
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


# --- 2. Evolution of mileage -----------------------------------------------
st.divider()
st.header(t("ltp.section.mileage"))

m_scale, m_gran, m_cum = _scale_and_granularity("ltp_mileage", with_cumulative=True)
mileage = aggregates.metric_series(
    activities, "distance_m", 0.001, seasons,
    mode=m_scale, granularity=m_gran, cumulative=m_cum, unassigned_label=unassigned,
)
col_plot, col_table = st.columns([3, 1])
with col_plot:
    fig = plotting.plot_season_curves(
        mileage,
        title=t("plot.ltp.mileage.title" if m_cum else "plot.ltp.mileage.periodic.title"),
        y_title=t("plot.ltp.mileage.y" if m_cum else "plot.ltp.mileage.periodic.y"),
        y_fmt=",.0f",
        hover_unit="km",
        overlay=(m_scale == "overlay"),
        step=m_cum,
        markers=not m_cum,
        x_max_months=x_max_months if m_scale == "overlay" else None,
        lang=lang,
    )
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        pd.DataFrame([
            {t("ltp.col.season"): name, t("ltp.mileage.col.total"): f"{total:,.0f} km"}
            for name, total in aggregates.season_totals(
                activities, "distance_m", 0.001, seasons
            )
        ]),
        hide_index=True,
        width="stretch",
    )


# --- 3. Evolution of elevation gain ----------------------------------------
st.divider()
st.header(t("ltp.section.elevation"))

e_scale, e_gran, e_cum = _scale_and_granularity("ltp_elevation", with_cumulative=True)
elevation = aggregates.metric_series(
    activities, "elevation_gain_m", 1.0, seasons,
    mode=e_scale, granularity=e_gran, cumulative=e_cum, unassigned_label=unassigned,
)
col_plot, col_table = st.columns([3, 1])
with col_plot:
    fig = plotting.plot_season_curves(
        elevation,
        title=t("plot.ltp.elevation.title" if e_cum else "plot.ltp.elevation.periodic.title"),
        y_title=t("plot.ltp.elevation.y" if e_cum else "plot.ltp.elevation.periodic.y"),
        y_fmt=",.0f",
        hover_unit="m",
        overlay=(e_scale == "overlay"),
        step=e_cum,
        markers=not e_cum,
        x_max_months=x_max_months if e_scale == "overlay" else None,
        lang=lang,
    )
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        pd.DataFrame([
            {t("ltp.col.season"): name, t("ltp.elevation.col.total"): f"{total:,.0f} m"}
            for name, total in aggregates.season_totals(
                activities, "elevation_gain_m", 1.0, seasons
            )
        ]),
        hide_index=True,
        width="stretch",
    )


# --- 4. Evolution of average gradient --------------------------------------
st.divider()
st.header(t("ltp.section.gradient"))
st.caption(t("ltp.section.gradient.help"))

g_scale, g_gran, _ = _scale_and_granularity("ltp_gradient")
gradient = aggregates.gradient_series(
    activities, seasons, mode=g_scale, granularity=g_gran, unassigned_label=unassigned,
)
col_plot, col_table = st.columns([3, 1])
with col_plot:
    fig = plotting.plot_season_curves(
        gradient,
        title=t("plot.ltp.gradient.title"),
        y_title=t("plot.ltp.gradient.y"),
        y_fmt=".1f",
        hover_unit="%",
        overlay=(g_scale == "overlay"),
        markers=True,
        x_max_months=x_max_months if g_scale == "overlay" else None,
        lang=lang,
    )
    st.plotly_chart(fig, use_container_width=True)
with col_table:
    st.dataframe(
        pd.DataFrame([
            {t("ltp.col.season"): name, t("ltp.gradient.col.avg"): f"{avg:.1f} %"}
            for name, avg in aggregates.season_gradient_averages(activities, seasons)
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
        format_func=lambda c: t(f"ltp.gran.{c}"),
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
    activities, from_date=from_dt, to_date_bound=to_dt, granularity=map_bin
)
if gmap.x:
    st.plotly_chart(plotting.plot_gradient_map(gmap, lang=lang), use_container_width=True)
else:
    st.info(t("ltp.no_data"))


# --- 6. Evolution of power-to-HR -------------------------------------------
st.divider()
st.header(t("ltp.section.power_hr"))
st.caption(t("ltp.section.power_hr.help"))

ph_dates = sorted(a.date.date() for a in activities if a.power_to_hr is not None)
if not ph_dates:
    if mass_kg is None:
        st.info(t("races.weight_needed"))
    else:
        st.info(t("ltp.no_data"))
else:
    ph_min, ph_max = ph_dates[0], ph_dates[-1]
    c_range, c_gran = st.columns([2, 1])
    with c_range:
        ph_range = st.date_input(
            t("ltp.power_hr.range_label"),
            value=(ph_min, ph_max),
            min_value=ph_min,
            max_value=ph_max,
            key="ltp_ph_range",
        )
    with c_gran:
        ph_gran = st.radio(
            t("ltp.gran_label"),
            options=_GRANULARITIES,
            index=1,  # week
            format_func=lambda c: t(f"ltp.gran.{c}"),
            horizontal=True,
            key="ltp_ph_gran",
        )

    # Smoothing of the final binned curve (windows in points), mirroring the
    # race comparator's filter controls.
    with st.expander(t("ltp.power_hr.smoothing"), expanded=False):
        st.caption(t("ltp.power_hr.smoothing.help"))
        f0, f1, f2, f3 = st.columns([1.1, 1, 1.2, 1])
        roll_on = f0.checkbox(t("ltp.power_hr.filter.rolling"), key="ltp_ph_roll")
        roll_w = f1.number_input(
            t("ltp.power_hr.filter.window_pts"),
            min_value=2, step=1, value=3,
            key="ltp_ph_rollw", disabled=not roll_on,
        )
        sav_on = f2.checkbox(t("ltp.power_hr.filter.savgol"), key="ltp_ph_sav")
        sav_w = f3.number_input(
            t("ltp.power_hr.filter.window_pts"),
            min_value=5, step=2, value=7,
            key="ltp_ph_savw", disabled=not sav_on,
        )

    # st.date_input returns a single date until both ends are picked.
    if isinstance(ph_range, (tuple, list)) and len(ph_range) == 2:
        ph_from = datetime.combine(ph_range[0], time_of_day.min)
        ph_to = datetime.combine(ph_range[1], time_of_day.max)
    else:
        ph_from = datetime.combine(ph_min, time_of_day.min)
        ph_to = datetime.combine(ph_max, time_of_day.max)

    power_hr = aggregates.power_hr_series(
        activities,
        seasons,
        granularity=ph_gran,
        from_date=ph_from,
        to_date_bound=ph_to,
        rolling_window=int(roll_w) if roll_on else None,
        savgol_window=int(sav_w) if sav_on else None,
        unassigned_label=unassigned,
    )
    if power_hr:
        fig = plotting.plot_season_curves(
            power_hr,
            title=t("plot.ltp.power_hr.title"),
            y_title=t("plot.ltp.power_hr.y"),
            y_fmt=".2f",
            hover_unit="W/bpm",
            overlay=False,
            markers=True,
            lang=lang,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(t("ltp.no_data"))
