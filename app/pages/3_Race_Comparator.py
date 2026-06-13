"""Race Comparator — pick up to 4 workouts from the loaded history and compare.

A workout selector (searchable by date, showing duration / distance / sport
type) feeds a cross-race analysis: a summary stats table plus evolution plots
(GAP pace, power, HR, power-to-HR) whose x-axis toggles between time and
distance. Everything runs on the already-fetched history — no re-fetching.
"""

from _helpers import add_repo_root_to_path, inject_theme_css

add_repo_root_to_path()

from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd
import streamlit as st

from src.domain.gap import theme
from src.domain.models.activity import ActivityStream
from src.domain.races.plotting import available_metric_keys, plot_metric_comparison
from src.domain.races.smoothing import (
    FilterConfig,
    SmoothingParams,
    default_smoothing_params,
)
from src.usecases.compare_races import CompareRaces, CompareRacesInput

MAX_RACES = 4

st.set_page_config(page_title="Race Comparator", layout="wide")
inject_theme_css()
st.title("Race Comparator")

# --- Gate: data must be loaded on the Home page first ----------------------
if "athlete_streams" not in st.session_state:
    st.warning(
        "No data loaded yet. Go to the **Home** page, connect Strava and click "
        "**Load my data** to unlock this analysis."
    )
    st.stop()

streams: List[ActivityStream] = st.session_state["athlete_streams"]

st.markdown(
    f"""
    Compare any number of races side by side — from a single workout up to
    **{MAX_RACES}** at once. Start by picking your workouts below. Use the date
    search to narrow things down; each option shows its duration, distance and
    sport type so you can tell similar sessions apart.
    """
)


# --- Workout summaries -----------------------------------------------------

def _duration_seconds(stream: ActivityStream) -> float:
    """Elapsed time of the activity, in seconds (Strava time stream is seconds)."""
    if stream.time is None or len(stream.time) == 0:
        return 0.0
    return float(stream.time[-1] - stream.time[0])


def _distance_meters(stream: ActivityStream) -> float:
    """Total distance covered, in meters (Strava distance stream is cumulative m)."""
    if stream.distance is None or len(stream.distance) == 0:
        return 0.0
    return float(stream.distance[-1] - stream.distance[0])


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _format_distance(meters: float) -> str:
    return f"{meters / 1000:.2f} km"


def _sport_name(sport_type) -> str:
    """Coerce a sport type to a plain string.

    Strava streams carry stravalib's ``RelaxedSportType`` (a pydantic model with
    a ``.root`` value), which Arrow/Streamlit can't serialize — unwrap it.
    """
    return str(getattr(sport_type, "root", sport_type))


def _summarize(stream: ActivityStream) -> dict:
    start = stream.start_date
    return {
        "activity_id": stream.activity_id,
        "date": start.date() if isinstance(start, datetime) else None,
        "sport_type": _sport_name(stream.sport_type),
        "duration_s": _duration_seconds(stream),
        "distance_m": _distance_meters(stream),
    }


summaries = [_summarize(s) for s in streams]
# Most recent first so the latest races surface at the top of the picker.
summaries.sort(key=lambda s: (s["date"] is not None, s["date"]), reverse=True)
summaries_by_id = {s["activity_id"]: s for s in summaries}


def _option_label(activity_id: int) -> str:
    s = summaries_by_id[activity_id]
    date_str = s["date"].strftime("%Y-%m-%d") if s["date"] else "unknown date"
    return (
        f"{date_str} · {s['sport_type']} · "
        f"{_format_distance(s['distance_m'])} · {_format_duration(s['duration_s'])}"
    )


# --- Selector --------------------------------------------------------------
st.subheader("Select workouts")

available_dates = sorted({s["date"] for s in summaries if s["date"] is not None})

search_by_date = st.checkbox(
    "Search by date",
    help="Narrow the workout list to a single day.",
)

candidate_ids = [s["activity_id"] for s in summaries]

if search_by_date and available_dates:
    selected_date = st.date_input(
        "Workout date",
        value=available_dates[-1],
        min_value=available_dates[0],
        max_value=available_dates[-1],
    )
    matches = [s for s in summaries if s["date"] == selected_date]
    candidate_ids = [s["activity_id"] for s in matches]

    if matches:
        st.caption(f"{len(matches)} workout(s) on {selected_date.strftime('%Y-%m-%d')}:")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Sport": m["sport_type"],
                        "Distance": _format_distance(m["distance_m"]),
                        "Duration": _format_duration(m["duration_s"]),
                    }
                    for m in matches
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No workouts on that date.")
elif search_by_date:
    st.info("No dated workouts in the loaded history.")

selected_ids = st.multiselect(
    f"Workouts to compare (up to {MAX_RACES})",
    options=candidate_ids,
    default=[
        sid
        for sid in st.session_state.get("comparator_selected_ids", [])
        if sid in candidate_ids
    ],
    format_func=_option_label,
    max_selections=MAX_RACES,
    help=f"Pick between 1 and {MAX_RACES} workouts.",
)

streams_by_id = {s.activity_id: s for s in streams}
# Keep stream order aligned with the user's selection order (and with labels).
st.session_state["comparator_selected_ids"] = selected_ids
st.session_state["comparator_selected_streams"] = [
    streams_by_id[sid] for sid in selected_ids
]

# --- Recap of the current selection ----------------------------------------
if selected_ids:
    st.subheader(f"Selected ({len(selected_ids)}/{MAX_RACES})")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Date": (
                        summaries_by_id[sid]["date"].strftime("%Y-%m-%d")
                        if summaries_by_id[sid]["date"]
                        else "—"
                    ),
                    "Sport": summaries_by_id[sid]["sport_type"],
                    "Distance": _format_distance(summaries_by_id[sid]["distance_m"]),
                    "Duration": _format_duration(summaries_by_id[sid]["duration_s"]),
                }
                for sid in selected_ids
            ]
        ),
        hide_index=True,
        width="stretch",
    )
else:
    st.info("Pick at least one workout above to get started.")
    st.stop()

# --- Smoothing settings ----------------------------------------------------
_SMOOTH_DEFAULTS = default_smoothing_params()


def _filter_controls(label: str, key: str, default: FilterConfig) -> FilterConfig:
    """Render rolling-average + Savitzky–Golay controls for one signal."""
    c0, c1, c2, c3, c4 = st.columns([1.4, 1, 1.1, 1.2, 1.1])
    c0.markdown(f"**{label}**")
    roll_on = c1.checkbox(
        "Rolling avg", value=default.rolling_window_s is not None, key=f"sm_{key}_roll"
    )
    roll_w = c2.number_input(
        "Window (s)", min_value=1.0, step=1.0,
        value=float(default.rolling_window_s or 15.0),
        key=f"sm_{key}_rollw", disabled=not roll_on,
    )
    sav_on = c3.checkbox(
        "Savitzky–Golay", value=default.savgol_window_m is not None, key=f"sm_{key}_sav"
    )
    sav_w = c4.number_input(
        "Window (m)", min_value=10.0, step=10.0,
        value=float(default.savgol_window_m or 200.0),
        key=f"sm_{key}_savw", disabled=not sav_on,
    )
    return FilterConfig(
        rolling_window_s=roll_w if roll_on else None,
        savgol_window_m=sav_w if sav_on else None,
    )


with st.expander("⚙️ Smoothing settings", expanded=False):
    st.caption(
        "Each signal can pass through a time-domain rolling average and/or a "
        "distance-domain Savitzky–Golay filter (applied in that order). "
        "Altitude smoothing drives the gradient and elevation gain."
    )
    smoothing_params = SmoothingParams(
        pace=_filter_controls("Pace / GAP", "pace", _SMOOTH_DEFAULTS.pace),
        altitude=_filter_controls("Altitude", "altitude", _SMOOTH_DEFAULTS.altitude),
        heartrate=_filter_controls("Heart rate", "hr", _SMOOTH_DEFAULTS.heartrate),
        power=_filter_controls("Power", "power", _SMOOTH_DEFAULTS.power),
    )

# --- Cross-race analysis ---------------------------------------------------
st.divider()
st.header("Analysis")

selected_streams = st.session_state["comparator_selected_streams"]
labels = [_option_label(sid) for sid in selected_ids]

usecase = CompareRaces()
result = usecase.execute(
    CompareRacesInput(
        streams=selected_streams,
        labels=labels,
        mass_kg=st.session_state.get("runner_weight_kg"),
        smoothing=smoothing_params,
    )
)

# 1. Summary stats table (best value per category highlighted) --------------
st.subheader("Summary stats")


def _format_pace(seconds_per_km: float) -> str:
    if seconds_per_km is None or not np.isfinite(seconds_per_km):
        return "—"
    minutes, secs = divmod(int(round(seconds_per_km)), 60)
    return f"{minutes}:{secs:02d} /km"


# "best" direction per column: True = highest is best, False = lowest is best.
_BEST_IS_MAX = {
    "Distance": True,
    "Elevation gain": True,
    "Time": False,
    "Avg pace": False,
    "Avg GAP pace": False,
    "Avg power": True,
}

stats = pd.DataFrame(
    [
        {
            "Distance": m.distance_km,
            "Elevation gain": m.elevation_gain_m,
            "Time": m.time_s,
            "Avg pace": m.avg_pace_s_per_km,
            "Avg GAP pace": m.avg_gap_pace_s_per_km,
            "Avg power": m.avg_power_w if m.avg_power_w is not None else np.nan,
        }
        for m in result.metrics
    ],
    index=[m.label for m in result.metrics],
)


def _highlight_best(column: pd.Series) -> List[str]:
    if column.name not in _BEST_IS_MAX or len(column) < 2 or column.isna().all():
        return [""] * len(column)
    target = column.max() if _BEST_IS_MAX[column.name] else column.min()
    return [
        f"background-color: {theme.MOSS}; color: white; font-weight: 700;"
        if v == target
        else ""
        for v in column
    ]


styler = (
    stats.style.format(
        {
            "Distance": "{:.2f} km",
            "Elevation gain": "{:.0f} m",
            "Time": lambda s: _format_duration(s),
            "Avg pace": _format_pace,
            "Avg GAP pace": _format_pace,
            "Avg power": lambda w: "—" if pd.isna(w) else f"{w:.0f} W",
        }
    ).apply(_highlight_best, axis=0)
)
st.dataframe(styler, width="stretch")
if stats["Avg power"].isna().all():
    st.caption(
        "Avg power is blank — set **your weight** on the Home page to enable "
        "power (and the power graphs below)."
    )

# 2-5. Evolution plots, with a shared time/distance x-axis toggle -----------
st.subheader("Evolution across the race")

col_x, col_gap = st.columns(2)
with col_x:
    x_axis_label = st.radio(
        "X axis",
        options=["Time", "Distance"],
        horizontal=True,
        help="Switch every graph below between elapsed time and distance covered.",
    )
with col_gap:
    gap_display = st.radio(
        "Show GAP as",
        options=["Pace", "Speed"],
        horizontal=True,
        help="Display the gradient-adjusted-pace graph as pace (min/km) or speed (km/h).",
    )
x_axis = "time" if x_axis_label == "Time" else "distance"
gap_as_speed = gap_display == "Speed"

# title shown above each metric's chart
_PLOT_TITLES = {
    "gap_pace": "Gradient-adjusted pace",
    "power": "Power",
    "heartrate": "Heart rate",
    "power_to_hr": "Power-to-HR",
}

plottable = available_metric_keys(result.series)
for metric_key in ("gap_pace", "power", "heartrate", "power_to_hr"):
    st.markdown(f"**{_PLOT_TITLES[metric_key]}**")
    if metric_key not in plottable:
        st.info("Set **your weight** on the Home page to enable this graph.")
        continue
    fig = plot_metric_comparison(
        result.series, metric_key=metric_key, x_axis=x_axis, gap_as_speed=gap_as_speed
    )
    st.pyplot(fig)
