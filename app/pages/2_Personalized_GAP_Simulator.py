"""Personalized GAP simulator — runs on the data loaded from the Home page.

All inputs here are filters/parameters applied to the already-fetched history,
so the simulation re-runs without ever hitting Strava again.

The simulator can compare several **time scales** (each a named date range) side
by side: define up to five and every figure overlays them, coloring each scale
distinctly. The user also picks which personalized models to fit (Efficiency /
Auto-Learning — at least one) and which reference curves to overlay.

All user-facing strings come from src/translations.py via the ``t()`` helper.
"""

from _helpers import (
    add_repo_root_to_path,
    get_lang,
    inject_theme_css,
    render_figure_with_download,
    render_run_loader,
    t,
)

add_repo_root_to_path()

from datetime import date, datetime, time

import streamlit as st

from src.domain.gap import theme
from src.domain.gap.efficiency_model import EfficiencyGapModel
from src.domain.gap.plotting import plot_gap_curves
from src.domain.gap.reference_curves import balanced_runner, kilian_jornet
from src.usecases.simulate_personalized_gap_model import (
    SimulatePersonalizedGapModel,
    SimulatePersonalizedGapModelInput,
)

MAX_TIME_SCALES = 5

# Line styles encode the model within a single figure; color encodes the time
# scale (see theme.TIME_SCALE_CYCLE). Reference curves keep their own dashed look.
EFFICIENCY_LINESTYLE = "-"
AUTO_LEARNING_LINESTYLE = "-."

st.set_page_config(page_title=t("page.gap.title"), layout="wide")
inject_theme_css()
st.title(t("page.gap.title"))

# --- Gate: data must be loaded on the Home page first ----------------------
if "athlete_streams" not in st.session_state:
    st.warning(t("gate.no_data"))
    st.stop()

streams = st.session_state["athlete_streams"]
runner_name = st.session_state.get("runner_name", t("common.you"))

st.markdown(t("gap.intro"))

# Date bounds come from what was actually fetched.
oldest = st.session_state.get("fetch_oldest_date")
newest = st.session_state.get("fetch_newest_date")
min_date = oldest.date() if isinstance(oldest, datetime) else date(2010, 1, 1)
max_date = newest.date() if isinstance(newest, datetime) else date.today()


# --- Time-scale list management --------------------------------------------
# We keep a list of stable integer ids in session_state and key every row widget
# by id, so adding/removing rows never carries stale values between positions.
st.session_state.setdefault("gap_ts_counter", 0)
st.session_state.setdefault("gap_ts_ids", None)
if st.session_state["gap_ts_ids"] is None:
    st.session_state["gap_ts_counter"] += 1
    st.session_state["gap_ts_ids"] = [st.session_state["gap_ts_counter"]]


def _add_time_scale() -> None:
    if len(st.session_state["gap_ts_ids"]) < MAX_TIME_SCALES:
        st.session_state["gap_ts_counter"] += 1
        st.session_state["gap_ts_ids"].append(st.session_state["gap_ts_counter"])


def _remove_time_scale(tid: int) -> None:
    st.session_state["gap_ts_ids"] = [
        x for x in st.session_state["gap_ts_ids"] if x != tid
    ]


def _unique_labels(scales):
    """Disambiguate duplicate time-scale names so legend keys never collide."""
    seen, labels = {}, []
    for sc in scales:
        base = sc["name"]
        if base in seen:
            seen[base] += 1
            labels.append(f"{base} ({seen[base]})")
        else:
            seen[base] = 0
            labels.append(base)
    return labels


with st.sidebar:
    st.header(t("gap.filters.header"))
    sport_types = st.multiselect(
        t("gap.session_types"), ["TrailRun", "Run"], default=["TrailRun"]
    )

    st.header(t("gap.timescales.header"))
    st.caption(t("gap.timescales.caption"))
    time_scales = []
    ids = st.session_state["gap_ts_ids"]
    for idx, tid in enumerate(ids):
        with st.container(border=True):
            c_name, c_rm = st.columns([5, 1])
            name = c_name.text_input(
                t("gap.timescales.name_label"),
                value=t("gap.timescales.name_default").format(n=idx + 1),
                key=f"ts_name_{tid}",
                placeholder=t("gap.timescales.name_placeholder"),
            )
            c_rm.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            c_rm.button(
                "✕",
                key=f"ts_rm_{tid}",
                help=t("gap.timescales.remove_help"),
                disabled=len(ids) <= 1,
                on_click=_remove_time_scale,
                args=(tid,),
            )
            c_from, c_to = st.columns(2)
            from_d = c_from.date_input(
                t("gap.timescales.from"),
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"ts_from_{tid}",
            )
            to_d = c_to.date_input(
                t("gap.timescales.to"),
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                key=f"ts_to_{tid}",
            )
        time_scales.append(
            {"name": name.strip(), "from_date": from_d, "to_date": to_d}
        )

    st.button(
        t("gap.timescales.add"),
        on_click=_add_time_scale,
        disabled=len(ids) >= MAX_TIME_SCALES,
        use_container_width=True,
    )

    st.header(t("gap.models.header"))
    st.caption(t("gap.models.caption"))
    show_efficiency = st.checkbox(t("gap.models.efficiency"), value=True)
    show_auto_learning = st.checkbox(t("gap.models.auto"), value=True)

    st.header(t("gap.refs.header"))
    st.caption(t("gap.refs.caption"))
    show_strava = st.checkbox(t("gap.refs.balanced"), value=True)
    show_kilian = st.checkbox(t("gap.refs.kilian"), value=True)

    st.header(t("gap.display.header"))
    show_std = st.checkbox(
        t("gap.display.show_std"),
        value=True,
        help=t("gap.display.show_std_help"),
    )

    st.header(t("gap.params.header"))
    split_min_time = st.number_input(t("gap.params.split_min_time"), min_value=1, value=10)
    hr_tolerance = st.number_input(t("gap.params.hr_tol"), min_value=1, value=3)
    efficiency_min_samples = st.number_input(
        t("gap.params.eff_min_samples"), min_value=10, value=250
    )
    efficiency_subset_min_samples = st.number_input(
        t("gap.params.eff_subset_min_samples"),
        min_value=5,
        value=50,
        help=t("gap.params.eff_subset_help"),
    )
    xgb_bin_width = st.number_input(t("gap.params.bin_width"), min_value=1, value=20)

    st.header(t("gap.intensity.header"))
    low_low = st.number_input(t("gap.intensity.low_min"), value=120)
    low_high = st.number_input(t("gap.intensity.low_max"), value=150)
    high_low = st.number_input(t("gap.intensity.high_min"), value=160)
    high_high = st.number_input(t("gap.intensity.high_max"), value=190)

    # --- Validation ---------------------------------------------------------
    named_scales = [ts for ts in time_scales if ts["name"]]
    inverted = [ts for ts in named_scales if ts["from_date"] > ts["to_date"]]
    valid_scales = [ts for ts in named_scales if ts["from_date"] <= ts["to_date"]]
    a_model_selected = show_efficiency or show_auto_learning
    can_run = bool(sport_types) and bool(valid_scales) and a_model_selected

    if not sport_types:
        st.warning(t("gap.validate.no_sport"))
    if not named_scales:
        st.warning(t("gap.validate.no_scale"))
    if inverted:
        st.warning(t("gap.validate.inverted"))
    if not a_model_selected:
        st.warning(t("gap.validate.no_model"))

    run = st.button(t("gap.run_button"), type="primary", disabled=not can_run)


def _to_datetime_start(d: date) -> datetime:
    return datetime.combine(d, time.min)


def _to_datetime_end(d: date) -> datetime:
    return datetime.combine(d, time.max)


if run:
    lang = get_lang()
    loader = st.empty()
    render_run_loader(loader, t("gap.loader"), frac=None)

    usecase = SimulatePersonalizedGapModel()
    scale_labels = _unique_labels(valid_scales)

    # Fit every model once per time scale. Reference curves are independent of the
    # time scale, so we add them once at the page level rather than per scale.
    ordered = []  # [{label, color, result}], preserving sidebar order.
    for i, (ts, label) in enumerate(zip(valid_scales, scale_labels)):
        params = SimulatePersonalizedGapModelInput(
            streams=streams,
            sport_types=sport_types,
            from_date=_to_datetime_start(ts["from_date"]),
            to_date=_to_datetime_end(ts["to_date"]),
            split_min_time=float(split_min_time),
            hr_tolerance=float(hr_tolerance),
            efficiency_min_samples_per_bucket=int(efficiency_min_samples),
            xgboost_bin_width=float(xgb_bin_width),
            fit_efficiency=show_efficiency,
            fit_xgboost=show_auto_learning,
            include_reference_curves=False,
            verbose=False,
        )
        ordered.append(
            {
                "label": label,
                "color": theme.TIME_SCALE_CYCLE[i % len(theme.TIME_SCALE_CYCLE)],
                "result": usecase.execute(params),
            }
        )

    loader.empty()

    # --- Main overlay: every model, every time scale ------------------------
    eff_label = t("gap.models.efficiency")
    auto_label = t("gap.models.auto")
    display_curves = {}
    for entry in ordered:
        result, color, label = entry["result"], entry["color"], entry["label"]
        eff = result.gap_curves.get("Efficiency Model")
        if eff is not None:
            eff.color = color
            eff.linestyle = EFFICIENCY_LINESTYLE
            display_curves[f"{label} – {eff_label}"] = eff
        auto = result.gap_curves.get("XGBoost Model")
        if auto is not None:
            auto.color = color
            auto.linestyle = AUTO_LEARNING_LINESTYLE
            display_curves[f"{label} – {auto_label}"] = auto

    if show_strava:
        display_curves[t("gap.refs.balanced")] = balanced_runner()
    if show_kilian:
        display_curves[t("gap.refs.kilian")] = kilian_jornet()

    splits_summary = ", ".join(
        t("gap.summary.item").format(
            label=e["label"], n=len(e["result"].dataset.speed)
        )
        for e in ordered
    )
    st.success(t("gap.summary").format(summary=splits_summary))

    st.subheader(t("gap.subheader.curves"))
    st.caption(t("gap.caption.main"))
    fig = plot_gap_curves(display_curves, show_std=show_std, lang=lang)
    render_figure_with_download(
        fig, display_curves, base_filename="gap_curves", key="dl-gap-curves"
    )

    # --- Intensity-stratified overlay: one column per selected model --------
    st.subheader(t("gap.subheader.intensity"))
    st.caption(t("gap.caption.intensity"))
    low_range = (float(low_low), float(low_high))
    high_range = (float(high_low), float(high_high))
    smoother = usecase.smoother
    low_word = t("gap.intensity.low")
    high_word = t("gap.intensity.high")

    def _intensity_curve(model_kind, result, hr_range):
        """Smoothed per-intensity GAP curve for one model on one time scale."""
        if model_kind == "efficiency":
            return smoother.smooth(
                EfficiencyGapModel(
                    min_samples_per_bucket=int(efficiency_subset_min_samples)
                )
                .fit_on_subset(result.dataset, heartrate_range=hr_range)
                .gap_curve()
            )
        return smoother.smooth(
            result.xgboost_model.gap_curve(
                heartrate_range=hr_range, bin_width=float(xgb_bin_width)
            )
        )

    active_models = []
    if show_efficiency:
        active_models.append(
            ("efficiency", t("gap.col.efficiency_heading").format(runner=runner_name))
        )
    if show_auto_learning:
        active_models.append(
            ("auto", t("gap.col.auto_heading").format(runner=runner_name))
        )

    for col, (model_kind, heading) in zip(st.columns(len(active_models)), active_models):
        with col:
            st.markdown(f"**{heading}**")
            curves = {}
            for entry in ordered:
                label, color, result = entry["label"], entry["color"], entry["result"]
                for word, hr_range, linestyle in (
                    (low_word, low_range, "-"),
                    (high_word, high_range, "--"),
                ):
                    try:
                        curve = _intensity_curve(model_kind, result, hr_range)
                        curve.color = color
                        curve.linestyle = linestyle
                        curves[f"{label} – {word}"] = curve
                    except Exception as e:
                        st.info(
                            t("gap.intensity.unavailable").format(
                                label=label, intensity=word.lower(), error=e
                            )
                        )
            if curves:
                fig_intensity = plot_gap_curves(curves, show_std=show_std, lang=lang)
                render_figure_with_download(
                    fig_intensity,
                    curves,
                    base_filename=f"{model_kind}_by_intensity",
                    key=f"dl-{model_kind}-intensity",
                )
else:
    st.info(t("gap.run_hint"))
