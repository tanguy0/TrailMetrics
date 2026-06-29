"""Shared Streamlit helpers."""

import sys
from pathlib import Path
from typing import Dict

# Ensure `from src...` works regardless of how Streamlit launches this file.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.domain.gap import theme
from src.domain.models.gap import GapCurve
from src.translations import DEFAULT_LANG, translate


def add_repo_root_to_path() -> None:
    """No-op kept for backwards compatibility; the path is set at import time."""
    return


# --- i18n ------------------------------------------------------------------

def get_lang() -> str:
    """Active UI language code ('fr' / 'en'), set by the Home page selector."""
    return st.session_state.get("lang", DEFAULT_LANG)


def t(key: str) -> str:
    """Translate ``key`` into the active UI language (see src/translations.py)."""
    return translate(key, get_lang())


# --- Theme / UI ------------------------------------------------------------

_THEME_CSS = f"""
<style>
/* Headings get a warm amber accent underline for extra contrast. */
h1 {{
  color: {theme.TEXT};
  border-bottom: 4px solid {theme.SUNRISE};
  padding-bottom: 0.25rem;
  display: inline-block;
}}
h2, h3 {{ color: {theme.PRIMARY}; }}
hr {{ border-color: {theme.GRID} !important; }}

/* Punchier metrics. */
[data-testid="stMetricValue"] {{ color: {theme.PRIMARY}; font-weight: 800; }}
[data-testid="stMetricLabel"] {{ color: {theme.TERRACOTTA}; font-weight: 600; }}

/* --- Fun "runner on a trail" loader --- */
.tm-loader {{ margin: 0.6rem 0 1.2rem 0; }}
.tm-loader-head {{
  display: flex; justify-content: space-between; align-items: baseline;
  font-size: 1.1rem; font-weight: 700; color: {theme.TEXT}; margin-bottom: 8px;
}}
.tm-loader-pct {{ font-size: 1.7rem; font-weight: 900; color: {theme.TERRACOTTA}; }}
.tm-track {{
  position: relative; height: 38px; border-radius: 20px;
  background: {theme.FIGURE_FACE};
  box-shadow: inset 0 2px 6px rgba(0,0,0,0.14);
  border: 1px solid {theme.SPINE}; overflow: hidden;
}}
.tm-fill {{
  position: absolute; top: 0; left: 0; height: 100%; border-radius: 20px;
  background: linear-gradient(90deg, {theme.PRIMARY}, {theme.MOSS}, {theme.TERRACOTTA}, {theme.SUNRISE});
  transition: width .35s ease;
}}
.tm-fill--indef {{
  width: 100% !important;
  background: repeating-linear-gradient(
    45deg, {theme.PRIMARY} 0 14px, {theme.MOSS} 14px 28px, {theme.SUNRISE} 28px 42px);
  background-size: 60px 100%;
  animation: tm-stripe 0.9s linear infinite;
  opacity: 0.85;
}}
.tm-runner {{
  position: absolute; top: 3px; font-size: 28px; transition: left .35s ease;
  animation: tm-bob .55s ease-in-out infinite; filter: drop-shadow(0 2px 1px rgba(0,0,0,.25));
}}
.tm-runner--indef {{ animation: tm-bob .55s ease-in-out infinite, tm-run 1.5s linear infinite; }}
.tm-goal {{ position: absolute; right: 8px; top: 5px; font-size: 26px; }}
@keyframes tm-bob {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-6px); }} }}
@keyframes tm-stripe {{ 0% {{ background-position: 0 0; }} 100% {{ background-position: 60px 0; }} }}
@keyframes tm-run {{ 0% {{ left: -7%; }} 100% {{ left: 100%; }} }}
</style>
"""


def inject_theme_css() -> None:
    """Inject the Trail/Earthy CSS (headings, metrics, loader). Call once per page."""
    st.markdown(_THEME_CSS, unsafe_allow_html=True)


def render_run_loader(placeholder, text: str, frac: float = None) -> None:
    """Render the fun runner loader into a placeholder (``st.empty()``).

    ``frac`` in [0, 1] shows a determinate trail; ``None`` shows an animated
    indeterminate trail (runner sprinting across) for unknown-length steps.
    """
    if frac is None:
        fill = '<div class="tm-fill tm-fill--indef"></div>'
        runner = '<div class="tm-runner tm-runner--indef">🏃</div>'
        pct_html = '<span class="tm-loader-pct">…</span>'
    else:
        pct = max(0, min(100, round(frac * 100)))
        fill = f'<div class="tm-fill" style="width:{pct}%"></div>'
        runner = f'<div class="tm-runner" style="left:calc({pct}% - 18px)">🏃</div>'
        pct_html = f'<span class="tm-loader-pct">{pct}%</span>'

    html = (
        '<div class="tm-loader">'
        f'<div class="tm-loader-head"><span>{text}</span>{pct_html}</div>'
        f'<div class="tm-track">{fill}{runner}<span class="tm-goal">🏔️</span></div>'
        "</div>"
    )
    placeholder.markdown(html, unsafe_allow_html=True)


def _curves_to_csv_bytes(curves: Dict[str, GapCurve]) -> bytes:
    """Long-format CSV: one row per (curve, bin_center) with mean/std/count."""
    rows = []
    for name, curve in curves.items():
        for center, mean, std, count in zip(
            curve.bin_centers, curve.means, curve.stds, curve.counts
        ):
            rows.append(
                {
                    "curve": name,
                    "bin_center_m_per_km": float(center),
                    "speed_adjuster_mean": float(mean),
                    "speed_adjuster_std": float(std),
                    "sample_count": int(count),
                }
            )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def render_figure_with_download(
    fig: go.Figure,
    curves: Dict[str, GapCurve],
    *,
    base_filename: str,
    key: str,
) -> None:
    """Render an interactive Plotly figure, then offer a CSV download of its data.

    The figure's own toolbar exports a PNG (camera icon); this adds a button for
    the underlying curve data as CSV.
    """
    st.plotly_chart(fig, use_container_width=True)

    st.download_button(
        label="Download data (CSV)",
        data=_curves_to_csv_bytes(curves),
        file_name=f"{base_filename}.csv",
        mime="text/csv",
        key=key,
    )
