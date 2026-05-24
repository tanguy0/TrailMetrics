"""Shared Streamlit helpers."""

import io
import sys
import zipfile
from pathlib import Path
from typing import Dict

# Ensure `from src...` works regardless of how Streamlit launches this file.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.domain.models.gap import GapCurve


def add_repo_root_to_path() -> None:
    """No-op kept for backwards compatibility; the path is set at import time."""
    return


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
    fig: plt.Figure,
    curves: Dict[str, GapCurve],
    *,
    base_filename: str,
    key: str,
) -> None:
    """Render a figure, then offer a single 'Download data' button that bundles PNG + CSV in a ZIP."""
    st.pyplot(fig)

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=150, bbox_inches="tight")
    png_buf.seek(0)

    csv_bytes = _curves_to_csv_bytes(curves)

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{base_filename}.png", png_buf.getvalue())
        zf.writestr(f"{base_filename}.csv", csv_bytes)
    zip_buf.seek(0)

    st.download_button(
        label="Download data",
        data=zip_buf,
        file_name=f"{base_filename}.zip",
        mime="application/zip",
        key=key,
    )
