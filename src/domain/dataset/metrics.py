"""Activity metric registry — the vocabulary of things you can plot.

Every activity-level plot (trend, distribution, scatter, table) takes a metric
key from here rather than hard-coding a column, so **adding one entry makes a new
quantity plottable in every one of them**, in every chart form, at every
granularity. This is where the app's analytical surface actually grows.

Two kinds of metric, and the distinction matters:

* **column** metrics aggregate one feature column (Σ distance, mean HR, min 10 km
  time);
* **ratio** metrics are Σ numerator ÷ Σ denominator × scale — average pace,
  average gradient, GAP pace. They must be computed *after* summing, per bin: a
  mean of per-activity paces silently over-weights short runs, and that bug is
  easy to write by hand and impossible to write here.

``value_kind`` carries presentation intent (a duration ticks as ``h:mm:ss``, a
pace additionally draws its axis reversed so faster sits higher) so plots don't
each re-derive axis formatting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.domain.dataset.features import band_column, best_column
from src.domain.progress.models import GRADIENT_BANDS, PR_DISTANCES

# Aggregations offered for column metrics.
AGGREGATIONS: Tuple[str, ...] = ("sum", "mean", "median", "max", "min", "count")

# Sentinel for "no second metric". A real value rather than an empty string so it
# survives a round-trip through a saved page spec and a <select> unambiguously.
NO_METRIC = "none"

# value_kind values.
NUMBER = "number"
DURATION = "duration"   # seconds, ticked as h:mm:ss
PACE = "pace"           # seconds per km, ticked as m:ss on a reversed axis
COUNT = "count"


@dataclass(frozen=True)
class ActivityMetric:
    """One measurable quantity over the activity feature table."""

    key: str
    label_key: str
    unit: str = ""
    # Column metric.
    column: Optional[str] = None
    # Ratio metric (Σ numerator ÷ Σ denominator × scale).
    numerator: Optional[str] = None
    denominator: Optional[str] = None
    scale: float = 1.0
    default_agg: str = "sum"
    # Which aggregations make sense; empty means the metric is fixed (ratios and
    # counts), so the form hides the aggregation control.
    allowed_aggs: Tuple[str, ...] = AGGREGATIONS
    value_kind: str = NUMBER
    decimals: int = 1
    # Drives best-value highlighting in comparison tables; None = no "best".
    higher_is_better: Optional[bool] = None
    # Metrics that only exist with per-second streams, so the UI can say why a
    # plot is empty instead of showing nothing.
    needs_streams: bool = False
    needs_weight: bool = False

    @property
    def is_ratio(self) -> bool:
        return self.numerator is not None and self.denominator is not None

    @property
    def is_count(self) -> bool:
        return self.value_kind == COUNT

    @property
    def is_fixed_agg(self) -> bool:
        return not self.allowed_aggs

    @property
    def columns_used(self) -> List[str]:
        if self.is_ratio:
            return [self.numerator, self.denominator]
        return [self.column] if self.column else []

    def aggregate(self, frame: pd.DataFrame, agg: Optional[str] = None) -> float:
        """Collapse ``frame``'s rows to one value for this metric."""
        if frame is None or frame.empty:
            return float("nan")
        if self.is_count:
            return float(len(frame))
        if self.is_ratio:
            numerator = _numeric(frame, self.numerator).sum()
            denominator = _numeric(frame, self.denominator).sum()
            if not denominator or not np.isfinite(denominator) or denominator <= 0:
                return float("nan")
            return float(numerator) / float(denominator) * self.scale

        values = _numeric(frame, self.column).dropna()
        if values.empty:
            return float("nan")
        how = agg or self.default_agg
        if how == "count":
            return float(values.count())
        if how not in ("sum", "mean", "median", "max", "min"):
            how = "sum"
        return float(getattr(values, how)()) * self.scale

    def values(self, frame: pd.DataFrame) -> pd.Series:
        """Per-**row** values, without aggregating — for scatter and histograms.

        Ratio metrics are evaluated row-wise here (this activity's own pace),
        which is the right thing for a distribution and the wrong thing for a
        binned trend; :meth:`aggregate` handles the latter.
        """
        if frame is None or frame.empty:
            return pd.Series(dtype="float64")
        if self.is_count:
            return pd.Series(np.ones(len(frame)), index=frame.index)
        if self.is_ratio:
            numerator = _numeric(frame, self.numerator)
            denominator = _numeric(frame, self.denominator)
            return (numerator / denominator.where(denominator > 0)) * self.scale
        return _numeric(frame, self.column) * self.scale

    def series_by_bin(
        self,
        frame: pd.DataFrame,
        bin_keys: Sequence,
        agg: Optional[str] = None,
    ) -> List[Tuple[object, float]]:
        """``(bin, value)`` pairs in bin order, empty/undefined bins dropped.

        ``bin_keys`` is one key per row of ``frame`` — whatever the caller binned
        by (a calendar date, an elapsed-bin index, a sport type, …).
        """
        if frame is None or frame.empty:
            return []
        grouped = frame.assign(_bin=list(bin_keys)).groupby("_bin", sort=True)
        out: List[Tuple[object, float]] = []
        for key, rows in grouped:
            value = self.aggregate(rows, agg)
            if value is None or not np.isfinite(value):
                continue
            out.append((key, float(value)))
        return out


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


# --- The registry ----------------------------------------------------------

def _volume_metrics() -> List[ActivityMetric]:
    return [
        ActivityMetric(
            key="distance_km", label_key="metric.distance_km", unit="km",
            column="distance_m", scale=0.001, default_agg="sum",
            decimals=1, higher_is_better=True,
        ),
        ActivityMetric(
            key="elevation_gain_m", label_key="metric.elevation_gain_m", unit="m",
            column="elevation_gain_m", default_agg="sum",
            decimals=0, higher_is_better=True,
        ),
        ActivityMetric(
            key="moving_time", label_key="metric.moving_time", unit="",
            column="moving_s", default_agg="sum",
            value_kind=DURATION, higher_is_better=True,
        ),
        ActivityMetric(
            key="activity_count", label_key="metric.activity_count", unit="",
            default_agg="count", allowed_aggs=(), value_kind=COUNT,
            decimals=0, higher_is_better=True,
        ),
    ]


def _intensity_metrics() -> List[ActivityMetric]:
    return [
        ActivityMetric(
            key="avg_pace", label_key="metric.avg_pace", unit="/km",
            numerator="moving_s", denominator="distance_m", scale=1000.0,
            allowed_aggs=(), value_kind=PACE, higher_is_better=False,
        ),
        ActivityMetric(
            key="avg_gap_pace", label_key="metric.avg_gap_pace", unit="/km",
            numerator="moving_s", denominator="gap_distance_m", scale=1000.0,
            allowed_aggs=(), value_kind=PACE, higher_is_better=False,
            needs_streams=True,
        ),
        ActivityMetric(
            key="avg_speed_kmh", label_key="metric.avg_speed_kmh", unit="km/h",
            numerator="distance_m", denominator="moving_s", scale=3.6,
            allowed_aggs=(), decimals=2, higher_is_better=True,
        ),
        ActivityMetric(
            key="avg_gradient_pct", label_key="metric.avg_gradient_pct", unit="%",
            numerator="elevation_gain_m", denominator="distance_m", scale=100.0,
            allowed_aggs=(), decimals=1,
        ),
        ActivityMetric(
            key="elevation_per_km", label_key="metric.elevation_per_km", unit="m/km",
            numerator="elevation_gain_m", denominator="distance_m", scale=1000.0,
            allowed_aggs=(), decimals=0,
        ),
        ActivityMetric(
            key="avg_hr", label_key="metric.avg_hr", unit="bpm",
            column="avg_hr", default_agg="mean",
            allowed_aggs=("mean", "median", "max", "min"), decimals=0,
            needs_streams=True,
        ),
        ActivityMetric(
            key="max_hr", label_key="metric.max_hr", unit="bpm",
            column="max_hr", default_agg="max",
            allowed_aggs=("max", "mean", "median"), decimals=0,
            needs_streams=True,
        ),
        ActivityMetric(
            key="avg_power_w", label_key="metric.avg_power_w", unit="W",
            column="avg_power_w", default_agg="mean",
            allowed_aggs=("mean", "median", "max", "min"), decimals=0,
            higher_is_better=True, needs_streams=True, needs_weight=True,
        ),
        ActivityMetric(
            key="power_to_hr", label_key="metric.power_to_hr", unit="W/bpm",
            column="power_to_hr", default_agg="mean",
            allowed_aggs=("mean", "median", "max", "min"), decimals=2,
            higher_is_better=True, needs_streams=True, needs_weight=True,
        ),
    ]


def _band_metrics() -> List[ActivityMetric]:
    """Time spent in each gradient band — one metric per band, summed."""
    return [
        ActivityMetric(
            key=f"time_{key}", label_key=f"ltp.band.{key}", unit="",
            column=band_column(key), default_agg="sum",
            allowed_aggs=("sum", "mean"), value_kind=DURATION,
            needs_streams=True,
        )
        for key, _, _ in GRADIENT_BANDS
    ]


def _best_effort_metrics() -> List[ActivityMetric]:
    """Best effort per PR distance — ``min`` over a bin is that bin's record."""
    return [
        ActivityMetric(
            key=f"best_{_slug(label)}",
            label_key=f"metric.best.{_slug(label)}",
            unit="",
            column=best_column(label), default_agg="min",
            allowed_aggs=("min", "mean", "median", "count"),
            value_kind=DURATION, higher_is_better=False, needs_streams=True,
        )
        for label, _ in PR_DISTANCES
    ]


def _slug(pr_label: str) -> str:
    return pr_label.lower().replace(" ", "_")


ACTIVITY_METRICS: Dict[str, ActivityMetric] = {
    m.key: m
    for m in (
        _volume_metrics()
        + _intensity_metrics()
        + _band_metrics()
        + _best_effort_metrics()
    )
}

# Order shown in pickers: volume first (what most people ask for), then
# intensity, then the long tail.
METRIC_ORDER: List[str] = list(ACTIVITY_METRICS)


def get_metric(key: str) -> Optional[ActivityMetric]:
    return ACTIVITY_METRICS.get(key)


def metric_or_default(key: Optional[str], fallback: str = "distance_km") -> ActivityMetric:
    """Never raise on a stale saved metric key — fall back to a sane default."""
    return ACTIVITY_METRICS.get(key or "") or ACTIVITY_METRICS[fallback]


def optional_metric(key: Optional[str]) -> Optional[ActivityMetric]:
    """A metric for an *opt-in* selector, or ``None`` when the user chose none.

    Distinct from :func:`metric_or_default` on purpose: for a second, optional
    series, an unrecognised or absent key means "don't draw it" rather than
    "fall back to distance", which would silently invent a series.
    """
    if not key or key == NO_METRIC:
        return None
    return ACTIVITY_METRICS.get(key)


def allowed_aggregations(metric: ActivityMetric) -> Tuple[str, ...]:
    return metric.allowed_aggs
