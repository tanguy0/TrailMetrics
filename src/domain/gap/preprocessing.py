from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from src.domain.models.activity import ActivityStream
from src.domain.models.gap import DownsampledDataset, ProcessedStream


class StreamPreprocessor(ABC):
    """Pipeline that turns raw activity streams into a model-ready dataset."""

    @abstractmethod
    def process_single(self, stream: ActivityStream) -> ProcessedStream:
        ...

    @abstractmethod
    def process_many(
        self,
        streams: List[ActivityStream],
        split_min_time: float,
        verbose: bool = True,
    ) -> DownsampledDataset:
        ...

    @abstractmethod
    def prepare_calibration_dataset(
        self,
        dataset: DownsampledDataset,
        flat_elevation_gain_range: Tuple[float, float] = (-10.0, 10.0),
        hr_tolerance: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ...


class DefaultStreamPreprocessor(StreamPreprocessor):
    """
    Default pipeline:
      - per-stream: derive instantaneous speed and elevation gain from time/distance/altitude.
      - across streams: cut warm-up, downsample by time, drop mixed-gradient splits, filter outliers.
      - calibration set: target each point at the mean speed of the similar-HR flat
        points, weighted by how many there were (see prepare_calibration_dataset).
    """

    DEFAULT_WARMUP_CUT_SECONDS: float = 60 * 15
    DEFAULT_SPEED_RANGE: Tuple[float, float] = (3.0, 22.0)
    DEFAULT_ELEVATION_RANGE: Tuple[float, float] = (-350.0, 350.0)

    def __init__(
        self,
        warmup_cut_seconds: float = DEFAULT_WARMUP_CUT_SECONDS,
        speed_range: Tuple[float, float] = DEFAULT_SPEED_RANGE,
        elevation_range: Tuple[float, float] = DEFAULT_ELEVATION_RANGE,
    ):
        self.warmup_cut_seconds = warmup_cut_seconds
        self.speed_range = speed_range
        self.elevation_range = elevation_range

    def process_single(self, stream: ActivityStream) -> ProcessedStream:
        # float64 explicitly: streams are *stored* (and held in memory) as float32
        # to halve their footprint, but `np.diff` on a cumulative distance in
        # float32 loses metres to cancellation — at 300 km the float32 spacing is
        # ~3 cm against a ~3 m step. Upcasting here costs one transient copy per
        # activity and keeps the derived speed exact.
        time = np.asarray(stream.time, dtype=float)
        distance = np.asarray(stream.distance, dtype=float)
        altitude = np.asarray(stream.altitude, dtype=float)
        heartrate = np.asarray(stream.heartrate, dtype=float)

        delta_dist = np.diff(distance)
        delta_time = np.diff(time)

        # m/s -> km/h
        speed = (delta_dist / delta_time) * 3.6

        # D+ meters per km
        elevation_gain = np.diff(altitude) / delta_dist * 1000

        return ProcessedStream(
            time=time[1:],
            distance=distance[1:],
            speed=speed,
            elevation_gain=elevation_gain,
            heartrate=heartrate[1:],
        )

    def process_many(
        self,
        streams: List[ActivityStream],
        split_min_time: float,
        verbose: bool = True,
    ) -> DownsampledDataset:
        speeds: List[np.ndarray] = []
        elevation_gains: List[np.ndarray] = []
        heartrates: List[np.ndarray] = []
        sport_types: List[np.ndarray] = []

        for i, stream in enumerate(streams):
            if verbose:
                print(f"Processing streams for activity {i + 1}/{len(streams)}")

            try:
                processed = self.process_single(stream)
                speed, elev, hr = self._downsample(processed, split_min_time)
                speeds.append(speed)
                elevation_gains.append(elev)
                heartrates.append(hr)
                sport_types.append(np.array([stream.sport_type] * len(speed)))
            except Exception as e:
                if verbose:
                    print(f"Error processing streams: {e}")
                continue

        all_speed = np.concatenate(speeds)
        all_elev = np.concatenate(elevation_gains)
        all_hr = np.concatenate(heartrates)
        all_sport = np.concatenate(sport_types)

        # Both GAP models are built on `heartrate / speed`, so a split without heart
        # rate cannot contribute to either — and must not merely be *ignored*, it has
        # to be removed. A single NaN reaching the efficiency model poisons every
        # curve it produces: the normalisation divides by `median(flat efficiencies)`,
        # and `np.median` over anything containing NaN is NaN, so one HR-less activity
        # in the flat band turns the whole fit into NaN and the plot reports "no sample
        # falls in this range" for a dataset that is mostly fine.
        #
        # Activities with no HR sensor arrive here as all-NaN (see `StravaClient`), and
        # the range comparisons below silently drop NaN speed and elevation already —
        # heart rate was the one column with no such guard.
        finite = np.isfinite(all_speed) & np.isfinite(all_elev) & np.isfinite(all_hr)
        mask = (
            finite
            & (all_elev >= self.elevation_range[0])
            & (all_elev <= self.elevation_range[1])
            & (all_speed >= self.speed_range[0])
            & (all_speed <= self.speed_range[1])
        )

        return DownsampledDataset(
            speed=all_speed[mask],
            elevation_gain=all_elev[mask],
            heartrate=all_hr[mask],
            sport_types=all_sport[mask],
        )

    def prepare_calibration_dataset(
        self,
        dataset: DownsampledDataset,
        flat_elevation_gain_range: Tuple[float, float] = (-10.0, 10.0),
        hr_tolerance: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Regression targets for the auto-learning model: **one weighted row per split**.

        The question the model answers is "how fast would this split have been on the
        flat, at this heart rate", so each split's target comes from the flat splits
        that share its heart rate (within ``hr_tolerance``).

        The obvious way to express that is one training row per (split, matching flat
        split) pair. Don't: it is quadratic. A year of running is tens of thousands
        of splits, each matching thousands of flat ones, and the pair list reaches
        tens of millions of rows — several GB before the model sees any of it, which
        is what used to OOM the container on exactly the athletes with the most data.

        The pairs are unnecessary. XGBoost minimises squared error, and for the ``m``
        pairs sharing one split's features ``x``::

            Σⱼ (f(x) − yⱼ)²  =  m · (f(x) − ȳ)²  +  Σⱼ (yⱼ − ȳ)²

        The second term does not involve ``f``, so it shifts the loss by a constant
        and changes no gradient, no split and no leaf value. One row carrying the
        *mean* matching flat speed and a ``sample_weight`` of ``m`` is therefore the
        same fit as ``m`` duplicated rows — not an approximation of it.

        Both are obtained without materialising a pair: sort the flat splits by heart
        rate once, and each split's matching window is a pair of ``searchsorted``
        bounds, its count their difference and its mean a difference of prefix sums.

        Returns ``(features, targets, weights)``; splits with no matching flat split
        contribute nothing and are dropped, so all three may be empty.
        """
        speeds = np.asarray(dataset.speed, dtype=float)
        elevation_gains = np.asarray(dataset.elevation_gain, dtype=float)
        heartrates = np.asarray(dataset.heartrate, dtype=float)

        flat_mask = (
            (elevation_gains > flat_elevation_gain_range[0])
            & (elevation_gains < flat_elevation_gain_range[1])
        )
        flat_hrs = heartrates[flat_mask]
        flat_speeds = speeds[flat_mask]

        empty = (np.empty((0, 3)), np.empty(0), np.empty(0))
        if flat_hrs.size == 0 or speeds.size == 0:
            return empty

        order = np.argsort(flat_hrs, kind="stable")
        sorted_hrs = flat_hrs[order]
        # prefix_sums[k] is the total speed of the k lowest-HR flat splits, so the
        # sum over any HR window is one subtraction.
        prefix_sums = np.concatenate(([0.0], np.cumsum(flat_speeds[order])))

        # Inclusive on both ends, matching `abs(flat_hr - hr) <= tolerance`.
        left = np.searchsorted(sorted_hrs, heartrates - hr_tolerance, side="left")
        right = np.searchsorted(sorted_hrs, heartrates + hr_tolerance, side="right")
        counts = (right - left).astype(float)

        matched = counts > 0
        if not matched.any():
            return empty

        targets = (prefix_sums[right[matched]] - prefix_sums[left[matched]]) / counts[matched]
        features = np.column_stack((
            speeds[matched], elevation_gains[matched], heartrates[matched],
        ))
        return features, targets, counts[matched]

    def _downsample(
        self,
        processed: ProcessedStream,
        split_min_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = processed.time
        speed = processed.speed
        elevation_gain = processed.elevation_gain
        heartrate = processed.heartrate

        idx_cut = (time < self.warmup_cut_seconds).sum()
        if idx_cut > 0:
            time = time[idx_cut:]
            speed = speed[idx_cut:]
            elevation_gain = elevation_gain[idx_cut:]
            heartrate = heartrate[idx_cut:]

        cuts: List[int] = []
        current_idx = 0
        for i in range(1, len(time)):
            if time[i] - time[current_idx] >= split_min_time:
                cuts.append(i)
                current_idx = i

        agg_speed: List[float] = []
        agg_elev: List[float] = []
        agg_hr: List[float] = []

        current_idx = 0
        for cut_idx in cuts:
            split_elev = elevation_gain[current_idx:cut_idx]
            # Drop splits with both positive and negative gradient (mixed up/down)
            if not (np.any(split_elev > 0) and np.any(split_elev < 0)):
                agg_speed.append(speed[current_idx:cut_idx].mean())
                agg_elev.append(split_elev.mean())
                agg_hr.append(heartrate[current_idx:cut_idx].mean())
            current_idx = cut_idx

        return np.array(agg_speed), np.array(agg_elev), np.array(agg_hr)
