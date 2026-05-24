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
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...


class DefaultStreamPreprocessor(StreamPreprocessor):
    """
    Default pipeline:
      - per-stream: derive instantaneous speed and elevation gain from time/distance/altitude.
      - across streams: cut warm-up, downsample by time, drop mixed-gradient splits, filter outliers.
      - calibration set: pair each point with similar-HR flat points to obtain a (X, y) regression target.
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
        time = np.asarray(stream.time)
        distance = np.asarray(stream.distance)
        altitude = np.asarray(stream.altitude)
        heartrate = np.asarray(stream.heartrate)

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

        mask = (
            (all_elev >= self.elevation_range[0])
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        speeds = dataset.speed
        elevation_gains = dataset.elevation_gain
        heartrates = dataset.heartrate

        flat_mask = (
            (elevation_gains > flat_elevation_gain_range[0])
            & (elevation_gains < flat_elevation_gain_range[1])
        )

        x_list: List[List[float]] = []
        y_list: List[float] = []

        flat_hrs = heartrates[flat_mask]
        flat_speeds = speeds[flat_mask]

        for i in range(len(speeds)):
            similar = np.abs(flat_hrs - heartrates[i]) <= hr_tolerance
            if np.any(similar):
                for matching_speed in flat_speeds[similar]:
                    x_list.append([speeds[i], elevation_gains[i], heartrates[i]])
                    y_list.append(matching_speed)

        return np.array(x_list), np.array(y_list)

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
