"""Encode / decode an :class:`ActivityStream` for blob storage.

Compressed ``.npz`` with the arrays as **float32**. That halves the size against
float64 and loses nothing that matters: distance resolves to a few centimetres
over a 300 km run, altitude and heart rate far finer than the sensors themselves.
A typical 2-hour activity lands around 30–60 kB.

Metadata (sport type, start date, the summary totals) rides along as a JSON blob
inside the same archive, so one object fully reconstructs the stream.
"""

import io
import json
from datetime import datetime
from typing import Optional

import numpy as np

from src.domain.models.activity import ActivityStream

# Bumped if the payload layout changes, so a reader can refuse what it can't parse.
FORMAT_VERSION = 1

# time/distance define the activity and have always been written. altitude/
# heartrate/watts are optional signals — an archive written before a given one
# was added simply lacks that key, so it decodes as NaN rather than KeyError-ing
# (see decode_stream). That keeps old blobs readable across a deploy that adds a
# new stream, with no need to bump FORMAT_VERSION or force an early resync.
_REQUIRED_ARRAYS = ("time", "distance")
_OPTIONAL_ARRAYS = ("altitude", "heartrate", "watts")
_ARRAYS = _REQUIRED_ARRAYS + _OPTIONAL_ARRAYS


def encode_stream(stream: ActivityStream) -> bytes:
    """Serialize one activity's per-second arrays plus its metadata."""
    metadata = {
        "format_version": FORMAT_VERSION,
        "activity_id": int(stream.activity_id),
        "sport_type": str(getattr(stream.sport_type, "root", stream.sport_type)),
        "start_date": stream.start_date.isoformat() if stream.start_date else None,
        "has_streams": bool(getattr(stream, "has_streams", True)),
        "summary_distance_m": stream.summary_distance_m,
        "summary_moving_time_s": stream.summary_moving_time_s,
        "summary_elevation_gain_m": stream.summary_elevation_gain_m,
    }
    arrays = {
        name: np.asarray(getattr(stream, name), dtype=np.float32)
        for name in _ARRAYS
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, meta=json.dumps(metadata), **arrays)
    return buffer.getvalue()


def decode_stream(payload: bytes) -> Optional[ActivityStream]:
    """Rebuild an :class:`ActivityStream`, or ``None`` if the payload is unusable."""
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["meta"]))
            if int(metadata.get("format_version", 0)) > FORMAT_VERSION:
                return None
            arrays = {
                name: np.asarray(archive[name], dtype=float)
                for name in _REQUIRED_ARRAYS
            }
            n = arrays["time"].size
            for name in _OPTIONAL_ARRAYS:
                arrays[name] = (
                    np.asarray(archive[name], dtype=float)
                    if name in archive
                    else np.full(n, np.nan)
                )
    except (ValueError, KeyError, OSError, json.JSONDecodeError):
        return None

    start_date = metadata.get("start_date")
    return ActivityStream(
        activity_id=int(metadata["activity_id"]),
        sport_type=metadata.get("sport_type") or "Run",
        start_date=datetime.fromisoformat(start_date) if start_date else None,
        has_streams=bool(metadata.get("has_streams", True)),
        summary_distance_m=metadata.get("summary_distance_m"),
        summary_moving_time_s=metadata.get("summary_moving_time_s"),
        summary_elevation_gain_m=metadata.get("summary_elevation_gain_m"),
        **arrays,
    )


def object_path(athlete_id: int, activity_id: int) -> str:
    """Stable, athlete-scoped key: ``streams/<athlete>/<activity>.npz``."""
    return f"streams/{int(athlete_id)}/{int(activity_id)}.npz"
