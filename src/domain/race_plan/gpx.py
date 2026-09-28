"""GPX → course points.

Deliberately minimal: a race plan needs the course's shape and its elevation, and
nothing else a GPX can carry (timestamps, heart rate, waypoints) changes the answer.
Track points are read in document order across every track and segment; a file
with no track falls back to its route points, which is what route planners export.
"""

from dataclasses import dataclass
from typing import List, Tuple
from xml.etree import ElementTree

import numpy as np


class GpxError(ValueError):
    """The file cannot be planned on, with a translatable reason key."""

    def __init__(self, reason_key: str):
        super().__init__(reason_key)
        self.reason_key = reason_key


@dataclass
class CoursePoints:
    lat: np.ndarray
    lon: np.ndarray
    elevation: np.ndarray


# Fewer points than this is not a course, it is a pin on a map.
MIN_POINTS = 10


def parse_gpx(payload: bytes) -> CoursePoints:
    # A GPX never legitimately declares entities, and refusing them outright is
    # the simplest guard against entity-expansion payloads in an upload.
    if b"<!ENTITY" in payload[:4096].upper():
        raise GpxError("race_plan.error.gpx_invalid")
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError:
        raise GpxError("race_plan.error.gpx_invalid")

    points = _points(root, "trkpt") or _points(root, "rtept")
    if len(points) < MIN_POINTS:
        raise GpxError("race_plan.error.gpx_no_points")

    with_elevation = [p for p in points if p[2] is not None]
    # A handful of points without <ele> is a glitch; most of them is a file that
    # simply has no elevation, and planning on a flat line would be a lie.
    if len(with_elevation) < 0.9 * len(points):
        raise GpxError("race_plan.error.gpx_no_elevation")

    array = np.array(with_elevation, dtype=float)
    return CoursePoints(lat=array[:, 0], lon=array[:, 1], elevation=array[:, 2])


def _points(root: ElementTree.Element, name: str) -> List[Tuple[float, float, float]]:
    out = []
    for element in root.iter():
        if _local(element.tag) != name:
            continue
        try:
            lat = float(element.get("lat"))
            lon = float(element.get("lon"))
        except (TypeError, ValueError):
            continue
        elevation = None
        for child in element:
            if _local(child.tag) == "ele" and child.text:
                try:
                    elevation = float(child.text)
                except ValueError:
                    pass
                break
        out.append((lat, lon, elevation))
    return out


def _local(tag: str) -> str:
    """``{http://www.topografix.com/GPX/1/1}trkpt`` → ``trkpt``, any GPX version."""
    return tag.rsplit("}", 1)[-1]
