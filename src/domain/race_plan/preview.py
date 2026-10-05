"""A saved plan's thumbnail: the course's route and elevation profile, downsampled.

The list of saved plans draws each course as a small map and a small profile. The
full GPX is hundreds of kilobytes; a few hundred numbers are enough at thumbnail
size, so they are computed once when the plan is saved and stored beside it.
"""

from typing import Any, Dict

import numpy as np

from src.domain.race_plan.gpx import CoursePoints
from src.domain.race_plan.planner import Course

ROUTE_POINTS = 150
PROFILE_POINTS = 120


def course_preview(points: CoursePoints, course: Course) -> Dict[str, Any]:
    """``{"route": [[lat, lon], …], "profile": [[km, m], …]}``, rounded for JSON."""
    route_at = _evenly(len(points.lat), ROUTE_POINTS)
    profile_at = _evenly(len(course.distance), PROFILE_POINTS)
    return {
        "route": [
            [round(float(points.lat[i]), 5), round(float(points.lon[i]), 5)] for i in route_at
        ],
        "profile": [
            [round(float(course.distance[i]) / 1000, 3), round(float(course.elevation_smooth[i]), 1)]
            for i in profile_at
        ],
    }


def _evenly(length: int, count: int) -> np.ndarray:
    """At most ``count`` indices spread over ``length``, first and last included."""
    if length <= count:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, count).round().astype(int))
