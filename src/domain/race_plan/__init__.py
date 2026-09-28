"""Race plan: a pace profile for a course, from a GPX and a target finish time.

The pipeline, in the order it runs:

* :mod:`.gpx` — the uploaded file into ``(lat, lon, elevation)`` points.
* :mod:`.planner` — those points onto a regular distance grid, a GAP curve onto
  that grid, the one constant GAP pace that lands exactly on the target time, and
  the course cut two ways: into climbs / descents / flats, and into legs between
  aid stations.
* :mod:`.output` — the plan as chart IR and tables, like any plot.
"""
