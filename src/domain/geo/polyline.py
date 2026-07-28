"""Google encoded-polyline decoding.

Strava ships an activity's route as an encoded polyline on the activity summary —
a compact string, no extra request. Decoding it here rather than in the browser
keeps the wire format plain coordinates, so the map component stays a renderer and
needs no decoding library of its own.

Hand-written because the algorithm is twenty lines and a dependency for it would
be the larger cost. It is specified at
https://developers.google.com/maps/documentation/utilities/polylinealgorithm
"""

from typing import List, Tuple

# The format stores coordinates as integers scaled by 1e5.
_SCALE = 1e5


def decode(encoded: str) -> List[Tuple[float, float]]:
    """Decode a polyline into ``(latitude, longitude)`` pairs.

    Malformed input yields the points decoded so far rather than raising: a route is
    decoration on a screen full of real data, and half a line beats a 500.
    """
    if not encoded:
        return []

    points: List[Tuple[float, float]] = []
    index = 0
    lat = 0
    lng = 0
    length = len(encoded)

    while index < length:
        try:
            delta, index = _next_value(encoded, index, length)
            lat += delta
            delta, index = _next_value(encoded, index, length)
            lng += delta
        except (IndexError, ValueError):
            break
        points.append((lat / _SCALE, lng / _SCALE))

    return points


def _next_value(encoded: str, index: int, length: int) -> Tuple[int, int]:
    """One zigzag-encoded varint, and where the next one starts.

    Each character carries five bits; the high bit marks "another chunk follows".
    The result is zigzag-encoded, so the low bit is the sign.
    """
    result = 0
    shift = 0
    while True:
        if index >= length:
            raise IndexError("truncated polyline")
        chunk = ord(encoded[index]) - 63
        if chunk < 0:
            raise ValueError("invalid polyline character")
        index += 1
        result |= (chunk & 0x1F) << shift
        shift += 5
        if chunk < 0x20:
            break
        if shift > 30:
            # A single coordinate delta never needs this many chunks; refusing to
            # loop further stops a corrupt string from spinning here.
            raise ValueError("polyline value too long")
    # Zigzag: even values are positive, odd ones are the negative of (n+1)/2.
    return (~(result >> 1) if result & 1 else (result >> 1)), index


def bounds(
    points: List[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """South-west and north-east corners, for fitting a map to the route."""
    lats = [lat for lat, _ in points]
    lngs = [lng for _, lng in points]
    return (min(lats), min(lngs)), (max(lats), max(lngs))
