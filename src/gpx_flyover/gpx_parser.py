"""Parse GPX files and extract track points."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gpxpy

MAX_GPX_SIZE = 50 * 1024 * 1024  # 50 MB


@dataclass
class TrackPoint:
    lat: float
    lon: float
    elevation: float  # meters
    time: Optional[datetime] = None


def parse_gpx(file_path: str) -> list[TrackPoint]:
    """Parse a GPX file and return a list of track points."""
    file_size = os.path.getsize(file_path)
    if file_size > MAX_GPX_SIZE:
        raise ValueError(
            f"GPX file too large ({file_size / 1024 / 1024:.1f} MB, max 50 MB)"
        )

    with open(file_path, "r") as f:
        gpx = gpxpy.parse(f)

    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(
                    TrackPoint(
                        lat=point.latitude,
                        lon=point.longitude,
                        elevation=point.elevation or 0.0,
                        time=point.time,
                    )
                )

    if len(points) < 2:
        raise ValueError("GPX file must contain at least 2 track points")

    return points
