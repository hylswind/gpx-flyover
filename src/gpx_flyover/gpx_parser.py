"""Parse GPX files and extract track points."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import gpxpy
import gpxpy.gpx

MAX_GPX_SIZE = 50 * 1024 * 1024  # 50 MB
_HTML_SIGNATURES = ["<!doctype html", "<html", "<head", "<body"]


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
        try:
            gpx = gpxpy.parse(f)
        except gpxpy.gpx.GPXXMLSyntaxException:
            # Sniff the file to give an actionable error message
            with open(file_path, "r", errors="replace") as sniff:
                head = sniff.read(500).lower()
            if any(sig in head for sig in _HTML_SIGNATURES):
                raise ValueError(
                    "The file appears to be an HTML web page, not a GPX file.\n"
                    "If you downloaded this from Strava or another activity tracker,\n"
                    "you need to export the GPX file first — the activity page URL\n"
                    "is not a direct download link."
                ) from None
            raise ValueError(
                f"Failed to parse '{file_path}' as GPX: the file is not valid XML.\n"
                "Ensure this is a .gpx file exported from your GPS device or app."
            ) from None

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
