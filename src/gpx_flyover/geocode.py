"""Fetch nearby landmarks via Overpass API for floating text overlays."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

from .gpx_parser import TrackPoint
from .route import compute_cumulative_distances

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
OVERPASS_CACHE_DIR = Path.home() / ".cache" / "gpx-flyover" / "overpass"
USER_AGENT = "gpx-flyover/0.2.0 (https://github.com/gpx-flyover)"

# ~500m in degrees (rough, good enough for padding)
_DEG_PAD = 0.005

# Overpass query: notable POI types for cycling/hiking routes
_OVERPASS_QUERY = """\
[timeout:30][out:json];
(
  nwr["tourism"~"^(viewpoint|attraction|museum|camp_site)$"]({bbox});
  nwr["amenity"="place_of_worship"]({bbox});
  nwr["leisure"~"^(park|nature_reserve)$"]({bbox});
  nwr["natural"~"^(peak|spring|waterfall|hot_spring|saddle)$"]({bbox});
  nwr["historic"]({bbox});
);
out center;
"""


@dataclass
class PlaceLabel:
    """Geocoded label for a route segment."""
    road: str
    area: str
    lat: float
    lon: float
    distance_m: float


def _bbox_key(min_lat: float, min_lon: float, max_lat: float, max_lon: float) -> str:
    """Deterministic cache key for a bounding box."""
    raw = f"{min_lat:.4f},{min_lon:.4f},{max_lat:.4f},{max_lon:.4f}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two GPS points."""
    R = 6_371_000
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
         * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _query_overpass(
    min_lat: float, min_lon: float, max_lat: float, max_lon: float,
    session: requests.Session,
) -> list[dict]:
    """Query Overpass API for POIs in bbox, with disk cache."""
    key = _bbox_key(min_lat, min_lon, max_lat, max_lon)
    cache_path = OVERPASS_CACHE_DIR / f"{key}.json"

    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    bbox = f"{min_lat},{min_lon},{max_lat},{max_lon}"
    query = _OVERPASS_QUERY.replace("{bbox}", bbox)

    elements = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = session.post(
                endpoint,
                data={"data": query},
                headers={"Accept-Language": "zh-TW,en"},
                timeout=60,
            )
            resp.raise_for_status()
            elements = resp.json().get("elements", [])
            break
        except (requests.RequestException, requests.Timeout):
            continue

    if elements is None:
        print("  Warning: Overpass API unavailable, skipping landmarks")
        return []

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(elements, ensure_ascii=False))
    return elements


def _extract_poi(element: dict) -> Optional[tuple[str, float, float]]:
    """Extract (name, lat, lon) from an Overpass element, or None."""
    tags = element.get("tags", {})
    name = tags.get("name", "")
    if not name:
        return None

    # Get coordinates (node has lat/lon directly, way/relation use center)
    lat = element.get("lat") or element.get("center", {}).get("lat")
    lon = element.get("lon") or element.get("center", {}).get("lon")
    if lat is None or lon is None:
        return None

    return (name, float(lat), float(lon))


def fetch_place_labels(
    points: list[TrackPoint],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[PlaceLabel]:
    """Fetch nearby landmark labels along the route via Overpass API."""
    if progress_callback:
        progress_callback(0, 3)

    # 1. Compute route bbox with padding
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    min_lat, max_lat = min(lats) - _DEG_PAD, max(lats) + _DEG_PAD
    min_lon, max_lon = min(lons) - _DEG_PAD, max(lons) + _DEG_PAD

    # 2. Query Overpass for POIs
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    elements = _query_overpass(min_lat, min_lon, max_lat, max_lon, session)

    if progress_callback:
        progress_callback(1, 3)

    # 3. Extract named POIs
    pois = []
    for el in elements:
        result = _extract_poi(el)
        if result:
            pois.append(result)

    if not pois:
        if progress_callback:
            progress_callback(3, 3)
        return []

    # 4. Build route arrays for distance computation
    cum_dist = compute_cumulative_distances(points)
    route_lats = np.array([p.lat for p in points])
    route_lons = np.array([p.lon for p in points])

    # 5. For each POI, find nearest route point and distance from route
    labels = []
    for name, poi_lat, poi_lon in pois:
        # Quick approximate distance to all route points
        dlat = route_lats - poi_lat
        dlon = (route_lons - poi_lon) * np.cos(np.radians(poi_lat))
        approx_dist = np.sqrt(dlat ** 2 + dlon ** 2) * 111_319
        nearest_idx = int(np.argmin(approx_dist))
        min_dist_m = float(approx_dist[nearest_idx])

        # Only keep POIs within 500m of route
        if min_dist_m > 500:
            continue

        labels.append(PlaceLabel(
            road=name,
            area="",
            lat=poi_lat,
            lon=poi_lon,
            distance_m=cum_dist[nearest_idx],
        ))

    if progress_callback:
        progress_callback(2, 3)

    # 6. Sort by distance along route and deduplicate (min 500m apart)
    labels.sort(key=lambda lb: lb.distance_m)
    deduped = []
    for lb in labels:
        if not deduped or (lb.distance_m - deduped[-1].distance_m) >= 500:
            deduped.append(lb)

    if progress_callback:
        progress_callback(3, 3)

    return deduped
