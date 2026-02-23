"""Route processing: smoothing, interpolation, bearing computation."""

import numpy as np
from scipy.interpolate import CubicSpline

from .gpx_parser import TrackPoint


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in meters between two lat/lon points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return float(2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def compute_cumulative_distances(points: list[TrackPoint]) -> np.ndarray:
    """Compute cumulative arc-length distance along the track."""
    dists = np.zeros(len(points))
    for i in range(1, len(points)):
        dists[i] = dists[i - 1] + haversine(
            points[i - 1].lat, points[i - 1].lon,
            points[i].lat, points[i].lon,
        )
    return dists


def interpolate_route(points: list[TrackPoint], num_samples: int) -> list[TrackPoint]:
    """Resample the route to evenly-spaced points using cubic spline interpolation."""
    cum_dist = compute_cumulative_distances(points)
    total_dist = cum_dist[-1]

    lats = np.array([p.lat for p in points])
    lons = np.array([p.lon for p in points])
    elevs = np.array([p.elevation for p in points])

    # Remove near-duplicate distance entries (stationary points)
    mask = np.ones(len(cum_dist), dtype=bool)
    mask[1:] = np.diff(cum_dist) > 0.1  # keep points >0.1m apart
    cum_dist_clean = cum_dist[mask]
    lats_clean = lats[mask]
    lons_clean = lons[mask]
    elevs_clean = elevs[mask]

    if len(cum_dist_clean) < 4:
        raise ValueError("Not enough unique track points for interpolation")

    cs_lat = CubicSpline(cum_dist_clean, lats_clean)
    cs_lon = CubicSpline(cum_dist_clean, lons_clean)
    cs_elev = CubicSpline(cum_dist_clean, elevs_clean)

    sample_dists = np.linspace(0, total_dist, num_samples)

    return [
        TrackPoint(
            lat=float(cs_lat(d)),
            lon=float(cs_lon(d)),
            elevation=float(cs_elev(d)),
        )
        for d in sample_dists
    ]


def compute_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Bearing in degrees (0=north, 90=east) from point 1 to point 2."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(lon2 - lon1)
    x = np.sin(dlam) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    bearing = np.degrees(np.arctan2(x, y))
    return float((bearing + 360) % 360)


def compute_bearings(points: list[TrackPoint]) -> list[float]:
    """Compute forward bearing at each point. Last point reuses the previous bearing."""
    bearings = []
    for i in range(len(points) - 1):
        b = compute_bearing(
            points[i].lat, points[i].lon,
            points[i + 1].lat, points[i + 1].lon,
        )
        bearings.append(b)
    bearings.append(bearings[-1])
    return bearings


def smooth_bearings(bearings: list[float], sigma: float = 0.0) -> list[float]:
    """Gaussian smoothing on bearings, handling 360/0 wraparound.

    If sigma is 0 (default), it is set to 2% of the total number of frames,
    which provides strong anticipatory smoothing for switchbacks.
    """
    from scipy.ndimage import gaussian_filter1d

    if sigma <= 0:
        sigma = max(5, len(bearings) * 0.02)

    # Convert bearings to sin/cos components to handle wraparound
    rads = np.radians(bearings)
    sin_b = np.sin(rads)
    cos_b = np.cos(rads)

    # Gaussian smooth both components independently
    sin_smooth = gaussian_filter1d(sin_b, sigma=sigma)
    cos_smooth = gaussian_filter1d(cos_b, sigma=sigma)

    # Convert back to degrees
    smoothed = np.degrees(np.arctan2(sin_smooth, cos_smooth)) % 360
    return smoothed.tolist()


def compute_cumulative_elevation_gain(points: list[TrackPoint]) -> np.ndarray:
    """Compute cumulative positive elevation gain at each point."""
    from scipy.ndimage import gaussian_filter1d

    elevs = np.array([p.elevation for p in points])
    # Smooth elevation to remove GPS noise before computing gain
    elevs = gaussian_filter1d(elevs, sigma=10)
    gain = np.zeros(len(elevs))
    for i in range(1, len(elevs)):
        delta = elevs[i] - elevs[i - 1]
        gain[i] = gain[i - 1] + max(0.0, delta)
    return gain


def compute_frame_elevation_gains(
    original_points: list[TrackPoint],
    frame_distances: np.ndarray,
) -> np.ndarray:
    """Compute cumulative elevation gain at each frame position from original GPS data.

    Computes gain from original (non-interpolated) GPS points, then maps to
    frame positions via CubicSpline to avoid spline-induced elevation noise.
    """
    cum_dist = compute_cumulative_distances(original_points)
    cum_gain = compute_cumulative_elevation_gain(original_points)

    # Remove near-duplicate distance entries for interpolation
    mask = np.ones(len(cum_dist), dtype=bool)
    mask[1:] = np.diff(cum_dist) > 0.1
    if np.sum(mask) < 4:
        return np.zeros(len(frame_distances))

    cs = CubicSpline(cum_dist[mask], cum_gain[mask])
    result = cs(frame_distances)
    # Ensure monotonically non-decreasing (CubicSpline can introduce slight dips)
    return np.maximum.accumulate(np.clip(result, 0, None))


def compute_frame_speeds(
    original_points: list[TrackPoint],
    frame_distances: np.ndarray,
) -> np.ndarray:
    """Compute speed (km/h) at each frame position using original GPS timestamps.

    Returns zeros if timestamps are not available.
    """
    from scipy.ndimage import gaussian_filter1d

    # Check if timestamps exist
    if not original_points[0].time or not original_points[-1].time:
        return np.zeros(len(frame_distances))

    cum_dist = compute_cumulative_distances(original_points)

    # Compute instantaneous speed at each original point
    speeds = np.zeros(len(original_points))
    for i in range(1, len(original_points)):
        dt = (original_points[i].time - original_points[i - 1].time).total_seconds()
        dd = cum_dist[i] - cum_dist[i - 1]
        if dt > 0:
            speeds[i] = (dd / dt) * 3.6  # m/s → km/h
    speeds[0] = speeds[1] if len(speeds) > 1 else 0.0

    # Smooth to reduce GPS noise
    speeds = gaussian_filter1d(speeds, sigma=5)
    speeds = np.clip(speeds, 0, None)

    # Remove near-duplicate distance entries for interpolation
    mask = np.ones(len(cum_dist), dtype=bool)
    mask[1:] = np.diff(cum_dist) > 0.1
    if np.sum(mask) < 4:
        return np.zeros(len(frame_distances))

    cs = CubicSpline(cum_dist[mask], speeds[mask])
    result = cs(frame_distances)
    return np.clip(result, 0, None)


def compute_bounds(points: list[TrackPoint]) -> dict:
    """Return bounding box as {sw: [lng, lat], ne: [lng, lat]}."""
    lats = [p.lat for p in points]
    lons = [p.lon for p in points]
    return {
        "sw": [min(lons), min(lats)],
        "ne": [max(lons), max(lats)],
    }
