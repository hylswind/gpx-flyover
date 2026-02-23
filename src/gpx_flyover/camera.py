"""3D camera system: generate per-frame camera positions following the route."""

from dataclasses import dataclass

import numpy as np

from .gpx_parser import TrackPoint

# Meters per degree of latitude
M_PER_DEG_LAT = 111_319.0


@dataclass
class Camera3DFrame:
    position: tuple[float, float, float]  # camera (x, y, z) in meters
    focal_point: tuple[float, float, float]  # look-at (x, y, z) in meters


def gps_to_local(
    lat: float,
    lon: float,
    elevation: float,
    origin_lat: float,
    origin_lon: float,
) -> tuple[float, float, float]:
    """Convert GPS coordinates to local 3D Cartesian (meters).

    Uses a simple planar approximation centered on origin.
    X = east, Y = north, Z = up.
    """
    x = (lon - origin_lon) * np.cos(np.radians(origin_lat)) * M_PER_DEG_LAT
    y = (lat - origin_lat) * M_PER_DEG_LAT
    z = elevation
    return (float(x), float(y), float(z))


def points_to_local(
    points: list[TrackPoint],
    origin_lat: float,
    origin_lon: float,
) -> list[tuple[float, float, float]]:
    """Convert a list of TrackPoints to local 3D coordinates."""
    return [
        gps_to_local(p.lat, p.lon, p.elevation, origin_lat, origin_lon)
        for p in points
    ]


def generate_camera_frames(
    route_points: list[TrackPoint],
    bearings: list[float],
    origin_lat: float,
    origin_lon: float,
    camera_distance: float = 3000.0,
    camera_height: float = 1800.0,
    look_ahead_m: float = 800.0,
    terrain_exaggeration: float = 1.5,
) -> list[Camera3DFrame]:
    """Generate a 3D camera frame for each route point.

    Camera is positioned behind the current point (opposite bearing),
    offset horizontally by camera_distance and vertically by camera_height.
    It looks toward a point ahead on the route (look_ahead_m meters ahead).
    Elevation values are scaled by terrain_exaggeration to match the rendered mesh.
    """
    coords = points_to_local(route_points, origin_lat, origin_lon)
    n = len(route_points)

    # Convert distance-based look-ahead to frame count
    if n > 1:
        coords_arr = np.array(coords)
        segment_dists = np.sqrt(np.sum(np.diff(coords_arr[:, :2], axis=0) ** 2, axis=1))
        dist_per_frame = float(np.sum(segment_dists)) / (n - 1)
        look_ahead = int(np.clip(look_ahead_m / dist_per_frame, 5, 120))
    else:
        look_ahead = 5

    frames = []

    for i in range(n):
        cx, cy, cz = coords[i]
        bearing_rad = np.radians(bearings[i])

        # Camera position: behind and above the current point
        # Bearing is 0=north, 90=east. In local coords: north=+Y, east=+X
        # "Behind" = opposite bearing direction
        cam_x = cx - np.sin(bearing_rad) * camera_distance
        cam_y = cy - np.cos(bearing_rad) * camera_distance
        cam_z = cz * terrain_exaggeration + camera_height

        # Focal point: midpoint between current position and look-ahead point
        # This keeps the rider visible in frame instead of behind the camera
        look_idx = min(i + look_ahead, n - 1)
        lx, ly, lz = coords[look_idx]
        fx = (cx + lx) / 2
        fy = (cy + ly) / 2
        fz = ((cz + lz) / 2) * terrain_exaggeration

        frames.append(
            Camera3DFrame(
                position=(float(cam_x), float(cam_y), float(cam_z)),
                focal_point=(float(fx), float(fy), float(fz)),
            )
        )

    return frames


def compute_birds_eye_frame(
    route_arr: np.ndarray,
    terrain_exaggeration: float = 1.5,
) -> Camera3DFrame:
    """Compute a bird's-eye camera showing the entire route from above."""
    center_x = float(route_arr[:, 0].mean())
    center_y = float(route_arr[:, 1].mean())
    center_z = float(route_arr[:, 2].mean())

    x_extent = float(route_arr[:, 0].max() - route_arr[:, 0].min())
    y_extent = float(route_arr[:, 1].max() - route_arr[:, 1].min())
    height = max(x_extent, y_extent) * 1.3

    return Camera3DFrame(
        position=(center_x, center_y, height),
        focal_point=(center_x, center_y, center_z),
    )
