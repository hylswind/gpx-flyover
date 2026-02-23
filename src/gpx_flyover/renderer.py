"""Renderer: PyVista-based 3D terrain rendering."""

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pyvista as pv
import srtm
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import RegularGridInterpolator

from .camera import Camera3DFrame, compute_birds_eye_frame, gps_to_local, points_to_local
from .geocode import PlaceLabel
from .gpx_parser import TrackPoint
from .route import (
    compute_bearing,
    compute_cumulative_distances,
    compute_cumulative_elevation_gain,
    compute_frame_elevation_gains,
    compute_frame_speeds,
)
from .tiles import fetch_terrain_texture
from .video import StreamingEncoder

# Cool tech color scheme
from matplotlib.colors import LinearSegmentedColormap
_TECH_CMAP = LinearSegmentedColormap.from_list("tech", ["#00D4FF", "#0066FF"])
_SPEED_CMAP = LinearSegmentedColormap.from_list(
    "speed", ["#00D4FF", "#44FF44", "#FFAA00", "#FF4444"]
)

# Suppress VTK output window
pv.global_theme.allow_empty_mesh = True


def _lat_to_merc_y(lat: float) -> float:
    """Convert latitude to Mercator Y coordinate."""
    return math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def _build_elevation_grid(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    origin_lat: float,
    origin_lon: float,
    grid_size: int = 200,
    terrain_exaggeration: float = 1.0,
) -> pv.StructuredGrid:
    """Build a 3D terrain mesh from SRTM elevation data."""
    elevation_data = srtm.get_data()

    lats = np.linspace(min_lat, max_lat, grid_size)
    lons = np.linspace(min_lon, max_lon, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Sample elevation for each grid point
    elev_grid = np.zeros_like(lat_grid)
    for i in range(grid_size):
        for j in range(grid_size):
            e = elevation_data.get_elevation(lats[i], lons[j])
            elev_grid[i, j] = e if e is not None else 0.0

    # Convert to local coordinates
    x_grid = (lon_grid - origin_lon) * np.cos(np.radians(origin_lat)) * 111_319.0
    y_grid = (lat_grid - origin_lat) * 111_319.0
    z_grid = elev_grid * terrain_exaggeration

    grid = pv.StructuredGrid(x_grid, y_grid, z_grid)
    grid["elevation"] = z_grid.ravel(order="F")

    # Build interpolant for sampling terrain Z at arbitrary (x, y)
    x_axis = x_grid[0, :]   # unique x values (columns)
    y_axis = y_grid[:, 0]    # unique y values (rows)
    terrain_interp = RegularGridInterpolator(
        (y_axis, x_axis), z_grid,
        method="linear", bounds_error=False, fill_value=0.0,
    )
    return grid, terrain_interp


def _world_to_screen(
    plotter: pv.Plotter, point: tuple[float, float, float],
    width: int, height: int,
) -> tuple[int, int] | None:
    """Project a 3D world point to 2D screen pixel coordinates."""
    renderer = plotter.renderer
    # Use VTK's SetWorldPoint/WorldToDisplay pipeline
    renderer.SetWorldPoint(point[0], point[1], point[2], 1.0)
    renderer.WorldToDisplay()
    dp = renderer.GetDisplayPoint()
    # dp = (display_x, display_y, depth)
    # depth near 0 = close to near plane, near 1 = far plane
    # Values outside [0,1] are behind the camera
    if dp[2] < 0 or dp[2] > 1:
        return None
    # VTK display coords: origin at bottom-left; PIL: origin at top-left
    px = int(dp[0])
    py = int(height - dp[1])
    # Check if within screen bounds (with some margin)
    margin = 50
    if -margin <= px <= width + margin and -margin <= py <= height + margin:
        return (px, py)
    return None


_FONT_CACHE_DIR = Path.home() / ".cache" / "gpx-flyover" / "fonts"
_FONT_DOWNLOAD_URL = (
    "https://github.com/google/fonts/raw/main/ofl/notosanssc/"
    "NotoSansSC%5Bwght%5D.ttf"
)
_FONT_SHA256 = "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"
_FONT_SEARCH_PATHS = [
    str(_FONT_CACHE_DIR / "NotoSansSC-Regular.ttf"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]


def _find_font() -> Optional[str]:
    """Find best available font with CJK support, downloading if needed."""
    for path in _FONT_SEARCH_PATHS:
        if Path(path).exists():
            return path

    # Auto-download Noto Sans SC (CJK + Latin)
    try:
        import hashlib
        import requests
        dest = _FONT_CACHE_DIR / "NotoSansSC-Regular.ttf"
        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_FONT_DOWNLOAD_URL, timeout=60)
        resp.raise_for_status()
        digest = hashlib.sha256(resp.content).hexdigest()
        if digest != _FONT_SHA256:
            warnings.warn(f"Font checksum mismatch: {digest}")
            return None
        dest.write_bytes(resp.content)
        return str(dest)
    except Exception:
        return None


def _draw_elevation_profile(
    img: Image.Image,
    elevations: np.ndarray,
    distances_km: np.ndarray,
    current_idx: int,
    font_path: Optional[str],
) -> int:
    """Draw elevation profile chart at the very bottom of the screen.

    Returns the chart top Y coordinate (for positioning stats above it).
    """
    w, h = img.size

    # Chart dimensions
    chart_h = max(50, h // 10)
    margin_x = max(40, w // 12)
    chart_w = w - 2 * margin_x
    margin_bottom = max(8, h // 120)
    chart_bottom = h - margin_bottom
    chart_top = chart_bottom - chart_h
    pad_y = chart_h * 0.1

    # Downsample elevations for drawing
    n_draw = min(len(elevations), chart_w)
    indices = np.linspace(0, len(elevations) - 1, n_draw).astype(int)
    elev_draw = elevations[indices]

    elev_min = float(elevations.min())
    elev_max = float(elevations.max())
    elev_range = max(elev_max - elev_min, 1.0)

    # Pixel coordinates for profile line
    xs = np.linspace(margin_x, margin_x + chart_w, n_draw)
    ys = chart_bottom - pad_y - (
        (elev_draw - elev_min) / elev_range * (chart_h - 2 * pad_y)
    )

    # Draw on RGBA overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Background
    draw.rectangle([0, chart_top - 4, w, h], fill=(10, 15, 40, 100))

    # Filled area below profile line
    fill_pts = [(int(xs[0]), chart_bottom)]
    for x, y in zip(xs, ys):
        fill_pts.append((int(x), int(y)))
    fill_pts.append((int(xs[-1]), chart_bottom))
    draw.polygon(fill_pts, fill=(0, 180, 255, 30))

    # Profile line
    line_pts = [(int(x), int(y)) for x, y in zip(xs, ys)]
    draw.line(line_pts, fill=(180, 210, 255, 180), width=max(1, h // 500))

    # Current position marker
    progress = current_idx / max(1, len(elevations) - 1)
    cx = int(margin_x + progress * chart_w)
    ci = min(int(progress * (n_draw - 1)), n_draw - 1)
    cy = int(ys[ci])

    # Vertical line
    draw.line([(cx, chart_top), (cx, chart_bottom)], fill=(0, 212, 255, 80), width=1)
    # Dot
    r = max(3, h // 250)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 212, 255, 255))

    # Composite overlay onto image
    result = Image.alpha_composite(img.convert("RGBA"), overlay)
    img.paste(result.convert("RGB"))

    # Elevation labels
    draw2 = ImageDraw.Draw(img)
    if font_path:
        label_font = ImageFont.truetype(font_path, size=max(10, h // 60))
    else:
        label_font = ImageFont.load_default()

    stroke = max(1, h // 500)
    label_color = (150, 180, 210)
    stroke_color = (0, 20, 50)

    draw2.text(
        (margin_x - 4, int(chart_top + pad_y)), f"{elev_max:.0f}m",
        fill=label_color, font=label_font, anchor="rt",
        stroke_width=stroke, stroke_fill=stroke_color,
    )
    draw2.text(
        (margin_x - 4, int(chart_bottom - pad_y)), f"{elev_min:.0f}m",
        fill=label_color, font=label_font, anchor="rb",
        stroke_width=stroke, stroke_fill=stroke_color,
    )

    return chart_top


def _draw_stats_row(
    img: Image.Image,
    speed_kmh: float,
    distance_km: float,
    elev_gain_m: float,
    bottom_y: int,
    font_path: Optional[str],
) -> None:
    """Draw speed, distance, elevation gain in a single row above the profile."""
    w, h = img.size

    if font_path:
        font_val = ImageFont.truetype(font_path, size=max(14, h // 30))
        font_unit = ImageFont.truetype(font_path, size=max(10, h // 50))
    else:
        font_val = ImageFont.load_default()
        font_unit = font_val

    stroke_w = max(2, h // 400)
    gap = max(6, h // 200)
    y_val = bottom_y - gap - font_val.size

    draw = ImageDraw.Draw(img)
    fill_color = "white"
    unit_color = (150, 190, 230)
    stroke_color = (0, 30, 80)

    stats = [
        (f"{speed_kmh:.1f}", "km/h"),
        (f"{distance_km:.1f}", "km"),
        (f"▲ {elev_gain_m:.0f}", "m"),
    ]

    col_w = w // 3
    for i, (val, unit) in enumerate(stats):
        cx = col_w * i + col_w // 2
        val_w = font_val.getlength(val)
        unit_w = font_unit.getlength(unit)
        total_w = val_w + 4 + unit_w
        x_start = cx - total_w / 2

        draw.text(
            (x_start, y_val), val,
            fill=fill_color, font=font_val,
            stroke_width=stroke_w, stroke_fill=stroke_color, anchor="lt",
        )
        draw.text(
            (x_start + val_w + 4, y_val + font_val.size - font_unit.size), unit,
            fill=unit_color, font=font_unit,
            stroke_width=stroke_w, stroke_fill=stroke_color, anchor="lt",
        )


@dataclass
class _Label3D:
    """Pre-computed 3D label data for floating road name rendering."""
    road: str
    x: float
    y: float
    z: float
    bearing: float  # degrees, 0=north


def _precompute_label_positions(
    labels: list[PlaceLabel],
    origin_lat: float,
    origin_lon: float,
    terrain_exaggeration: float,
) -> list[_Label3D]:
    """Convert geocoded labels to 3D positions with bearings."""
    elevation_data = srtm.get_data()
    result = []
    for i, lb in enumerate(labels):
        if not lb.road:
            continue
        x, y, _ = gps_to_local(lb.lat, lb.lon, 0, origin_lat, origin_lon)
        elev = elevation_data.get_elevation(lb.lat, lb.lon)
        if elev is None:
            elev = 0.0
        z = elev * terrain_exaggeration + 30  # float above terrain

        # Compute bearing from this label to the next (for text rotation)
        if i < len(labels) - 1:
            bearing = compute_bearing(lb.lat, lb.lon,
                                      labels[i + 1].lat, labels[i + 1].lon)
        elif i > 0:
            bearing = compute_bearing(labels[i - 1].lat, labels[i - 1].lon,
                                      lb.lat, lb.lon)
        else:
            bearing = 0.0
        result.append(_Label3D(road=lb.road, x=x, y=y, z=z, bearing=bearing))
    return result


def _draw_floating_labels(
    img: Image.Image,
    labels_3d: list[_Label3D],
    plotter: pv.Plotter,
    width: int,
    height: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Draw road name labels at their 3D map positions."""
    stroke_w = max(2, height // 300)
    min_spacing = max(120, height // 8)  # generous spacing to avoid clutter
    placed: list[tuple[int, int, str]] = []  # (x, y, road_name)

    for lb in labels_3d:
        pos = _world_to_screen(plotter, (lb.x, lb.y, lb.z), width, height)
        if pos is None:
            continue
        px, py = pos

        # Skip if too close to screen edges
        edge_margin = 80
        if px < edge_margin or px > width - edge_margin:
            continue
        if py < edge_margin or py > height - edge_margin:
            continue

        # Skip if same road name already placed
        if any(name == lb.road for _, _, name in placed):
            continue

        # Skip if too close to another placed label
        too_close = False
        for ox, oy, _ in placed:
            if abs(px - ox) < min_spacing and abs(py - oy) < min_spacing // 2:
                too_close = True
                break
        if too_close:
            continue

        # Compute screen-space rotation from bearing
        ahead_dist = 50  # meters ahead
        bearing_rad = math.radians(lb.bearing)
        ax = lb.x + math.sin(bearing_rad) * ahead_dist
        ay = lb.y + math.cos(bearing_rad) * ahead_dist
        ahead_pos = _world_to_screen(plotter, (ax, ay, lb.z), width, height)
        if ahead_pos is not None:
            dx = ahead_pos[0] - px
            dy = ahead_pos[1] - py
            angle_deg = -math.degrees(math.atan2(dy, dx))
        else:
            angle_deg = 0.0

        # Cap rotation to ±30° for readability
        angle_deg = max(-30.0, min(30.0, angle_deg))

        # Render text on a temporary RGBA image, then rotate and paste
        bbox = font.getbbox(lb.road)
        tw = bbox[2] - bbox[0] + stroke_w * 4
        th = bbox[3] - bbox[1] + stroke_w * 4
        pad = 4
        tmp = Image.new("RGBA", (tw + pad * 2, th + pad * 2), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)

        # White text with dark navy stroke
        tmp_draw.text(
            (pad + stroke_w * 2, pad + stroke_w * 2 - bbox[1]),
            lb.road,
            fill="white",
            font=font,
            stroke_width=stroke_w,
            stroke_fill=(0, 20, 60),
        )

        # Rotate and paste
        rotated = tmp.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
        rw, rh = rotated.size
        paste_x = px - rw // 2
        paste_y = py - rh // 2
        img.paste(rotated, (paste_x, paste_y), rotated)

        placed.append((px, py, lb.road))


# ── Intro / Outro helpers ───────────────────────────────────────────────


def _ease_in_out(t: float) -> float:
    """Smoothstep ease-in-out: 0→1 with zero derivative at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _interpolate_camera(
    start: Camera3DFrame, end: Camera3DFrame, t: float,
) -> Camera3DFrame:
    """Interpolate two camera frames with ease-in-out."""
    s = _ease_in_out(t)
    pos = tuple(a + (b - a) * s for a, b in zip(start.position, end.position))
    foc = tuple(a + (b - a) * s for a, b in zip(start.focal_point, end.focal_point))
    return Camera3DFrame(position=pos, focal_point=foc)


def _compute_duration_str(points: list[TrackPoint]) -> str:
    """Human-readable ride duration from timestamps, or empty string."""
    if not points or not points[0].time or not points[-1].time:
        return ""
    total_sec = int((points[-1].time - points[0].time).total_seconds())
    if total_sec <= 0:
        return ""
    hours, remainder = divmod(total_sec, 3600)
    minutes = remainder // 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def _render_intro(
    plotter: pv.Plotter,
    encoder: StreamingEncoder,
    birds_eye: Camera3DFrame,
    first_frame: Camera3DFrame,
    num_frames: int,
    width: int,
    height: int,
    distance_km: float,
    elev_gain_m: float,
    font_path: Optional[str],
    progress_callback: Optional[Callable[[int, int], None]],
    progress_offset: int,
    total_frames: int,
) -> None:
    """Render intro: animated zoom-in from bird's-eye to starting camera."""
    if font_path:
        font = ImageFont.truetype(font_path, size=max(16, height // 25))
    else:
        font = ImageFont.load_default()

    stroke_w = max(2, height // 400)
    stroke_color = (0, 30, 80)
    fade_in_end = num_frames // 3       # first third: fade from black
    fade_out_start = num_frames * 2 // 3  # last third: text fades out

    stats_text = f"{distance_km:.1f} km   |   ▲ {elev_gain_m:.0f} m"

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        cam = _interpolate_camera(birds_eye, first_frame, t)

        plotter.camera_position = [cam.position, cam.focal_point, (0.0, 0.0, 1.0)]
        plotter.render()
        plotter.reset_camera_clipping_range()
        img = Image.fromarray(plotter.screenshot(return_img=True))

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Fade from black (first third)
        if i < fade_in_end:
            black_alpha = int(255 * (1.0 - i / max(1, fade_in_end)))
            draw.rectangle([0, 0, width, height], fill=(0, 0, 0, black_alpha))

        # Stats text with fade
        if i < fade_out_start:
            text_alpha = 255
        else:
            text_alpha = int(
                255 * (1.0 - (i - fade_out_start) / max(1, num_frames - 1 - fade_out_start))
            )
        text_alpha = max(0, min(255, text_alpha))

        if text_alpha > 0:
            text_color = (255, 255, 255, text_alpha)
            cx = width // 2
            cy = height // 2
            draw.text(
                (cx, cy), stats_text,
                fill=text_color, font=font, anchor="mm",
                stroke_width=stroke_w, stroke_fill=(*stroke_color, text_alpha),
            )

        result = Image.alpha_composite(img.convert("RGBA"), overlay)
        encoder.write_frame(np.asarray(result.convert("RGB")).tobytes())

        if progress_callback:
            progress_callback(progress_offset + i + 1, total_frames)


def _render_outro(
    plotter: pv.Plotter,
    encoder: StreamingEncoder,
    last_frame: Camera3DFrame,
    num_frames: int,
    width: int,
    height: int,
    total_distance_km: float,
    total_elev_gain_m: float,
    avg_speed_kmh: float,
    duration_str: str,
    font_path: Optional[str],
    progress_callback: Optional[Callable[[int, int], None]],
    progress_offset: int,
    total_frames: int,
) -> None:
    """Render outro: stats summary card over the last 3D frame."""
    # Render the last frame once and cache it
    plotter.camera_position = [
        last_frame.position, last_frame.focal_point, (0.0, 0.0, 1.0),
    ]
    plotter.render()
    plotter.reset_camera_clipping_range()
    bg_img = Image.fromarray(plotter.screenshot(return_img=True))

    # Font setup
    if font_path:
        title_font = ImageFont.truetype(font_path, size=max(16, height // 25))
        label_font = ImageFont.truetype(font_path, size=max(12, height // 45))
        value_font = ImageFont.truetype(font_path, size=max(14, height // 22))
    else:
        title_font = label_font = value_font = ImageFont.load_default()

    stroke_w = max(2, height // 400)
    stroke_color = (0, 30, 80)
    text_delay = num_frames // 6  # text starts after ~0.5 seconds

    # Build stat rows: (label, value)
    stat_rows = [
        ("Distance", f"{total_distance_km:.1f} km"),
        ("Elevation Gain", f"▲ {total_elev_gain_m:.0f} m"),
    ]
    if avg_speed_kmh > 0:
        stat_rows.append(("Avg Speed", f"{avg_speed_kmh:.1f} km/h"))
    if duration_str:
        stat_rows.append(("Duration", duration_str))

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Dark overlay fades in
        overlay_alpha = int(160 * _ease_in_out(t))
        draw.rectangle([0, 0, width, height], fill=(10, 15, 40, overlay_alpha))

        # Text fades in (delayed)
        if i < text_delay:
            text_alpha = 0
        else:
            text_alpha = int(
                255 * _ease_in_out((i - text_delay) / max(1, num_frames - 1 - text_delay))
            )
        text_alpha = max(0, min(255, text_alpha))

        if text_alpha > 0:
            cx = width // 2
            cy_start = height // 2 - height // 6  # start above center

            # "RIDE COMPLETE" title
            title_color = (0, 212, 255, text_alpha)
            draw.text(
                (cx, cy_start), "RIDE COMPLETE",
                fill=title_color, font=title_font, anchor="mm",
                stroke_width=stroke_w, stroke_fill=(*stroke_color, text_alpha),
            )

            # Divider line
            line_w = width * 2 // 5
            line_y = cy_start + title_font.size // 2 + height // 60
            draw.line(
                [(cx - line_w // 2, line_y), (cx + line_w // 2, line_y)],
                fill=(0, 212, 255, text_alpha // 2), width=max(1, height // 500),
            )

            # Stat rows
            row_y = line_y + height // 30
            row_spacing = height // 12
            label_color = (150, 190, 230, text_alpha)
            value_color = (255, 255, 255, text_alpha)

            for label, value in stat_rows:
                draw.text(
                    (cx, row_y), label,
                    fill=label_color, font=label_font, anchor="mm",
                    stroke_width=stroke_w, stroke_fill=(*stroke_color, text_alpha),
                )
                draw.text(
                    (cx, row_y + label_font.size + height // 80), value,
                    fill=value_color, font=value_font, anchor="mm",
                    stroke_width=stroke_w, stroke_fill=(*stroke_color, text_alpha),
                )
                row_y += row_spacing

        result = Image.alpha_composite(bg_img.convert("RGBA"), overlay)
        encoder.write_frame(np.asarray(result.convert("RGB")).tobytes())

        if progress_callback:
            progress_callback(progress_offset + i + 1, total_frames)


class TerrainRenderer:
    """Renders 3D terrain flyover using PyVista."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        terrain_exaggeration: float = 2.0,
    ):
        self.width = width
        self.height = height
        self.terrain_exaggeration = terrain_exaggeration

    def render(
        self,
        route_points: list[TrackPoint],
        camera_frames: list[Camera3DFrame],
        origin_lat: float,
        origin_lon: float,
        encoder: StreamingEncoder,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        terrain_callback: Optional[Callable[[], None]] = None,
        tile_callback: Optional[Callable[[int, int], None]] = None,
        tile_zoom: int = 16,
        tile_source: str = "esri-satellite",
        place_labels: Optional[list[PlaceLabel]] = None,
        original_points: Optional[list[TrackPoint]] = None,
        intro: bool = True,
        outro: bool = True,
        intro_duration: float = 3.0,
        outro_duration: float = 3.0,
        fps: int = 24,
    ) -> None:
        """Render all frames and stream to the encoder."""
        # Compute bounds with padding
        lats = [p.lat for p in route_points]
        lons = [p.lon for p in route_points]
        lat_pad = (max(lats) - min(lats)) * 0.3 + 0.02
        lon_pad = (max(lons) - min(lons)) * 0.3 + 0.02
        min_lat, max_lat = min(lats) - lat_pad, max(lats) + lat_pad
        min_lon, max_lon = min(lons) - lon_pad, max(lons) + lon_pad

        # Build terrain mesh
        if terrain_callback:
            terrain_callback()
        terrain, terrain_interp = _build_elevation_grid(
            min_lat, max_lat, min_lon, max_lon,
            origin_lat, origin_lon,
            grid_size=350,
            terrain_exaggeration=self.terrain_exaggeration,
        )

        # Fetch map tiles for texture
        tile_result = fetch_terrain_texture(
            min_lat, max_lat, min_lon, max_lon,
            zoom=tile_zoom,
            progress_callback=tile_callback,
            tile_source=tile_source,
        )

        # Crop stitched tile image to exact terrain bounds using Mercator Y.
        # The tile image is Mercator-projected, so latitude→pixel mapping
        # must use Mercator math, not linear interpolation.
        img = tile_result.image
        img_w, img_h = img.size

        lon_range = tile_result.se_lon - tile_result.nw_lon
        nw_merc = _lat_to_merc_y(tile_result.nw_lat)
        se_merc = _lat_to_merc_y(tile_result.se_lat)
        merc_range = nw_merc - se_merc

        crop_left = int((min_lon - tile_result.nw_lon) / lon_range * img_w)
        crop_right = int((max_lon - tile_result.nw_lon) / lon_range * img_w)
        crop_top = int((nw_merc - _lat_to_merc_y(max_lat)) / merc_range * img_h)
        crop_bottom = int((nw_merc - _lat_to_merc_y(min_lat)) / merc_range * img_h)

        # Clamp to image bounds
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(img_w, crop_right)
        crop_bottom = min(img_h, crop_bottom)

        texture_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))

        # Re-project from Mercator to equirectangular (linear latitude).
        # The mesh Y coordinates are linear in latitude, so the texture rows
        # must also be linear in latitude for correct alignment.
        crop_nw_lat = max_lat  # top of crop = max_lat
        crop_se_lat = min_lat  # bottom of crop = min_lat
        crop_nw_merc = _lat_to_merc_y(crop_nw_lat)
        crop_se_merc = _lat_to_merc_y(crop_se_lat)
        tex_arr = np.asarray(texture_img)
        h_tex, w_tex = tex_arr.shape[:2]
        # For each output row (linear latitude), find the source Mercator row
        out_lats = np.linspace(crop_nw_lat, crop_se_lat, h_tex)
        out_merc = np.log(np.tan(np.pi / 4 + np.radians(out_lats) / 2))
        src_rows = ((crop_nw_merc - out_merc) / (crop_nw_merc - crop_se_merc)
                     * (h_tex - 1))
        src_rows = np.clip(src_rows.astype(int), 0, h_tex - 1)
        tex_arr = tex_arr[src_rows]
        texture_img = Image.fromarray(tex_arr)

        # No explicit vertical flip needed — PyVista's Texture._from_array()
        # internally flips Y to convert numpy (top=row0) to VTK (bottom=origin).

        # Cap texture at GPU max dimension; no POT resize needed (modern OpenGL)
        max_dim = 16384
        w, h = texture_img.size
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            texture_img = texture_img.resize(
                (int(w * scale), int(h * scale)), Image.LANCZOS)

        # Explicit horizontal texture mapping plane (z=0) to prevent
        # auto-detected plane tilt from exaggerated terrain heights
        bounds = terrain.bounds
        terrain.texture_map_to_plane(
            origin=(bounds[0], bounds[2], 0.0),
            point_u=(bounds[1], bounds[2], 0.0),
            point_v=(bounds[0], bounds[3], 0.0),
            inplace=True,
        )
        tex = pv.numpy_to_texture(np.asarray(texture_img))
        tex.SetMipmap(False)
        tex.SetInterpolate(True)

        # Pre-compute stats arrays (needed for route coloring and overlays)
        frame_distances = compute_cumulative_distances(route_points)
        frame_elev_gains = compute_frame_elevation_gains(
            original_points or route_points, frame_distances,
        )
        frame_speeds = compute_frame_speeds(
            original_points or route_points, frame_distances,
        )
        elevations = np.array([p.elevation for p in route_points])

        # Build route line — sample terrain Z so route sits on the surface
        coords_3d = points_to_local(route_points, origin_lat, origin_lon)
        route_arr = np.array(coords_3d)
        terrain_z = terrain_interp(np.column_stack([route_arr[:, 1], route_arr[:, 0]]))
        route_arr[:, 2] = terrain_z + 3
        route_line = pv.Spline(route_arr, n_points=max(1000, len(route_arr) * 2))

        # Set up plotter
        plotter = pv.Plotter(
            off_screen=True,
            window_size=(self.width, self.height),
        )
        plotter.set_background("#1a1a3e", top="#0a0a1a")  # dark cool sky gradient

        # Add terrain with OSM texture
        plotter.add_mesh(
            terrain,
            texture=tex,
            show_scalar_bar=False,
            lighting=False,
            smooth_shading=False,
        )

        # Add route line colored by speed (fallback to progress gradient if no timestamps)
        has_speed = bool(np.any(frame_speeds > 0))
        if has_speed:
            progress = np.linspace(0, 1, route_line.n_points)
            speed_interp = np.interp(
                progress, np.linspace(0, 1, len(frame_speeds)), frame_speeds,
            )
            route_line["speed"] = speed_interp
            route_tube = route_line.tube(radius=3)
            plotter.add_mesh(
                route_tube, scalars="speed", cmap=_SPEED_CMAP,
                show_scalar_bar=False, opacity=0.9,
            )
        else:
            route_line["progress"] = np.linspace(0, 1, route_line.n_points)
            route_tube = route_line.tube(radius=3)
            plotter.add_mesh(
                route_tube, scalars="progress", cmap=_TECH_CMAP,
                show_scalar_bar=False, opacity=0.9,
            )

        # Lighting: angled sun + fill light for terrain relief
        center_x = route_arr[:, 0].mean()
        center_y = route_arr[:, 1].mean()
        center_z = route_arr[:, 2].mean()
        max_z = route_arr[:, 2].max()

        sun = pv.Light(
            position=(center_x + 8000, center_y - 5000, max_z + 5000),
            focal_point=(center_x, center_y, center_z),
            intensity=0.8,
        )
        plotter.add_light(sun)

        fill = pv.Light(
            position=(center_x - 4000, center_y + 3000, max_z + 2000),
            focal_point=(center_x, center_y, center_z),
            intensity=0.4,
        )
        plotter.add_light(fill)

        # Rider marker settings (2D circle drawn on image via PIL)
        rider_radius = max(6, min(self.width, self.height) // 120)
        rider_stroke = max(2, rider_radius // 3)

        # Prepare text overlay data
        font_path = _find_font()
        if font_path is None:
            warnings.warn("No CJK font found; text overlay may not render correctly")

        # Pre-compute floating label data
        labels_3d = None
        label_font = None
        if place_labels:
            labels_3d = _precompute_label_positions(
                place_labels, origin_lat, origin_lon,
                self.terrain_exaggeration,
            )
            if font_path:
                label_font = ImageFont.truetype(
                    font_path, size=max(14, self.height // 50))
            else:
                label_font = ImageFont.load_default()

        # Compute intro/outro frame counts and stats
        intro_frames = int(intro_duration * fps) if intro else 0
        outro_frames = int(outro_duration * fps) if outro else 0
        total = len(camera_frames) + intro_frames + outro_frames

        total_distance_km = frame_distances[-1] / 1000.0
        total_elev_gain_m = float(frame_elev_gains[-1])
        valid_speeds = frame_speeds[frame_speeds > 0]
        avg_speed_kmh = float(np.mean(valid_speeds)) if len(valid_speeds) > 0 else 0.0

        # Render intro sequence
        if intro_frames > 0:
            birds_eye = compute_birds_eye_frame(route_arr, self.terrain_exaggeration)
            _render_intro(
                plotter, encoder, birds_eye, camera_frames[0],
                intro_frames, self.width, self.height,
                total_distance_km, total_elev_gain_m,
                font_path, progress_callback, 0, total,
            )

        # Render main frames
        for i, frame in enumerate(camera_frames):
            # Update camera
            plotter.camera_position = [
                frame.position,
                frame.focal_point,
                (0.0, 0.0, 1.0),  # view-up: Z is up
            ]

            # Force re-render and recalculate clipping planes for the new camera
            plotter.render()
            plotter.reset_camera_clipping_range()

            # Capture frame
            img_array = plotter.screenshot(return_img=True)
            img = Image.fromarray(img_array)

            # Draw 2D rider marker (circle)
            rider_pos = route_arr[min(i, len(route_arr) - 1)]
            screen_pos = _world_to_screen(
                plotter, tuple(rider_pos), self.width, self.height,
            )
            if screen_pos is not None:
                draw = ImageDraw.Draw(img)
                px, py = screen_pos
                r = rider_radius
                # Cyan glow
                draw.ellipse(
                    [px - r - 3, py - r - 3, px + r + 3, py + r + 3],
                    fill=(0, 212, 255, 80),
                )
                # White fill with cyan stroke
                draw.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill="white",
                    outline="#00D4FF",
                    width=rider_stroke,
                )

            # Draw floating road name labels on 3D map
            if labels_3d is not None:
                _draw_floating_labels(
                    img, labels_3d, plotter,
                    self.width, self.height, label_font,
                )

            # Draw elevation profile at bottom, then stats at corners above it
            idx = min(i, len(frame_distances) - 1)
            profile_top = _draw_elevation_profile(
                img,
                elevations=elevations,
                distances_km=frame_distances / 1000.0,
                current_idx=idx,
                font_path=font_path,
            )
            _draw_stats_row(
                img,
                speed_kmh=float(frame_speeds[idx]),
                distance_km=frame_distances[idx] / 1000.0,
                elev_gain_m=float(frame_elev_gains[idx]),
                bottom_y=profile_top,
                font_path=font_path,
            )

            encoder.write_frame(np.asarray(img).tobytes())

            if progress_callback:
                progress_callback(intro_frames + i + 1, total)

        # Render outro sequence
        if outro_frames > 0:
            duration_str = _compute_duration_str(original_points or route_points)
            _render_outro(
                plotter, encoder, camera_frames[-1],
                outro_frames, self.width, self.height,
                total_distance_km, total_elev_gain_m,
                avg_speed_kmh, duration_str,
                font_path, progress_callback,
                intro_frames + len(camera_frames), total,
            )

        plotter.close()
