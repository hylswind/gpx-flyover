"""CLI entry point for gpx-flyover."""

import shutil

import click
from tqdm import tqdm

from .camera import generate_camera_frames
from .gpx_parser import parse_gpx
from .renderer import TerrainRenderer
from .route import compute_bearings, interpolate_route, smooth_bearings
from .video import StreamingEncoder

QUALITY_PRESETS = {
    "fast": {"fps": 15, "preset": "veryfast", "crf": 23},
    "medium": {"fps": 24, "preset": "medium", "crf": 18},
    "high": {"fps": 30, "preset": "slow", "crf": 15},
}

FORMAT_PRESETS = {
    "instagram": {"width": 1080, "height": 1920},
    "youtube": {"width": 1920, "height": 1080},
    "tiktok": {"width": 1080, "height": 1920},
    "square": {"width": 1080, "height": 1080},
}


@click.command()
@click.argument("gpx_file", type=click.Path(exists=True))
@click.argument("output", type=click.Path(), default="output.mp4")
@click.option("--width", default=1080, help="Video width in pixels.")
@click.option("--height", default=1920, help="Video height in pixels.")
@click.option("--fps", default=None, type=int, help="Frames per second (overrides quality preset).")
@click.option("--duration", default=60, help="Video duration in seconds.")
@click.option("--quality", type=click.Choice(["fast", "medium", "high"]), default="medium",
              help="Quality preset: fast=15fps, medium=24fps, high=30fps.")
@click.option("--camera-distance", default=3000.0, help="Camera distance behind rider (meters).")
@click.option("--camera-height", default=1800.0, help="Camera height above terrain (meters).")
@click.option("--terrain-exaggeration", default=1.5, help="Terrain height multiplier.")
@click.option("--tile-zoom", default=16, help="Map tile zoom level (higher=sharper, more tiles).")
@click.option("--labels/--no-labels", default=True,
              help="Enable/disable text overlay labels (road names, elevation, distance).")
@click.option("--intro/--no-intro", default=True,
              help="Enable/disable animated intro sequence.")
@click.option("--outro/--no-outro", default=True,
              help="Enable/disable outro stats card.")
@click.option("--format", "video_format",
              type=click.Choice(["instagram", "youtube", "tiktok", "square"]),
              default=None, help="Output format preset (overrides --width/--height).")
def main(
    gpx_file: str,
    output: str,
    width: int,
    height: int,
    fps: int | None,
    duration: int,
    quality: str,
    camera_distance: float,
    camera_height: float,
    terrain_exaggeration: float,
    tile_zoom: int,
    labels: bool,
    intro: bool,
    outro: bool,
    video_format: str | None,
) -> None:
    """Generate a 3D flyover video from a GPX file."""
    # Pre-flight: check FFmpeg
    if not shutil.which("ffmpeg"):
        raise click.UsageError(
            "FFmpeg not found. Install it:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )

    # Apply format preset
    if video_format:
        fmt = FORMAT_PRESETS[video_format]
        width = fmt["width"]
        height = fmt["height"]

    preset = QUALITY_PRESETS[quality]
    effective_fps = fps if fps is not None else preset["fps"]

    click.echo(f"Parsing GPX file: {gpx_file}")
    points = parse_gpx(gpx_file)
    if len(points) < 10:
        raise click.UsageError(
            f"GPX file has only {len(points)} track points (need at least 10)."
        )
    click.echo(f"  Found {len(points)} track points")

    total_frames = effective_fps * duration
    click.echo(f"Interpolating route to {total_frames} frames ({effective_fps}fps x {duration}s)...")
    smooth_points = interpolate_route(points, total_frames)

    click.echo("Computing camera path...")
    bearings = compute_bearings(smooth_points)
    bearings = smooth_bearings(bearings)

    # Use route center as coordinate origin
    lats = [p.lat for p in smooth_points]
    lons = [p.lon for p in smooth_points]
    origin_lat = (min(lats) + max(lats)) / 2
    origin_lon = (min(lons) + max(lons)) / 2

    camera_frames = generate_camera_frames(
        smooth_points,
        bearings,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        camera_distance=camera_distance,
        camera_height=camera_height,
        terrain_exaggeration=terrain_exaggeration,
    )

    # Fetch nearby landmark labels
    place_labels = None
    if labels:
        from .geocode import fetch_place_labels

        landmark_bar = tqdm(total=3, unit="step", desc="Finding landmarks", leave=False)

        def on_landmarks(current: int, total: int) -> None:
            landmark_bar.total = total
            landmark_bar.n = current
            landmark_bar.refresh()

        place_labels = fetch_place_labels(
            points,
            progress_callback=on_landmarks,
        )
        landmark_bar.close()
        click.echo(f"  Found {len(place_labels)} landmarks")

    intro_frames = int(3.0 * effective_fps) if intro else 0
    outro_frames = int(3.0 * effective_fps) if outro else 0
    total_video_frames = total_frames + intro_frames + outro_frames
    extra_secs = (3 if intro else 0) + (3 if outro else 0)
    click.echo(
        f"Quality: {quality} | {total_video_frames} frames at {width}x{height}"
        + (f" ({duration}s + {extra_secs}s intro/outro)" if extra_secs else "")
    )

    render_bar = tqdm(total=total_video_frames, unit="frame", desc="Rendering")
    last_rendered = [0]

    def on_progress(current: int, total: int) -> None:
        delta = current - last_rendered[0]
        render_bar.update(delta)
        last_rendered[0] = current

    def on_terrain() -> None:
        click.echo("Building terrain mesh from SRTM elevation data...")

    tile_bar = tqdm(total=1, unit="tile", desc="Downloading map tiles", leave=False)

    def on_tiles(current: int, total: int) -> None:
        tile_bar.total = total
        tile_bar.n = current
        tile_bar.refresh()

    encoder = StreamingEncoder(
        output,
        width=width,
        height=height,
        fps=effective_fps,
        crf=preset["crf"],
        preset=preset["preset"],
    )
    renderer = TerrainRenderer(
        width=width,
        height=height,
        terrain_exaggeration=terrain_exaggeration,
    )
    renderer.render(
        smooth_points,
        camera_frames,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        encoder=encoder,
        progress_callback=on_progress,
        terrain_callback=on_terrain,
        tile_callback=on_tiles,
        tile_zoom=tile_zoom,
        tile_source="esri-satellite",
        place_labels=place_labels,
        original_points=points,
        intro=intro,
        outro=outro,
        fps=effective_fps,
    )
    tile_bar.close()
    encoder.finalize()
    render_bar.close()

    click.echo(f"Video saved to: {output}")


if __name__ == "__main__":
    main()
