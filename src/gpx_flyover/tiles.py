"""Map tile fetcher: download and stitch map tiles for terrain texture."""

import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

import requests
from PIL import Image

Image.MAX_IMAGE_PIXELS = 300_000_000  # Allow large stitched tile images (zoom 16+)

TILE_SIZE = 256
USER_AGENT = "gpx-flyover/0.2.0 (https://github.com/gpx-flyover)"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "gpx-flyover" / "tiles"

TILE_SOURCES = {
    "esri-satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}


def lat_lon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    """Convert latitude/longitude to OSM tile coordinates."""
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_to_lat_lon(x: int, y: int, zoom: int) -> tuple[float, float]:
    """Convert tile coordinates to lat/lon (northwest corner of tile)."""
    n = 2 ** zoom
    lon = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)
    return lat, lon


def _download_tile(
    z: int, x: int, y: int, cache_dir: Path, session: requests.Session,
    tile_source: str = "esri-satellite",
) -> tuple[int, int, Optional[Image.Image]]:
    """Download a single tile, using cache if available."""
    cache_path = cache_dir / tile_source / str(z) / str(x) / f"{y}.png"

    if cache_path.exists():
        try:
            return x, y, Image.open(cache_path).convert("RGB")
        except Exception as e:
            logger.debug("Corrupt tile cache %s: %s", cache_path, e)

    url = TILE_SOURCES[tile_source].format(z=z, x=x, y=y)
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(resp.content)
            return x, y, img
    except Exception as e:
        logger.warning("Failed to download tile %d/%d/%d: %s", z, x, y, e)

    # Return a grey placeholder on failure
    return x, y, Image.new("RGB", (TILE_SIZE, TILE_SIZE), (200, 200, 200))


class TileResult:
    """Stitched tile image with its actual geographic bounds."""
    def __init__(self, image: Image.Image,
                 nw_lat: float, nw_lon: float,
                 se_lat: float, se_lon: float):
        self.image = image
        self.nw_lat = nw_lat  # north-west corner
        self.nw_lon = nw_lon
        self.se_lat = se_lat  # south-east corner
        self.se_lon = se_lon


def fetch_terrain_texture(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    zoom: int = 13,
    cache_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tile_source: str = "esri-satellite",
) -> TileResult:
    """Fetch map tiles for the bounding box and stitch into one texture image."""
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    # Convert bounds to tile range
    min_tx, min_ty = lat_lon_to_tile(max_lat, min_lon, zoom)  # NW corner
    max_tx, max_ty = lat_lon_to_tile(min_lat, max_lon, zoom)  # SE corner

    # Actual geographic bounds of the tile grid
    nw_lat, nw_lon = tile_to_lat_lon(min_tx, min_ty, zoom)
    se_lat, se_lon = tile_to_lat_lon(max_tx + 1, max_ty + 1, zoom)

    tiles_x = max_tx - min_tx + 1
    tiles_y = max_ty - min_ty + 1
    total_tiles = tiles_x * tiles_y

    MAX_TILES = 5000
    if total_tiles > MAX_TILES:
        raise ValueError(
            f"Tile request too large: {total_tiles} tiles at zoom {zoom}. "
            f"Reduce --tile-zoom or use a smaller GPX area (max {MAX_TILES})."
        )

    # Stitched image
    width = tiles_x * TILE_SIZE
    height = tiles_y * TILE_SIZE
    combined = Image.new("RGB", (width, height))

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for tx in range(min_tx, max_tx + 1):
            for ty in range(min_ty, max_ty + 1):
                fut = pool.submit(
                    _download_tile, zoom, tx, ty, cache_dir, session, tile_source,
                )
                futures[fut] = (tx, ty)

        for fut in as_completed(futures):
            tx, ty, img = fut.result()
            if img:
                paste_x = (tx - min_tx) * TILE_SIZE
                paste_y = (ty - min_ty) * TILE_SIZE
                combined.paste(img, (paste_x, paste_y))
            done += 1
            if progress_callback:
                progress_callback(done, total_tiles)

    return TileResult(combined, nw_lat, nw_lon, se_lat, se_lon)
