# GPX Flyover — CLAUDE.md

## Project Overview

Generate 3D flyover videos from GPX cycling/hiking recordings. Renders terrain from SRTM elevation data with satellite imagery, and overlays route lines, rider markers, landmark labels, and stats.

## Quick Reference

```bash
# Setup
source .venv/bin/activate
pip install .                    # NOT pip install -e . (no setup.py)

# Run
gpx-flyover test.gpx output.mp4
gpx-flyover test.gpx output.mp4 --quality fast --duration 10   # quick test
gpx-flyover test.gpx output.mp4 --no-labels                    # no overlay labels
```

## Architecture

```
GPX File → parse → interpolate (cubic spline) → bearings → camera path
                                                              ↓
SRTM elevation    → terrain mesh (PyVista StructuredGrid 350×350)
Satellite tiles   → stitch → Mercator→equirect reproject → texture
                                                              ↓
                                          PyVista off-screen render loop:
                                            camera position → render → screenshot
                                            → PIL overlays (rider, labels, stats)
                                            → FFmpeg stdin (streaming H.264 encode)
                                                              ↓
                                                           MP4 video
```

## Source Files

| File | Purpose |
|---|---|
| `src/gpx_flyover/cli.py` | Click CLI entry point, quality presets, orchestration |
| `src/gpx_flyover/gpx_parser.py` | GPX parsing → `TrackPoint(lat, lon, elevation)` |
| `src/gpx_flyover/route.py` | Haversine, cubic spline interpolation, bearing math, Gaussian smoothing |
| `src/gpx_flyover/camera.py` | GPS→local 3D coords, camera frame generation (behind + above rider) |
| `src/gpx_flyover/tiles.py` | ESRI satellite tile download, disk cache, parallel stitching |
| `src/gpx_flyover/geocode.py` | Overpass API POI query, disk cache, `PlaceLabel` |
| `src/gpx_flyover/renderer.py` | PyVista 3D rendering, texture mapping, PIL overlays, label projection |
| `src/gpx_flyover/video.py` | FFmpeg subprocess streaming encoder (PNG→H.264) |

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--width` / `--height` | 1080 / 1920 | Video dimensions (portrait for mobile) |
| `--fps` | from preset | Override frames per second |
| `--duration` | 60 | Video length in seconds |
| `--quality` | medium | Preset: fast (15fps/crf23), medium (24fps/crf18), high (30fps/crf15) |
| `--camera-distance` | 3000 | Horizontal distance behind rider (meters) |
| `--camera-height` | 1800 | Height above terrain (meters) |
| `--terrain-exaggeration` | 1.5 | Elevation multiplier |
| `--tile-zoom` | 16 | Map tile detail (max practical: 17–18; 18+ may OOM) |
| `--labels` / `--no-labels` | on | Floating landmark labels + stats overlay |
| `--intro` / `--no-intro` | on | Animated intro sequence (3s) |
| `--outro` / `--no-outro` | on | Outro stats card (3s) |

## Critical Technical Notes

### PyVista Texture Y-Axis
**Do NOT manually flip textures** before `pv.numpy_to_texture()`. PyVista internally flips Y via `np.flip(image.swapaxes(0,1), axis=1)` to convert numpy (top-left origin) to VTK (bottom-left origin). An additional flip = inverted map.

### texture_map_to_plane
**Always specify explicit plane** with `origin`/`point_u`/`point_v`. Without it, VTK auto-detects via PCA on mesh vertices — with terrain exaggeration, the auto-detected plane tilts, causing 100s of meters of texture UV offset.

### World-to-Screen Projection
Use VTK's `SetWorldPoint()/WorldToDisplay()/GetDisplayPoint()` pipeline. PyVista has no `renderer.world_to_view()` method.

### PyVista Gotchas
- `pv.Light(light_type="ambient")` → ValueError. Use directional lights only.
- Texture power-of-2 resize: resize each dimension independently, not to a square.
- Terrain mesh uses `lighting=False` to prevent light wash-out on satellite imagery.

## External Services & Cache

| Service | Purpose | Cache Location |
|---|---|---|
| SRTM | Elevation data | Managed by `SRTM.py` library |
| ESRI World Imagery | Satellite tiles | `~/.cache/gpx-flyover/tiles/esri-satellite/{z}/{x}/{y}.png` |
| Overpass API | Nearby POI/landmark query | `~/.cache/gpx-flyover/overpass/{hash}.json` |
| Google Fonts | Noto Sans SC (CJK) | `~/.cache/gpx-flyover/fonts/NotoSansSC-Regular.ttf` |

**FFmpeg** must be installed and in PATH.

## Coordinate Systems

- **GPS**: `(lat, lon, elevation_m)` — WGS84
- **Local Cartesian**: `(x_east, y_north, z_up)` in meters, relative to route center origin. Conversion: `M_PER_DEG_LAT = 111,319`, longitude scaled by `cos(origin_lat)`.
- **Web Mercator**: For tile indexing and texture projection
- **VTK Screen**: `(display_x, display_y, depth)` — origin bottom-left; PIL is top-left (Y-flip needed)

## Test Data

- `test.gpx` / `test01.gpx`: 13,732 points, Zhonghe↔Lengshui Keng cycling route through Yangmingshan, Taipei
- `examples/sample.gpx`: 20 points, short Taipei route

## User Preferences

- **Before making any change that could degrade map quality, warn the user first.** Do not apply texture filtering, tile downscaling, resolution reduction, or CRF changes without explicit confirmation.
