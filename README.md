# GPX Flyover

Generate 3D flyover videos from GPX cycling/hiking recordings. Renders terrain from SRTM elevation data with satellite imagery, and overlays route lines, rider markers, landmark labels, and stats.

## Quick Start

```bash
# 1. Install FFmpeg (required)
# macOS: brew install ffmpeg
# Ubuntu: sudo apt install ffmpeg

# 2. Install gpx-flyover
python -m venv .venv
source .venv/bin/activate
pip install .

# 3. Generate a video
gpx-flyover your-route.gpx output.mp4
```

## Usage

```bash
gpx-flyover INPUT.gpx OUTPUT.mp4 [OPTIONS]
```

### Quick Examples

```bash
# Fast preview (10 seconds)
gpx-flyover route.gpx preview.mp4 --quality fast --duration 10

# YouTube landscape format
gpx-flyover route.gpx youtube.mp4 --format youtube

# Instagram Reel
gpx-flyover route.gpx reel.mp4 --format instagram --duration 30

# High quality, close camera
gpx-flyover route.gpx hq.mp4 --quality high --camera-distance 1500 --camera-height 1000
```

## CLI Options

| Option | Default | Description |
|---|---|---|
| `--width` / `--height` | 1080 / 1920 | Video dimensions (pixels) |
| `--format` | — | Output preset: `instagram` (1080x1920), `youtube` (1920x1080), `tiktok` (1080x1920), `square` (1080x1080). Overrides width/height. |
| `--fps` | from quality | Override frames per second |
| `--duration` | 60 | Video length in seconds |
| `--quality` | medium | `fast` (15fps), `medium` (24fps), `high` (30fps) |
| `--camera-distance` | 3000 | Horizontal distance behind rider (meters) |
| `--camera-height` | 1800 | Height above terrain (meters) |
| `--terrain-exaggeration` | 1.5 | Elevation multiplier (1.0 = real scale) |
| `--tile-zoom` | 16 | Map tile detail level (max practical: 17-18) |
| `--labels` / `--no-labels` | on | Floating landmark labels and stats overlay |
| `--intro` / `--no-intro` | on | Animated intro sequence |
| `--outro` / `--no-outro` | on | Outro stats card |

## Features

- 3D terrain from SRTM elevation data with satellite imagery
- Route line colored by speed (red = slow, cyan = fast)
- Real-time elevation profile chart
- Live stats overlay (speed, distance, elevation gain)
- Floating landmark labels (nearby POIs via OpenStreetMap)
- Animated intro (zoom-in from bird's-eye) and outro (ride stats)

## Cloud Usage (GitHub Actions)

Generate videos without installing anything — just use GitHub Actions:

1. **Fork** this repository
2. **Upload** your GPX file to any file sharing service and get a direct download URL
3. Go to **Actions** → **Generate Video** → **Run workflow**
4. Paste the GPX download URL, choose duration/quality/format
5. When the workflow finishes (~5-10 min), download `gpx-flyover-output` from the **Artifacts** section

## Requirements

- Python 3.10+
- FFmpeg (must be in PATH)

## GPX Sources

Export GPX files from: Garmin Connect, Komoot, RideWithGPS, or any GPS device/app.

## License

MIT
