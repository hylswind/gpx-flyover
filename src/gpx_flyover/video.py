"""Video encoding: stream raw pixel frames to FFmpeg."""

import subprocess
import threading


class StreamingEncoder:
    """Streams raw RGB frames to FFmpeg via stdin pipe for constant memory usage."""

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: int = 24,
        crf: int = 23,
        preset: str = "medium",
    ):
        self.output_path = output_path
        self._stderr_chunks: list[bytes] = []
        self._stderr_size = 0
        self._MAX_STDERR = 1024 * 1024  # 1 MB cap
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",  # overwrite output
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-framerate", str(fps),
                "-i", "-",  # stdin
                "-c:v", "libx264",
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Drain stderr in a background thread to prevent pipe buffer deadlock
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        """Read stderr continuously so FFmpeg never blocks on a full pipe."""
        for chunk in iter(lambda: self.process.stderr.read(4096), b""):
            if self._stderr_size < self._MAX_STDERR:
                self._stderr_chunks.append(chunk)
                self._stderr_size += len(chunk)

    def write_frame(self, raw_bytes: bytes) -> None:
        """Write a single raw RGB frame to the encoder."""
        self.process.stdin.write(raw_bytes)

    def finalize(self) -> None:
        """Close input and wait for FFmpeg to finish encoding."""
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        self.process.wait()
        self._stderr_thread.join(timeout=5)
        if self.process.returncode != 0:
            stderr = b"".join(self._stderr_chunks).decode(errors="replace")
            raise RuntimeError(f"FFmpeg encoding failed:\n{stderr}")
