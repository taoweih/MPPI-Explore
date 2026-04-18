"""Utilities for recording MuJoCo videos using FFmpeg."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from typing import Optional


class VideoRecorder:
    """Simple FFmpeg-backed video recorder for raw RGB frames."""

    def __init__(
        self,
        output_dir: str,
        width: int = 720,
        height: int = 480,
        fps: float = 30.0,
        crf: int = 2,
        preset: str = "slow",
    ) -> None:
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = int(crf)
        self.preset = preset

        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.video_path: Optional[str] = None
        self.is_recording = False

    def start(self) -> bool:
        """Start recording and open the FFmpeg stdin pipe."""
        if self.is_recording:
            return True

        os.makedirs(self.output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(
            self.output_dir, f"simulation_{timestamp}.mp4"
        )

        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Warning: FFmpeg not found. Video recording disabled.")
            self.is_recording = False
            return False

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{self.width}x{self.height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-vcodec",
            "h264",
            "-crf",
            str(self.crf),
            "-preset",
            self.preset,
            "-movflags",
            "+faststart",
            "-pix_fmt",
            "yuv420p",
            "-loglevel",
            "error",
            self.video_path,
        ]

        self.ffmpeg_process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.is_recording = True
        print(f"Recording video to {self.video_path}")
        return True

    def add_frame(self, frame: bytes) -> bool:
        """Append one RGB frame."""
        if (
            not self.is_recording
            or self.ffmpeg_process is None
            or self.ffmpeg_process.stdin is None
        ):
            return False

        try:
            self.ffmpeg_process.stdin.write(frame)
            return True
        except (BrokenPipeError, IOError):
            self.is_recording = False
            return False

    def stop(self) -> bool:
        """Finalize recording and close FFmpeg process."""
        if not self.is_recording or self.ffmpeg_process is None:
            return False

        try:
            if self.ffmpeg_process.stdin is not None:
                self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.is_recording = False
            print(f"Video saved to {self.video_path}")
            return True
        except (subprocess.TimeoutExpired, BrokenPipeError, IOError):
            try:
                self.ffmpeg_process.terminate()
            except OSError:
                pass
            self.is_recording = False
            return False
