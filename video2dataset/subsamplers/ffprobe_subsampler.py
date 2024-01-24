"""extracts basic video compression metadata."""
from ffmpeg.nodes import Stream
import json
import os
import subprocess
import tempfile
from typing import Tuple

from video2dataset.types import Metadata


# TODO: figuer out why this is so slow (12 samples/s)
class FFProbe:
    """
    Extracts metadata from bytes.
    Args:
        extract_keyframes (bool): Whether to extract keyframe timestamps.
    """

    def __init__(self, extract_keyframes=False):
        self.extract_keyframes = extract_keyframes

    def __call__(self, filepath: str) -> Metadata:
        command = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            filepath,
        ]
        if self.extract_keyframes:
            command.extend(["-select_streams", "v:0", "-show_entries", "packet=pts_time,flags"])

        process = subprocess.run(command, capture_output=True, text=True, check=True)
        video_metadata = json.loads(process.stdout)
        if self.extract_keyframes:
            keyframe_info = [entry for entry in video_metadata["packets"] if "K" in entry.get("flags", "")]
            keyframe_timestamps = [float(entry["pts_time"]) for entry in keyframe_info]
            if "duration" in video_metadata["format"]:
                duration = float(video_metadata["format"]["duration"])
                keyframe_timestamps.append(duration)
            video_metadata["keyframe_timestamps"] = keyframe_timestamps
            video_metadata.pop("packets")  # Don't need it anymore

        return video_metadata
