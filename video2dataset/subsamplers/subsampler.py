"""Base subsampler and probe class"""
from abc import abstractmethod
from typing import Tuple, List, Optional

from video2dataset.types import Error, FFmpegStream, Metadata


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    @abstractmethod
    def __call__(self, ffmpeg_stream: FFmpegStream, metadata: Metadata, tmpdir: str) -> Tuple[List[FFmpegStream], List[Metadata], Error]:
        raise NotImplementedError("Subsampler should not be called")


class Probe:
    """Gathers metadata about a file"""

    @abstractmethod
    def __call__(self, video_filepath: str, metadata: Optional[Metadata] = None) -> Tuple[Metadata, Error]:
        raise NotImplementedError("Subsampler should not be called")
