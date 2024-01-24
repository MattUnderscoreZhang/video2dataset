"""Base subsampler class"""
from abc import abstractmethod
from ffmpeg.nodes import Stream
from typing import Tuple

from video2dataset.types import Metadata


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    @abstractmethod
    def __call__(self, stream: Stream, metadata: Metadata) -> Tuple[Stream, Metadata]:
        raise NotImplementedError("Subsampler should not be called")
