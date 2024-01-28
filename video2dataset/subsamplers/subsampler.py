"""Base subsampler and probe class"""
from abc import abstractmethod
from typing import Tuple, Optional

from video2dataset.types import Metadata, Error


class Subsampler:
    """Subsamples input and returns in same format (stream dict + metadata)"""

    @abstractmethod
    def __call__(self, streams, metadata):
        raise NotImplementedError("Subsampler should not be called")


class Probe:
    """Gathers metadata about a file"""

    @abstractmethod
    def __call__(self, video_filepath: str, metadata: Optional[Metadata] = None) -> Tuple[Metadata, Error]:
        raise NotImplementedError("Subsampler should not be called")
