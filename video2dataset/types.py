"""Type definitions for video2dataset."""
from ffmpeg.nodes import FilterableStream
from typing import TypedDict, Optional


class EncodeFormats(TypedDict, total=False):
    video: str
    audio: str


class ByteStreams(TypedDict, total=False):
    video: bytes
    audio: bytes


# TODO: make more structured
Metadata = dict


Error = Optional[str]


# this is here because ffmpeg objects aren't type annotated correctly
class FFmpegStream(FilterableStream):
    def filter(self, *args, **kwargs) -> FFmpegStream:
        ...

    def afilter(self, *args, **kwargs) -> FFmpegStream:
        ...

    def output(self, *args, **kwargs) -> FFmpegStream:
        ...

    def run(self, *args, **kwargs) -> None:
        ...
