"""
frame subsampler adjusts the fps of the videos to some constant value
"""
from typing import Tuple, List

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error, FFmpegStream


class AudioRateSubsampler(Subsampler):
    """
    Adjusts the sampling rate of the audio to the specified rate.
    Args:
        sample_rate (int): Target sample rate of the audio.
        encode_format (str): Format to encode in (i.e. m4a)
    """

    def __init__(self, sample_rate, encode_format, n_audio_channels=None):
        self.sample_rate = sample_rate
        self.encode_format = encode_format
        self.n_audio_channels = n_audio_channels

    def __call__(self, ffmpeg_stream: FFmpegStream, metadata: Metadata, tmpdir: str) -> Tuple[List[FFmpegStream], List[Metadata], Error]:
        ffmpeg_stream = (
            ffmpeg_stream
            .afilter("aresample", self.sample_rate)
            .afilter("aformat", **{"channel_layouts": "mono" if self.n_audio_channels == 1 else "stereo"})
        )
        return [ffmpeg_stream], [metadata], None
