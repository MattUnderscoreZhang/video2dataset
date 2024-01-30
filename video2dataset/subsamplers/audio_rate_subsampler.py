"""
frame subsampler adjusts the fps of the videos to some constant value
"""
from typing import Tuple

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

    def __call__(self, ffmpeg_streams: List[FFmpegStream], metadatas: List[Metadata]) -> Tuple[List[FFmpegStream], List[Metadata], Error]:
        ext = self.encode_format
        # TODO: for now assuming m4a, change this
        ffmpeg_args = {"ar": str(self.sample_rate), "f": ext}
        if self.n_audio_channels is not None:
            ffmpeg_args["ac"] = str(self.n_audio_channels)
        ffmpeg_stream = ffmpeg_stream.output(f"{tmpdir}/output.{ext}", **ffmpeg_args)
        return ffmpeg_stream, metadata, None
