"""
frame subsampler adjusts the fps of the videos to some constant value
"""
import copy
from typing import Tuple, List

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import Metadata, Error, FFmpegStream


class FrameSubsampler(Subsampler):
    """
    Adjusts the frame rate of the videos to the specified frame rate.
    Subsamples the frames of the video in terms of spacing and quantity (frame_rate, which ones etc.)
    Args:
        frame_rate (int): Target frame rate of the videos.
        downsample_method (str): determiens how to downsample the video frames:
            fps: decreases the framerate but sample remains a valid video
            first_frame: only use the first frame of a video of a video and output as image
            yt_subtitle: temporary special case where you want a frame at the beginning of each yt_subtitle
                         we will want to turn this into something like frame_timestamps and introduce
                         this as a fusing option with clipping_subsampler
        encode_format (str): Format to encode in (i.e. mp4)

    TODO: n_frame
    TODO: generalize interface, should be like (frame_rate, n_frames, sampler, output_format)
    # frame_rate - spacing
    # n_frames - quantity
    # sampler - from start, end, center out
    # output_format - save as video, or images
    """

    def __init__(self, frame_rate, downsample_method="fps", encode_format=None):
        self.frame_rate = frame_rate
        self.downsample_method = downsample_method
        if encode_format is None:
            encode_format = "mp4" if downsample_method == "fps" else "jpg"
        self.encode_format = encode_format

    def __call__(self, ffmpeg_stream: FFmpegStream, metadata: Metadata) -> Tuple[List[FFmpegStream], List[Metadata], Error]:
        if self.downsample_method == "fps":
            ffmpeg_stream = (
                ffmpeg_stream
                .filter("fps", fps=self.frame_rate)
                .output(f"{tmpdir}/output.{self.encode_format}", reset_timestamps=1)
            )
        elif "frame" in self.downsample_method:
            ffmpeg_stream = (
                ffmpeg_stream
                .filter("select", "eq(n,0)")
                .output(f"{tmpdir}/output.{self.encode_format}")
            )
        elif self.downsample_method == "yt_subtitle":
            try:
                subtitles = metadata["yt_meta_dict"]["subtitles"]
                frame_ffmpeg_streams: List[FFmpegStream] = []
                frame_metadatas: List[Metadata] = []
                for frame_id, frame_subtitles in enumerate(subtitles):
                    frame_metadata = copy.deepcopy(metadata)
                    frame_metadata["frame_time"] = frame_subtitles["start"]
                    frame_metadata["frame_subtitle"] = frame_subtitles["lines"]
                    frame_metadata["key"] = f"{frame_metadata['key']}_{frame_id:04d}"
                    frame_metadatas.append(frame_metadata)
                    frame_ffmpeg_streams.append(
                        ffmpeg_stream
                        .output(f"{tmpdir}/frame_{frame_id}.jpg", vframes=1, **{"q:v": 2})
                    )
                ffmpeg_stream = frame_ffmpeg_streams
                metadata = frame_metadatas
            except Exception as err:  # pylint: disable=broad-except
                return [], [], str(err)
        return [ffmpeg_stream], [metadata], None
