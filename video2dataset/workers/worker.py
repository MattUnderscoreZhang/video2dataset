"""Standard worker for video2dataset."""
from dataclasses import dataclass, field
import ffmpeg
import glob
import numpy as np
import os
import tempfile
from typing import Any, List, Dict, Tuple, Optional
import uuid

from video2dataset.logger import CappedCounter
from video2dataset.subsamplers import (
    ClippingSubsampler,
    CutDetectionSubsampler,
    FrameSubsampler,
    FFProbeSubsampler,
    NoOpSubsampler,
    ResolutionSubsampler,
    AudioRateSubsampler,
)
from video2dataset.types import EncodeFormats, ByteStreams, Metadata, FFmpegStream


@dataclass
class Subsamplers:
    """Subsamplers used in processing"""

    ffprobe_subsampler: Optional[FFProbeSubsampler] = None
    modal_subsamplers: dict = field(default_factory=dict)
    cut_detection_subsampler: Optional[CutDetectionSubsampler] = None
    cuts_are_clips: bool = False


def get_subsamplers(
    config: dict,
    input_encode_formats: EncodeFormats,
    do_clipping: bool = False,
) -> Tuple[Subsamplers, EncodeFormats]:
    """Initialize all subsamplers using config"""

    clipping_subsampler = ClippingSubsampler(
        oom_clip_count=5,
        encode_formats=input_encode_formats,
        **config["subsampling"].get("ClippingSubsampler", {"args": {}})["args"],
    )
    need_keyframes = clipping_subsampler.precision == "keyframe_adjusted"

    cut_detection_subsampler = None
    cuts_are_clips = False
    if "CutDetectionSubsampler" in config["subsampling"]:
        if "args" in config["subsampling"]["CutDetectionSubsampler"]:
            cut_detection_subsampler = CutDetectionSubsampler(**config["subsampling"]["CutDetectionSubsampler"]["args"])
        cuts_are_clips = config["subsampling"]["CutDetectionSubsampler"].get("cuts_are_clips", False)

    ffprobe_subsampler = None
    if "FFProbeSubsampler" in config["subsampling"] or need_keyframes:
        ffprobe_subsampler = FFProbeSubsampler(**config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"])
        ffprobe_subsampler.extract_keyframes |= need_keyframes

    broadcast_subsampler = (
        clipping_subsampler
        if (do_clipping or config["storage"]["captions_are_subtitles"] or cuts_are_clips)
        else NoOpSubsampler()
    )

    video_subsamplers: List[Any] = []
    video_subsamplers.append(broadcast_subsampler)
    if "ResolutionSubsampler" in config["subsampling"]:
        video_subsamplers.append(ResolutionSubsampler(**config["subsampling"]["ResolutionSubsampler"]["args"]))
    if "FrameSubsampler" in config["subsampling"]:
        video_subsamplers.append(FrameSubsampler(**config["subsampling"]["FrameSubsampler"]["args"]))

    audio_subsamplers: List[Any] = []
    if "AudioRateSubsampler" in config["subsampling"]:
        audio_subsamplers.append(AudioRateSubsampler(**config["subsampling"]["AudioRateSubsampler"]["args"]))

    modal_subsamplers = {"video": video_subsamplers, "audio": audio_subsamplers}

    # output encoding formats
    output_encode_formats = input_encode_formats.copy()
    if modal_subsamplers["audio"]:
        assert (
            len({s.encode_format for s in modal_subsamplers["audio"]}) == 1
        )  # assert that all audio subsamplers have the same output format
        output_encode_formats["audio"] = modal_subsamplers["audio"][0].encode_format
    if modal_subsamplers["video"]:
        assert (
            len({s.encode_format for s in modal_subsamplers["video"]}) == 1
        )  # assert that all video subsamplers have the same output format
        output_encode_formats["video"] = modal_subsamplers["video"][0].encode_format

    return (
        Subsamplers(
            ffprobe_subsampler=ffprobe_subsampler,
            modal_subsamplers=modal_subsamplers,
            cut_detection_subsampler=cut_detection_subsampler,
            cuts_are_clips=cuts_are_clips,
        ),
        output_encode_formats,
    )


@dataclass
class ShardStatus:
    """Shard processing status"""

    successes: int = 0
    failed: dict = field(
        default_factory=lambda: {
            "failed_to_download": 0,
            "failed_to_subsample": 0,
        }
    )
    status_dict: CappedCounter = field(default_factory=CappedCounter)
    error_message: Optional[str] = None
    count: int = 0
    bytes_downloaded: int = 0


def extract_video_metadata(
    subsamplers: Subsamplers,
    shard_status: ShardStatus,
    metadata: Metadata,
    video_filepath: str,
    captions_are_subtitles: bool,
):
    """Add additional metadata keys for video file"""

    if subsamplers.ffprobe_subsampler is not None:
        metadata, shard_status.error_message = subsamplers.ffprobe_subsampler(video_filepath, metadata)
        assert shard_status.error_message is None

    if captions_are_subtitles:  # create clips
        subtitles = metadata["yt_meta_dict"]["subtitles"]
        metadata["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
    elif subsamplers.cut_detection_subsampler is not None:  # apply cut detection to get clips
        metadata, shard_status.error_message = subsamplers.cut_detection_subsampler(video_filepath, metadata)
        assert shard_status.error_message is None
        cuts = metadata["cuts"]
        assert cuts is not None
        if subsamplers.cuts_are_clips:
            metadata["clips"] = (np.array(cuts["cuts_original_fps"]) / cuts["original_fps"]).tolist()

    return metadata


def process_sample(
    subsamplers: Subsamplers,
    shard_status: ShardStatus,
    byte_streams: ByteStreams,
    key: str,
    caption: str,
    metadata: Metadata,
    captions_are_subtitles: bool,
    shard_sample_writer: Any,  # TODO: type correctly
):
    """Process a single video"""
    # TODO: don't split video into different modalities before passing to this function
    # it's completely unnecessary and makes the code slower and more complex
    # you can extract and process audio and video portions of ffmpeg_stream on the fly as needed
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # temporarily dump streams to file
            # TODO: don't have the dataloader read these streams to memory in the first place
            filepaths: Dict[str, str] = {}
            for modality in byte_streams:
                stream_uuid = str(uuid.uuid4())
                temp_filepath = os.path.join(tmpdir, stream_uuid)
                with open(temp_filepath, "wb") as f:
                    f.write(byte_streams[modality])
                filepaths[modality] = temp_filepath

            # extract video metadata about keyframes and cuts
            if "video" in filepaths:
                metadata = extract_video_metadata(
                    subsamplers=subsamplers,
                    shard_status=shard_status,
                    metadata=metadata,
                    video_filepath=filepaths["video"],
                    captions_are_subtitles=captions_are_subtitles,
                )

            # create input ffmpeg streams and metadatas - start with single-element lists to represent one input
            ffmpeg_streams: Dict[str, List[FFmpegStream]] = {}
            for modality in filepaths:
                ffmpeg_streams[modality] = [ffmpeg.input(filepaths[modality])]
            metadatas = [metadata]

            # chain ffmpeg streams
            for modality in filepaths:
                # each subsampler operates on one input stream and can generate one or more output streams
                for modality_subsampler in subsamplers.modal_subsamplers[modality]:
                    # TODO: use a datastructure to combine stream modes and metadata, instead of matching list indices
                    output_ffmpeg_streams = []
                    output_metadatas = []
                    # TODO: don't use metadata keys to share information across subsamplers, it's crazy
                    for ffmpeg_stream, metadata in zip(ffmpeg_streams[modality], metadatas):
                        clip_ffmpeg_streams, clip_metadatas, shard_status.error_message = modality_subsampler(
                            ffmpeg_stream, metadata, tmpdir,
                        )
                        assert shard_status.error_message is None
                        output_ffmpeg_streams += clip_ffmpeg_streams
                        output_metadatas += clip_metadatas
                    ffmpeg_streams[modality] = output_ffmpeg_streams
                    metadatas = output_metadatas

            # run all streams and output bytestreams of all results
            output_byte_streams: ByteStreams = {}
            for modality in ffmpeg_streams:
                for ffmpeg_stream in ffmpeg_streams[modality]:
                    ffmpeg_stream.run(capture_stdout=True, quiet=True)
                output_filepaths = glob.glob(f"{tmpdir}/output*")
                output_byte_streams[modality] = []
                for output_filepath in output_filepaths:
                    with open(output_filepath, "rb") as f:
                        output_byte_streams[modality].append(f.read())

            # note success
            shard_status.successes += 1
            shard_status.status_dict.increment("success")

            # group modalities by index
            print("done", len(output_byte_streams["video"]))
            output_byte_streams_list: List[ByteStreams] = [
                dict(zip(output_byte_streams, values))
                for values in zip(*output_byte_streams.values())
            ]

            # write output dataset and update status
            if len(output_byte_streams_list) == 0:  # no audio or video, just write metadata
                metadata["status"] = "success"
                shard_sample_writer.write(
                    {},
                    key,
                    caption,
                    metadata,
                )
                return
            for output_byte_streams, metadata in zip(output_byte_streams_list, metadatas):
                metadata["status"] = "success"
                text_caption = caption
                if captions_are_subtitles:
                    clip_subtitles = metadata.get("clip_subtitles")
                    first_clip_subtitles = clip_subtitles[0] if clip_subtitles else None
                    subtitle_lines = first_clip_subtitles["lines"] if first_clip_subtitles else None
                    text_caption = subtitle_lines[0] if subtitle_lines else text_caption
                shard_sample_writer.write(
                    output_byte_streams,  # TODO: writer expects {modality: byte_stream}
                    metadata["key"],
                    text_caption,
                    metadata,
                )
    except Exception as err:  # pylint: disable=broad-except
        print(err)
        shard_status.failed["failed_to_subsample"] += 1
        shard_status.status_dict.increment(shard_status.error_message)
        metadata["status"] = "failed_to_subsample"
        metadata["error_message"] = shard_status.error_message
        shard_sample_writer.write(
            {},
            key,
            caption,
            metadata,
        )
