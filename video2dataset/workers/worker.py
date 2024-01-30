"""Standard worker for video2dataset."""
from dataclasses import dataclass, field
import ffmpeg
import numpy as np
import os
import tempfile
from typing import Any, List, Tuple, Optional, Literal, cast
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
    Subsampler,
)
from video2dataset.types import EncodeFormats, Streams, Metadata, TempFilepaths


@dataclass
class Subsamplers:
    """Subsamplers used in processing"""

    ffprobe_subsampler: Optional[FFProbeSubsampler] = None
    modal_subsamplers: dict = field(default_factory=dict)
    cut_detection_subsampler: Optional[CutDetectionSubsampler] = None
    cuts_are_clips: bool = False
    broadcast_subsampler: Subsampler = field(default_factory=NoOpSubsampler)


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

    broadcast_subsampler = (
        clipping_subsampler
        if (do_clipping or config["storage"]["captions_are_subtitles"] or cuts_are_clips)
        else NoOpSubsampler()
    )

    ffprobe_subsampler = None
    if "FFProbeSubsampler" in config["subsampling"] or need_keyframes:
        ffprobe_subsampler = FFProbeSubsampler(**config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"])
        ffprobe_subsampler.extract_keyframes |= need_keyframes

    video_subsamplers: List[Any] = []
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
            broadcast_subsampler=broadcast_subsampler,
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
    streams: Streams,
    key: str,
    caption: str,
    metadata: Metadata,
    captions_are_subtitles: bool,
    shard_sample_writer: Any,  # TODO: type correctly
):
    """Process a single video"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # save temp stream dumps
            filepaths: TempFilepaths = {}
            for modality in streams:
                modality = cast(Literal["video", "audio"], modality)
                filepaths[modality] = []
                for stream in streams[modality]:
                    stream_uuid = str(uuid.uuid4())
                    temp_filepath = os.path.join(tmpdir, stream_uuid)
                    with open(temp_filepath, "wb") as f:
                        f.write(stream)
                    filepaths[modality].append(temp_filepath)

            # add info to video metadata about keyframes and cuts
            # this is pre-broadcast, so there should only be one video
            assert "video" in filepaths
            assert len(filepaths["video"]) == 1
            video_filepath = filepaths["video"][0]
            metadata = extract_video_metadata(
                subsamplers=subsamplers,
                shard_status=shard_status,
                metadata=metadata,
                video_filepath=video_filepath,
                captions_are_subtitles=captions_are_subtitles,
            )

            # 1 video -> many videos (either clipping or noop which does identity broadcasting)
            subsample_filepaths, subsample_metadatas, shard_status.error_message = subsamplers.broadcast_subsampler(filepaths, metadata)
            if shard_status.error_message is not None:
                metadata["clips"] = []
                assert False

            # create ffmpeg process for each file, then run to generate output file
            for modality in list(subsample_filepaths.keys()):
                for subsample_filepath, subsample_metadata in zip(subsample_filepaths[modality], subsample_metadatas):
                    ffmpeg_stream = ffmpeg.input(subsample_filepath)
                    for modality_subsampler in subsamplers.modal_subsamplers[modality]:
                        ffmpeg_stream, subsample_metadata, shard_status.error_message = modality_subsampler(
                            ffmpeg_stream, subsample_metadata, tmpdir,
                        )
                        assert shard_status.error_message is None
                    ffmpeg_stream.run(capture_stdout=True, quiet=True)

            shard_status.successes += 1
            shard_status.status_dict.increment("success")

            subsample_filepaths_list = [dict(zip(subsample_filepaths, s)) for s in zip(*subsample_filepaths.values())]
            if len(subsample_filepaths_list) == 0:  # no audio or video, just write metadata
                metadata["status"] = "success"
                shard_sample_writer.write(
                    {},
                    key,
                    caption,
                    metadata,
                )
                return
            for subsample_filepaths, subsample_metadata in zip(subsample_filepaths_list, subsample_metadatas):
                subsample_metadata["status"] = "success"
                text_caption = caption
                if captions_are_subtitles:
                    clip_subtitles = subsample_metadata.get("clip_subtitles")
                    first_clip_subtitles = clip_subtitles[0] if clip_subtitles else None
                    subtitle_lines = first_clip_subtitles["lines"] if first_clip_subtitles else None
                    text_caption = subtitle_lines[0] if subtitle_lines else text_caption
                shard_sample_writer.write(
                    # TODO: read filepath and extract stream
                    subsample_stream,
                    subsample_metadata["key"],
                    text_caption,
                    subsample_metadata,
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
