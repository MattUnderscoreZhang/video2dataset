"""creates a subset of an existing dataset inside the sample dimension"""
from dataclasses import dataclass, field
import ffmpeg
from ffmpeg.nodes import Stream
import fsspec
import json
import numpy as np
import pyarrow as pa
import time
import traceback
from typing import List, Tuple, Any, Optional, Literal, cast
import webdataset as wds

from video2dataset.dataloader import get_video_dataset
from video2dataset.logger import CappedCounter, write_stats
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
from video2dataset.types import EncodeFormats, Streams


def get_subsamplers(config: dict, input_encode_formats: EncodeFormats) -> Tuple[List[Subsampler], EncodeFormats]:
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
        clipping_subsampler if (config["storage"]["captions_are_subtitles"] or cuts_are_clips) else NoOpSubsampler()
    )

    ffprobe_subsampler = None
    if "FFProbeSubsampler" in config["subsampling"] or need_keyframes:
        ffprobe_subsampler = FFProbeSubsampler(**config["subsampling"].get("FFProbeSubsampler", {"args": {}})["args"])
        ffprobe_subsampler.extract_keyframes |= need_keyframes

    video_subsamplers: List[Any] = []
    if "FrameSubsampler" in config["subsampling"]:
        video_subsamplers.append(FrameSubsampler(**config["subsampling"]["FrameSubsampler"]["args"]))
    if "ResolutionSubsampler" in config["subsampling"]:
        video_subsamplers.append(ResolutionSubsampler(**config["subsampling"]["ResolutionSubsampler"]["args"]))

    audio_subsamplers: List[Any] = []
    if "AudioRateSubsampler" in config["subsampling"]:
        audio_subsamplers.append(AudioRateSubsampler(**config["subsampling"]["AudioRateSubsampler"]["args"]))

    # calculate output encoding
    output_encode_formats = input_encode_formats.copy()
    if len(video_subsamplers) > 0:
        assert (
            len({s.encode_format for s in video_subsamplers}) == 1
        )  # assert that all video subsamplers have the same output format
        output_encode_formats["video"] = video_subsamplers[0].encode_format
    if len(audio_subsamplers) > 0:
        assert (
            len({s.encode_format for s in audio_subsamplers}) == 1
        )  # assert that all audio subsamplers have the same output format
        output_encode_formats["audio"] = audio_subsamplers[0].encode_format

    return [
        subsampler
        for subsampler in [
            ffprobe_subsampler,
            *video_subsamplers,
            *audio_subsamplers,
            cut_detection_subsampler,
            broadcast_subsampler,
        ]
        if subsampler is not None
    ], output_encode_formats


@dataclass
class ShardStatus:
    successes: int = 0
    failed_to_subsample: int = 0
    status_dict: CappedCounter = field(default_factory=CappedCounter)
    error_message: Optional[str] = None
    count: int = 0


class SubsetWorker:
    """The loader class reads the shards, then the selected data is chosen and writen by the writer"""

    def __init__(
        self,
        sample_writer_class,
        output_folder,
        encode_formats: EncodeFormats,
        config,
    ) -> None:
        self.sample_writer_class = sample_writer_class
        self.output_folder = output_folder
        self.config = config
        self.input_encode_formats = encode_formats
        self.subsamplers, self.output_encode_formats = get_subsamplers(config, encode_formats)

    def __call__(
        self,
        row,
    ):
        try:
            shard, shard_id = row
            self.process_shard(shard, shard_id)
            return (True, row)
        except Exception as err:  # pylint: disable=broad-except
            traceback.print_exc()
            print(f"shard {row[0]} failed with error {err}")
            return (False, row)

    def get_shard_processors(
        self,
        shard: str,
        shard_id: int,
    ):
        try:
            fs, shard_path = fsspec.core.url_to_fs(shard[: -len(".tar")] + ".parquet")
            with fs.open(shard_path, "rb") as f:
                df = pa.parquet.read_table(f)
                schema = df.schema
        except Exception:  # pylint: disable=broad-except
            fields = [
                pa.field("key", pa.string()),
                pa.field("status", pa.string()),
                pa.field("error_message", pa.string()),
            ]
            schema = pa.schema(fields)
        shard_sample_writer = self.sample_writer_class(
            shard_id,
            self.output_folder,
            True,  # save_caption
            self.config["storage"]["oom_shard_count"],
            schema,
            self.output_encode_formats,
        )
        shard_dataloader = get_video_dataset(
            urls=shard,
            batch_size=1,
            decoder_kwargs={},
            enforce_additional_keys=[],
            handler=wds.warn_and_continue,
        )
        return shard_sample_writer, shard_dataloader

    def process_shard(
        self,
        shard: str,
        shard_id: int,
    ):
        """Function to start an video processing in one process"""

        start_time = time.time()
        shard_sample_writer, shard_dataloader = self.get_shard_processors(shard, shard_id)
        shard_status = ShardStatus()

        for sample in shard_dataloader:
            shard_status.count += 1
            key = sample["__key__"]
            try:
                caption = sample.get("txt", b"").decode("utf-8")
                metadata = json.loads(sample.get("json", b"{}").decode("utf-8"))
                stream_node: Stream = ffmpeg.input(sample)
            except Exception as err:  # pylint: disable=broad-except
                traceback.print_exc()
                print(f"Sample {key} failed to download: {err}")
                return

            try:
                metadata["video_metadata"] = video_metadata

                for subsampler in self.subsamplers:
                    stream_node, metadata = subsampler(stream_node, metadata)
                stream_node.run(capture_stdout=True, capture_stderr=True)

                shard_status.successes += 1
                status = "success"
                shard_status.status_dict.increment(status)

                """
                if self.ffprobe_subsampler is not None:
                    streams, metadata, shard_status.error_message = self.ffprobe_subsampler(streams, metadata)
                    assert shard_status.error_message is None

                if self.config["storage"]["captions_are_subtitles"]:  # create clips
                    subtitles = metadata["yt_meta_dict"]["subtitles"]
                    metadata["clips"] = [[line_dict["start"], line_dict["end"]] for line_dict in subtitles]
                elif self.cut_detection_subsampler is not None:  # apply cut detection to get clips
                    streams, cuts, shard_status.error_message = self.cut_detection_subsampler(streams)
                    assert shard_status.error_message is None
                    metadata["cuts"] = cuts
                    assert cuts is not None
                    if self.cuts_are_clips:
                        metadata["clips"] = (np.array(cuts["cuts_original_fps"]) / cuts["original_fps"]).tolist()

                # 1 video -> many videos (either clipping or noop which does identity broadcasting)
                subsampled_streams, metas, shard_status.error_message = self.broadcast_subsampler(streams, metadata)
                if shard_status.error_message is not None:
                    metadata["clips"] = []
                    assert False

                for modality in list(subsampled_streams.keys()):
                    for modality_subsampler in self.modal_subsamplers[modality]:
                        subsampled_streams, metas, shard_status.error_message = modality_subsampler(
                            subsampled_streams, metas
                        )
                        assert shard_status.error_message is None

                subsampled_streams_list = [dict(zip(subsampled_streams, s)) for s in zip(*subsampled_streams.values())]
                if len(subsampled_streams_list) == 0:  # no audio or video, just write metadata
                    metadata["status"] = status
                    shard_sample_writer.write(
                        {},
                        key,
                        caption,
                        metadata,
                    )
                    continue
                for subsampled_streams, metadata in zip(subsampled_streams_list, metas):
                    metadata["status"] = status
                    text_caption = caption
                    if self.config["storage"]["captions_are_subtitles"]:
                        text_caption = metadata.get("clip_subtitles")[0]["lines"][0]
                    shard_sample_writer.write(
                        subsampled_streams,
                        metadata["key"],
                        text_caption,
                        metadata,
                    )
                """
            except Exception:  # pylint: disable=broad-except
                shard_status.failed_to_subsample += 1
                shard_status.status_dict.increment(shard_status.error_message)
                metadata["status"] = "failed_to_subsample"
                metadata["error_message"] = shard_status.error_message
                shard_sample_writer.write(
                    {},
                    key,
                    caption,
                    metadata,
                )

        shard_sample_writer.close()
        end_time = time.time()

        write_stats(
            self.output_folder,
            shard_id,
            shard_status.count,
            shard_status.successes,
            0,  # failed to download
            shard_status.failed_to_subsample,
            0,  # bytes downloaded
            start_time,
            end_time,
            shard_status.status_dict,
            self.config["storage"]["oom_shard_count"],
        )
