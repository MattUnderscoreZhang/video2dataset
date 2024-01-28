"""
clipping subsampler turns full videos into clips of videos according to clip_col
"""
import copy
import datetime
import ffmpeg
import glob
import os
import tempfile
from typing import Any, List, Tuple, Dict, Literal, Union, Optional, cast

from video2dataset.subsamplers.subsampler import Subsampler
from video2dataset.types import EncodeFormats, Streams, Metadata, Error


################################
# HELPER FUNCTIONS AND CLASSES #
################################


ClipSpan = List[float]  # [start, end]


def _convert_time_str_to_float(t: Union[str, float]) -> float:
    """Converts time from string to seconds"""
    if not isinstance(t, str):
        return float(t)  # already seconds
    time_format = "%H:%M:%S.%f"  # TODO: maybe parameterize this?
    t_obj = datetime.datetime.strptime(t, time_format).time()
    return t_obj.second + t_obj.microsecond / 1e6 + t_obj.minute * 60 + t_obj.hour * 3600


def _convert_time_float_to_str(t_sec: float) -> str:
    """Converts time from seconds to string"""
    hour = int(t_sec // 3600)
    minute = int((t_sec // 60) % 60)
    second = int(t_sec % 60)
    # Use round to solve machine error problem (e.g. t_sec=13.6)
    microsecond = round((t_sec - int(t_sec)) * 1000)
    return f"{hour:02d}:{minute:02d}:{second:02d}.{microsecond:03d}"


#############################
# FIND CLIP SPLITTING TIMES #
#############################


def _shorten_clip_to_keyframes(clip_span: ClipSpan, keyframes: List[float]) -> Optional[ClipSpan]:
    """Shorten clip to span from first to last keyframe"""
    start, end = clip_span
    keyframes_in_range = [k for k in keyframes if start <= k <= end]
    if keyframes_in_range:
        adjusted_start = min(keyframes_in_range)
        adjusted_end = max(keyframes_in_range)
        if adjusted_start != adjusted_end:
            return [adjusted_start, adjusted_end]
    return None


def _split_clip_into_segments(s: float, e: float, min_length: float, max_length: float) -> List[ClipSpan]:
    """Splits clip into as many clips of max length as possible, followed by up to one clip of greater than min length"""
    time_d = e - s
    n_full_clips = int(time_d // max_length)
    clip_spans = [[s + i * max_length, s + (i + 1) * max_length] for i in range(n_full_clips)] + (
        [[s + (n_full_clips) * max_length, e]] if time_d % max_length > min_length else []
    )
    return clip_spans


def _combine_clip_splitting_data(clip_spans: List[ClipSpan]) -> Tuple[str, List[int]]:
    """
    Collates clip spans into a single string for ffmpeg and a list of clip idxs.
    e.g. '0.0,1.0,3.5,5.0' -> [0, 2].
    This example would indicate valid clips from [0.0, 1.0] and [3.5, 5.0].
    """
    clip_times = []
    clip_idxs = []
    e_prev = 0.0
    clip_idx = 0

    for s, e in clip_spans:
        if s == e_prev:  # clip starts where last one left off
            clip_times += [e]
            clip_idxs.append(clip_idx)
            clip_idx += 1
        else:  # next clip skips over some time
            clip_times += [s, e]
            clip_idxs.append(clip_idx + 1)
            clip_idx += 2
        e_prev = e

    clip_times_str = ",".join([str(time) for time in clip_times])
    return clip_times_str, clip_idxs


def _get_clip_splitting_data(
    original_clip_spans: List[ClipSpan],
    keyframe_timestamps: Union[List[float], None],
    min_length: float,
    max_length: float,
    max_length_strategy: str,
) -> Tuple[List[ClipSpan], str, List[int]]:
    """
    Splits clips into segments of the requested lengths, using keyframes if available.
    """
    shortened_clip_spans = (
        [
            shortened_clip
            for clip_span in original_clip_spans
            if (shortened_clip := _shorten_clip_to_keyframes(clip_span, keyframe_timestamps)) is not None
        ]
        if keyframe_timestamps
        else original_clip_spans
    )

    all_split_clip_spans = []
    for s, e in shortened_clip_spans:
        split_clip_spans = _split_clip_into_segments(s, e, min_length, max_length)
        if max_length_strategy == "first":
            split_clip_spans = split_clip_spans[:1]
        all_split_clip_spans += split_clip_spans

    clip_times, clip_idxs = _combine_clip_splitting_data(all_split_clip_spans)

    return all_split_clip_spans, clip_times, clip_idxs


####################
# EXTRACT METADATA #
####################


def _get_subtitles(clip_span: ClipSpan, meta_clip: Metadata) -> List[Metadata]:
    """Extracts subtitles and groups them by language"""
    clip_subtitles: List[dict] = []
    s_c, e_c = clip_span
    for lang_id, (lang, subtitles) in enumerate(meta_clip["yt_meta_dict"]["subtitles"].items()):
        idx = 0
        for line in subtitles:
            line_dict = {lang: line["lines"]}
            s, e = _convert_time_str_to_float(line["start"]), _convert_time_str_to_float(line["end"])
            if max(s_c, s) < min(e_c, e):
                if lang_id != 0:
                    clip_subtitles[idx]["lines"].update(line_dict)
                    idx += 1
                else:
                    temp_line = copy.deepcopy(line)
                    temp_line["lines"] = line_dict
                    clip_subtitles.append(temp_line)
            elif s > e_c:
                break
    return clip_subtitles


def _get_clip_metadatas(
    clip_spans: List[ClipSpan],
    clip_idxs: List[int],
    metadata: Metadata,
    oom_clip_count: int,
    strtime_formatting: bool,
) -> List[Metadata]:
    """Gets metadata for each clip"""
    metadata_clips = []
    for clip_id, (clip_span, _) in enumerate(zip(clip_spans, clip_idxs)):
        clip_key = "{clip_id:0{oom_clip_count}d}".format(  # pylint: disable=consider-using-f-string
            clip_id=clip_id, oom_clip_count=oom_clip_count
        )
        meta_clip = copy.deepcopy(metadata)
        # set the timeframe of this clip
        if strtime_formatting:
            #  Keep clip_spans in the original format to be compatible with the data schema.
            meta_clip["clips"] = [(_convert_time_float_to_str(clip_span[0]), _convert_time_float_to_str(clip_span[1]))]
        else:
            meta_clip["clips"] = [clip_span]
        meta_clip["key"] = f"{meta_clip['key']}_{clip_key}"

        yt_md_dict = meta_clip.get("yt_meta_dict", {})
        if (yt_md_dict is not None) and (yt_md_dict.get("subtitles", None) is not None):
            meta_clip["clip_subtitles"] = _get_subtitles(clip_span, meta_clip)
        metadata_clips.append(meta_clip)

    # remove redundant metadata from clips after the first
    for m_clips in metadata_clips[1:]:
        m_clips["yt_meta_dict"] = {}

    return metadata_clips


###############
# SPLIT CLIPS #
###############


def _process_stream(
    tmpdir: Any,  # BytesPath
    stream_bytes: bytes,
    encode_format: str,
    ffmpeg_kwargs: dict,
) -> List[str]:
    """Processes a stream into clips using ffmpeg"""
    # TODO: we need to put the extension into the metadata
    # TODO: This can be done better using pipes I just don't feel like sinking too much time into this rn
    with open(os.path.join(tmpdir, f"input.{encode_format}"), "wb") as f:
        f.write(stream_bytes)
    try:
        (
            ffmpeg.input(f"{tmpdir}/input.{encode_format}")
            .output(f"{tmpdir}/clip_%d.{encode_format}", **ffmpeg_kwargs)
            .run(capture_stdout=True, quiet=True)
        )
    except Exception as err:  # pylint: disable=broad-except
        raise err
    stream_clips = glob.glob(f"{tmpdir}/clip*.{encode_format}")
    stream_clips.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return stream_clips


def _get_clips(
    streams: Streams,
    encode_formats: EncodeFormats,
    precision: str,
    clip_times: str,
    clip_idxs: List[int],
) -> Streams:
    """Gets clips from streams"""
    ffmpeg_kwargs = {
        "map": 0,
        "f": "segment",
        "segment_times": clip_times,
        "reset_timestamps": 1,
    }
    if precision == "exact":
        ffmpeg_kwargs["force_key_frames"] = clip_times
    else:
        ffmpeg_kwargs["c"] = "copy"

    clips: Streams = {}
    for k in streams.keys():
        k = cast(Literal["audio", "video"], k)
        with tempfile.TemporaryDirectory() as tmpdir:
            stream_bytes = streams[k][0]  # pre-broadcast so only one
            if stream_bytes is None:
                continue
            try:
                stream_clips = _process_stream(
                    tmpdir=tmpdir,
                    stream_bytes=stream_bytes,
                    encode_format=encode_formats[k],
                    ffmpeg_kwargs=ffmpeg_kwargs,
                )
            except Exception as err:  # pylint: disable=broad-except
                raise err

            clips[k] = []
            for clip_idx in clip_idxs:
                with open(stream_clips[clip_idx], "rb") as vid_f:
                    clip_bytes = vid_f.read()
                    clips[k].append(clip_bytes)

    return clips


class ClippingSubsampler(Subsampler):
    """
    Cuts videos up into segments according to the 'clips' metadata

    Parameters:
    oom_clip_count: int
        The number of orders of magnitude for clip count, used for formatting clip keys.
    encode_formats: dict
        A dictionary mapping stream keys to their corresponding file extensions, e.g., {"video": "mp4", "audio": "mp3"}.
    min_length: float optional (default=0.0)
        Minimum length in seconds of a clip. Below this the subsampler will reject the clips
    max_length: float optional (default=999999.0)
        Maximum clip length, if exceeded resolve according to max_length_strategy
    max_length_strategy: str optional (defaul="all")
        "all" - cut up long clip into as many clips of max_length as possible
        "first" - take the first max_length clip from the long clip
    precision: str, optional (default="low")
        "low" - splits can be imprecise in any direction
        "keyframe_adjusted" - translates cuts into the vocab of existing keyframes (a good middlepoint)
            useful if you need to do fast clipping but information can't cross cut boundries
        "exact" - keyframes are inserted to get exact splits (warning, slow)

    expects:
    - clips to be sorted in increasing order and non-overlapping
    - time to be in the format "%H:%M:%S.%f", or a number representing the second of the timestamp
    """

    def __init__(
        self,
        oom_clip_count: int,
        encode_formats: EncodeFormats,
        min_length: float = 0.0,
        max_length: float = 999999.0,
        max_length_strategy: Literal["all", "first"] = "all",
        precision: Literal["low", "keyframe_adjusted", "exact"] = "low",
    ):
        assert max_length_strategy in ["all", "first"]
        assert precision in ["exact", "low", "keyframe_adjusted"]
        self.oom_clip_count = oom_clip_count
        self.encode_formats = encode_formats
        self.min_length = min_length
        self.max_length = max_length
        self.max_length_strategy = max_length_strategy
        self.precision = precision

    def __call__(self, streams: Streams, metadata: Metadata) -> Tuple[Streams, List[Metadata], Error]:
        original_clip_spans = metadata.pop("clips")

        clip_spans, clip_times, clip_idxs = _get_clip_splitting_data(
            original_clip_spans=[
                [_convert_time_str_to_float(s), _convert_time_str_to_float(e)]
                for [s, e] in original_clip_spans
            ],
            keyframe_timestamps=(
                # TODO: make it so if keyframe timestamps not present, get it yourself
                metadata["video_metadata"].pop("keyframe_timestamps")
                if self.precision == "keyframe_adjusted"
                else None
            ),
            min_length=self.min_length,
            max_length=self.max_length,
            max_length_strategy=self.max_length_strategy,
        )
        if len(clip_spans) == 0:
            return {}, [], f"Video had no clip_spans longer than {self.min_length}"

        try:
            clip_metadatas = _get_clip_metadatas(
                clip_spans=clip_spans,
                clip_idxs=clip_idxs,
                metadata=metadata,
                oom_clip_count=self.oom_clip_count,
                strtime_formatting=isinstance(original_clip_spans[0][0], str),
            )
            clips = _get_clips(
                streams=streams,
                encode_formats=self.encode_formats,
                precision=self.precision,
                clip_times=clip_times,
                clip_idxs=clip_idxs,
            )
        except Exception as err:  # pylint: disable=broad-except
            return {}, [], str(err)

        return clips, clip_metadatas, None
