"""
Transform video and audio by reducing fps, extracting videos, changing resolution, reducing bitrate, etc.
"""

from .audio_rate_subsampler import AudioRateSubsampler
from .caption_subsampler import CaptionSubsampler
from .clipping_subsampler import ClippingSubsampler, _get_seconds, _split_time_frame, Streams
from .cut_detection_subsampler import CutDetectionSubsampler
from .ffprobe_subsampler import FFProbeSubsampler
from .frame_subsampler import FrameSubsampler
from .noop_subsampler import NoOpSubsampler
from .optical_flow_subsampler import OpticalFlowSubsampler
from .resolution_subsampler import ResolutionSubsampler
from .whisper_subsampler import WhisperSubsampler

from .subsampler import Subsampler, Probe
