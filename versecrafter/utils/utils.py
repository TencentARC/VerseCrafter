# VerseCrafter-specific utility overrides.
#
# Thin wrappers around the upstream VideoX-Fun utilities that add VerseCrafter
# behavior without modifying the pristine upstream submodule.
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
videox_fun_path = os.path.join(project_root, "third_party/VideoX-Fun")
if videox_fun_path not in sys.path:
    sys.path.insert(0, videox_fun_path)

from videox_fun.utils.utils import (
    get_video_to_video_latent as _upstream_get_video_to_video_latent,
)


def get_video_to_video_latent(*args, **kwargs):
    """VerseCrafter variant of ``get_video_to_video_latent``.

    Identical to upstream, except that when no reference image is provided the
    first frame of the input video is used as the reference image (matching the
    behavior baked into the original VerseCrafter fork).
    """
    input_video, input_video_mask, ref_image, clip_image = _upstream_get_video_to_video_latent(
        *args, **kwargs
    )
    if ref_image is None:
        ref_image = input_video[:, :, :1]
    return input_video, input_video_mask, ref_image, clip_image
