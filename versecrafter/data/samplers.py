# VerseCrafter-specific batch samplers.
#
# These subclass the upstream VideoX-Fun samplers and only override __iter__ to
# use VerseCrafter's video-first defaults and metadata keys ('video' /
# 'qwen_prompt'). Everything else (constructors, attributes) is inherited from
# upstream so the upstream submodule stays pristine.
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
videox_fun_path = os.path.join(project_root, "third_party/VideoX-Fun")
if videox_fun_path not in sys.path:
    sys.path.insert(0, videox_fun_path)

import cv2

from videox_fun.data.bucket_sampler import get_image_size_without_loading
from videox_fun.data.bucket_sampler import (
    AspectRatioBatchImageVideoSampler as _UpstreamAspectRatioBatchImageVideoSampler,
)
from videox_fun.data.dataset_image_video import (
    ImageVideoSampler as _UpstreamImageVideoSampler,
)


class AspectRatioBatchImageVideoSampler(_UpstreamAspectRatioBatchImageVideoSampler):
    """VerseCrafter variant: defaults content type to 'video' and reads the
    VerseCrafter metadata keys ('video' / 'qwen_prompt') for the size fallback."""

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset[idx].get('type', 'video')
            if content_type == 'image':
                try:
                    image_dict = self.dataset[idx]

                    width, height = image_dict.get("width", None), image_dict.get("height", None)
                    if width is None or height is None:
                        image_id, name = image_dict['file_path'], image_dict['text']
                        if self.train_folder is None:
                            image_dir = image_id
                        else:
                            image_dir = os.path.join(self.train_folder, image_id)

                        width, height = get_image_size_without_loading(image_dir)

                        ratio = height / width  # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width  # self.dataset[idx]
                except Exception as e:
                    print(e, self.dataset[idx], "This item is error, please check it.")
                    continue
                # find the closest aspect ratio
                closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket['image'][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]
            else:
                try:
                    video_dict = self.dataset[idx]
                    width, height = video_dict.get("width", None), video_dict.get("height", None)

                    if width is None or height is None:
                        video_id, name = video_dict['video'], video_dict['qwen_prompt']
                        if self.train_folder is None:
                            video_dir = video_id
                        else:
                            video_dir = os.path.join(self.train_folder, video_id)
                        cap = cv2.VideoCapture(video_dir)

                        # 获取视频尺寸
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 浮点数转换为整数
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 浮点数转换为整数

                        ratio = height / width  # self.dataset[idx]
                    else:
                        height = int(height)
                        width = int(width)
                        ratio = height / width  # self.dataset[idx]
                except Exception as e:
                    print(e, self.dataset[idx], "This item is error, please check it.")
                    continue
                # find the closest aspect ratio
                closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
                if closest_ratio not in self.current_available_bucket_keys:
                    continue
                bucket = self.bucket['video'][closest_ratio]
                bucket.append(idx)
                # yield a batch of indices in the same aspect ratio group
                if len(bucket) == self.batch_size:
                    yield bucket[:]
                    del bucket[:]


class ImageVideoSampler(_UpstreamImageVideoSampler):
    """VerseCrafter variant: defaults content type to 'video'."""

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset.dataset[idx].get('type', 'video')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]
