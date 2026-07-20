# VerseCrafter-specific video-control datasets.
#
# These classes (VideoControlDataset_sekai / _spatialvid) are the
# VerseCrafter additions to the upstream VideoX-Fun dataset module. They are kept
# here (instead of inside the upstream package) so that the upstream submodule at
# third_party/VideoX-Fun stays pristine. All unchanged helpers are imported from
# the upstream package.
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
videox_fun_path = os.path.join(project_root, "third_party/VideoX-Fun")
if videox_fun_path not in sys.path:
    sys.path.insert(0, videox_fun_path)

import csv
import json
import random
from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from func_timeout import FunctionTimedOut, func_timeout
from PIL import Image
from torch.utils.data.dataset import Dataset

from videox_fun.data.dataset_image_video import (
    VideoReader_contextmanager,
    get_random_mask,
    get_video_reader_batch,
    padding_image,
    process_pose_file,
)

# VerseCrafter uses a longer decord read timeout than the upstream default (20s).
VIDEO_READER_TIMEOUT = 60


class VideoControlDataset_sekai(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        control_video_filename=["background_RGB.mp4", "background_depth.mp4", "3D_gaussian_RGB.mp4", "3D_gaussian_depth.mp4"],
        mask_video_filename="merged_mask.mp4",
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r', encoding='utf-8-sig') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
        
        print(f"Control Videos: {control_video_filename}")
        print(f"Mask Videos: {mask_video_filename}")
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'video') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'video') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        self.control_video_filename = control_video_filename
        self.mask_video_filename = mask_video_filename
        # control_video_filename_0 = ["background_rgb.mp4", "gaussian_projection.mp4", "foreground_bbox_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]
        # control_video_filename_1 = ["background_rgb.mp4", "gaussian_projection.mp4", "background_foreground_ellipsoid_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]
        # control_video_filename_2 = ["background_gaussian_projection.mp4", "background_foreground_ellipsoid_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]
        # control_video_filename_3 = ["background_rgb.mp4", "background_depth_bw.mp4", "gaussian_projection.mp4", "foreground_bbox_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_id, crowd_density, text = data_info['clipPath'], data_info['crowdDensity'], data_info['qwen_prompt']

        if data_info.get('type', 'video')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                video_dir = os.path.join(self.data_root, f"vstreams_{crowd_density}_clip", '_'.join(video_id.split('_')[:-4]), video_id)
                                         
            # GT video
            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                # start_idx   = 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    ) # [F, H, W, C] [0,255]
                    # resized_frames = []
                    # for i in range(len(pixel_values)):
                    #     frame = pixel_values[i]
                    #     resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                    #     resized_frames.append(resized_frame)
                    # pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''


            control_pixel_values_list = []

            for control_filename in self.control_video_filename:
                control_video_path = os.path.join(self.data_root, f"vstreams_{crowd_density}_clip_annotation", '_'.join(video_id.split('_')[:-4]), video_id.split('.')[0], control_filename)

                if os.path.exists(control_video_path):
                    with VideoReader_contextmanager(control_video_path, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values_single = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            control_pixel_values_single = torch.from_numpy(control_pixel_values_single).permute(0, 3, 1, 2).contiguous()
                            control_pixel_values_single = control_pixel_values_single / 255.
                            del control_video_reader
                        else:
                            control_pixel_values_single = control_pixel_values_single
                        
                        if not self.enable_bucket:
                            control_pixel_values_single = self.video_transforms(control_pixel_values_single)
                        
                        control_pixel_values_list.append(control_pixel_values_single)
                else:
                    if not self.enable_bucket:
                        control_pixel_values_list.append(torch.zeros_like(pixel_values))
                    else:
                        control_pixel_values_list.append(np.zeros_like(pixel_values))
                    control_camera_values = None



            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                mask_video_path = os.path.join(self.data_root, f"vstreams_{crowd_density}_clip_annotation", '_'.join(video_id.split('_')[:-4]), video_id.split('.')[0], self.mask_video_filename)
                if os.path.exists(mask_video_path):
                    with VideoReader_contextmanager(mask_video_path, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            mask_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            # resized_frames = []
                            # for i in range(len(control_pixel_values)):
                            #     frame = control_pixel_values[i]
                            #     resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            #     resized_frames.append(resized_frame)
                            # control_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            mask_pixel_values = torch.from_numpy(mask_pixel_values).permute(0, 3, 1, 2).contiguous()
                            mask_pixel_values = mask_pixel_values / 255.
                            del control_video_reader
                        else:
                            mask_pixel_values = mask_pixel_values

                        if not self.enable_bucket:
                            mask_pixel_values = self.video_transforms(mask_pixel_values)
                            
                            # Convert from [-1, 1] to [0, 1]
                            mask_pixel_values = (mask_pixel_values + 1.0) / 2.0
                            
                            # Apply Gaussian blur
                            f, c, height, width = mask_pixel_values.shape
                            kernel_size = min(height, width) // 16
                            if kernel_size % 2 == 0:
                                kernel_size += 1
                            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                            
                            # Apply gaussian blur to each frame
                            mask_pixel_values = torchvision.transforms.functional.gaussian_blur(
                                mask_pixel_values, 
                                kernel_size=kernel_size, 
                                sigma=sigma
                            )
                            # mask_pixel_values = mask_pixel_values[:, :1]
                else:
                    if not self.enable_bucket:
                        mask_pixel_values = torch.zeros_like(pixel_values)
                    else:
                        mask_pixel_values = np.zeros_like(pixel_values)
                control_pixel_values_list.append(mask_pixel_values)
                control_camera_values = None
            
            if self.enable_subject_info:
                # if not self.enable_bucket:
                #     visual_height, visual_width = pixel_values.shape[-2:]
                # else:
                #     visual_height, visual_width = pixel_values.shape[1:3]

                # subject_id = data_info.get('object_file_path', [])
                # shuffle(subject_id)
                # subject_images = []
                # for i in range(min(len(subject_id), 4)):
                #     subject_image = Image.open(subject_id[i])
                #     width, height = subject_image.size
                #     total_pixels = width * height

                #     img = padding_image(subject_image, visual_width, visual_height)
                    
                #     if random.random() < 0.5:
                #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
                #     subject_images.append(img)
                # subject_image = np.array(subject_images)

                ref_pixel_values = pixel_values[0:1]  # Keep shape as (1, C, H, W)

            else:
                ref_pixel_values = None

            return pixel_values, control_pixel_values_list, ref_pixel_values, control_camera_values, text, "video", video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i])
                    width, height = subject_image.size
                    total_pixels = width * height

                    img = padding_image(subject_image, visual_width, visual_height)
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(img)
                subject_image = np.array(subject_images)
            else:
                subject_image = None

            return image, control_image, subject_image, None, text, 'image', image_path
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values_list, ref_pixel_values, control_camera_values, name, data_type, file_path = self.get_batch(idx)

                background_rgb, background_depth_bw, gaussian_projection, foreground_ellipsoid_depth_bw, background_foreground_ellipsoid_mask = control_pixel_values_list

                # 将point cloud render video first frame 替换为 GT video first frame
                background_rgb[0] = pixel_values[0]
                background_foreground_ellipsoid_mask[0]=0.0

                sample["pixel_values"] = pixel_values
                sample["background_rgb"] = background_rgb
                sample["background_depth_bw"] = background_depth_bw
                sample["gaussian_projection"] = gaussian_projection
                sample["foreground_ellipsoid_depth_bw"] = foreground_ellipsoid_depth_bw
                sample["mask_pixel_values"] = background_foreground_ellipsoid_mask
                # sample["ref_pixel_values"] = ref_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample

class VideoControlDataset_sekai_spatialvid(Dataset):
    def __init__(
        self,
        ann_path, data_root=None,
        video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
        image_sample_size=512,
        video_repeat=0,
        text_drop_ratio=0.1,
        enable_bucket=False,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=False,
        enable_camera_info=False,
        return_file_name=False,
        enable_subject_info=False,
        control_video_filename=["background_RGB.mp4", "background_depth.mp4", "3D_gaussian_RGB.mp4", "3D_gaussian_depth.mp4"],
        mask_video_filename="merged_mask.mp4",
    ):
        # Loading annotations from files
        print(f"loading annotations from {ann_path} ...")
        if ann_path.endswith('.csv'):
            with open(ann_path, 'r', encoding='utf-8-sig') as csvfile:
                dataset = list(csv.DictReader(csvfile))
        elif ann_path.endswith('.json'):
            dataset = json.load(open(ann_path))
    
        self.data_root = data_root

        # It's used to balance num of images and videos.
        if video_repeat > 0:
            self.dataset = []
            for data in dataset:
                if data.get('type', 'video') != 'video':
                    self.dataset.append(data)
                    
            for _ in range(video_repeat):
                for data in dataset:
                    if data.get('type', 'video') == 'video':
                        self.dataset.append(data)
        else:
            self.dataset = dataset
        del dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        # TODO: enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint
        self.enable_camera_info = enable_camera_info
        self.enable_subject_info = enable_subject_info
        self.return_file_name = return_file_name

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        if self.enable_camera_info:
            self.video_transforms_camera = transforms.Compose(
                [
                    transforms.Resize(min(self.video_sample_size)),
                    transforms.CenterCrop(self.video_sample_size)
                ]
            )

        self.control_video_filename = control_video_filename
        self.mask_video_filename = mask_video_filename
        # control_video_filename_0 = ["background_rgb.mp4", "gaussian_projection.mp4", "foreground_bbox_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]
        # control_video_filename_1 = ["background_rgb.mp4", "gaussian_projection.mp4", "background_foreground_ellipsoid_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]
        # control_video_filename_2 = ["background_gaussian_projection.mp4", "background_foreground_ellipsoid_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]
        # control_video_filename_3 = ["background_rgb.mp4", "background_depth_bw.mp4", "gaussian_projection.mp4", "foreground_bbox_depth_bw.mp4", "background_foreground_ellipsoid_mask.mp4"]

        # Image params
        self.image_sample_size  = tuple(image_sample_size) if not isinstance(image_sample_size, int) else (image_sample_size, image_sample_size)
        self.image_transforms   = transforms.Compose([
            transforms.Resize(min(self.image_sample_size)),
            transforms.CenterCrop(self.image_sample_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
        ])

        self.larger_side_of_image_and_video = max(min(self.image_sample_size), min(self.video_sample_size))
    
    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        dataset_name, video_id, crowd_density, text = data_info['dataset'], data_info['clipPath'], data_info['crowdDensity'], data_info['qwen_prompt']

        if data_info.get('type', 'video')=='video':
            if self.data_root is None:
                video_dir = video_id
            else:
                if dataset_name == "sekai":
                    video_dir = os.path.join(self.data_root, "sekai_train_10k", f"vstreams_{crowd_density}_clip", '_'.join(video_id.split('_')[:-4]), video_id)
                elif dataset_name == "spatialvid_hq":
                    video_dir = os.path.join(self.data_root, "spatialvid", f"{crowd_density}_clip", video_id)
                else:
                    raise ValueError(f"Unknown dataset name {dataset_name}")
                
            # GT video
            with VideoReader_contextmanager(video_dir, num_threads=2) as video_reader:
                min_sample_n_frames = min(
                    self.video_sample_n_frames, 
                    int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
                )
                if min_sample_n_frames == 0:
                    raise ValueError(f"No Frames in video.")

                video_length = int(self.video_length_drop_end * len(video_reader))
                clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)
                start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
                # start_idx   = 0
                batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

                try:
                    sample_args = (video_reader, batch_index)
                    pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                    ) # [F, H, W, C] [0,255]
                    # resized_frames = []
                    # for i in range(len(pixel_values)):
                    #     frame = pixel_values[i]
                    #     resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                    #     resized_frames.append(resized_frame)
                    # pixel_values = np.array(resized_frames)
                except FunctionTimedOut:
                    raise ValueError(f"Read {idx} timeout.")
                except Exception as e:
                    raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                if not self.enable_bucket:
                    pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.
                    del video_reader
                else:
                    pixel_values = pixel_values

                if not self.enable_bucket:
                    pixel_values = self.video_transforms(pixel_values)
                
                # Random use no text generation
                if random.random() < self.text_drop_ratio:
                    text = ''


            control_pixel_values_list = []

            for control_filename in self.control_video_filename:
                if dataset_name == "sekai":
                    control_video_path = os.path.join(self.data_root, "sekai_train_10k", f"vstreams_{crowd_density}_clip_annotation", '_'.join(video_id.split('_')[:-4]), video_id.split('.')[0], control_filename)
                elif dataset_name == "spatialvid_hq":
                    control_video_path = os.path.join(self.data_root, "spatialvid", "clip_annotation", video_id.split('.')[0], control_filename)
                else:
                    raise ValueError(f"Unknown dataset name {dataset_name}")

                if os.path.exists(control_video_path):
                    with VideoReader_contextmanager(control_video_path, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            control_pixel_values_single = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            control_pixel_values_single = torch.from_numpy(control_pixel_values_single).permute(0, 3, 1, 2).contiguous()
                            control_pixel_values_single = control_pixel_values_single / 255.
                            del control_video_reader
                        else:
                            control_pixel_values_single = control_pixel_values_single
                        
                        if not self.enable_bucket:
                            control_pixel_values_single = self.video_transforms(control_pixel_values_single)
                        
                        control_pixel_values_list.append(control_pixel_values_single)
                else:
                    if not self.enable_bucket:
                        control_pixel_values_list.append(torch.zeros_like(pixel_values))
                    else:
                        control_pixel_values_list.append(np.zeros_like(pixel_values))
                    control_camera_values = None



            if self.enable_camera_info:
                if control_video_id.lower().endswith('.txt'):
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0])
                        control_camera_values = torch.from_numpy(control_camera_values).permute(0, 3, 1, 2).contiguous()
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)
                        control_camera_values = self.video_transforms_camera(control_camera_values)
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)

                        control_camera_values = process_pose_file(control_video_id, width=self.video_sample_size[1], height=self.video_sample_size[0], return_poses=True)
                        control_camera_values = torch.from_numpy(np.array(control_camera_values)).unsqueeze(0).unsqueeze(0)
                        control_camera_values = F.interpolate(control_camera_values, size=(len(video_reader), control_camera_values.size(3)), mode='bilinear', align_corners=True)[0][0]
                        control_camera_values = np.array([control_camera_values[index] for index in batch_index])
                else:
                    if not self.enable_bucket:
                        control_pixel_values = torch.zeros_like(pixel_values)
                        control_camera_values = None
                    else:
                        control_pixel_values = np.zeros_like(pixel_values)
                        control_camera_values = None
            else:
                if dataset_name == "sekai":
                    mask_video_path = os.path.join(self.data_root, "sekai_train_10k", f"vstreams_{crowd_density}_clip_annotation", '_'.join(video_id.split('_')[:-4]), video_id.split('.')[0], self.mask_video_filename)
                elif dataset_name == "spatialvid_hq":
                    mask_video_path = os.path.join(self.data_root, "spatialvid", "clip_annotation", video_id.split('.')[0], self.mask_video_filename)
                else:
                    raise ValueError(f"Unknown dataset name {dataset_name}")
                if os.path.exists(mask_video_path):
                    with VideoReader_contextmanager(mask_video_path, num_threads=2) as control_video_reader:
                        try:
                            sample_args = (control_video_reader, batch_index)
                            mask_pixel_values = func_timeout(
                                VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                            )
                            # resized_frames = []
                            # for i in range(len(control_pixel_values)):
                            #     frame = control_pixel_values[i]
                            #     resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                            #     resized_frames.append(resized_frame)
                            # control_pixel_values = np.array(resized_frames)
                        except FunctionTimedOut:
                            raise ValueError(f"Read {idx} timeout.")
                        except Exception as e:
                            raise ValueError(f"Failed to extract frames from video. Error is {e}.")

                        if not self.enable_bucket:
                            mask_pixel_values = torch.from_numpy(mask_pixel_values).permute(0, 3, 1, 2).contiguous()
                            mask_pixel_values = mask_pixel_values / 255.
                            del control_video_reader
                        else:
                            mask_pixel_values = mask_pixel_values

                        if not self.enable_bucket:
                            mask_pixel_values = self.video_transforms(mask_pixel_values)
                            
                            # Convert from [-1, 1] to [0, 1]
                            mask_pixel_values = (mask_pixel_values + 1.0) / 2.0
                            
                            # Apply Gaussian blur
                            f, c, height, width = mask_pixel_values.shape
                            kernel_size = min(height, width) // 16
                            if kernel_size % 2 == 0:
                                kernel_size += 1
                            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                            
                            # Apply gaussian blur to each frame
                            mask_pixel_values = torchvision.transforms.functional.gaussian_blur(
                                mask_pixel_values, 
                                kernel_size=kernel_size, 
                                sigma=sigma
                            )
                            # mask_pixel_values = mask_pixel_values[:, :1]
                else:
                    if not self.enable_bucket:
                        mask_pixel_values = torch.zeros_like(pixel_values)
                    else:
                        mask_pixel_values = np.zeros_like(pixel_values)
                control_pixel_values_list.append(mask_pixel_values)
                control_camera_values = None
            
            if self.enable_subject_info:
                # if not self.enable_bucket:
                #     visual_height, visual_width = pixel_values.shape[-2:]
                # else:
                #     visual_height, visual_width = pixel_values.shape[1:3]

                # subject_id = data_info.get('object_file_path', [])
                # shuffle(subject_id)
                # subject_images = []
                # for i in range(min(len(subject_id), 4)):
                #     subject_image = Image.open(subject_id[i])
                #     width, height = subject_image.size
                #     total_pixels = width * height

                #     img = padding_image(subject_image, visual_width, visual_height)
                    
                #     if random.random() < 0.5:
                #         img = img.transpose(Image.FLIP_LEFT_RIGHT)
                #     subject_images.append(img)
                # subject_image = np.array(subject_images)

                ref_pixel_values = pixel_values[0:1]  # Keep shape as (1, C, H, W)

            else:
                ref_pixel_values = None

            return pixel_values, control_pixel_values_list, ref_pixel_values, control_camera_values, text, "video", video_dir
        else:
            image_path, text = data_info['file_path'], data_info['text']
            if self.data_root is not None:
                image_path = os.path.join(self.data_root, image_path)
            image = Image.open(image_path).convert('RGB')
            if not self.enable_bucket:
                image = self.image_transforms(image).unsqueeze(0)
            else:
                image = np.expand_dims(np.array(image), 0)

            if random.random() < self.text_drop_ratio:
                text = ''

            control_image_id = data_info['control_file_path']

            if self.data_root is None:
                control_image_id = control_image_id
            else:
                control_image_id = os.path.join(self.data_root, control_image_id)

            control_image = Image.open(control_image_id).convert('RGB')
            if not self.enable_bucket:
                control_image = self.image_transforms(control_image).unsqueeze(0)
            else:
                control_image = np.expand_dims(np.array(control_image), 0)
            
            if self.enable_subject_info:
                if not self.enable_bucket:
                    visual_height, visual_width = image.shape[-2:]
                else:
                    visual_height, visual_width = image.shape[1:3]

                subject_id = data_info.get('object_file_path', [])
                shuffle(subject_id)
                subject_images = []
                for i in range(min(len(subject_id), 4)):
                    subject_image = Image.open(subject_id[i])
                    width, height = subject_image.size
                    total_pixels = width * height

                    img = padding_image(subject_image, visual_width, visual_height)
                    if random.random() < 0.5:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    subject_images.append(img)
                subject_image = np.array(subject_images)
            else:
                subject_image = None

            return image, control_image, subject_image, None, text, 'image', image_path
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        data_type = data_info.get('type', 'video')
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                data_type_local = data_info_local.get('type', 'video')
                if data_type_local != data_type:
                    raise ValueError("data_type_local != data_type")

                pixel_values, control_pixel_values_list, ref_pixel_values, control_camera_values, name, data_type, file_path = self.get_batch(idx)

                background_rgb, background_depth_bw, gaussian_projection, foreground_ellipsoid_depth_bw, background_foreground_ellipsoid_mask = control_pixel_values_list

                # 将point cloud render video first frame 替换为 GT video first frame
                background_rgb[0] = pixel_values[0]
                background_foreground_ellipsoid_mask[0]=0.0

                sample["pixel_values"] = pixel_values
                sample["background_rgb"] = background_rgb
                sample["background_depth_bw"] = background_depth_bw
                sample["gaussian_projection"] = gaussian_projection
                sample["foreground_ellipsoid_depth_bw"] = foreground_ellipsoid_depth_bw
                sample["mask_pixel_values"] = background_foreground_ellipsoid_mask
                # sample["ref_pixel_values"] = ref_pixel_values
                sample["text"] = name
                sample["data_type"] = data_type
                sample["idx"] = idx
                if self.return_file_name:
                    sample["file_name"] = os.path.basename(file_path)

                if self.enable_camera_info:
                    sample["control_camera_values"] = control_camera_values

                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        if self.enable_inpaint and not self.enable_bucket:
            mask = get_random_mask(pixel_values.size())
            mask_pixel_values = pixel_values * (1 - mask) + torch.zeros_like(pixel_values) * mask
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask

            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values

        return sample
