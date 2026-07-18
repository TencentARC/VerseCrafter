import os
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GSAM2_HOME = PROJECT_ROOT / "third_party" / "Grounded-SAM-2"
GSAM2_HOME = Path(os.environ.get("GSAM2_HOME", str(DEFAULT_GSAM2_HOME))).expanduser().resolve()
if not GSAM2_HOME.exists():
    raise FileNotFoundError(
        f"Grounded-SAM-2 path not found: {GSAM2_HOME}. "
        "Please initialize submodule or set GSAM2_HOME."
    )
if str(GSAM2_HOME) not in sys.path:
    sys.path.insert(0, str(GSAM2_HOME))

import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from utils.track_utils import sample_points_from_masks
from ffmpeg_video_utils import create_video_from_images_with_ffmpeg as create_video_from_images


def _resolve_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(GSAM2_HOME / path)


def _resolve_sam2_config(config_value: str) -> str:
    config_path = Path(config_value)
    if not config_path.is_absolute():
        return config_value
    try:
        return str(config_path.relative_to(GSAM2_HOME))
    except ValueError:
        return str(config_path)


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(str(path))
    try:
        yield
    finally:
        os.chdir(str(previous))


def load_models_and_init(
    grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth",
    sam2_checkpoint="checkpoints/sam2.1_hiera_large.pt",
    sam2_model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
    device=None
):
    """
    Load GroundingDINO and SAM2 models and return the relevant objects.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    sam2_cfg = _resolve_sam2_config(sam2_model_cfg)
    sam2_ckpt = _resolve_path(sam2_checkpoint)

    grounding_model = load_model(
        model_config_path=_resolve_path(grounding_dino_config),
        model_checkpoint_path=_resolve_path(grounding_dino_checkpoint),
        device=device
    )
    # SAM2's hydra compose resolves configs under Grounded-SAM-2 working directory.
    with _pushd(GSAM2_HOME):
        video_predictor = build_sam2_video_predictor(sam2_cfg, sam2_ckpt)
        sam2_image_model = build_sam2(sam2_cfg, sam2_ckpt)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    return grounding_model, video_predictor, image_predictor, device

def run_video_segmentation(
    video_path,
    grounding_model,
    video_predictor,
    image_predictor,
    device,
    text_prompt="person . animal . car . moving object . dynamic object .",
    box_threshold=0.4,
    text_threshold=0.25,
    prompt_type_for_video="box",
    visualize=False,
    save_tracking_results_dir="./tracking_results",
    output_video_path="./output_tracking_demo.mp4",
    temp_frame_dir="./custom_video_frames_temp",
    max_frames=None
):
    """
    Run segmentation on an input video and return `video_segments`.
    Optionally visualize and save tracking results.

    Args:
        max_frames: maximum number of frames to process (None means all frames)
    """
    # Step 1: extract video frames
    video_info = sv.VideoInfo.from_video_path(video_path)
    print(video_info)
    
    # 确保 max_frames 不超过实际帧数
    actual_frame_count = video_info.total_frames
    if max_frames is not None:
        end_frame = min(max_frames, actual_frame_count)
    else:
        end_frame = None
    
    print(f"Total frames: {actual_frame_count}, Processing frames: {end_frame if end_frame else 'all'}")
    frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=end_frame)
    source_frames = Path(temp_frame_dir)
    source_frames.mkdir(parents=True, exist_ok=True)
    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)
    frame_names = [
        p for p in os.listdir(temp_frame_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = video_predictor.init_state(video_path=temp_frame_dir)
    ann_frame_idx = 0
    # Step 2: get boxes from GroundingDINO
    img_path = os.path.join(temp_frame_dir, frame_names[ann_frame_idx])
    image_source, image = load_image(img_path)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        remove_combined=True
    )
    # 添加检测结果检查
    if boxes.shape[0] == 0:
        print(f"[WARNING] No objects detected in frame {ann_frame_idx}. Returning empty segments.")
        return {}
    
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])

    # keep top 6 largest boxes by area
    if boxes.shape[0] > 0:
        areas = boxes[:, 2] * boxes[:, 3]  # w * h
        topk = min(6, boxes.shape[0])
        topk_idx = torch.topk(areas, topk).indices
        boxes = boxes[topk_idx]
        confidences = confidences[topk_idx]
        labels = [labels[i] for i in topk_idx.tolist()]

    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    confidences = confidences.numpy().tolist()
    class_names = labels
    image_predictor.set_image(image_source)
    OBJECTS = class_names
    device_type = "cuda" if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _autocast():
        if device_type == "cuda":
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return nullcontext()

    with _autocast():
        masks, scores, logits = image_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    masks_tensor = torch.from_numpy(masks) if isinstance(masks, np.ndarray) else masks
    if masks_tensor.ndim == 4:
        masks_tensor = masks_tensor.squeeze(1)
    image_area = h * w
    mask_areas = masks_tensor.flatten(1).float().sum(dim=1)
    mask_area_ratios = mask_areas / image_area
    print(f"mask_area_ratios: {mask_area_ratios}")
    MIN_RATIO, MAX_RATIO = 0.005, 0.2
    valid_mask = (mask_area_ratios >= MIN_RATIO) & (mask_area_ratios <= MAX_RATIO)
    if not torch.any(valid_mask):
        print(f"[WARNING] No masks within [{MIN_RATIO}, {MAX_RATIO}] area ratio. Returning empty segments.")
        return {}
    valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1).cpu().tolist()
    masks = masks[valid_indices]
    input_boxes = input_boxes[valid_indices]
    confidences = [confidences[i] for i in valid_indices]
    class_names = [class_names[i] for i in valid_indices]
    OBJECTS = class_names
    # Step 3: register prompts for the video predictor
    assert prompt_type_for_video in ["point", "box", "mask"], "SAM 2 video predictor only support point/box/mask prompt"
    if prompt_type_for_video == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif prompt_type_for_video == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif prompt_type_for_video == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    else:
        raise NotImplementedError("SAM 2 video predictor only support point/box/mask prompts")
    # Step 4: inference / propagate masks through the video
    with _autocast():
        video_segments = {}
        _video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
            if visualize:
                _video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0)
                for i, out_obj_id in enumerate(out_obj_ids)
            }
    # Step 5: visualization (optional)
    if visualize:
        if not os.path.exists(save_tracking_results_dir):
            os.makedirs(save_tracking_results_dir)
        ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
        for frame_idx, segments in _video_segments.items():
            img = cv2.imread(os.path.join(temp_frame_dir, frame_names[frame_idx]))
            object_ids = list(segments.keys())
            masks = list(segments.values())
            masks = np.concatenate(masks, axis=0)
            detections = sv.Detections(
                xyxy=sv.mask_to_xyxy(masks),
                mask=masks,
                class_id=np.array(object_ids, dtype=np.int32),
            )
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)
            label_annotator = sv.LabelAnnotator()
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=[ID_TO_OBJECTS[i] for i in object_ids])
            mask_annotator = sv.MaskAnnotator()
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
            cv2.imwrite(os.path.join(save_tracking_results_dir, f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
        create_video_from_images(save_tracking_results_dir, output_video_path)

    return video_segments 

if __name__ == "__main__":
    mp4_path = "/path/to/input_video.mp4"
    save_tracking_results_dir = "results/tracking_results_demo"
    output_video_path = "results/output_tracking_demo.mp4"
    temp_frame_dir = "results/custom_video_frames_temp_demo"
    max_frames = 100 

    print("[INFO] Loading models...")
    grounding_model, video_predictor, image_predictor, device = load_models_and_init()
    print("[INFO] Models loaded. Running video segmentation...")
    video_segments = run_video_segmentation(
        video_path=mp4_path,
        grounding_model=grounding_model,
        video_predictor=video_predictor,
        image_predictor=image_predictor,
        device=device,
        text_prompt="person . animal . automobile . moving object . dynamic object .",
        box_threshold=0.3,
        text_threshold=0.25,
        prompt_type_for_video="box",
        visualize=True,
        save_tracking_results_dir=save_tracking_results_dir,
        output_video_path=output_video_path,
        temp_frame_dir=temp_frame_dir,
        max_frames=max_frames
    )
    print(f"[INFO] Segmentation finished.") 