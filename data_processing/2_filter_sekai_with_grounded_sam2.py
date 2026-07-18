import os
import sys
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
from grounding_dino.groundingdino.util.inference import load_model, predict
from utils.track_utils import sample_points_from_masks
from ffmpeg_video_utils import create_video_from_images_with_ffmpeg as create_video_from_images
import pandas as pd
import random
from torchvision.ops import nms
import re
import ast
import subprocess
import tempfile
import grounding_dino.groundingdino.datasets.transforms as T
from decord import VideoReader, cpu, bridge
import argparse
import traceback

"""
Hyperparam for Ground and Tracking
"""
GROUNDING_DINO_CONFIG = str(GSAM2_HOME / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = str(GSAM2_HOME / "gdino_checkpoints/groundingdino_swint_ogc.pth")
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
TEXT_PROMPT = "person . human . car. animal ."
PROMPT_TYPE_FOR_VIDEO = "box" # choose from ["point", "box", "mask"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCENEDETECT_CSV = os.environ.get("GSAM2_SCENEDETECT_CSV")
VIDEO_ROOT = os.environ.get("GSAM2_VIDEO_ROOT")
OUTPUT_CLIP_DIR = os.environ.get("GSAM2_OUTPUT_CLIP_DIR", "./gsam2_valid_clips")
OUTPUT_CSV = os.environ.get("GSAM2_OUTPUT_CSV", "./gsam2_valid_clips.csv")
SEG_LEN = 81
STEP = 10
MAX_SAMPLES = None  # None means no limit; set to 10 for quick debugging.

# ========================= Person Completeness Parameters =========================
# Small-person thresholds (relative to image size)
H_SMALL = 0.20  # Height ratio < 0.20 is treated as a small person.
W_SMALL = 0.05  # Width ratio < 0.05 is treated as a small person.

# Normal aspect-ratio band (AR = height / width). Outside this range is abnormal.
AR_MIN = 2.1
AR_MAX = 4

# Near-edge rule: when the minimum distance to image border d_edge <= e.
EDGE_MARGIN_PCT = 0.01  # Relative to the shorter image side.
EDGE_MARGIN_PX_MIN = 3   # Minimum pixel threshold.

# Mask/BBox area ratio rule: mask_area / bbox_area below this is incomplete.
MASK_BBOX_RATIO_MIN = 0.4  # Typical range is 0.4-0.6.

# Person count constraints
SMALL_MIN, SMALL_MAX = 0, 0
BIG_MIN, BIG_MAX = 0, 5
# BIG_INCOMPLETE must be 0.
# ---------------------------------------------------------------

# build grounding dino model from local path
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG, 
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE
)

# init sam image predictor and video predictor model
sam2_checkpoint = str(GSAM2_HOME / "checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

# SAM2's Hydra config loading expects configs to be resolved under Grounded-SAM-2 cwd.
os.chdir(str(GSAM2_HOME))

video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
image_predictor = SAM2ImagePredictor(sam2_image_model)

def parse_video_file(vf: str):
    return "_".join(vf.split("_")[:-2])
def _parse_timestamps_cell(ts_cell):
    try:
        return ast.literal_eval(ts_cell)
    except Exception:
        return []

def _build_video_path(video_root, video_file):
    return os.path.join(
        video_root,
        "_".join(video_file.split("_")[:-2]),
        video_file,
    )

def _export_clip_ffmpeg(in_path, start_frame, end_frame, fps, width, height, out_path):
    """
    Export clip by frame index range [start_frame, end_frame) using ffmpeg filters.
    Frame-accurate cutting with select/setpts; no time-based -ss/-t.
    """
    
    s = int(start_frame)
    e = int(end_frame)
    if e <= s:
        raise ValueError("end_frame must be greater than start_frame")
    vf = (
        f"fps={fps},"
        f"select='between(n,{s},{e-1})',"
        f"setpts=N/({fps}*TB),"
        f"scale={int(width)}:{int(height)}:flags=lanczos,"
        f"format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", in_path,
        "-vf", vf,
        "-r", f"{fps}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-an", "-y", out_path,
    ]
    subprocess.run(cmd, check=True)

def load_image(img_or_path):
    """
    Load image for GroundingDINO:
    - Accepts ndarray (RGB) or file path.
    - Returns (image_source_rgb_numpy, transformed_tensor).
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if isinstance(img_or_path, str):
        pil = Image.open(img_or_path).convert("RGB")
        image_np = np.asarray(pil)
        image_transformed, _ = transform(pil, None)
        return image_np, image_transformed

    # # Compatibility path for decord NDArray (when bridge is not configured).
    # if hasattr(img_or_path, "asnumpy"):
    #     img_or_path = img_or_path.asnumpy()

    if isinstance(img_or_path, np.ndarray):
        arr = img_or_path  # RGB
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        image_transformed, _ = transform(pil, None)
        return arr, image_transformed

    raise TypeError("load_image expects a file path or a numpy ndarray.")

def _read_frame_rgb_decord(vr, frame_idx):
    """
    Read frame by index using decord.VideoReader, return RGB ndarray or None.
    """
    try:
        frame = vr[int(frame_idx)]
        if hasattr(frame, "asnumpy"):
            frame = frame.asnumpy()
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame  # RGB
    except Exception:
        return None

def _visualize_detection_results(image_source, boxes, masks, labels, confidences, prompt, video_path):
    """
    Visualize detection results.
    """
    # Create output directory.
    debug_dir = "./debug_visualization_sekai"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Convert image format for visualization.
    img_bgr = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    
    # Build detection object.
    detections = sv.Detections(
        xyxy=np.array(boxes),
        mask=np.array(masks, dtype=bool),  # Ensure mask is boolean.
        class_id=np.arange(len(labels)),
        confidence=np.array(confidences)
    )
    
    # Add bounding boxes.
    box_annotator = sv.BoxAnnotator()
    annotated_frame = box_annotator.annotate(scene=img_bgr.copy(), detections=detections)
    
    # Add labels.
    label_annotator = sv.LabelAnnotator()
    labels_with_conf = [f"{label}: {conf:.2f}" for label, conf in zip(labels, confidences)]
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels_with_conf)
    
    # Add masks.
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(annotated_frame, detections=detections)
    
    # Save result.
    prompt_clean = prompt.replace(" ", "_").replace(".", "").replace(",", "")
    output_path = os.path.join(debug_dir, f"detection_{os.path.basename(video_path).split('.')[0]}_{prompt_clean}.jpg")
    cv2.imwrite(output_path, annotated_frame)

def _near_edge_flag(x1: int, y1: int, x2: int, y2: int, W_img: int, H_img: int) -> bool:
    """Whether the bbox is near the image boundary (assuming bbox is inside image)."""
    d_edge = min(x1, y1, W_img - x2, H_img - y2)
    e = max(EDGE_MARGIN_PX_MIN, int(EDGE_MARGIN_PCT * min(W_img, H_img)))
    return d_edge <= e

def _classify_person_by_ar_edge(box_xyxy: np.ndarray, mask: np.ndarray, img_h: int, img_w: int) -> str:
    """Classify one person by AR + Edge + Mask/BBox ratio: return 'SMALL' | 'BIG' | 'BIG_INCOMPLETE'.
    box_xyxy: [x1, y1, x2, y2] (float/np)
    mask: 2D boolean/0-1 array, shape (img_h, img_w)
    """
    x1, y1, x2, y2 = box_xyxy.astype(int)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    H = bh / float(img_h)
    W = bw / float(img_w)
    AR = bh / float(bw)

    # 1) Small?
    if (H < H_SMALL) or (W < W_SMALL):
        return 'SMALL'

    # 2) Near-Edge?
    NE = _near_edge_flag(x1, y1, x2, y2, img_w, img_h)

    # 3) AR abnormal?
    AR_abnormal = (AR < AR_MIN) or (AR > AR_MAX)

    # 4) Mask area ratio inside bbox.
    bbox_area = bw * bh
    mask_area = np.sum(mask[y1:y2, x1:x2] > 0)  # Only count mask area inside bbox.
    mask_bbox_ratio = mask_area / max(1, bbox_area)
    
    # Too small mask ratio indicates incomplete person.
    Mask_incomplete = mask_bbox_ratio < MASK_BBOX_RATIO_MIN

    # 5) Incomplete if (near-edge and abnormal AR) or mask ratio is too small.
    Incomplete = (NE and AR_abnormal) or Mask_incomplete
    return 'BIG_INCOMPLETE' if Incomplete else 'BIG'

def _is_valid_by_gsam2_with_visualization(
    first_frame_path_or_rgb,
    prompt: str,
    max_object_num: int = 4,
    min_object_num: int = 0,
    min_area_pct: float = 1.0,
    max_area_pct: float = 20.0,
    check_person_completeness: bool = False,
    debug: bool = False,
    video_path: str = None,
):
    """
    When `check_person_completeness=True` (person prompts), use AR+Edge-based completeness
    with scene-level count constraints: 0<=Small<=3, 1<=Big<=5, Big_Incomplete==0.

    For non-person prompts (car/animal), keep the original mask-area filtering logic.
    Returns: (is_valid: bool, detection_results: dict | None, num_object: int)
    """
    # 1) Load image and run GroundingDINO prediction.
    image_source, image = load_image(first_frame_path_or_rgb)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        remove_combined=True,
    )

    # 2) Convert box coordinates and run NMS.
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")

    # NMS
    if input_boxes.shape[0] > 0:
        keep = nms(input_boxes, confidences, iou_threshold=0.5)
        input_boxes = input_boxes[keep].cpu().numpy()
        confidences = confidences[keep].cpu().numpy()
        labels = [labels[i] for i in keep.cpu().numpy()]
    else:
        input_boxes = np.zeros((0, 4))
        confidences = np.array([])
        labels = []

    num = input_boxes.shape[0]
    if not (min_object_num <= num <= max_object_num):
        return False, None, 0

    # If count is valid and equals 0, return directly without SAM2.
    if num == 0:
        return True, None, 0

    # 3) Generate SAM2 masks (used for visualization and non-person area filtering).
    image_predictor.set_image(image_source)
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    if debug:
        _visualize_detection_results(
            image_source,
            input_boxes,
            masks,
            labels,
            confidences,
            prompt,
            video_path,
        )

    # 4) Branch A (person): AR+Edge+Mask/BBox ratio + count constraints.
    if check_person_completeness:
        classes = []  # 'SMALL' | 'BIG' | 'BIG_INCOMPLETE'
        for i in range(len(input_boxes)):
            c = _classify_person_by_ar_edge(input_boxes[i], masks[i], h, w)
            classes.append(c)

        Ns = sum(1 for c in classes if c == 'SMALL')
        Nb = sum(1 for c in classes if c == 'BIG')
        Nbi = sum(1 for c in classes if c == 'BIG_INCOMPLETE')

        ALLOW_EMPTY = True   # If True, allow frames with zero accepted people.
        # Scene-level acceptance rules, from hard constraints to soft constraints.
        if Nbi != 0:
            return False, None, 0  # Reject immediately when incomplete person exists.
        if not (SMALL_MIN <= Ns <= SMALL_MAX):
            return False, None, 0
        if not (BIG_MIN <= Nb <= BIG_MAX):
            return False, None, 0
        # Do not allow "small-only" persons.
        if Ns > 0 and Nb == 0:
            return False, None, 0
        # Whether empty person frames are allowed.
        if not ALLOW_EMPTY and (Ns + Nb) == 0:
            return False, None, 0

        # Optionally return detections for BIG+SMALL classes.
        keep_idx = [i for i, c in enumerate(classes) if c in ('SMALL', 'BIG')]
        detection_results = {
            'boxes': np.array([input_boxes[i] for i in keep_idx]) if keep_idx else np.zeros((0,4)),
            'masks': np.array([masks[i] for i in keep_idx]) if keep_idx else np.zeros((0,)),
            'labels': [labels[i] for i in keep_idx],
            'confidences': np.array([confidences[i] for i in keep_idx]) if keep_idx else np.array([]),
            'person_classes': [classes[i] for i in keep_idx],
        }
        num_object = detection_results['boxes'].shape[0]
        return True, detection_results, num_object

    # 5) Branch B (non-person): keep original area-percentage filtering.
    total_pixels = masks.shape[1] * masks.shape[2]
    mask_areas = np.sum(masks, axis=(1, 2))
    mask_area_percentages = (mask_areas / total_pixels) * 100.0

    valid_masks = []
    valid_boxes = []
    valid_labels = []
    valid_confidences = []

    for i, area_pct in enumerate(mask_area_percentages):
        if not (min_area_pct <= area_pct <= max_area_pct):
            continue
        valid_masks.append(masks[i])
        valid_boxes.append(input_boxes[i])
        valid_labels.append(labels[i])
        valid_confidences.append(float(confidences[i]))

    if len(valid_boxes) == 0:
        return False, None, 0

    detection_results = {
        'boxes': np.array(valid_boxes),
        'masks': np.array(valid_masks),
        'labels': valid_labels,
        'confidences': np.array(valid_confidences),
    }
    num_object = detection_results['boxes'].shape[0]
    return True, detection_results, num_object

def is_video_frame_valid(frame_rgb: np.ndarray, video_path: str = None, debug: bool = False) -> bool:
    """
    Encapsulated Step 2:
    - Run three detectors on one frame: person (with completeness rules), car, animal.
    - Return True only when all enabled checks pass and total objects are in [1, 6].

    Args:
        frame_rgb: RGB ndarray with shape (H, W, 3), dtype uint8.
        video_path: Optional path used for visualization naming.
        debug: Whether to save visualization images.

    Returns:
        bool: Whether this frame is valid.
    """
    try:
        # 1) Person: AR+Edge completeness rules; max 6 persons.
        ok_person, det_person, n_person = _is_valid_by_gsam2_with_visualization(
            frame_rgb,
            "person . human .",
            max_object_num=6,
            min_object_num=0,
            min_area_pct=0.0,   # Area check is replaced by completeness checks.
            max_area_pct=20.0,
            check_person_completeness=True,
            debug=debug,
            video_path=video_path,
        )
        if not ok_person:
            return False

        # 2) Car: no completeness check; max 6 cars; area filter [0, 30]%.
        ok_car, det_car, n_car = _is_valid_by_gsam2_with_visualization(
            frame_rgb,
            "car .",
            max_object_num=6,
            min_object_num=0,
            min_area_pct=5.0,
            max_area_pct=30.0,
            check_person_completeness=False,
            debug=debug,
            video_path=video_path,
        )
        if not ok_car:
            return False

        # 3) Total object count constraint: 1 <= (person + car + animal) <= 6.
        total_objs = int(n_person) + int(n_car)
        return 1 <= total_objs <= 6

    except Exception:
        # Treat any exception as invalid frame to keep batch processing robust.
        traceback.print_exc()
        return False

def _ensure_required_path_arg(parser, value, cli_flag, env_var):
    if value is None or str(value).strip() == "":
        parser.error(f"{cli_flag} is required unless environment variable {env_var} is set.")

def parse_args():
    parser = argparse.ArgumentParser(description="Filter Sekai clips with GroundedSAM2 and export valid windows.")
    parser.add_argument(
        "--scenedetect_csv",
        type=str,
        default=SCENEDETECT_CSV,
        help="Input SceneDetect CSV path (or set GSAM2_SCENEDETECT_CSV).",
    )
    parser.add_argument(
        "--video_root",
        type=str,
        default=VIDEO_ROOT,
        help="Root directory of videos (or set GSAM2_VIDEO_ROOT).",
    )
    parser.add_argument(
        "--output_clip_dir",
        type=str,
        default=OUTPUT_CLIP_DIR,
        help="Directory to save exported clips (or set GSAM2_OUTPUT_CLIP_DIR).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=OUTPUT_CSV,
        help="Output CSV path (or set GSAM2_OUTPUT_CSV).",
    )
    # New: partitioning controls
    parser.add_argument("--num_parts", type=int, default=1, help="Split input CSV into this many parts")
    parser.add_argument("--part_idx", type=int, default=0, help="Zero-based index of the part to process [0..num_parts-1]")
    args = parser.parse_args()
    _ensure_required_path_arg(parser, args.scenedetect_csv, "--scenedetect_csv", "GSAM2_SCENEDETECT_CSV")
    _ensure_required_path_arg(parser, args.video_root, "--video_root", "GSAM2_VIDEO_ROOT")
    return args

def main(scenedetect_csv, video_root, output_clip_dir, output_csv, num_parts=1, part_idx=0):
    # Read timestamp CSV generated by 1_run_scenedetect.py
    ts_df = pd.read_csv(scenedetect_csv)
    if MAX_SAMPLES is not None:
        ts_df = ts_df.head(MAX_SAMPLES)

    if num_parts > 1:
        parts = np.array_split(ts_df, num_parts)
        ts_df = parts[part_idx].copy()

    orig_cols = list(ts_df.columns)

    results = []
    os.makedirs(output_clip_dir, exist_ok=True)

    for row_idx, row in enumerate(tqdm(ts_df.itertuples(index=False), total=len(ts_df), desc="Scanning timestamps")):
        video_file = getattr(row, "videoFile", None)
        timestamps_cell = getattr(row, "timestamp", None)

        video_path = _build_video_path(video_root, video_file)
        if not os.path.exists(video_path):
            print(f"Not found: {video_path}")
            continue

        # Use decord to read video meta (no OpenCV)
        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            first = vr[0]
            height, width = first.shape[:2]
        except Exception as e:
            print(f"Decord open error: {video_path}, {e}")
            continue

        ts_list = _parse_timestamps_cell(timestamps_cell)
        if not ts_list:
            continue

        base = Path(video_file).stem
        video_id = parse_video_file(base)

        # Iterate through timestamp segments:
        # (start_seconds, end_seconds, start_frames, end_frames, start_timecode, end_timecode, framerate)
        for t in ts_list:
            start_fr = int(float(t[2]))
            end_fr = int(float(t[3]))
            fps = int(float(t[-1]))

            start_fr = max(0, start_fr)
            end_fr = min(total_frames, end_fr)
            if end_fr - start_fr < SEG_LEN:
                continue

            i = start_fr
            last_start = end_fr - SEG_LEN

            while i <= last_start:
                frame_rgb = _read_frame_rgb_decord(vr, i)
                if frame_rgb is None:
                    break

                is_valid = is_video_frame_valid(frame_rgb, video_path=video_path, debug=True)

                if is_valid:
                    out_subdir = os.path.join(output_clip_dir, video_id)
                    out_name = f"{base}_{i:07d}_{i+SEG_LEN:07d}.mp4"
                    out_path = os.path.join(out_subdir, out_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    
                    # Check whether output file already exists and is valid.
                    skip_export = False
                    if os.path.exists(out_path):
                        try:
                            # Validate output with VideoReader.
                            test_vr = VideoReader(out_path, ctx=cpu(0))
                            if len(test_vr) > 0:  # File exists and is readable.
                                skip_export = True
                                print(f"File already exists and valid, skipping: {out_path}")
                        except Exception:
                            # Existing file is corrupted or unreadable; regenerate.
                            print(f"File exists but invalid, will regenerate: {out_path}")
        
                    export_success = True
                    if not skip_export:
                        try:
                            _export_clip_ffmpeg(
                                video_path,
                                start_frame=i,
                                end_frame=i + SEG_LEN,
                                fps=fps,
                                width=width,
                                height=height,
                                out_path=out_path,
                            )
                            print(f"Exported clip: {out_path}")
                        except subprocess.CalledProcessError as e:
                            print(f"ffmpeg failed: {video_path} [{i}-{i+SEG_LEN}]: {e}")
                            export_success = False
                    
                    # Add row only if export was skipped or successfully finished.
                    if skip_export or export_success:
                        # Write original row plus generated clipPath.
                        row_dict = row._asdict()
                        row_dict = dict(row_dict)
                        row_dict["clipPath"] = out_name
                        results.append(row_dict)
                        if skip_export:
                            print(f"Added existing clip to results: {out_path}")
                    
                    i += SEG_LEN  # Skip one full segment length after a valid hit.
                else:
                    i += STEP

    res_df = pd.DataFrame(results)
    if len(res_df) > 0:
        res_df.to_csv(output_csv, index=False)
        print(f"Saved {len(res_df)} clips to {output_csv}")
    else:
        print("No valid clips found.")

if __name__ == '__main__':
    args = parse_args()
    main(
        scenedetect_csv=args.scenedetect_csv,
        video_root=args.video_root,
        output_clip_dir=args.output_clip_dir,
        output_csv=args.output_csv,
        num_parts=args.num_parts,
        part_idx=args.part_idx,
    )