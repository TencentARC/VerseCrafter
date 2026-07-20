import argparse
import random
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from torchvision.io import read_video, write_video
from transformers import AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
VIDEOX_FUN_PATH = PROJECT_ROOT / "third_party" / "VideoX-Fun"

for import_path in (PROJECT_ROOT, VIDEOX_FUN_PATH):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

from videox_fun.dist import set_multi_gpus_devices, shard_model
from videox_fun.models import AutoencoderKLWan, WanT5EncoderModel
from videox_fun.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from videox_fun.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from videox_fun.utils.fp8_optimization import (
    convert_model_weight_to_float8,
    convert_weight_dtype_wrapper,
    replace_parameters_by_name,
)
from videox_fun.utils.lora_utils import merge_lora, unmerge_lora
from videox_fun.utils.utils import filter_kwargs, get_video_to_video_latent
from versecrafter.models import VerseCrafterWanTransformer3DModel
from versecrafter.pipeline import WanVerseCrafterPipeline

TEACACHE_COEFFICIENTS = [
    8.10705460e03,
    2.13393892e03,
    -3.72934672e02,
    1.66203073e01,
    -4.17769401e-02,
]


def load_csv_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def sample_data(df: pd.DataFrame, sample_size: int, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    if sample_size >= len(df):
        return df
    return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)


def split_data_into_parts(df: pd.DataFrame, num_parts: int) -> list[pd.DataFrame]:
    part_size = len(df) // num_parts
    parts = []
    for i in range(num_parts):
        start_idx = i * part_size
        end_idx = len(df) if i == num_parts - 1 else (i + 1) * part_size
        parts.append(df.iloc[start_idx:end_idx].reset_index(drop=True))
    return parts


def video_to_frames(video_path: str, target_height: int = 480, target_width: int = 832) -> list[np.ndarray]:
    video_tensor, _, _ = read_video(video_path, pts_unit="sec")

    if video_tensor.shape[1] != target_height or video_tensor.shape[2] != target_width:
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        video_tensor = torch.nn.functional.interpolate(
            video_tensor.float(),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        video_tensor = video_tensor.permute(0, 2, 3, 1).byte()

    return [frame.numpy() for frame in video_tensor]


def frames_to_video(frames: list[np.ndarray] | np.ndarray, output_path: str, fps: int = 15) -> None:
    if isinstance(frames, (list, tuple)):
        if not frames:
            return
        frames = np.stack(frames, axis=0)
    elif isinstance(frames, np.ndarray):
        if frames.size == 0 or frames.shape[0] == 0:
            return
    else:
        return

    video_tensor = torch.from_numpy(frames)
    if video_tensor.dtype != torch.uint8:
        video_tensor = video_tensor.byte()

    write_video(output_path, video_tensor, fps=fps, video_codec="h264")


def create_comparison_grid(
    videos_dict: dict[str, str | np.ndarray | list[np.ndarray]],
    output_path: str,
    fps: int = 15,
    max_frames_limit: int = 81,
    target_height: int = 480,
    target_width: int = 832,
) -> None:
    all_frames: dict[str, list[np.ndarray]] = {}
    max_frames = 0

    for key, data in videos_dict.items():
        if isinstance(data, np.ndarray):
            frames = [data[i] for i in range(data.shape[0])]
        elif isinstance(data, list):
            frames = data
        else:
            path = Path(data)
            if not path.exists():
                print(f"Warning: file not found: {path}")
                continue
            frames = video_to_frames(str(path), target_height=target_height, target_width=target_width)
        all_frames[key] = frames
        max_frames = max(max_frames, len(frames))

    max_frames = min(max_frames, max_frames_limit)

    for key in all_frames:
        frames = all_frames[key]
        if len(frames) > max_frames:
            all_frames[key] = frames[:max_frames]
        elif len(frames) < max_frames:
            last_frame = frames[-1] if frames else np.zeros((target_height, target_width, 3), dtype=np.uint8)
            frames.extend([last_frame] * (max_frames - len(frames)))
            all_frames[key] = frames

    for key in all_frames:
        resized_frames = []
        for frame in all_frames[key]:
            if frame.shape[0] != target_height or frame.shape[1] != target_width:
                frame = cv2.resize(frame, (target_width, target_height))
            resized_frames.append(frame)
        all_frames[key] = resized_frames

    order = [
        "background_RGB",
        "3D_gaussian_RGB",
        "background_depth",
        "3D_gaussian_depth",
        "merged_mask",
        "generated_video",
        "gt_video",
    ]

    grid_frames = []
    for i in range(max_frames):
        row1, row2, row3 = [], [], []
        for j, key in enumerate(order):
            frame = all_frames[key][i] if key in all_frames else np.zeros((target_height, target_width, 3), dtype=np.uint8)
            if j < 3:
                row1.append(frame)
            elif j < 6:
                row2.append(frame)
            else:
                row3.append(frame)

        while len(row3) < 3:
            row3.append(np.zeros((target_height, target_width, 3), dtype=np.uint8))

        grid_frames.append(np.vstack([np.hstack(row1), np.hstack(row2), np.hstack(row3)]))

    frames_to_video(grid_frames, output_path, fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch VerseCrafter inference on Sekai with comparison grid output")
    parser.add_argument("--csv", default="path/to/sekai_test_split.csv", help="CSV path containing clip metadata")
    parser.add_argument("--data-dir", default="/path/to/sekai-codebase", help="Sekai root used by build scripts")
    parser.add_argument("--output-dir", default="samples/batch_inference/sekai", help="Output directory")
    parser.add_argument("--sample-size-limit", type=int, default=None, help="Randomly sample at most this many rows")
    parser.add_argument("--num-parts", type=int, default=1, help="Split CSV into N parts")
    parser.add_argument("--part", type=int, default=0, help="Part index to process (0-based)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config-path", default="config/wan2.1/wan_civitai.yaml", help="Config path")
    parser.add_argument("--model-name", default="model/Wan2.1-T2V-14B", help="Base Wan model path")
    parser.add_argument("--transformer-path", default="model/VerseCrafter", help="VerseCrafter checkpoint path")
    parser.add_argument("--vae-path", default=None, help="Optional VAE checkpoint")
    parser.add_argument("--lora-path", default=None, help="Optional LoRA checkpoint")
    parser.add_argument("--lora-weight", type=float, default=0.55, help="LoRA weight")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG guidance scale")
    parser.add_argument("--video-length", type=int, default=81, help="Video frames")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--ulysses-degree", type=int, default=2, help="Ulysses degree")
    parser.add_argument("--ring-degree", type=int, default=2, help="Ring degree")
    parser.add_argument("--fps", type=int, default=16, help="Output FPS")
    parser.add_argument("--geoada-context-scale", type=float, default=1.0, help="GeoAdapter context scale")
    parser.add_argument("--geoada-in-dim", type=int, default=128, help="GeoAdapter input dim")
    parser.add_argument(
        "--negative-prompt",
        default=(
            "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
            "paintings, images, static, overall gray, worst quality, low quality, JPEG "
            "compression residue, ugly, incomplete, extra fingers, poorly drawn hands, "
            "poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
            "still picture, messy background, three legs, many people in the background, "
            "walking backwards"
        ),
        help="Negative prompt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading CSV from {args.csv}")
    df = load_csv_data(args.csv)
    print(f"Total rows: {len(df)}")

    if args.sample_size_limit is not None:
        df = sample_data(df, args.sample_size_limit, args.seed)
        print(f"Using sampled rows: {len(df)}")

    parts = split_data_into_parts(df, args.num_parts)
    if args.part >= len(parts):
        print(f"Error: part {args.part} is out of range (0-{len(parts) - 1})")
        return

    current_part = parts[args.part]
    print(f"Processing part {args.part} with {len(current_part)} rows")

    gpu_memory_mode = "model_full_load"
    ulysses_degree = args.ulysses_degree
    ring_degree = args.ring_degree
    fsdp_dit = False
    fsdp_text_encoder = False
    compile_dit = False

    enable_teacache = True
    teacache_threshold = 0.05
    num_skip_start_steps = 5
    teacache_offload = False

    cfg_skip_ratio = 0
    enable_riflex = False
    riflex_k = 6
    sampler_name = "Flow_Unipc"
    shift = 16
    weight_dtype = torch.bfloat16

    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(args.config_path)

    transformer_additional_kwargs = OmegaConf.to_container(config["transformer_additional_kwargs"])
    if args.geoada_in_dim is not None:
        transformer_additional_kwargs["geoada_in_dim"] = args.geoada_in_dim

    if args.transformer_path is not None and Path(args.transformer_path).is_dir():
        print(f"Loading transformer from checkpoint directory: {args.transformer_path}")
        transformer = VerseCrafterWanTransformer3DModel.from_pretrained(
            args.transformer_path,
            transformer_additional_kwargs=transformer_additional_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = VerseCrafterWanTransformer3DModel.from_pretrained(
            str(Path(args.model_name) / config["transformer_additional_kwargs"].get("transformer_subpath", "transformer")),
            transformer_additional_kwargs=transformer_additional_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=weight_dtype,
        )

        if args.transformer_path is not None:
            print(f"Loading transformer weights from checkpoint: {args.transformer_path}")
            if args.transformer_path.endswith("safetensors"):
                from safetensors.torch import load_file

                state_dict = load_file(args.transformer_path)
            else:
                state_dict = torch.load(args.transformer_path, map_location="cpu")
            state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

            missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
            print(f"missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")

    print("Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        str(Path(args.model_name) / config["vae_kwargs"].get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config["vae_kwargs"]),
    ).to(weight_dtype)

    if args.vae_path is not None:
        print(f"Loading VAE checkpoint from: {args.vae_path}")
        if args.vae_path.endswith("safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(args.vae_path)
        else:
            state_dict = torch.load(args.vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        missing_keys, unexpected_keys = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_name) / config["text_encoder_kwargs"].get("tokenizer_subpath", "tokenizer"))
    )

    print("Loading text encoder...")
    text_encoder = WanT5EncoderModel.from_pretrained(
        str(Path(args.model_name) / config["text_encoder_kwargs"].get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config["text_encoder_kwargs"]),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    chosen_scheduler = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name in {"Flow_Unipc", "Flow_DPM++"}:
        config["scheduler_kwargs"]["shift"] = 1
    scheduler = chosen_scheduler(**filter_kwargs(chosen_scheduler, OmegaConf.to_container(config["scheduler_kwargs"])))

    print("Creating inference pipeline...")
    pipeline = WanVerseCrafterPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
    )

    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial

        transformer.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)
            print("Enabled FSDP for DIT")
        if fsdp_text_encoder:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.text_encoder = shard_fn(pipeline.text_encoder)
            print("Enabled FSDP for text encoder")

    if compile_dit:
        for i in range(len(pipeline.transformer.blocks)):
            pipeline.transformer.blocks[i] = torch.compile(pipeline.transformer.blocks[i])
        print("Enabled torch.compile")

    if gpu_memory_mode == "sequential_cpu_offload":
        replace_parameters_by_name(transformer, ["modulation"], device=device)
        transformer.freqs = transformer.freqs.to(device=device)
        pipeline.enable_sequential_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_cpu_offload":
        pipeline.enable_model_cpu_offload(device=device)
    elif gpu_memory_mode == "model_full_load_and_qfloat8":
        convert_model_weight_to_float8(transformer, exclude_module_name=["modulation"], device=device)
        convert_weight_dtype_wrapper(transformer, weight_dtype)
        pipeline.to(device=device)
    else:
        pipeline.to(device=device)

    coefficients = TEACACHE_COEFFICIENTS if enable_teacache else None
    if coefficients is not None:
        print(
            f"Enabling TeaCache with threshold {teacache_threshold}, "
            f"skip first {num_skip_start_steps} steps."
        )
        pipeline.transformer.enable_teacache(
            coefficients,
            args.num_inference_steps,
            teacache_threshold,
            num_skip_start_steps=num_skip_start_steps,
            offload=teacache_offload,
        )

    if cfg_skip_ratio is not None:
        print(f"Enabling cfg_skip_ratio {cfg_skip_ratio}.")
        pipeline.transformer.enable_cfg_skip(cfg_skip_ratio, args.num_inference_steps)

    if args.lora_path is not None:
        pipeline = merge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)

    for idx, (_, row) in enumerate(current_part.iterrows()):
        video_name = str(row.get("clipPath", "unknown")).replace(".mp4", "")
        try:
            clip_path = str(row["clipPath"])
            clip_stem = Path(clip_path).stem
            parts = clip_stem.split("_")
            video_id = "_".join(parts[:-4]) if len(parts) > 4 else clip_stem
            crowd_density = str(row["crowdDensity"]).strip()

            generated_video_path = output_dir / f"{clip_stem}_generated.mp4"
            comparison_video_path = output_dir / f"{clip_stem}_comparison.mp4"
            if generated_video_path.exists() and comparison_video_path.exists():
                print(f"Skipping existing outputs: {clip_stem}")
                continue

            print(f"\nProcessing {idx + 1}/{len(current_part)}: {clip_stem}")
            data_dir = Path(args.data_dir)
            annotation_dir = data_dir / f"vstreams_{crowd_density}_clip_annotation" / video_id / clip_stem
            required_files = {
                "gt_video": data_dir / f"vstreams_{crowd_density}_clip" / video_id / Path(clip_path).name,
                "background_RGB": annotation_dir / "background_RGB.mp4",
                "background_depth": annotation_dir / "background_depth.mp4",
                "3D_gaussian_RGB": annotation_dir / "3D_gaussian_RGB.mp4",
                "3D_gaussian_depth": annotation_dir / "3D_gaussian_depth.mp4",
                "merged_mask": annotation_dir / "merged_mask.mp4",
            }

            missing_files = [name for name, path in required_files.items() if not path.exists()]
            if missing_files:
                print(f"Skipping {clip_stem}: missing files {missing_files}")
                continue

            sample_size = [args.height, args.width]
            video_length = args.video_length

            with torch.no_grad():
                if video_length != 1:
                    ratio = vae.config.temporal_compression_ratio
                    video_length = int((video_length - 1) // ratio * ratio) + 1
                latent_frames = (video_length - 1) // vae.config.temporal_compression_ratio + 1

                if enable_riflex:
                    pipeline.transformer.enable_riflex(k=riflex_k, L_test=latent_frames)

                gt_video, _, _, _ = get_video_to_video_latent(
                    str(required_files["gt_video"]),
                    video_length=video_length,
                    sample_size=sample_size,
                    fps=args.fps,
                    ref_image=None,
                )

                control_filenames = ["background_RGB", "background_depth", "3D_gaussian_RGB", "3D_gaussian_depth"]
                control_videos = []
                for control_name in control_filenames:
                    input_video, _, _, _ = get_video_to_video_latent(
                        str(required_files[control_name]),
                        video_length=video_length,
                        sample_size=sample_size,
                        fps=args.fps,
                        ref_image=None,
                    )
                    control_videos.append(input_video)

                input_video_mask, _, _, _ = get_video_to_video_latent(
                    str(required_files["merged_mask"]),
                    video_length=video_length,
                    sample_size=sample_size,
                    fps=args.fps,
                    ref_image=None,
                )
                input_video_mask = input_video_mask[:, :1]
                input_video_mask[:, :, 0] = 0.0

                control_videos[0][:, :, 0] = gt_video[:, :, 0]
                generator = torch.Generator(device=device).manual_seed(args.seed)
                prompt = row.get("qwen_prompt", row.get("prompt", ""))

                sample = pipeline(
                    prompt,
                    num_frames=video_length,
                    negative_prompt=args.negative_prompt,
                    height=sample_size[0],
                    width=sample_size[1],
                    generator=generator,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    video=None,
                    mask_video=input_video_mask,
                    control_video=control_videos,
                    subject_ref_images=None,
                    shift=shift,
                    geoada_context_scale=args.geoada_context_scale,
                ).videos

                video_frames = sample[0]
                if video_frames.shape[0] == 3:
                    video_frames = video_frames.permute(1, 2, 3, 0)
                video_frames_rgb = (video_frames.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                frames_to_video(video_frames_rgb, str(generated_video_path), fps=args.fps)
                print(f"Saved generated video: {generated_video_path}")

                videos_dict = {
                    "background_RGB": str(required_files["background_RGB"]),
                    "3D_gaussian_RGB": str(required_files["3D_gaussian_RGB"]),
                    "background_depth": str(required_files["background_depth"]),
                    "3D_gaussian_depth": str(required_files["3D_gaussian_depth"]),
                    "merged_mask": str(required_files["merged_mask"]),
                    "generated_video": video_frames_rgb,
                    "gt_video": str(required_files["gt_video"]),
                }
                create_comparison_grid(
                    videos_dict,
                    str(comparison_video_path),
                    fps=args.fps,
                    max_frames_limit=video_length,
                    target_height=args.height,
                    target_width=args.width,
                )
                print(f"Saved comparison video: {comparison_video_path}")
        except Exception as exc:
            traceback.print_exc()
            print(f"Error when processing {video_name}: {exc}")
            continue

    if args.lora_path is not None:
        pipeline = unmerge_lora(pipeline, args.lora_path, args.lora_weight, device=device, dtype=weight_dtype)

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
