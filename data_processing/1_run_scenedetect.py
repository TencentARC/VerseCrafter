import argparse
import logging
import os
from multiprocessing import Pool
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from scenedetect import ContentDetector, detect
from tqdm import tqdm


def configure_logging(log_level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


configure_logging()

DEFAULT_VIDEO_PATH_COLUMNS = [
    "video path",
    "video_path",
    "videoFile",
    "video",
    "clipPath",
]


def infer_video_path_column(df: pd.DataFrame, user_specified: Optional[str]) -> str:
    if user_specified:
        if user_specified not in df.columns:
            raise ValueError(f"Specified video path column '{user_specified}' not found in CSV columns: {list(df.columns)}")
        return user_specified

    for col in DEFAULT_VIDEO_PATH_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not infer video path column. Tried {DEFAULT_VIDEO_PATH_COLUMNS}, available columns: {list(df.columns)}"
    )


def resolve_video_path(video_root_dir: str, video_path_value: str) -> str:
    video_path_value = str(video_path_value).strip()
    if os.path.isabs(video_path_value):
        return video_path_value
    return os.path.join(video_root_dir, video_path_value)


def scene_to_tuple(
    start_tc: Any,
    end_tc: Any,
    start_remove_frames: int,
    end_remove_frames: int,
    min_frames: int,
) -> Optional[Tuple[float, float, int, int, str, str, float]]:
    start_frame = int(start_tc.get_frames()) + start_remove_frames
    end_frame = int(end_tc.get_frames()) - end_remove_frames
    if end_frame - start_frame < min_frames:
        return None

    framerate = float(start_tc.framerate)
    start_seconds = start_frame / framerate
    end_seconds = end_frame / framerate

    start_timecode = str(start_tc + start_remove_frames)
    end_timecode = str(end_tc - end_remove_frames)

    return (
        start_seconds,
        end_seconds,
        start_frame,
        end_frame,
        start_timecode,
        end_timecode,
        framerate,
    )


def detect_timestamps_for_video(
    video_path: str,
    frame_skip: int,
    start_remove_frames: int,
    end_remove_frames: int,
    min_frames: int,
) -> Tuple[List[Tuple[float, float, int, int, str, str, float]], Optional[str]]:
    if not os.path.exists(video_path):
        return [], f"video not found: {video_path}"

    try:
        scene_list = detect(
            video_path,
            ContentDetector(),
            frame_skip=frame_skip,
            start_in_scene=True,
        )
    except Exception as exc:
        return [], f"scenedetect failed: {exc}"

    timestamps: List[Tuple[float, float, int, int, str, str, float]] = []
    for start_tc, end_tc in scene_list:
        scene_tuple = scene_to_tuple(
            start_tc=start_tc,
            end_tc=end_tc,
            start_remove_frames=start_remove_frames,
            end_remove_frames=end_remove_frames,
            min_frames=min_frames,
        )
        if scene_tuple is not None:
            timestamps.append(scene_tuple)
    return timestamps, None


def worker_process(item: Dict[str, Any]) -> Dict[str, Any]:
    idx = item["idx"]
    video_path = item["video_path"]
    timestamps, err = detect_timestamps_for_video(
        video_path=video_path,
        frame_skip=item["frame_skip"],
        start_remove_frames=item["start_remove_frames"],
        end_remove_frames=item["end_remove_frames"],
        min_frames=item["min_frames"],
    )
    return {"idx": idx, "timestamps": timestamps, "error": err}


def build_output_csv_path(input_csv_path: str, output_csv_path: Optional[str]) -> str:
    if output_csv_path:
        return output_csv_path
    stem, ext = os.path.splitext(input_csv_path)
    return f"{stem}_timestamp{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SceneDetect and generate timestamp CSV.")
    parser.add_argument("--csv_path", type=str, required=True, help="Input metadata CSV path.")
    parser.add_argument("--video_root_dir", type=str, required=True, help="Root directory for videos.")
    parser.add_argument("--output_csv", type=str, default=None, help="Output CSV path. Defaults to <input>_timestamp.csv")
    parser.add_argument("--video_path_column", type=str, default=None, help="Column name containing relative/absolute video path.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for multiprocessing.")
    parser.add_argument("--frame_skip", type=int, default=2, help="Frame skip for SceneDetect.")
    parser.add_argument("--start_remove_frames", type=int, default=0, help="Frames removed from each scene start.")
    parser.add_argument("--end_remove_frames", type=int, default=0, help="Frames removed from each scene end.")
    parser.add_argument("--min_frames", type=int, default=81, help="Minimum valid segment length in frames.")
    parser.add_argument("--disable_parallel", action="store_true", help="Disable multiprocessing and run single-process.")
    parser.add_argument("--output_dir", type=str, default=None, help="Reserved for compatibility; currently unused.")
    args = parser.parse_args()

    output_csv = build_output_csv_path(args.csv_path, args.output_csv)
    logging.info(f"Loading CSV: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    if len(df) == 0:
        logging.warning("Input CSV is empty. Writing empty output.")
        df["timestamp"] = []
        df.to_csv(output_csv, index=False)
        return

    video_col = infer_video_path_column(df, args.video_path_column)
    logging.info(f"Using video path column: {video_col}")

    tasks: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        tasks.append(
            {
                "idx": idx,
                "video_path": resolve_video_path(args.video_root_dir, row[video_col]),
                "frame_skip": args.frame_skip,
                "start_remove_frames": args.start_remove_frames,
                "end_remove_frames": args.end_remove_frames,
                "min_frames": args.min_frames,
            }
        )

    results: List[Dict[str, Any]] = []
    if args.disable_parallel or args.num_workers <= 1:
        for task in tqdm(tasks, desc="SceneDetect", dynamic_ncols=True):
            results.append(worker_process(task))
    else:
        with Pool(processes=args.num_workers) as pool:
            for out in tqdm(pool.imap_unordered(worker_process, tasks), total=len(tasks), desc="SceneDetect", dynamic_ncols=True):
                results.append(out)

    timestamps_col: List[str] = ["[]"] * len(df)
    error_count = 0
    valid_scene_rows = 0

    for res in results:
        idx = res["idx"]
        timestamps = res["timestamps"]
        err = res["error"]
        timestamps_col[idx] = str(timestamps)
        if err:
            error_count += 1
            logging.warning(f"[row {idx}] {err}")
        if len(timestamps) > 0:
            valid_scene_rows += 1

    out_df = df.copy()
    out_df["timestamp"] = timestamps_col
    out_df.to_csv(output_csv, index=False)

    logging.info("=" * 80)
    logging.info(f"Saved output CSV: {output_csv}")
    logging.info(f"Total rows: {len(df)}")
    logging.info(f"Rows with >=1 valid segment: {valid_scene_rows}")
    logging.info(f"Rows with errors: {error_count}")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()
