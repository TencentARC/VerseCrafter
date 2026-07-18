#!/usr/bin/env bash

# Set SEKAI_ROOT when running this script, for example:
# cd data_processing
# SEKAI_ROOT=/path/to/sekai-codebase bash build_versecontrol4d_sekai.sh
SEKAI_ROOT="${SEKAI_ROOT:-/path/to/sekai-codebase}"
if [[ "${SEKAI_ROOT}" == "/path/to/sekai-codebase" ]]; then
  echo "Please set SEKAI_ROOT to your local sekai-codebase path before running."
  exit 1
fi

# Step 0: Split Sekai metadata by crowdDensity (scattered/moderate).
INPUT_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq.csv"
SCATTERED_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq_scattered.csv"
MODERATE_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq_moderate.csv"
SCATTERED_TIMESTAMP_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq_scattered_timestamp.csv"
MODERATE_TIMESTAMP_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq_moderate_timestamp.csv"
SCATTERED_CLIP_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq_scattered_clippath_0.csv"
MODERATE_CLIP_CSV="${SEKAI_ROOT}/train/sekai-real-walking-hq_moderate_clippath_0.csv"

python 0_filter_by_crowdDensity.py \
  --input_csv "${INPUT_CSV}" \
  --crowd_density scattered

python 0_filter_by_crowdDensity.py \
  --input_csv "${INPUT_CSV}" \
  --crowd_density moderate


# Step 1: Use https://github.com/Lixsp11/sekai-codebase for clip extraction.


# Step 2: Filter Sekai clips with Grounded-SAM2.
python 2_filter_sekai_with_grounded_sam2.py \
  --scenedetect_csv "${SCATTERED_TIMESTAMP_CSV}" \
  --video_root "${SEKAI_ROOT}/vstreams_scattered" \
  --output_clip_dir "${SEKAI_ROOT}/vstreams_scattered_clip" \
  --output_csv "${SCATTERED_CLIP_CSV}" \
  --num_parts 1 \
  --part_idx 0


python 2_filter_sekai_with_grounded_sam2.py \
  --scenedetect_csv "${MODERATE_TIMESTAMP_CSV}" \
  --video_root "${SEKAI_ROOT}/vstreams_moderate" \
  --output_clip_dir "${SEKAI_ROOT}/vstreams_moderate_clip" \
  --output_csv "${MODERATE_CLIP_CSV}" \
  --num_parts 1 \
  --part_idx 0

# Step 3: Use https://github.com/zbw001/TAPIP3D to predict depth, camera pose, and export NPZ files.

# Step 4: Render control maps for clips.
python render_video_pipeline_sekai.py \
  --max_samples 1000 \
  --num_parts 1 \
  --part_idx 0 \
  --csv_path "${SCATTERED_CLIP_CSV}" \
  --video_root "${SEKAI_ROOT}" \
  --output_root "${SEKAI_ROOT}/vstreams_scattered_clip_annotation"

python render_video_pipeline_sekai.py \
  --max_samples 1000 \
  --num_parts 1 \
  --part_idx 0 \
  --csv_path "${MODERATE_CLIP_CSV}" \
  --video_root "${SEKAI_ROOT}" \
  --output_root "${SEKAI_ROOT}/vstreams_moderate_clip_annotation"