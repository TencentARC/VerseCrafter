#!/usr/bin/env bash

# Set SPATIALVID_ROOT when running this script, for example:
# cd data_processing
# SPATIALVID_ROOT=/path/to/SpatialVID-HQ bash build_versecontrol4d_spatialvid.sh
SPATIALVID_ROOT="${SPATIALVID_ROOT:-/path/to/SpatialVID-HQ}"
if [[ "${SPATIALVID_ROOT}" == "/path/to/SpatialVID-HQ" ]]; then
  echo "Please set SPATIALVID_ROOT to your local SpatialVID-HQ path before running."
  exit 1
fi

# Step 0: Split SpatialVID metadata by crowdDensity (Sparse/Moderate).
SPATIALVID_METADATA_DIR="${SPATIALVID_ROOT}/data/train"
SPATIALVID_VIDEO_ROOT="${SPATIALVID_ROOT}/SpatialVid/HQ"
INPUT_CSV="${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata.csv"

python 0_filter_by_crowdDensity.py \
  --input_csv "${INPUT_CSV}" \
  --crowd_density Sparse

python 0_filter_by_crowdDensity.py \
  --input_csv "${INPUT_CSV}" \
  --crowd_density Moderate

# Step 1: Run SceneDetect for SpatialVID clip extraction.
python 1_run_scenedetect.py \
  --csv_path "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Moderate.csv" \
  --video_root_dir "${SPATIALVID_VIDEO_ROOT}" \
  --num_workers 64 \
  --frame_skip 2 \
  --start_remove_frames 75 \
  --end_remove_frames 75 \
  --min_frames 81

python 1_run_scenedetect.py \
  --csv_path "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Sparse.csv" \
  --video_root_dir "${SPATIALVID_VIDEO_ROOT}" \
  --num_workers 64 \
  --frame_skip 2 \
  --start_remove_frames 75 \
  --end_remove_frames 75 \
  --min_frames 81

# Step 2: Filter SpatialVID clips with Grounded-SAM2.
python 2_filter_spatialvid_with_grounded_sam2.py \
  --scenedetect_csv "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Moderate_timestamp.csv" \
  --video_root "${SPATIALVID_VIDEO_ROOT}/" \
  --output_clip_dir "${SPATIALVID_VIDEO_ROOT}/Moderate_clip" \
  --output_csv "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Moderate_clippath_0.csv" \
  --num_parts 1 \
  --part_idx 0

python 2_filter_spatialvid_with_grounded_sam2.py \
  --scenedetect_csv "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Sparse_timestamp.csv" \
  --video_root "${SPATIALVID_VIDEO_ROOT}/" \
  --output_clip_dir "${SPATIALVID_VIDEO_ROOT}/Sparse_clip" \
  --output_csv "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Sparse_clippath_0.csv" \
  --num_parts 1 \
  --part_idx 0

# Step 3: Use https://github.com/zbw001/TAPIP3D to predict depth, camera pose, and export NPZ files.

# Step 4: Render control maps for clips.
python render_video_pipeline_spatialvid.py \
  --max_samples 1000 \
  --num_parts 1 \
  --part_idx 0 \
  --csv_path "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Sparse_clippath_0.csv" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --output_root "${SPATIALVID_ROOT}/clip_annotation"

python render_video_pipeline_spatialvid.py \
  --max_samples 1000 \
  --num_parts 1 \
  --part_idx 0 \
  --csv_path "${SPATIALVID_METADATA_DIR}/SpatialVID_HQ_metadata_Moderate_clippath_0.csv" \
  --video_root "${SPATIALVID_VIDEO_ROOT}" \
  --output_root "${SPATIALVID_ROOT}/clip_annotation"