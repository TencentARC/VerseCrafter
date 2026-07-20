#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   conda activate versecrafter
#   bash batch_inference/run_batch_inference_spatialvid.sh 0
#
# The first argument is the part index (0-based) used with --num-parts.

PART_NUM=${1:-}
if [[ -z "${PART_NUM}" ]]; then
  echo "Usage: bash batch_inference/run_batch_inference_spatialvid.sh <part_idx>"
  exit 1
fi

torchrun --nproc-per-node=8 --master-port 29511 batch_inference/predict_v2v_control_spatialvid_batch.py \
  --csv /path/to/spatialvid_test_split.csv \
  --data-dir /path/to/SpatialVID-HQ/SpatialVid/HQ \
  --annotation-dir /path/to/SpatialVID-HQ/clip_annotation \
  --output-dir samples/batch_inference/spatialvid \
  --height 720 \
  --width 1280 \
  --num-inference-steps 50 \
  --num-parts 1 \
  --part "${PART_NUM}" \
  --model-name model/Wan2.1-T2V-14B \
  --transformer-path model/VerseCrafter \
  --ulysses-degree 2 \
  --ring-degree 4 \
  --seed 42

echo "SpatialVID batch inference started for part ${PART_NUM}."
