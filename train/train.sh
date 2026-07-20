#!/usr/bin/env bash
# Multi-node training launcher for VerseCrafter (GeoAdapter on a frozen Wan2.1-T2V-14B backbone).
#
# Usage (run from anywhere; the script cd's to the repo root):
#   bash train/train.sh
#
# For multi-node training, set MASTER_ADDR / WORLD_SIZE / NUM_PROCESS / RANK
# (per machine) as environment variables before launching.

cd "$(dirname "$0")/.." || exit 1
set -Eeuo pipefail

# ---- NCCL (edit NCCL_SOCKET_IFNAME to match your network interfaces) ----
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=brainpf0,brainpf1,brainpf2,brainpf3,brainpf4,brainpf5,brainpf6,brainpf7
export NCCL_P2P_LEVEL=NVL
# export NCCL_DEBUG=INFO   # enable when debugging

# ---- Distributed launch config (override via environment variables) ----
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # rank-0 machine IP (set for multi-node)
export MASTER_PORT="${MASTER_PORT:-29500}"
export WORLD_SIZE="${WORLD_SIZE:-1}"             # number of machines
export NUM_PROCESS="${NUM_PROCESS:-8}"           # total processes = WORLD_SIZE * GPUs_per_node
export RANK="${RANK:-0}"                         # rank of THIS machine (0 .. WORLD_SIZE-1)

# ---- Paths (edit to match your setup) ----
export MODEL_NAME="model/Wan2.1-T2V-14B"
export DATASET_NAME="dataset"
export DATASET_META_NAME="dataset/sekai_and_spatialvid_merged_train.csv"

# ---- Activate your Python environment (edit path / env name) ----
# source /path/to/anaconda3/bin/activate
# conda activate videoxfun

accelerate launch --mixed_precision="bf16" --main_process_ip=$MASTER_ADDR --main_process_port=$MASTER_PORT --num_machines=$WORLD_SIZE --num_processes=$NUM_PROCESS --machine_rank=$RANK \
    --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json --deepspeed_multinode_launcher standard \
    train/train.py \
    --config_path="config/wan2.1/wan_civitai.yaml" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$DATASET_NAME \
    --train_data_meta=$DATASET_META_NAME \
    --image_sample_size=720 \
    --video_sample_size 720 1280 \
    --token_sample_size=512 \
    --video_sample_stride=1 \
    --video_sample_n_frames=81 \
    --train_batch_size=1 \
    --video_repeat=0 \
    --gradient_accumulation_steps=1 \
    --dataloader_num_workers=4 \
    --num_train_epochs=100 \
    --checkpointing_steps=50 \
    --learning_rate=2e-05 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=100 \
    --seed=42 \
    --output_dir="results/exp_ours_14B" \
    --gradient_checkpointing \
    --adam_weight_decay=3e-2 \
    --adam_epsilon=1e-10 \
    --vae_mini_batch=1 \
    --max_grad_norm=0.05 \
    --geoada_in_dim=128 \
    --uniform_sampling \
    --control_ref_image="first_frame" \
    --trainable_modules="geoada" \
    --low_vram \
    --use_deepspeed \
    --resume_from_checkpoint="latest" \
    --report_to="tensorboard" \
    --max_train_steps=5000 \
    --validation_steps=50 \
    --validation_paths="3CRJWy5zXvg_0139474_0141274_0001576_0001657" \
    --validation_prompts="The video shows the interior of a well-lit shopping mall with a modern and inviting atmosphere. The ceiling is high, with recessed lighting that illuminates the space evenly. On the left side of the frame, there are glass storefronts displaying various items, possibly from a store selling home goods or kitchenware. The right side features a food kiosk with a sign that reads Evie's Cookies, showcasing an assortment of baked goods behind a glass display case. A large cookie prop stands next to the sign, adding a playful touch.  In the foreground, two individuals are walking away from the camera. One person is wearing a black shirt with a graphic design on the back and blue pants, while the other is dressed in a striped shirt and dark pants. They appear to be casually strolling through the mall, possibly window shopping or heading towards another destination within the complex. The overall ambiance suggests a relaxed and pleasant shopping experience."
