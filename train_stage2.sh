
export WANDB_MODE="disabled"
export WANDB_API_KEY=""

export TOKENIZERS_PARALLELISM=true
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}
NUM_PROCESSES=$((8 * WORLD_SIZE))

accelerate launch \
  --config_file scripts/accelerate_configs/multi_node_example_zero2.yaml \
  --main_process_ip ${MASTER_ADDR} \
  --main_process_port ${MASTER_PORT} \
  --machine_rank ${RANK} \
  --num_machines ${WORLD_SIZE} \
  --num_processes ${NUM_PROCESSES} \
  train_denoiser_tf_mask.py scripts/denoiser/flux_qwen2p5vl_7b_vlm_stage2_224_3b_tf_finetune_mask_all.yaml