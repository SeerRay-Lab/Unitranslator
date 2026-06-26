#!/bin/bash
set -euo pipefail

echo "开始执行任务 (DE -> EN 推理 + 评测)"

# -----------------------------
# Config
# -----------------------------
TOTAL_GPUS=8
OUTPUT_DIR="results/outputs_3b_tf_mask_en"
JSONL_DIR="results/jsonl_tf_mask_en"

INPUT_DIR="/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/test_de"
REF_DIR="/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/test_en"

SOURCE_LANGUAGE="German"
TARGET_LANGUAGE="English"


BASE_MODEL_PATH="UniWorld_Kontext_3b_TF"
FLUX_FINETUNE_PATH="./checkpoint-10000/univa"
LORA_ADAPTER_PATH="./checkpoint-10000/lora"


# -----------------------------
# Prepare dirs
# -----------------------------
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${JSONL_DIR}"

# -----------------------------
# Launch inference (0..7)
# -----------------------------
for GPU_ID in $(seq 0 $((TOTAL_GPUS - 1))); do
  echo "Launching inference on GPU ${GPU_ID}/${TOTAL_GPUS} ..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    python infer_dir_tf_256.py \
      --gpu_id "${GPU_ID}" \
      --total_gpus "${TOTAL_GPUS}" \
      --output_dir "${OUTPUT_DIR}" \
      --base_model_path "${BASE_MODEL_PATH}" \
      --flux_finetune_path "${FLUX_FINETUNE_PATH}" \
      --lora_adapter_path "${LORA_ADAPTER_PATH}" \
      --input_dir "${INPUT_DIR}" \
      --source_language "${SOURCE_LANGUAGE}" \
      --target_language "${TARGET_LANGUAGE}" \
    &
done

wait
echo "All inference jobs completed."

# -----------------------------
# Collect jsonl
# -----------------------------
shopt -s nullglob
jsonl_files=("${OUTPUT_DIR}"/*.jsonl)
if (( ${#jsonl_files[@]} > 0 )); then
  mv "${OUTPUT_DIR}"/*.jsonl "${JSONL_DIR}/"
  echo "Moved ${#jsonl_files[@]} jsonl files to ${JSONL_DIR}"
else
  echo "No jsonl files found under ${OUTPUT_DIR}"
fi
shopt -u nullglob

# -----------------------------
# EasyOCR cache (保持你原逻辑)
# -----------------------------
if [ -d ".EasyOCR" ]; then
  cp -r .EasyOCR/ /root/
else
  echo "Warning: .EasyOCR not found, skip copying."
fi

# -----------------------------
# Eval (DE)
# -----------------------------
python eval/structure_bleu_v2.py \
  --generate_dir "${OUTPUT_DIR}/" \
  --ref_dir "${REF_DIR}" \
  --lang en

echo "Done."
