#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import uuid
import json
import argparse
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from peft import PeftModel
from transformers import AutoProcessor

from qwen_vl_utils import process_vision_info
from univa.utils.denoiser_prompt_embedding_flux import encode_prompt
from univa.utils.flux_pipeline import FluxKontextPipeline
from univa.models.qwen2p5vl.modeling_univa_qwen2p5vl_tf_v2 import (
    TFUnivaQwen2p5VLForConditionalGeneration,
)


# -------------------------
# Determinism helpers
# -------------------------
def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic flags
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# IO helpers
# -------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def list_images(input_dir: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    # 去重 + 排序保证多机/多卡一致
    paths = sorted(list(set(paths)))
    return paths


def shard_by_gpu(items: List[str], gpu_id: int, total_gpus: int) -> List[str]:
    """
    简单的 round-robin 分片，确保每个 GPU 处理不相交子集：
    idx % total_gpus == gpu_id
    """
    return [p for i, p in enumerate(items) if (i % total_gpus) == gpu_id]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# -------------------------
# Model / weights loading
# -------------------------
def prepare_condition_images(image_paths: List[str], device: str):
    if not image_paths:
        return None
    imgs = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        imgs.append(t)
    return torch.stack(imgs).to(device=device, dtype=torch.float32)


def load_flux_weights_from_checkpoint(model, ckpt_path: str):
    """
    从给定 checkpoint 目录中抽取包含 "denoise_tower." 的权重，加载到 model。
    """
    all_files = glob.glob(os.path.join(ckpt_path, "*.safetensors")) + \
                glob.glob(os.path.join(ckpt_path, "*.bin"))

    flux_sd = {}
    for f in all_files:
        sd = load_file(f, device="cpu") if f.endswith(".safetensors") else torch.load(f, map_location="cpu")
        for k, v in sd.items():
            if "denoise_tower." in k:
                # 你的 ckpt 里是 base_model.model.xxx 的前缀，这里照你原逻辑 strip
                flux_sd[k.replace("base_model.model.", "")] = v

    if flux_sd:
        print(f"--- Attempting to load {len(flux_sd)} keys into denoise_tower ---")
        missing_keys, unexpected_keys = model.load_state_dict(flux_sd, strict=False)

        if missing_keys:
            print("\n❌ WARNING: Some weights of the model were not initialized from the checkpoint:")
            for key in missing_keys:
                if "denoise_tower" in key:
                    print(f"  - {key}")

        if unexpected_keys:
            print("\n❌ WARNING: Some weights of the checkpoint were not used:")
            for key in unexpected_keys:
                print(f"  - {key}")

        if (not missing_keys) and (not unexpected_keys):
            print("✅ All checkpoint keys were successfully matched and loaded.")

    return model


def load_hybrid_model(
    base_model_path: str,
    lora_adapter_path: Optional[str],
    flux_finetune_path: Optional[str],
    device: str,
    dtype: torch.dtype,
):
    """
    1) load base model
    2) (optional) load finetuned denoise_tower weights
    3) (optional) load + merge LoRA
    """
    print(f"--- Step 1/3: Loading Base Model from: {base_model_path} ---")
    model = TFUnivaQwen2p5VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    ).to(device)
    print("✅ Base model loaded.")

    if flux_finetune_path:
        print(f"--- Step 2/3: Loading finetuned FLUX weights from: {flux_finetune_path} ---")
        model = load_flux_weights_from_checkpoint(model, flux_finetune_path)
        print("✅ Finetuned FLUX weights loaded.")
    else:
        print("--- Step 2/3: Skipping finetuned FLUX loading (no path provided). ---")

    if lora_adapter_path:
        print(f"--- Step 3/3: Loading and merging LoRA Adapter from: {lora_adapter_path} ---")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()
        print("✅ LoRA adapter merged.")
    else:
        print("--- Step 3/3: Skipping LoRA loading (no path provided). ---")

    model.eval()
    print("\nHybrid model loading complete!")
    return model


# -------------------------
# Core inference
# -------------------------
@torch.no_grad()
def run_unified_inference_single(
    model: TFUnivaQwen2p5VLForConditionalGeneration,
    processor: AutoProcessor,
    pipe: FluxKontextPipeline,
    source_image_path: str,
    text_prompt: str,
    device: str,
    dtype: torch.dtype,
    seed: int = 42,
    joint_with_t5: bool = True,
    max_new_tokens: int = 256,
    t5_max_length: int = 256,
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 20,
    guidance_scale: float = 5.0,
) -> Tuple[str, Image.Image]:
    """
    返回 (translation_text, final_image)
    """
    # Step 0: build convo
    convo = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {
                "type": "image",
                "image": source_image_path,
                "min_pixels": 448 * 448,
                "max_pixels": 448 * 448,
            },
        ],
    }]

    chat_text = processor.apply_chat_template(
        convo, tokenize=False, add_generation_prompt=True
    )
    # 保持你原逻辑：去掉第一段 system 之类内容
    # chat_text = "<|im_end|>\n".join(chat_text.split("<|im_end|>\n")[1:])

    image_inputs, video_inputs = process_vision_info(convo)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Step 1: generate translation
    # 每张图单独 seed，保证可复现（同一张图在同卡同 seed 下）
    torch.manual_seed(seed)

    generated_ids = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        pixel_values=inputs.pixel_values.to(dtype),
        image_grid_thw=inputs.image_grid_thw,
        max_new_tokens=max_new_tokens,
        # temperature=0.3,
        # top_k=10,
        # top_p=0.7,
        # repetition_penalty=1.2,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    new_token_ids = generated_ids[0][inputs.input_ids.shape[1]:]
    translation_text = processor.tokenizer.decode(
        new_token_ids, skip_special_tokens=True
    ).strip().replace('"', '')

    # Step 2: teacher forcing labels
    original_prompt_ids = inputs.input_ids

    translation_ids = processor.tokenizer(
        translation_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device)

    teacher_forcing_input_ids = torch.cat([original_prompt_ids, translation_ids], dim=1)
    teacher_forcing_attention_mask = (teacher_forcing_input_ids != processor.tokenizer.pad_token_id).long()

    prompt_length = original_prompt_ids.shape[1]
    labels = teacher_forcing_input_ids.clone()
    labels[:, :prompt_length] = -100

    # Step 3: forward to get denoise_embeds
    denoise_out = model(
        input_ids=teacher_forcing_input_ids,
        attention_mask=teacher_forcing_attention_mask,
        pixel_values=inputs.pixel_values.to(dtype),
        image_grid_thw=inputs.image_grid_thw,
        labels=labels,
        output_type="denoise_embeds",
        use_teacher_forcing=True,
    )
    detailed_prompt_embeds = denoise_out["model_pred"]  # 你原脚本假设 key 为 model_pred

    # Step 4: pooled embeds from FLUX text encoders
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]

    prm_embeds, pooled_prompt_embeds = encode_prompt(
        text_encoders,
        tokenizers,
        # f"Replace all texts with {translation_text}",
        # f"{text_prompt.replace('Translate', 'Replace')}: '{translation_text}'",
        f'''Replace all text in the image with the exact string: "{translation_text}". Keep layout, font style, and perspective.''',
        # f'''{translation_text}''',
        t5_max_length,
        device,
        1,
    )

    prompt_embeds = torch.cat([detailed_prompt_embeds, prm_embeds], dim=1)
    # Step 5: FLUX pipeline
    condition_images = prepare_condition_images([source_image_path], device)
    generator = torch.Generator(device=device).manual_seed(seed)

    images = pipe(
        image=condition_images,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=torch.zeros_like(pooled_prompt_embeds),
        # pooled_prompt_embeds=pooled_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images

    return translation_text, images[0]


# -------------------------
# Main
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser("Batch inference for UniWorld/UniVA translation + FLUX editing")

    parser.add_argument("--base_model_path", type=str, default="./UniWorld_Kontext",
                        help="Path to the BASE UniVA model checkpoint.")
    parser.add_argument("--lora_adapter_path", type=str,
                        default="./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_7b_onlyde2en_tf_v2_de_en_test/checkpoint-10000/lora",
                        help="Path to LoRA adapter dir. Empty means no LoRA.")
    parser.add_argument("--flux_finetune_path", type=str,
                        default="./checkpoints/flux_kontext_qwenvl_stage2_256_adamw_transv_7b_onlyde2en_tf_v2_de_en_test/checkpoint-10000/univa",
                        help="Path to checkpoint containing finetuned FLUX weights. Empty means skip.")
    parser.add_argument("--flux_base_path", type=str, default="/mnt/vlm-ks3/ljh/hf-model/FLUX.1-Kontext-dev",
                        help="Path to the original FLUX pipeline.")
    parser.add_argument("--input_dir", type=str,
                        default="/mnt/vlm-ks3/ljh/data/translationV/iwslt14.de-en-images/test_de",
                        help="Directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="results/outputs_3b_tf_lora",
                        help="Directory to save outputs.")

    # -- Parallelization --
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use.")
    parser.add_argument("--total_gpus", type=int, required=True, help="Total number of GPUs.")

    # -- Inference options --
    parser.add_argument("--seed", type=int, default=42, help="Base seed.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="Compute dtype.")
    parser.add_argument("--source_language", type=str, default="German")
    parser.add_argument("--target_language", type=str, default="English")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--t5_max_length", type=int, default=256)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--joint_with_t5", action="store_true", help="Keep compatibility with your joint flag (does not change pipe inputs).")

    parser.add_argument("--skip_existing", action="store_true",
                        help="If output already exists in jsonl map, skip.")
    parser.add_argument("--limit", type=int, default=-1, help="Process at most N images on this GPU shard. -1 means no limit.")

    return parser.parse_args()


def dtype_from_str(s: str) -> torch.dtype:
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()

    # GPU binding
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = dtype_from_str(args.dtype)

    # deterministic
    set_global_seed(args.seed)

    ensure_dir(args.output_dir)
    jsonl_path = os.path.join(args.output_dir, f"results_gpu{args.gpu_id}_of_{args.total_gpus}.jsonl")

    # interpret empty paths as None
    lora_path = args.lora_adapter_path.strip() if args.lora_adapter_path is not None else ""
    lora_path = lora_path if len(lora_path) > 0 else None

    flux_ft_path = args.flux_finetune_path.strip() if args.flux_finetune_path is not None else ""
    flux_ft_path = flux_ft_path if len(flux_ft_path) > 0 else None

    # Load model & processor
    model = load_hybrid_model(
        base_model_path=args.base_model_path,
        lora_adapter_path=lora_path,
        flux_finetune_path=flux_ft_path,
        device=device,
        dtype=dtype,
    )
    processor = AutoProcessor.from_pretrained(args.base_model_path)

    # Load FLUX pipeline, plug in your denoiser
    print(f"--- Loading FLUX Pipeline from: {args.flux_base_path} ---")
    pipe = FluxKontextPipeline.from_pretrained(
        args.flux_base_path,
        transformer=model.denoise_tower.denoiser,
        torch_dtype=dtype,
    ).to(device)
    print("✅ FLUX Pipeline loaded.")

    # Build prompt
    prompt = f"Translate all {args.source_language} texts into {args.target_language}."

    # List + shard images
    all_images = list_images(args.input_dir)
    shard_images = shard_by_gpu(all_images, args.gpu_id, args.total_gpus)

    if args.limit and args.limit > 0:
        shard_images = shard_images[: args.limit]

    print(f"[INFO] Found {len(all_images)} images in input_dir.")
    print(f"[INFO] GPU shard {args.gpu_id}/{args.total_gpus} will process {len(shard_images)} images.")

    # Optional: skip already processed by reading jsonl
    done_set = set()
    if args.skip_existing and os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done_set.add(obj.get("input_image", ""))
                except Exception:
                    continue
        print(f"[INFO] skip_existing enabled, loaded {len(done_set)} done entries from {jsonl_path}")

    # Run
    with open(jsonl_path, "a", encoding="utf-8") as fout:
        for idx, img_path in enumerate(shard_images):
            if args.skip_existing and img_path in done_set:
                print(f"[SKIP] ({idx+1}/{len(shard_images)}) already done: {img_path}")
                continue

            # 你可以选择 per-sample seed：base_seed + global_index
            # 为保证多 GPU 分片一致性，我们用 all_images 的 index 做稳定 seed
            global_index = all_images.index(img_path)
            sample_seed = args.seed

            try:
                translation, out_img = run_unified_inference_single(
                    model=model,
                    processor=processor,
                    pipe=pipe,
                    source_image_path=img_path,
                    text_prompt=prompt,
                    device=device,
                    dtype=dtype,
                    seed=sample_seed,
                    joint_with_t5=True,
                    max_new_tokens=args.max_new_tokens,
                    t5_max_length=args.t5_max_length,
                    height=args.height,
                    width=args.width,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                )

                out_name = f"{os.path.splitext(os.path.basename(img_path))[0]}.jpg"
                out_path = os.path.join(args.output_dir, out_name)
                out_img.save(out_path)

                rec = {
                    "input_image": img_path,
                    "output_image": out_path,
                    "translation": translation,
                    "prompt": prompt,
                    "seed": sample_seed,
                    "gpu_id": args.gpu_id,
                    "total_gpus": args.total_gpus,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                print(f"[OK] ({idx+1}/{len(shard_images)}) {img_path}")
                print(f"     translation: {translation}")
                print(f"     saved: {out_path}")

            except Exception as e:
                err = {
                    "input_image": img_path,
                    "error": repr(e),
                    "gpu_id": args.gpu_id,
                    "total_gpus": args.total_gpus,
                }
                fout.write(json.dumps(err, ensure_ascii=False) + "\n")
                fout.flush()
                print(f"[ERR] ({idx+1}/{len(shard_images)}) {img_path} -> {e}")


if __name__ == "__main__":
    main()
